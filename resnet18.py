# -*- coding: utf-8 -*-
"""
Urban Sound Classification with ResNet-18
Clean implementation inspired by cnn.py

Key features:
- ResNet-18 architecture (18 layers)
- SE attention blocks
- SpecAugment + Mixup
- Cosine annealing LR
- Label smoothing
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/scratch/tl3735/MLFA/Kaggle_Data")
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"
TEST_AUDIO_DIR = AUDIO_DIR / "test"
CACHE_DIR = BASE_DIR / "mel_cache"

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_CONFIG = {
    "sr": 22050, "n_mels": 128, "n_fft": 2048,
    "hop_length": 512, "duration": 4.0, "fmax": 8000,
}

# =============================================================================
# Feature Extraction (3-channel: Mel + Delta + Delta-Delta)
# =============================================================================

def extract_mel_3ch(audio_path, config=AUDIO_CONFIG):
    """
    Convert audio to 3-channel feature:
    - Channel 0: Log-Mel spectrogram
    - Channel 1: Delta (1st derivative) 
    - Channel 2: Delta-Delta (2nd derivative)
    
    Peak normalization: ref=np.max
    """
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    # Pad/truncate to fixed duration
    target_len = int(config["sr"] * config["duration"])
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))))
    
    # Compute mel spectrogram with peak normalization
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Peak normalization
    
    # Compute Delta and Delta-Delta
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    
    # Normalize each channel to [0, 1]
    def normalize(x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
    
    # Stack into 3 channels: (3, n_mels, time)
    features = np.stack([normalize(mel_db), normalize(delta), normalize(delta2)], axis=0)
    return features


def extract_mel_1ch(audio_path, config=AUDIO_CONFIG):
    """Single channel version for comparison."""
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    target_len = int(config["sr"] * config["duration"])
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))))
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

# =============================================================================
# Data Augmentation
# =============================================================================

class SpecAugment:
    """Time and frequency masking for spectrograms (supports multi-channel)."""
    def __init__(self, freq_mask=15, time_mask=25, n_freq=2, n_time=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time
    
    def __call__(self, x):
        # x: (C, H, W) or (H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        c, n_mels, time_steps = x.shape
        out = x.clone()
        
        # Apply same mask to all channels
        for _ in range(self.n_freq):
            f = random.randint(0, min(self.freq_mask, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            out[:, f0:f0+f, :] = 0
        
        for _ in range(self.n_time):
            t = random.randint(0, min(self.time_mask, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            out[:, :, t0:t0+t] = 0
        
        return out


class TimeShift:
    """Random time shift augmentation."""
    def __init__(self, max_shift=0.2):
        self.max_shift = max_shift  # Fraction of total time
    
    def __call__(self, x):
        # x: (C, H, W) or (H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        _, _, time_steps = x.shape
        shift = int(time_steps * self.max_shift * (random.random() * 2 - 1))
        return torch.roll(x, shifts=shift, dims=-1)


class AudioAugment:
    """Combined augmentation: SpecAugment + TimeShift."""
    def __init__(self):
        self.spec_aug = SpecAugment(freq_mask=15, time_mask=25, n_freq=2, n_time=2)
        self.time_shift = TimeShift(max_shift=0.1)
    
    def __call__(self, x):
        # Apply time shift (30% chance)
        if random.random() < 0.3:
            x = self.time_shift(x)
        # Apply spec augment (50% chance)
        if random.random() < 0.5:
            x = self.spec_aug(x)
        return x


def mixup(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1
    
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

# =============================================================================
# Dataset (3-channel support)
# =============================================================================

class AudioDataset(Dataset):
    """Dataset with 3-channel features and augmentation."""
    
    def __init__(self, csv_path, audio_dir, folds=None, cache_dir=None, 
                 augment=False, use_3ch=True):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        if folds:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.augment = augment
        self.use_3ch = use_3ch
        self.aug = AudioAugment() if augment else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold, fname, label = row["fold"], row["slice_file_name"], int(row["classID"])
        
        # Cache path (different for 1ch vs 3ch)
        suffix = "_3ch" if self.use_3ch else "_1ch"
        cache = self.cache_dir / f"fold{fold}_{Path(fname).stem}{suffix}.npy" if self.cache_dir else None
        
        if cache and cache.exists():
            features = np.load(cache)
        else:
            audio_path = self.audio_dir / f"fold{fold}" / fname
            if self.use_3ch:
                features = extract_mel_3ch(audio_path)  # (3, H, W)
            else:
                features = extract_mel_1ch(audio_path)  # (H, W)
            if cache:
                np.save(cache, features)
        
        x = torch.from_numpy(features)
        if not self.use_3ch:
            x = x.unsqueeze(0)  # (1, H, W)
        
        # Apply augmentation
        if self.augment and self.aug:
            x = self.aug(x)
        
        return x, torch.tensor(label)


class TestDataset(Dataset):
    """Test dataset with 3-channel support."""
    
    def __init__(self, csv_path, audio_dir, cache_dir=None, use_3ch=True):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_3ch = use_3ch
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["slice_file_name"]
        suffix = "_3ch" if self.use_3ch else "_1ch"
        cache = self.cache_dir / f"test_{Path(fname).stem}{suffix}.npy" if self.cache_dir else None
        
        if cache and cache.exists():
            features = np.load(cache)
        else:
            if self.use_3ch:
                features = extract_mel_3ch(self.audio_dir / fname)
            else:
                features = extract_mel_1ch(self.audio_dir / fname)
            if cache:
                np.save(cache, features)
        
        x = torch.from_numpy(features)
        if not self.use_3ch:
            x = x.unsqueeze(0)
        
        return x, fname

# =============================================================================
# ResNet-18 Model
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """ResNet basic block with optional SE attention."""
    
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        
        # Shortcut for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 for audio classification.
    
    Architecture: [2, 2, 2, 2] blocks = 18 layers
    Channels: 64 -> 128 -> 256 -> 512
    
    Modifications for audio:
    - Supports 1 or 3 input channels (Mel / Mel+Delta+Delta2)
    - 3x3 initial conv (not 7x7)
    - No initial maxpool
    - SE attention in later layers
    """
    
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, use_se=True):
        super().__init__()
        
        # Initial conv (supports 1 or 3 channels)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1, use_se=False)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, use_se=use_se)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride, use_se):
        layers = [BasicBlock(in_ch, out_ch, stride, use_se)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, use_se))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """Train one epoch with optional mixup."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        # Mixup
        if use_mixup and random.random() > 0.5:
            x, y_a, y_b, lam = mixup(x, y)
            out = model(x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            correct += (lam * (out.argmax(1) == y_a).sum().item() +
                       (1 - lam) * (out.argmax(1) == y_b).sum().item())
        else:
            out = model(x)
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total += y.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            total_loss += criterion(out, y).item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    score = 0.8 * acc + 0.2 * f1
    return total_loss / total, acc, f1, score


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, 
                label_smoothing=0.1, use_mixup=True, save_path="resnet18_best.pth"):
    """Full training loop."""
    model = model.to(DEVICE)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_score = 0
    print(f"\nTraining on {DEVICE}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Epochs: {epochs} | LR: {lr} | Mixup: {use_mixup}")
    print("=" * 60)
    
    for epoch in range(epochs):
        lr_now = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} (lr={lr_now:.6f})")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, use_mixup
        )
        val_loss, val_acc, val_f1, val_score = evaluate(
            model, val_loader, val_criterion, DEVICE
        )
        
        scheduler.step()
        
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}, score={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"*** Best model saved (score={best_score:.4f}) ***")
    
    print(f"\nTraining complete! Best score: {best_score:.4f}")
    return model

# =============================================================================
# Inference
# =============================================================================

def predict(model, loader):
    """Generate predictions."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for x, fnames in tqdm(loader, desc="Predicting"):
            preds = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            results.extend(zip(fnames, preds))
    
    return results


def save_submission(predictions, path="resnet18.csv"):
    """Save predictions to CSV."""
    df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [p for _, p in predictions]
    }).astype(int)
    df.to_csv(path, index=False)
    
    print(f"\nSubmission saved: {path}")
    print(f"Predictions: {len(df)}")
    print(df["TARGET"].value_counts().sort_index())
    return df

# =============================================================================
# Main Pipeline
# =============================================================================

def run(
    train_folds=(1, 2, 3, 4, 5, 6),
    val_folds=(7, 8),
    batch_size=32,
    epochs=50,
    lr=1e-3,
    num_workers=4,
    use_se=True,
    use_mixup=True,
    label_smoothing=0.1,
    use_3ch=True,  # 3-channel features (Mel + Delta + Delta2)
):
    """Run complete pipeline: train + inference."""
    
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    in_channels = 3 if use_3ch else 1
    
    print("=" * 60)
    print("ResNet-18 Audio Classification")
    print("=" * 60)
    print(f"Train folds: {train_folds}")
    print(f"Val folds: {val_folds}")
    print(f"Features: {in_channels}-channel {'(Mel+Delta+Delta2)' if use_3ch else '(Mel only)'}")
    print(f"SE attention: {use_se}")
    print(f"Mixup: {use_mixup} | Label smoothing: {label_smoothing}")
    
    # Data loaders
    train_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, list(train_folds), CACHE_DIR, 
                     augment=True, use_3ch=use_3ch),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, list(val_folds), CACHE_DIR, 
                     augment=False, use_3ch=use_3ch),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Model
    model = ResNet18(in_channels=in_channels, num_classes=NUM_CLASSES, use_se=use_se)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Train
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                label_smoothing=label_smoothing, use_mixup=use_mixup)
    
    # Inference
    if TEST_CSV.exists():
        print("\nRunning inference...")
        model.load_state_dict(torch.load("resnet18_best.pth", map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        predictions = predict(model, test_loader)
        save_submission(predictions, "resnet18.csv")


def run_cv_ensemble(
    batch_size=32,
    epochs=60,
    lr=1e-3,
    num_workers=4,
    use_se=True,
    use_mixup=True,
    label_smoothing=0.1,
    use_3ch=True,
):
    """
    4-fold cross-validation with ensemble prediction.
    Each fold uses 2 validation folds (as requested).
    
    Fold splits:
    - Round 1: val=(1,2), train=(3,4,5,6,7,8)
    - Round 2: val=(3,4), train=(1,2,5,6,7,8)
    - Round 3: val=(5,6), train=(1,2,3,4,7,8)
    - Round 4: val=(7,8), train=(1,2,3,4,5,6)
    """
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    in_channels = 3 if use_3ch else 1
    
    # Define 4 fold splits (2 val folds each)
    cv_splits = [
        {"val": (1, 2), "train": (3, 4, 5, 6, 7, 8)},
        {"val": (3, 4), "train": (1, 2, 5, 6, 7, 8)},
        {"val": (5, 6), "train": (1, 2, 3, 4, 7, 8)},
        {"val": (7, 8), "train": (1, 2, 3, 4, 5, 6)},
    ]
    
    print("=" * 60)
    print("ResNet-18 - 4-Fold CV Ensemble (2 val folds each)")
    print("=" * 60)
    print(f"Features: {in_channels}-channel")
    print(f"Epochs per fold: {epochs}")
    print(f"Total models: {len(cv_splits)}")
    print("=" * 60)
    
    model_paths = []
    val_scores = []
    
    # Train 4 models
    for fold_idx, split in enumerate(cv_splits):
        val_folds = split["val"]
        train_folds = split["train"]
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/4")
        print(f"Val: {val_folds} | Train: {train_folds}")
        print("=" * 60)
        
        # Data loaders
        train_loader = DataLoader(
            AudioDataset(TRAIN_CSV, AUDIO_DIR, list(train_folds), CACHE_DIR,
                        augment=True, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            AudioDataset(TRAIN_CSV, AUDIO_DIR, list(val_folds), CACHE_DIR,
                        augment=False, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        # Model
        model = ResNet18(in_channels=in_channels, num_classes=NUM_CLASSES, use_se=use_se)
        save_path = f"resnet18_fold{fold_idx + 1}.pth"
        
        # Train
        train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                   label_smoothing=label_smoothing, use_mixup=use_mixup, save_path=save_path)
        
        # Load best and evaluate
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        _, val_acc, val_f1, val_score = evaluate(model, val_loader, 
                                                  nn.CrossEntropyLoss(), DEVICE)
        
        model_paths.append(save_path)
        val_scores.append(val_score)
        print(f"Fold {fold_idx + 1} best score: {val_score:.4f}")
    
    # Print CV results
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)
    print(f"Mean CV Score: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    print(f"Individual: {[f'{s:.4f}' for s in val_scores]}")
    
    # Ensemble inference
    if TEST_CSV.exists():
        print("\n" + "=" * 60)
        print("Ensemble Inference (averaging 4 models)")
        print("=" * 60)
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        # Collect predictions from all models
        all_probs = []
        file_names = None
        
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/{len(model_paths)}...")
            model = ResNet18(in_channels=in_channels, num_classes=NUM_CLASSES, use_se=use_se)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model = model.to(DEVICE)
            model.eval()
            
            probs = []
            names = []
            with torch.no_grad():
                for x, fnames in tqdm(test_loader, desc=f"Model {i+1}"):
                    out = model(x.to(DEVICE))
                    prob = F.softmax(out, dim=1)
                    probs.append(prob.cpu().numpy())
                    names.extend(fnames)
            
            all_probs.append(np.vstack(probs))
            if file_names is None:
                file_names = names
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        predictions = [(name, int(np.argmax(avg_probs[i]))) 
                      for i, name in enumerate(file_names)]
        
        # Save submission
        save_submission(predictions, "resnet18_ensemble.csv")
        print(f"\nEnsemble submission saved!")
    
    return val_scores


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cv":
        # 4-fold CV ensemble (recommended)
        run_cv_ensemble(
            batch_size=32,
            epochs=60,
            lr=1e-3,
            use_se=True,
            use_mixup=True,
            label_smoothing=0.1,
            use_3ch=True,
        )
    else:
        # Single run
        run(
            train_folds=(1, 2, 3, 4, 5, 6),
            val_folds=(7, 8),
            batch_size=32,
            epochs=60,
            lr=1e-3,
            use_se=True,
            use_mixup=True,
            label_smoothing=0.1,
            use_3ch=True,
        )

