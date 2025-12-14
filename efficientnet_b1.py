# -*- coding: utf-8 -*-
"""
Urban Sound Classification with EfficientNet-B1
Clean implementation based on resnet18.py structure

Key features:
- EfficientNet-B1 architecture (~7.8M params, efficient)
- 3-channel features (Mel + Delta + Delta-Delta)
- SpecAugment + Mixup + TimeShift
- Cosine annealing LR
- Label smoothing
- 8-fold CV with weighted ensemble
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
    """
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
    
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    
    def normalize(x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
    
    features = np.stack([normalize(mel_db), normalize(delta), normalize(delta2)], axis=0)
    return features


def extract_mel_1ch(audio_path, config=AUDIO_CONFIG):
    """Single channel version."""
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
    """Time and frequency masking for spectrograms."""
    def __init__(self, freq_mask=15, time_mask=25, n_freq=2, n_time=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time
    
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        c, n_mels, time_steps = x.shape
        out = x.clone()
        
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
        self.max_shift = max_shift
    
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, _, time_steps = x.shape
        shift = int(time_steps * self.max_shift * (random.random() * 2 - 1))
        return torch.roll(x, shifts=shift, dims=-1)


class AudioAugment:
    """Combined augmentation."""
    def __init__(self):
        self.spec_aug = SpecAugment(freq_mask=15, time_mask=25, n_freq=2, n_time=2)
        self.time_shift = TimeShift(max_shift=0.1)
    
    def __call__(self, x):
        if random.random() < 0.3:
            x = self.time_shift(x)
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
# Dataset
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
        
        suffix = "_3ch" if self.use_3ch else "_1ch"
        cache = self.cache_dir / f"fold{fold}_{Path(fname).stem}{suffix}.npy" if self.cache_dir else None
        
        if cache and cache.exists():
            features = np.load(cache)
        else:
            audio_path = self.audio_dir / f"fold{fold}" / fname
            if self.use_3ch:
                features = extract_mel_3ch(audio_path)
            else:
                features = extract_mel_1ch(audio_path)
            if cache:
                np.save(cache, features)
        
        x = torch.from_numpy(features)
        if not self.use_3ch:
            x = x.unsqueeze(0)
        
        if self.augment and self.aug:
            x = self.aug(x)
        
        return x, torch.tensor(label)


class TestDataset(Dataset):
    """Test dataset."""
    
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
# EfficientNet-B1 Model
# =============================================================================

class EfficientNetB1(nn.Module):
    """
    EfficientNet-B1 for audio classification.
    
    Architecture:
    - ~7.8M parameters (lighter than ResNet-18's ~11M)
    - Compound scaling (depth, width, resolution)
    - MBConv blocks with squeeze-and-excitation
    - Swish activation
    
    Modifications for audio:
    - Custom first conv for 1 or 3 channels
    - Custom classifier head
    """
    
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        
        # Load EfficientNet-B1 architecture (without pretrained weights)
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        
        # Create model without weights
        self.efficientnet = efficientnet_b1(weights=None)
        
        # Modify first conv to accept 1 or 3 channels (audio spectrograms)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels, 32, 
            kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Modify classifier
        # Original: Linear(1280, 1000)
        in_features = self.efficientnet.classifier[1].in_features  # 1280
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Initialize weights for the modified layers
        self._init_weights()
    
    def _init_weights(self):
        # Initialize the modified first conv
        nn.init.kaiming_normal_(
            self.efficientnet.features[0][0].weight, 
            mode='fan_out', nonlinearity='relu'
        )
        # Initialize classifier
        nn.init.normal_(self.efficientnet.classifier[1].weight, 0, 0.01)
        nn.init.constant_(self.efficientnet.classifier[1].bias, 0)
    
    def forward(self, x):
        return self.efficientnet(x)

# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """Train one epoch with optional mixup."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
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
                label_smoothing=0.1, use_mixup=True, save_path="efficientnet_b1_best.pth"):
    """Full training loop."""
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer (slightly lower LR for EfficientNet)
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


def save_submission(predictions, path="efficientnet_b1.csv"):
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
    lr=5e-4,  # Slightly lower LR for EfficientNet
    num_workers=4,
    use_mixup=True,
    label_smoothing=0.1,
    use_3ch=True,
):
    """Run complete pipeline: train + inference."""
    
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    in_channels = 3 if use_3ch else 1
    
    print("=" * 60)
    print("EfficientNet-B1 Audio Classification")
    print("=" * 60)
    print(f"Train folds: {train_folds}")
    print(f"Val folds: {val_folds}")
    print(f"Features: {in_channels}-channel {'(Mel+Delta+Delta2)' if use_3ch else '(Mel only)'}")
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
    model = EfficientNetB1(in_channels=in_channels, num_classes=NUM_CLASSES)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Train
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                label_smoothing=label_smoothing, use_mixup=use_mixup)
    
    # Inference
    if TEST_CSV.exists():
        print("\nRunning inference...")
        model.load_state_dict(torch.load("efficientnet_b1_best.pth", map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        predictions = predict(model, test_loader)
        save_submission(predictions, "efficientnet_b1.csv")


def run_cv_ensemble(
    batch_size=32,
    epochs=60,
    lr=5e-4,  # Slightly lower LR for EfficientNet
    num_workers=4,
    use_mixup=True,
    label_smoothing=0.1,
    use_3ch=True,
):
    """
    8-fold cross-validation with weighted ensemble prediction.
    """
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    in_channels = 3 if use_3ch else 1
    
    print("=" * 60)
    print("EfficientNet-B1 - 8-Fold CV Ensemble")
    print("=" * 60)
    print(f"Features: {in_channels}-channel (Mel+Delta+Delta2)")
    print(f"Epochs per fold: {epochs}")
    print(f"Total models: 8")
    print("=" * 60)
    
    model_paths = []
    val_scores = []
    
    # Train 8 models
    for val_fold in range(1, 9):
        train_folds = [f for f in range(1, 9) if f != val_fold]
        
        print(f"\n{'='*60}")
        print(f"FOLD {val_fold}/8")
        print(f"Val: [{val_fold}] | Train: {train_folds}")
        print("=" * 60)
        
        train_loader = DataLoader(
            AudioDataset(TRAIN_CSV, AUDIO_DIR, train_folds, CACHE_DIR,
                        augment=True, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            AudioDataset(TRAIN_CSV, AUDIO_DIR, [val_fold], CACHE_DIR,
                        augment=False, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        model = EfficientNetB1(in_channels=in_channels, num_classes=NUM_CLASSES)
        if val_fold == 1:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,}")
        
        save_path = f"efficientnet_b1_fold{val_fold}.pth"
        
        train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                   label_smoothing=label_smoothing, use_mixup=use_mixup, save_path=save_path)
        
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        _, val_acc, val_f1, val_score = evaluate(model, val_loader, 
                                                  nn.CrossEntropyLoss(), DEVICE)
        
        model_paths.append(save_path)
        val_scores.append(val_score)
        print(f"Fold {val_fold} best score: {val_score:.4f}")
    
    # Print CV results
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)
    print(f"Mean CV Score: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    print(f"Individual: {[f'{s:.4f}' for s in val_scores]}")
    
    # Ensemble inference with weighted average
    if TEST_CSV.exists():
        print("\n" + "=" * 60)
        print("Ensemble Inference (weighted by val_score)")
        print("=" * 60)
        
        scores_arr = np.array(val_scores)
        temperature = 10.0
        weights = np.exp(scores_arr * temperature)
        weights = weights / weights.sum()
        
        print("Model weights (based on val_score):")
        for i, (score, w) in enumerate(zip(val_scores, weights)):
            print(f"  Fold {i+1}: score={score:.4f}, weight={w:.4f}")
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR, use_3ch=use_3ch),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        all_probs = []
        file_names = None
        
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/{len(model_paths)}...")
            model = EfficientNetB1(in_channels=in_channels, num_classes=NUM_CLASSES)
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
        
        # Weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for probs, w in zip(all_probs, weights):
            weighted_probs += w * probs
        
        predictions = [(name, int(np.argmax(weighted_probs[i]))) 
                      for i, name in enumerate(file_names)]
        
        save_submission(predictions, "efficientnet_b1_ensemble.csv")
        print(f"\nWeighted 8-fold ensemble submission saved!")
    
    return val_scores


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cv":
        # 8-fold CV ensemble (recommended)
        run_cv_ensemble(
            batch_size=32,
            epochs=60,
            lr=5e-4,
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
            lr=5e-4,
            use_mixup=True,
            label_smoothing=0.1,
            use_3ch=True,
        )

