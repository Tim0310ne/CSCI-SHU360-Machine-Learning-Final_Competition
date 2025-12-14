# -*- coding: utf-8 -*-
"""
Urban Sound Classification with Improved CNN
Full pipeline: Data -> Feature Extraction -> Training -> Inference

Improvements over baseline:
- Deeper network (6 conv layers)
- SE attention blocks
- SpecAugment data augmentation
- Label smoothing
- Cosine annealing LR scheduler
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

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]

AUDIO_CONFIG = {
    "sr": 22050, "n_mels": 128, "n_fft": 2048,
    "hop_length": 512, "duration": 4.0, "fmax": 8000,
}

# =============================================================================
# Data Augmentation
# =============================================================================

class SpecAugment:
    """
    SpecAugment: Time and frequency masking for spectrograms.
    Helps model generalize by randomly masking portions of input.
    """
    def __init__(self, freq_mask_param=15, time_mask_param=25, 
                 num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, mel_spec):
        """Apply augmentation to mel spectrogram tensor."""
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        _, n_mels, time_steps = mel_spec.shape
        augmented = mel_spec.clone()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            augmented[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            augmented[:, :, t0:t0 + t] = 0
        
        return augmented

# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mel_spectrogram(audio_path, config=AUDIO_CONFIG):
    """Convert audio file to normalized Log-Mel spectrogram."""
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    if config["duration"]:
        target_len = int(config["sr"] * config["duration"])
        y = y[:target_len] if len(y) > target_len else np.pad(y, (0, max(0, target_len - len(y))))
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

# =============================================================================
# Dataset Classes
# =============================================================================

class AudioDataset(Dataset):
    """Dataset for training/validation with optional augmentation."""
    
    def __init__(self, csv_path, audio_dir, folds=None, cache_dir=None, augment=False):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        if folds:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold, fname, label = row["fold"], row["slice_file_name"], int(row["classID"])
        
        cache_path = self.cache_dir / f"fold{fold}_{Path(fname).stem}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            mel = np.load(cache_path)
        else:
            mel = extract_mel_spectrogram(self.audio_dir / f"fold{fold}" / fname)
            if cache_path:
                np.save(cache_path, mel)
        
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        
        # Apply augmentation during training
        if self.augment and self.spec_augment and random.random() > 0.5:
            mel_tensor = self.spec_augment(mel_tensor)
        
        return mel_tensor, torch.tensor(label)


class TestDataset(Dataset):
    """Dataset for test inference (no labels)."""
    
    def __init__(self, csv_path, audio_dir, cache_dir=None):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["slice_file_name"]
        
        cache_path = self.cache_dir / f"test_{Path(fname).stem}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            mel = np.load(cache_path)
        else:
            mel = extract_mel_spectrogram(self.audio_dir / fname)
            if cache_path:
                np.save(cache_path, mel)
        
        return torch.from_numpy(mel).unsqueeze(0), fname

# =============================================================================
# SE (Squeeze-and-Excitation) Block
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =============================================================================
# Improved CNN Model
# =============================================================================

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> (optional SE) -> Pool"""
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        x = self.pool(x)
        return x


class AudioCNNv2(nn.Module):
    """
    Improved CNN for audio classification.
    
    Improvements:
    - 6 conv layers (deeper)
    - More channels (32->64->128->256->512->512)
    - SE attention blocks in later layers
    - Two FC layers with dropout
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Feature extractor: 6 conv blocks
        # Input: (1, 128, 173) -> After all pools: (512, 2, 2)
        self.features = nn.Sequential(
            ConvBlock(1, 32, use_se=False),      # -> (32, 64, 86)
            ConvBlock(32, 64, use_se=False),     # -> (64, 32, 43)
            ConvBlock(64, 128, use_se=True),     # -> (128, 16, 21)
            ConvBlock(128, 256, use_se=True),    # -> (256, 8, 10)
            ConvBlock(256, 512, use_se=True),    # -> (512, 4, 5)
            ConvBlock(512, 512, use_se=True),    # -> (512, 2, 2)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with 2 FC layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# Keep original model for comparison
class AudioCNN(nn.Module):
    """Original simple CNN (baseline)."""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

# =============================================================================
# Label Smoothing Loss
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for regularization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return (-true_dist * log_preds).sum(dim=-1).mean()

# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
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


def train(model, train_loader, val_loader, epochs=50, lr=5e-4, 
          label_smoothing=0.1, warmup_epochs=5, save_path="best_model.pth"):
    """
    Full training loop with improvements:
    - Label smoothing
    - Linear warmup + Cosine annealing LR scheduler
    - AdamW optimizer
    """
    model = model.to(DEVICE)
    
    # Label smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # No smoothing for validation
    
    # AdamW optimizer (better weight decay handling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing (smooth decay, no restarts for stability)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    
    best_score = 0
    print(f"Training on {DEVICE} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Learning rate: {lr}, Warmup epochs: {warmup_epochs}, Label smoothing: {label_smoothing}")
    
    for epoch in range(epochs):
        # Linear warmup for first few epochs
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} (lr={current_lr:.6f})")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_score = evaluate(model, val_loader, val_criterion, DEVICE)
        
        # Only apply cosine scheduler after warmup
        if epoch >= warmup_epochs:
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

def predict(model, test_loader):
    """Generate predictions for test set."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for x, fnames in tqdm(test_loader, desc="Predicting"):
            preds = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            results.extend(zip(fnames, preds))
    
    return results


def generate_submission(predictions, output_path="cnn.csv"):
    """Save predictions to CSV in Kaggle submission format."""
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    
    submission_df["ID"] = submission_df["ID"].astype(int)
    submission_df["TARGET"] = submission_df["TARGET"].astype(int)
    
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 rows:")
    print(submission_df.head(10).to_string(index=False))
    print(f"\nClass distribution in predictions:")
    print(submission_df["TARGET"].value_counts().sort_index())
    
    return submission_df

# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    train_folds=None,
    val_folds=None,
    batch_size=48,
    num_workers=4,
    epochs=60,
    lr=5e-4,              # Lower LR for larger model
    label_smoothing=0.1,
    use_augment=True,
    model_version="v2",   # "v1" for original, "v2" for improved
):
    """
    Run complete pipeline: train + inference.
    
    Args:
        model_version: "v1" for original CNN, "v2" for improved CNN with SE blocks
        use_augment: Whether to use SpecAugment during training
        label_smoothing: Label smoothing factor (0 = no smoothing)
    """
    if train_folds is None:
        train_folds = TRAIN_FOLDS
    if val_folds is None:
        val_folds = VAL_FOLDS
    
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    # Create data loaders
    print(f"Loading data... Train folds: {train_folds}, Val folds: {val_folds}")
    print(f"Config: batch_size={batch_size}, num_workers={num_workers}, augment={use_augment}")
    
    train_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, train_folds, CACHE_DIR, augment=use_augment),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, val_folds, CACHE_DIR, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Initialize model
    if model_version == "v2":
        model = AudioCNNv2()
        print("Using AudioCNNv2 (improved with SE blocks)")
    else:
        model = AudioCNN()
        print("Using AudioCNN (original baseline)")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Train
    train(model, train_loader, val_loader, epochs=epochs, lr=lr, 
          label_smoothing=label_smoothing, save_path="best_model.pth")
    
    # Inference
    if TEST_CSV.exists():
        print("\nRunning inference...")
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        predictions = predict(model, test_loader)
        generate_submission(predictions, "cnn.csv")
    else:
        print(f"Warning: {TEST_CSV} not found, skipping inference")


if __name__ == "__main__":
    # Run with improved model
    # Note: Lower LR (5e-4) for larger model, with 5 warmup epochs
    run_pipeline(
        batch_size=48,
        epochs=60,
        lr=5e-4,           # Lower LR for larger model (was 1e-3)
        label_smoothing=0.1,
        use_augment=True,
        model_version="v2",  # Use improved CNN
    )
