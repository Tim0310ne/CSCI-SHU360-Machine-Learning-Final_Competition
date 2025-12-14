# -*- coding: utf-8 -*-
"""
Urban Sound Classification with CNN
Full pipeline: Data -> Feature Extraction -> Training -> Inference
"""

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

# =============================================================================
# Train/Validation Split Configuration (Fold-based, NO random shuffle!)
# =============================================================================
# IMPORTANT: Data is pre-organized into 8 folds to prevent data leakage.
# Related audio segments from the same source are split across files.
# Using random shuffle (e.g., train_test_split with shuffle=True) will cause
# data leakage and artificially inflate validation scores.
#
# Dataset distribution (total ~7000 samples):
#   - Folds 1-6: ~75% for training
#   - Folds 7-8: ~25% for validation

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]  # Training set: Folds 1-6
VAL_FOLDS = [7, 8]                 # Validation set: Folds 7-8

AUDIO_CONFIG = {
    "sr": 22050, "n_mels": 128, "n_fft": 2048,
    "hop_length": 512, "duration": 4.0, "fmax": 8000,
}

# =============================================================================
# Data Split Utilities
# =============================================================================

def get_fold_split(csv_path=TRAIN_CSV):
    """
    Create train/val DataFrames based on fold configuration.
    NO random shuffling - strictly fold-based to prevent data leakage.
    
    Returns:
        train_df: DataFrame with training samples (folds 1-6)
        val_df: DataFrame with validation samples (folds 7-8)
    """
    df = pd.read_csv(csv_path)
    
    train_df = df[df["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    val_df = df[df["fold"].isin(VAL_FOLDS)].reset_index(drop=True)
    
    return train_df, val_df


def show_fold_distribution(csv_path=TRAIN_CSV):
    """Display fold and class distribution for verification."""
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("Dataset Fold Distribution")
    print("=" * 60)
    
    # Fold distribution
    fold_counts = df["fold"].value_counts().sort_index()
    print("\nSamples per fold:")
    for fold, count in fold_counts.items():
        split = "TRAIN" if fold in TRAIN_FOLDS else "VAL"
        print(f"  Fold {fold}: {count:4d} samples [{split}]")
    
    # Train/Val split summary
    train_df, val_df = get_fold_split(csv_path)
    total = len(df)
    print(f"\nTrain/Val Split:")
    print(f"  Training (folds {TRAIN_FOLDS}): {len(train_df):4d} samples ({100*len(train_df)/total:.1f}%)")
    print(f"  Validation (folds {VAL_FOLDS}): {len(val_df):4d} samples ({100*len(val_df)/total:.1f}%)")
    
    # Class distribution
    print(f"\nClass distribution:")
    class_counts = df.groupby("class")["classID"].first().reset_index()
    for _, row in df["class"].value_counts().items():
        pass  # Skip detailed class print for brevity
    print(f"  Total classes: {df['classID'].nunique()}")
    print(f"  Total samples: {total}")
    
    return train_df, val_df


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mel_spectrogram(audio_path, config=AUDIO_CONFIG):
    """Convert audio file to normalized Log-Mel spectrogram."""
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    # Pad or truncate to fixed duration
    if config["duration"]:
        target_len = int(config["sr"] * config["duration"])
        y = y[:target_len] if len(y) > target_len else np.pad(y, (0, max(0, target_len - len(y))))
    
    # Compute Log-Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

# =============================================================================
# Dataset Classes
# =============================================================================

class AudioDataset(Dataset):
    """Dataset for training/validation with labels."""
    
    def __init__(self, csv_path, audio_dir, folds=None, cache_dir=None):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        if folds:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold, fname, label = row["fold"], row["slice_file_name"], int(row["classID"])
        
        # Try loading from cache
        cache_path = self.cache_dir / f"fold{fold}_{Path(fname).stem}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            mel = np.load(cache_path)
        else:
            mel = extract_mel_spectrogram(self.audio_dir / f"fold{fold}" / fname)
            if cache_path:
                np.save(cache_path, mel)
        
        return torch.from_numpy(mel).unsqueeze(0), torch.tensor(label)


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
# CNN Model
# =============================================================================

class AudioCNN(nn.Module):
    """Simple CNN for audio classification."""
    
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


def train(model, train_loader, val_loader, epochs=50, lr=1e-3, save_path="best_model.pth"):
    """Full training loop."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_score = 0
    print(f"Training on {DEVICE} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_score = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_score)
        
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
    """Save predictions to CSV."""
    df = pd.DataFrame(predictions, columns=["slice_file_name", "classID"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(df)} predictions)")
    return df

# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    train_folds=None,
    val_folds=None,
    batch_size=32,
    epochs=50,
    lr=1e-3,
):
    """
    Run complete pipeline: train + inference.
    
    Args:
        train_folds: List of fold numbers for training (default: TRAIN_FOLDS = [1,2,3,4,5,6])
        val_folds: List of fold numbers for validation (default: VAL_FOLDS = [7,8])
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
    
    Note: We use fold-based splitting (NOT random shuffle) to prevent data leakage.
    """
    # Use global defaults if not specified
    if train_folds is None:
        train_folds = TRAIN_FOLDS
    if val_folds is None:
        val_folds = VAL_FOLDS
    
    # Check data exists
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    # Create data loaders (shuffle=True only shuffles within training set, not across folds)
    print(f"Loading data... Train folds: {train_folds}, Val folds: {val_folds}")
    train_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, train_folds, CACHE_DIR),
        batch_size=batch_size, shuffle=True, num_workers=0  # shuffle within train set is OK
    )
    val_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, val_folds, CACHE_DIR),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Initialize and train model
    model = AudioCNN()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    train(model, train_loader, val_loader, epochs=epochs, lr=lr)
    
    # Inference on test set
    if TEST_CSV.exists():
        print("\nRunning inference...")
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        model = model.to(DEVICE)
        
        test_loader = DataLoader(
            TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        predictions = predict(model, test_loader)
        generate_submission(predictions, "cnn.csv")
    else:
        print(f"Warning: {TEST_CSV} not found, skipping inference")


if __name__ == "__main__":
    run_pipeline()

