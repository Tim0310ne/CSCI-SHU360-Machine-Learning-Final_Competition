#!/usr/bin/env python3
"""
ResNet34 audio classification training (log-mel input).

This script trains a 1-channel ResNet34 on fixed-size log-mel spectrograms.
It keeps the same hyperparameter values and learning recipe as the original
version in this workspace, but is reorganized for readability and easier reuse.

Notes:
- If you are using this for coursework/research, add appropriate attribution
  and describe what you changed. Avoid presenting others' work as your own.
"""

import os
import warnings
import random
import numpy as np
import pandas as pd
import librosa
import pickle
import time
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable, Dict

warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# Progress bar
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
import argparse

# Fix random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# =============================
# CONFIG (values kept identical)
# =============================
@dataclass(frozen=True)
class PathsConfig:
    # Defaults aligned with MLFA/resnet18.py (HPC layout)
    base_dir: Path = Path("/scratch/tl3735/MLFA/Kaggle_Data")
    model_dir: Path = Path("/scratch/tl3735/MLFA/pth")

    @property
    def audio_dir(self) -> Path:
        return self.base_dir / "audio"

    @property
    def test_audio_dir(self) -> Path:
        return self.audio_dir / "test"

    @property
    def cache_dir(self) -> Path:
        # Align with resnet18.py style: keep cached mel features under base_dir.
        return self.base_dir / "mel_cache"

    @property
    def train_csv(self) -> Path:
        return self.base_dir / "metadata" / "kaggle_train.csv"

    @property
    def test_csv(self) -> Path:
        return self.base_dir / "metadata" / "kaggle_test.csv"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    target_frames: int = 130


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 24
    epochs: int = 45
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 10
    num_classes: int = 10
    use_label_smoothing: bool = True


PATHS = PathsConfig()
AUDIO = AudioConfig()
TRAIN = TrainConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"Using device: {DEVICE}")
print("-" * 50)

# =============================
# AUDIO PROCESSING
# =============================
def melspectrogram_enhanced(
    audio_path: str,
    audio_cfg: AudioConfig,
    target_frames: Optional[int] = None,
    augment: bool = False,
) -> np.ndarray:
    """Log-mel spectrogram extraction with light waveform augmentation."""
    if target_frames is None:
        target_frames = audio_cfg.target_frames
    try:
        y, sr = librosa.load(audio_path, sr=audio_cfg.sample_rate)
        
        # Enhanced preprocessing
        if augment:
            # Random time stretching (light)
            if np.random.random() < 0.2:
                stretch_factor = np.random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=stretch_factor)
            
            # Random pitch shifting (light)
            if np.random.random() < 0.2:
                pitch_shift = np.random.randint(-1, 2)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        
        # Improved mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=audio_cfg.n_fft,
            hop_length=audio_cfg.hop_length,
            n_mels=audio_cfg.n_mels,
            fmin=0,
            fmax=sr//2
        )
        
        # Convert to log scale with improved dynamic range
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize to [-1, 1] range
        mel_spec_db = mel_spec_db / 40.0
        
        # Handle length with improved padding/truncation
        if mel_spec_db.shape[1] < target_frames:
            # Use reflection padding
            pad_width = target_frames - mel_spec_db.shape[1]
            if pad_width <= mel_spec_db.shape[1]:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='reflect')
            else:
                repeats = (pad_width // mel_spec_db.shape[1]) + 1
                mel_spec_extended = np.tile(mel_spec_db, (1, repeats))
                mel_spec_db = mel_spec_extended[:, :target_frames]
        elif mel_spec_db.shape[1] > target_frames:
            if augment:
                # Random cropping for training
                start = np.random.randint(0, mel_spec_db.shape[1] - target_frames + 1)
                mel_spec_db = mel_spec_db[:, start:start + target_frames]
            else:
                # Center cropping for validation
                start = (mel_spec_db.shape[1] - target_frames) // 2
                mel_spec_db = mel_spec_db[:, start:start + target_frames]
        
        # Ensure exact shape
        if mel_spec_db.shape != (audio_cfg.n_mels, target_frames):
            mel_spec_db = np.resize(mel_spec_db, (audio_cfg.n_mels, target_frames))
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros((audio_cfg.n_mels, target_frames))

def apply_light_augmentation(mel_spec, prob=0.6):
    """Apply light augmentation techniques (reduced strength)"""
    if np.random.random() > prob:
        return mel_spec
    
    mel_spec = mel_spec.copy()
    
    # SpecAugment - Time masking (reduced strength)
    if np.random.random() < 0.25:
        t_mask_size = np.random.randint(3, 8)  # Smaller masks
        t_start = np.random.randint(0, max(1, mel_spec.shape[1] - t_mask_size))
        mel_spec[:, t_start:t_start + t_mask_size] = mel_spec.min()
    
    # SpecAugment - Frequency masking (reduced strength)
    if np.random.random() < 0.25:
        f_mask_size = np.random.randint(2, 6)  # Smaller masks
        f_start = np.random.randint(0, max(1, mel_spec.shape[0] - f_mask_size))
        mel_spec[f_start:f_start + f_mask_size, :] = mel_spec.min()
    
    # Gaussian noise (reduced strength)
    if np.random.random() < 0.15:
        noise_factor = np.random.uniform(0.003, 0.015)
        noise = np.random.normal(0, noise_factor, mel_spec.shape)
        mel_spec = mel_spec + noise
    
    # Random gain (light)
    if np.random.random() < 0.2:
        gain = np.random.uniform(0.9, 1.1)
        mel_spec = mel_spec * gain
    
    return mel_spec

# =============================
# DATASET (cache-first, aligned with resnet18.py)
# =============================
def _audio_path(audio_dir: Path, fold: int, fname: str) -> Path:
    return Path(audio_dir) / f"fold{int(fold)}" / str(fname)


def _cache_path(cache_root: Path, fold: int, fname: str, variant: str) -> Path:
    # Similar spirit to resnet18.py: cache per file, keyed by fold + stem (+ variant).
    stem = Path(fname).stem
    return Path(cache_root) / variant / f"fold{int(fold)}_{stem}.npy"


def _filter_existing(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    """Match original behavior: skip missing audio files."""
    keep_rows = []
    for row in df.itertuples(index=False):
        ap = _audio_path(audio_dir, getattr(row, "fold"), getattr(row, "slice_file_name"))
        if ap.exists():
            keep_rows.append(True)
        else:
            keep_rows.append(False)
    out = df.loc[keep_rows].reset_index(drop=True)
    if len(out) != len(df):
        print(f"Filtered missing audio files: {len(df) - len(out)} removed, {len(out)} remaining")
    return out


def precompute_mel_cache(
    df: pd.DataFrame,
    audio_dir: Path,
    audio_cfg: AudioConfig,
    cache_root: Path,
    variant: str,
    augment: bool,
    overwrite: bool = False,
) -> None:
    """
    Precompute and store mel features to disk in deterministic row order.

    This preserves the original script's behavior where waveform-level augment/cropping
    occurs once per sample at dataset construction time (not every epoch).
    """
    target_dir = Path(cache_root) / variant
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing cache: variant='{variant}', augment={augment}, overwrite={overwrite}")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        fold = getattr(row, "fold")
        fname = getattr(row, "slice_file_name")
        ap = _audio_path(audio_dir, fold, fname)
        cp = _cache_path(cache_root, fold, fname, variant)

        if cp.exists() and not overwrite:
            continue
        if not ap.exists():
            # Should have been filtered already; keep a safe-guard.
            continue

        mel = melspectrogram_enhanced(str(ap), audio_cfg=audio_cfg, augment=augment)
        if mel.shape != (audio_cfg.n_mels, audio_cfg.target_frames):
            mel = np.resize(mel, (audio_cfg.n_mels, audio_cfg.target_frames))
        np.save(cp, mel.astype(np.float32))


def fit_scaler_from_cache(
    df: pd.DataFrame,
    cache_root: Path,
    variant: str,
) -> StandardScaler:
    """Fit StandardScaler over all mel values (same as original), but streaming from cache."""
    scaler = StandardScaler()
    first = True
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Fitting scaler"):
        fold = getattr(row, "fold")
        fname = getattr(row, "slice_file_name")
        cp = _cache_path(cache_root, fold, fname, variant)
        if not cp.exists():
            continue
        x = np.load(cp).reshape(-1, 1)
        if first:
            scaler.partial_fit(x)
            first = False
        else:
            scaler.partial_fit(x)
    return scaler


class CachedAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cache_root: Path,
        variant: str,
        is_train: bool,
        scaler: Optional[StandardScaler],
    ):
        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.variant = variant
        self.is_train = is_train
        self.scaler = scaler
        self.labels = self.df["classID"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold = int(row["fold"])
        fname = str(row["slice_file_name"])
        label = int(row["classID"])

        cp = _cache_path(self.cache_root, fold, fname, self.variant)
        mel_spec = np.load(cp).astype(np.float32)
        
        # Apply additional augmentation for training
        if self.is_train:
            mel_spec = apply_light_augmentation(mel_spec)
        
        # Normalize if scaler is provided
        if self.scaler is not None:
            original_shape = mel_spec.shape
            mel_spec_flat = mel_spec.reshape(-1, 1)
            mel_spec_scaled = self.scaler.transform(mel_spec_flat)
            mel_spec = mel_spec_scaled.reshape(original_shape)
        
        # Add channel dimension for CNN: (1, H, W)
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
        label = torch.LongTensor([label]).squeeze()
        
        return mel_spec, label

# =============================
# DATA LOADING
# =============================
def load_and_split_data(train_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata and split by fold (train: 1-6, val: 7-8)."""
    print("Loading training metadata...")
    df = pd.read_csv(train_csv)
    
    print(f"Dataset info:")
    print(f"Total samples: {len(df)}")
    print(f"Unique folds: {sorted(df['fold'].unique())}")
    print(f"Class distribution:")
    print(df['classID'].value_counts().sort_index())
    
    # Enhanced fold-aware split
    print("Using enhanced fold-aware train/val split...")
    train_folds = [1, 2, 3, 4, 5, 6]  
    val_folds = [7, 8]                
    
    train_df = df[df['fold'].isin(train_folds)].reset_index(drop=True)
    val_df = df[df['fold'].isin(val_folds)].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)} (folds {train_folds})")
    print(f"Val samples: {len(val_df)} (folds {val_folds})")
    
    return train_df, val_df

def prepare_enhanced_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    audio_dir: Path,
    audio_cfg: AudioConfig,
    train_cfg: TrainConfig,
    model_dir: Path,
    cache_root: Path,
    overwrite_cache: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare data loaders (scaler + weighted sampling)."""

    # Filter missing audio files (same behavior as original preloading dataset).
    train_df = _filter_existing(train_df, audio_dir)
    val_df = _filter_existing(val_df, audio_dir)

    # Cache variants (separate to preserve original semantics):
    # - train_aug0: used ONLY for fitting scaler (original used is_train=False temp dataset)
    # - train_aug1: used for training dataset (original used augment=True during dataset construction)
    # - val_aug0: validation features (augment=False)
    train_for_scaler_variant = "resnet34_train_aug0"
    train_variant = "resnet34_train_aug1"
    val_variant = "resnet34_val_aug0"

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    precompute_mel_cache(
        train_df, audio_dir, audio_cfg, cache_root,
        variant=train_for_scaler_variant, augment=False, overwrite=overwrite_cache
    )

    print("Enhanced feature standardization...")
    scaler = fit_scaler_from_cache(train_df, cache_root, train_for_scaler_variant)
    
    # Save scaler
    scaler_path = model_dir / "resnet34_scaler.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"ResNet34 scaler saved to: {scaler_path}")

    # Precompute cached features for train/val (aligned with original order: train then val).
    precompute_mel_cache(
        train_df, audio_dir, audio_cfg, cache_root,
        variant=train_variant, augment=True, overwrite=overwrite_cache
    )
    precompute_mel_cache(
        val_df, audio_dir, audio_cfg, cache_root,
        variant=val_variant, augment=False, overwrite=overwrite_cache
    )

    # Create cache-backed datasets (no huge RAM spike).
    train_dataset = CachedAudioDataset(train_df, cache_root, train_variant, is_train=True, scaler=scaler)
    val_dataset = CachedAudioDataset(val_df, cache_root, val_variant, is_train=False, scaler=scaler)
    
    print("Train class distribution:", np.bincount(train_dataset.labels))
    print("Val class distribution:", np.bincount(val_dataset.labels))
    
    # Enhanced weighted sampling
    print("Using enhanced WeightedRandomSampler")
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    sample_weights = class_weights[train_dataset.labels]
    
    print("Enhanced class weights:", class_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders with SLURM-compatible configuration
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.batch_size, 
        sampler=sampler,
        num_workers=0,  # SLURM-safe
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,  # SLURM-safe
        pin_memory=True
    )
    
    return train_loader, val_loader

# =============================
# ENHANCED RESNET34 MODEL
# =============================
class EnhancedAudioResNet34(nn.Module):
    """
    Enhanced ResNet34 optimized for audio classification
    ResNet34 often performs better than ResNet50 on small datasets
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(EnhancedAudioResNet34, self).__init__()

        # Load ResNet34 architecture
        base_model = models.resnet34(pretrained=False)

        # Audio-optimized stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Use ResNet34 layers (lighter than ResNet50)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Simple and effective pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optimized classifier for ResNet34
        in_features = base_model.fc.in_features  # 512 for ResNet34
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Optimized forward pass
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.classifier(x)
        return x

# =============================
# EVALUATION FUNCTIONS
# =============================
def test_accuracy(model, loader, device):
    """计算准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def test_macro_f1(model, loader, device):
    """计算Macro-F1分数"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return f1_score(all_labels, all_predictions, average='macro') * 100

def kaggle_score(model, loader, device):
    """计算Kaggle评分"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return f1_score(all_labels, all_predictions, average='weighted') * 100

# =============================
# TRAINING FUNCTION
# =============================
def train_resnet34_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    train_cfg: TrainConfig,
    model_dir: Path,
) -> nn.Module:
    """Train ResNet34 model with the same recipe (AdamW + ReduceLROnPlateau + early stop)."""
    
    print("Using model: EnhancedAudioResNet34")
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Optimized loss function
    if train_cfg.use_label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimized optimizer for ResNet34
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []
    val_kaggle_scores = []
    
    best_kaggle_score = 0
    best_accuracy = 0
    patience_counter = 0
    
    print("Starting ResNet34 training...")
    
    start_time = time.time()
    
    for epoch in range(train_cfg.epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"RESNET34 EPOCH {epoch + 1}/{train_cfg.epochs}")
        print(f"{'='*70}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_acc = test_accuracy(model, val_loader, device)
        val_f1 = test_macro_f1(model, val_loader, device)
        val_kaggle = kaggle_score(model, val_loader, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_kaggle_scores.append(val_kaggle)
        
        # Progress display
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = train_cfg.epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print("_" * 70)
        print(f"RESNET34 EPOCH {epoch + 1}/{train_cfg.epochs} COMPLETED")
        print(f"Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"Elapsed Time: {elapsed_time/60:.1f} minutes")
        print(f"Estimated Remaining: {estimated_remaining/60:.1f} minutes")
        print("_" * 70)
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Validation Macro-F1: {val_f1:.2f}%")
        print(f"Validation Kaggle Score: {val_kaggle:.2f}%")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best models
        if val_kaggle > best_kaggle_score:
            best_kaggle_score = val_kaggle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_kaggle_path = model_dir / f"audio_resnet34_best_kaggle_{timestamp}.pth"
            torch.save(model.state_dict(), best_kaggle_path)
            print(f"New best Kaggle score model saved! Epoch {epoch+1}, Score: {val_kaggle:.2f}%")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_acc_path = model_dir / f"audio_resnet34_best_acc_{timestamp}.pth"
            torch.save(model.state_dict(), best_acc_path)
            print(f"New best accuracy model saved! Epoch {epoch+1}, Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_kaggle)
        
        # Early stopping
        if val_kaggle > best_kaggle_score - 0.5:
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= train_cfg.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation
    final_train_acc = test_accuracy(model, train_loader, device)
    final_val_acc = test_accuracy(model, val_loader, device)
    final_val_f1 = test_macro_f1(model, val_loader, device)
    final_val_kaggle = kaggle_score(model, val_loader, device)
    
    print("_" * 70)
    print("RESNET34 FINAL RESULTS:")
    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Final Validation Macro-F1: {final_val_f1:.2f}%")
    print(f"Final Validation Kaggle Score: {final_val_kaggle:.2f}%")
    
    # Classification report
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nResNet34 Classification Report:")
    class_names = [f"Class_{i}" for i in range(train_cfg.num_classes)]
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = model_dir / f"audio_resnet34_{timestamp}.pth"
    torch.save(model.state_dict(), final_model_path)
    print("_" * 70)
    print(f"ResNet34 final model saved: {final_model_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_f1s, val_kaggle_scores)
    
    return model

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_f1s, val_kaggle_scores):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('ResNet34 Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('ResNet34 Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 score
    ax3.plot(epochs, val_f1s, 'g-', label='Validation Macro-F1', linewidth=2)
    ax3.set_title('ResNet34 Validation Macro-F1 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Kaggle score
    ax4.plot(epochs, val_kaggle_scores, 'm-', label='Validation Kaggle Score', linewidth=2)
    best_epoch = np.argmax(val_kaggle_scores) + 1
    best_score = max(val_kaggle_scores)
    ax4.annotate(f'Best: {best_score:.2f}% (Epoch {best_epoch})', 
                xy=(best_epoch, best_score), xytext=(best_epoch+2, best_score+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold')
    ax4.set_title('ResNet34 Validation Kaggle Score', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Kaggle Score (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path("/scratch/tl3735/MLFA/resnet34_training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"ResNet34 training metrics plot saved: {plot_path}")
    
    plt.close()

# =============================
# MAIN FUNCTION
# =============================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ResNet34 on log-mel spectrograms (audio).")
    p.add_argument("--base_dir", type=str, default=str(PATHS.base_dir))
    p.add_argument("--model_dir", type=str, default=str(PATHS.model_dir))
    p.add_argument("--cache_dir", type=str, default=str(PATHS.cache_dir))
    p.add_argument("--overwrite_cache", action="store_true", help="Recompute cached mel features.")
    p.add_argument("--seed", type=int, default=42)
    return p

def main():
    """Main training function"""
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    paths = PathsConfig(base_dir=Path(args.base_dir), model_dir=Path(args.model_dir))

    print("Enhanced ResNet34 Audio Classification Training")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nResNet34 Training Configuration:")
    print(f"- Model: ResNet34 (lighter than ResNet50, often better on small datasets)")
    print(f"- Parameters: ~21M (vs ResNet50's ~25M)")
    print(f"- Batch Size: {TRAIN.batch_size}")
    print(f"- Learning Rate: {TRAIN.learning_rate}")
    print(f"- Weight Decay: {TRAIN.weight_decay}")
    print(f"- Epochs: {TRAIN.epochs}")
    print(f"- Patience: {TRAIN.patience}")
    print()
    
    # Load and split data
    train_df, val_df = load_and_split_data(paths.train_csv)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_enhanced_data_loaders(
        train_df,
        val_df,
        audio_dir=paths.audio_dir,
        audio_cfg=AUDIO,
        train_cfg=TRAIN,
        model_dir=paths.model_dir,
        cache_root=Path(args.cache_dir),
        overwrite_cache=bool(args.overwrite_cache),
    )
    
    print(f"ResNet34 training set: {len(train_loader.dataset)} samples")
    print(f"ResNet34 validation set: {len(val_loader.dataset)} samples")
    
    # Create ResNet34 model
    model = EnhancedAudioResNet34(num_classes=TRAIN.num_classes).to(DEVICE)
    
    # Train model
    trained_model = train_resnet34_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        train_cfg=TRAIN,
        model_dir=paths.model_dir,
    )
    
    print("ResNet34 training completed!")

if __name__ == "__main__":
    main()