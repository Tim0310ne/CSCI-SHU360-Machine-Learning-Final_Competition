# -*- coding: utf-8 -*-
"""
Urban Sound Classification - High Score Solution (Target: 88+)

Key techniques:
1. 8-fold Cross-Validation Ensemble
2. ResNet-style architecture (optimized for audio)
3. Mixup + SpecAugment data augmentation
4. Class weights for imbalanced data
5. Probability averaging for ensemble prediction
"""

import os
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
# Reproducibility
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

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
# Data Augmentation
# =============================================================================

class SpecAugment:
    """Time and frequency masking for spectrograms."""
    def __init__(self, freq_mask=10, time_mask=20, n_masks=1):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_masks = n_masks
    
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, n_mels, n_time = x.shape
        x = x.clone()
        
        for _ in range(self.n_masks):
            # Frequency mask
            f = random.randint(0, min(self.freq_mask, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            x[:, f0:f0+f, :] = 0
            # Time mask
            t = random.randint(0, min(self.time_mask, n_time - 1))
            t0 = random.randint(0, n_time - t)
            x[:, :, t0:t0+t] = 0
        
        return x


def mixup_data(x, y, alpha=0.2):
    """Mixup: creates virtual training examples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mel_spectrogram(audio_path, config=AUDIO_CONFIG):
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
# Dataset
# =============================================================================

class AudioDataset(Dataset):
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
        
        if self.augment and self.spec_augment and random.random() > 0.5:
            mel_tensor = self.spec_augment(mel_tensor)
        
        return mel_tensor, torch.tensor(label)


class TestDataset(Dataset):
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
# Model: Audio ResNet (Optimized for Mel Spectrograms)
# =============================================================================

class BasicBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class AudioResNet(nn.Module):
    """
    ResNet optimized for audio mel spectrograms.
    - Smaller initial kernel (3x3) to preserve frequency detail
    - 4 residual stages with [2,2,2,2] blocks
    - ~11M parameters
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Initial conv (3x3 instead of 7x7)
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Residual stages
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# =============================================================================
# Training Functions
# =============================================================================

def compute_class_weights(folds, csv_path=TRAIN_CSV):
    """Compute class weights for imbalanced data."""
    df = pd.read_csv(csv_path)
    df = df[df["fold"].isin(folds)]
    counts = df["classID"].value_counts().sort_index()
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        
        # Mixup with 50% probability
        if use_mixup and random.random() > 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            _, pred = out.max(1)
            correct += (lam * pred.eq(y_a).sum().item() + (1-lam) * pred.eq(y_b).sum().item())
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
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
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


def train_fold(fold_idx, train_folds, val_fold, num_epochs=30, batch_size=32, lr=1e-3):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx} (val_fold={val_fold})")
    print("="*60)
    
    # Data loaders
    train_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, train_folds, CACHE_DIR, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        AudioDataset(TRAIN_CSV, AUDIO_DIR, [val_fold], CACHE_DIR, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model
    model = AudioResNet().to(DEVICE)
    
    # Class weights
    class_weights = compute_class_weights(train_folds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_score = 0
    save_path = f"model_fold{val_fold}.pth"
    
    for epoch in range(num_epochs):
        lr_now = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_score = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | lr={lr_now:.6f} | "
              f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | score={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"  *** Best model saved (score={best_score:.4f}) ***")
    
    return save_path, best_score

# =============================================================================
# 8-Fold Cross-Validation Ensemble
# =============================================================================

def run_cv_ensemble(n_folds=8, num_epochs=30, batch_size=32, lr=1e-3):
    """
    8-fold CV ensemble - THE KEY TO HIGH SCORES!
    
    Each fold:
    - Train on 7 folds, validate on 1 fold
    - Save best model
    
    Final prediction:
    - Average probabilities from all 8 models
    """
    print("="*60)
    print("8-Fold Cross-Validation Ensemble (Target: 88+)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs per fold: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    model_paths = []
    val_scores = []
    
    # Train 8 models
    for val_fold in range(1, n_folds + 1):
        train_folds = [f for f in range(1, n_folds + 1) if f != val_fold]
        save_path, best_score = train_fold(
            fold_idx=val_fold,
            train_folds=train_folds,
            val_fold=val_fold,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr
        )
        model_paths.append(save_path)
        val_scores.append(best_score)
    
    # Print CV results
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    mean_score = np.mean(val_scores)
    std_score = np.std(val_scores)
    print(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual scores: {[f'{s:.4f}' for s in val_scores]}")
    
    # Ensemble inference
    print("\n" + "="*60)
    print("Ensemble Inference")
    print("="*60)
    
    test_loader = DataLoader(
        TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    all_probs = []
    file_names = None
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        model = AudioResNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        probs = []
        names = []
        with torch.no_grad():
            for x, fnames in tqdm(test_loader, desc=f"Model {i+1}", leave=False):
                x = x.to(DEVICE)
                out = F.softmax(model(x), dim=1)
                probs.append(out.cpu().numpy())
                names.extend(fnames)
        
        all_probs.append(np.vstack(probs))
        if file_names is None:
            file_names = names
    
    # Weighted average based on validation scores (better models contribute more)
    weights = np.array(val_scores)
    weights = weights / weights.sum()  # Normalize to sum to 1
    print(f"\nEnsemble weights: {[f'{w:.3f}' for w in weights]}")
    
    # Weighted average of probabilities
    avg_probs = np.zeros_like(all_probs[0])
    for i, (probs, w) in enumerate(zip(all_probs, weights)):
        avg_probs += w * probs
    
    predictions = [(name, int(np.argmax(avg_probs[i]))) for i, name in enumerate(file_names)]
    
    # Generate submission
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    submission_df.to_csv("cnn.csv", index=False)
    
    print(f"\nSubmission saved to: cnn.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df["TARGET"].value_counts().sort_index())
    
    return mean_score, val_scores

# =============================================================================
# Inference Only (使用已训练好的模型)
# =============================================================================

def run_inference_only(val_scores=None, n_folds=8):
    """
    只运行推理，使用已经训练好的模型。
    
    Args:
        val_scores: 每个fold的验证分数列表，用于加权平均
        n_folds: fold数量
    """
    print("="*60)
    print("Ensemble Inference Only (使用已训练模型)")
    print("="*60)
    
    # 默认验证分数（如果没有提供）
    if val_scores is None:
        val_scores = [1.0] * n_folds  # 简单平均
        print("Using simple average (equal weights)")
    else:
        print(f"Using weighted average based on val_scores")
    
    model_paths = [f"model_fold{i}.pth" for i in range(1, n_folds + 1)]
    
    # 检查模型文件是否存在
    for path in model_paths:
        if not Path(path).exists():
            print(f"Error: Model file not found: {path}")
            return
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        TestDataset(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    all_probs = []
    file_names = None
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        model = AudioResNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        probs = []
        names = []
        with torch.no_grad():
            for x, fnames in tqdm(test_loader, desc=f"Model {i+1}", leave=False):
                x = x.to(DEVICE)
                out = F.softmax(model(x), dim=1)
                probs.append(out.cpu().numpy())
                names.extend(fnames)
        
        all_probs.append(np.vstack(probs))
        if file_names is None:
            file_names = names
    
    # 加权平均
    weights = np.array(val_scores)
    weights = weights / weights.sum()
    print(f"\nEnsemble weights: {[f'{w:.3f}' for w in weights]}")
    
    avg_probs = np.zeros_like(all_probs[0])
    for i, (probs, w) in enumerate(zip(all_probs, weights)):
        avg_probs += w * probs
    
    predictions = [(name, int(np.argmax(avg_probs[i]))) for i, name in enumerate(file_names)]
    
    # 生成提交文件
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    submission_df.to_csv("cnn.csv", index=False)
    
    print(f"\nSubmission saved to: cnn.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df["TARGET"].value_counts().sort_index())
    
    return submission_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        # 只运行推理（使用已训练好的模型）
        # 使用你之前的验证分数
        val_scores = [0.7346, 0.7288, 0.6656, 0.7651, 0.7791, 0.7785, 0.7844, 0.7440]
        run_inference_only(val_scores=val_scores)
    else:
        # 完整训练 + 推理
        run_cv_ensemble(
            n_folds=8,
            num_epochs=30,
            batch_size=32,
            lr=1e-3,
        )

