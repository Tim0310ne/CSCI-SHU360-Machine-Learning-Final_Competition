# -*- coding: utf-8 -*-
"""
Multi-Architecture Ensemble
Combines predictions from multiple model architectures for better accuracy.

Models:
- AudioResNet (1-channel): model_fold1.pth ~ model_fold8.pth
- ResNet18 (3-channel): resnet18_fold1.pth ~ resnet18_fold8.pth

Expected improvement: +2-4% over single architecture
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/scratch/tl3735/MLFA/Kaggle_Data")
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
TEST_AUDIO_DIR = BASE_DIR / "audio" / "test"
CACHE_DIR = BASE_DIR / "mel_cache"

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_CONFIG = {
    "sr": 22050, "n_mels": 128, "n_fft": 2048,
    "hop_length": 512, "duration": 4.0, "fmax": 8000,
}

# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mel_1ch(audio_path, config=AUDIO_CONFIG):
    """1-channel mel spectrogram for AudioResNet."""
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    target_len = int(config["sr"] * config["duration"])
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, max(0, target_len - len(y))))
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def extract_mel_3ch(audio_path, config=AUDIO_CONFIG):
    """3-channel mel spectrogram (mel + delta + delta2) for ResNet18."""
    y, _ = librosa.load(str(audio_path), sr=config["sr"])
    
    target_len = int(config["sr"] * config["duration"])
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, max(0, target_len - len(y))))
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=config["sr"], n_mels=config["n_mels"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], fmax=config["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    
    def normalize(x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
    
    return np.stack([normalize(mel_db), normalize(delta), normalize(delta2)], axis=0)

# =============================================================================
# Datasets
# =============================================================================

class TestDataset1ch(Dataset):
    """Test dataset for 1-channel models."""
    def __init__(self, csv_path, audio_dir, cache_dir=None):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["slice_file_name"]
        cache_path = self.cache_dir / f"test_{Path(fname).stem}.npy" if self.cache_dir else None
        
        if cache_path and cache_path.exists():
            mel = np.load(cache_path)
        else:
            mel = extract_mel_1ch(self.audio_dir / fname)
            if cache_path:
                np.save(cache_path, mel)
        
        return torch.from_numpy(mel).unsqueeze(0), fname


class TestDataset3ch(Dataset):
    """Test dataset for 3-channel models."""
    def __init__(self, csv_path, audio_dir, cache_dir=None):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(csv_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["slice_file_name"]
        cache_path = self.cache_dir / f"test_3ch_{Path(fname).stem}.npy" if self.cache_dir else None
        
        if cache_path and cache_path.exists():
            mel = np.load(cache_path)
        else:
            mel = extract_mel_3ch(self.audio_dir / fname)
            if cache_path:
                np.save(cache_path, mel)
        
        return torch.from_numpy(mel), fname

# =============================================================================
# Model Architectures
# =============================================================================

# --- AudioResNet (1-channel) ---
class BasicBlock(nn.Module):
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
    """AudioResNet for 1-channel input (from cnn_ensemble.py)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    
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


# --- ResNet18 with SE (3-channel) - Matching resnet18.py exactly ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention (matching resnet18.py)."""
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


class BasicBlockSE(nn.Module):
    """ResNet basic block with optional SE attention (matching resnet18.py)."""
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
    """ResNet-18 for audio (matching resnet18.py exactly)."""
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # [2, 2, 2, 2] blocks, SE in layers 2-4
        self.layer1 = self._make_layer(64, 64, 2, stride=1, use_se=False)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, use_se=use_se)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride, use_se):
        layers = [BasicBlockSE(in_ch, out_ch, stride, use_se)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockSE(out_ch, out_ch, 1, use_se))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# =============================================================================
# Multi-Architecture Ensemble
# =============================================================================

def run_multi_ensemble():
    """
    Combine predictions from AudioResNet (1ch) and ResNet18 (3ch).
    Total: 16 models (8 + 8)
    """
    print("="*60)
    print("Multi-Architecture Ensemble (16 models)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Model configurations
    models_config = [
        {
            "name": "AudioResNet",
            "model_class": AudioResNet,
            "model_kwargs": {},
            "paths": [f"model_fold{i}.pth" for i in range(1, 9)],
            "dataset_class": TestDataset1ch,
            "val_scores": [0.7530, 0.7275, 0.6908, 0.7685, 0.7712, 0.8241, 0.7726, 0.7717],  # 你的CV分数
        },
        {
            "name": "ResNet18",
            "model_class": ResNet18,
            "model_kwargs": {"in_channels": 3, "use_se": True},
            "paths": [f"resnet18_fold{i}.pth" for i in range(1, 9)],
            "dataset_class": TestDataset3ch,
            "val_scores": None,  # 如果有分数可以填入
        },
    ]
    
    all_probs = []
    all_weights = []
    file_names = None
    
    for config in models_config:
        print(f"\n--- {config['name']} ---")
        
        # Check if model files exist
        existing_paths = [p for p in config["paths"] if Path(p).exists()]
        if not existing_paths:
            print(f"  No model files found, skipping...")
            continue
        
        print(f"  Found {len(existing_paths)} models")
        
        # Create data loader
        test_loader = DataLoader(
            config["dataset_class"](TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
            batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Get predictions from each model
        for i, model_path in enumerate(existing_paths):
            print(f"  Loading {model_path}...")
            model = config["model_class"](**config["model_kwargs"]).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            
            probs = []
            names = []
            with torch.no_grad():
                for x, fnames in tqdm(test_loader, desc=f"  {config['name']} {i+1}", leave=False):
                    x = x.to(DEVICE)
                    out = F.softmax(model(x), dim=1)
                    probs.append(out.cpu().numpy())
                    names.extend(fnames)
            
            all_probs.append(np.vstack(probs))
            
            # Weight based on val score if available
            if config["val_scores"] and i < len(config["val_scores"]):
                all_weights.append(config["val_scores"][i])
            else:
                all_weights.append(1.0)
            
            if file_names is None:
                file_names = names
    
    if not all_probs:
        print("Error: No models loaded!")
        return
    
    # Normalize weights
    weights = np.array(all_weights)
    weights = weights / weights.sum()
    
    print(f"\n--- Ensemble ---")
    print(f"Total models: {len(all_probs)}")
    print(f"Weights: {[f'{w:.3f}' for w in weights]}")
    
    # Weighted average
    avg_probs = np.zeros_like(all_probs[0])
    for probs, w in zip(all_probs, weights):
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
    
    return submission_df


if __name__ == "__main__":
    run_multi_ensemble()

