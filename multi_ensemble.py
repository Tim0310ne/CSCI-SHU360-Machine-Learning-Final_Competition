# -*- coding: utf-8 -*-
"""
Multi-Architecture Ensemble (40 models)
Combines predictions from 5 different architectures for maximum accuracy.

Models:
- AudioResNet (1-channel): model_fold1.pth ~ model_fold8.pth (8 models)
- ResNet18 (3-channel): resnet18_fold1.pth ~ resnet18_fold8.pth (8 models)
- ResNet34 (3-channel): resnet34_fold1.pth ~ resnet34_fold8.pth (8 models)
- ResNet101 (3-channel): resnet101_fold1.pth ~ resnet101_fold8.pth (8 models)
- EfficientNet-B1 (3-channel): efficientnet_b1_fold1.pth ~ efficientnet_b1_fold8.pth (8 models)

Total: 40 models
Expected improvement: +3-5% over single architecture
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
    """3-channel mel spectrogram (mel + delta + delta2)."""
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
# Model 1: AudioResNet (1-channel) - from cnn_ensemble.py
# =============================================================================

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
    """AudioResNet for 1-channel input."""
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


# =============================================================================
# Model 2: ResNet18 with SE (3-channel) - from resnet18.py
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


class BasicBlockSE(nn.Module):
    """ResNet basic block with optional SE attention."""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        
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
    """ResNet-18 for audio (matching resnet18.py)."""
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
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
# Model 3: ResNet34 (3-channel) - from resnet34_new.py
# =============================================================================

class ResNet34(nn.Module):
    """ResNet-34 for audio (matching resnet34_new.py). [3,4,6,3] blocks."""
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # [3, 4, 6, 3] blocks for ResNet-34
        self.layer1 = self._make_layer(64, 64, 3, stride=1, use_se=False)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, use_se=use_se)
        
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
# Model 4: ResNet101 (3-channel) - from resnet101.py
# =============================================================================

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-101."""
    expansion = 4
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(out_ch * self.expansion) if use_se else nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet101(nn.Module):
    """ResNet-101 for audio (matching resnet101.py)."""
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1, use_se=False)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(512, 256, 23, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, use_se=use_se)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride, use_se):
        downsample = None
        if stride != 1 or in_ch != out_ch * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * Bottleneck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * Bottleneck.expansion)
            )
        layers = [Bottleneck(in_ch, out_ch, stride, downsample, use_se)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_ch * Bottleneck.expansion, out_ch, use_se=use_se))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
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
# Model 4: EfficientNet-B1 (3-channel) - from efficientnet_b1.py
# =============================================================================

class EfficientNetB1(nn.Module):
    """EfficientNet-B1 for audio (matching efficientnet_b1.py)."""
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        
        from torchvision.models import efficientnet_b1
        
        self.efficientnet = efficientnet_b1(weights=None)
        
        # Modify first conv
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Modify classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


# =============================================================================
# Multi-Architecture Ensemble
# =============================================================================

def run_multi_ensemble():
    """
    Combine predictions from 3 architectures (24 models total).
    """
    print("="*60)
    print("Multi-Architecture Ensemble (24 models)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Model configurations (3 architectures, 24 models total)
    models_config = [
        {
            "name": "AudioResNet",
            "model_class": AudioResNet,
            "model_kwargs": {},
            "paths": [f"model_fold{i}.pth" for i in range(1, 9)],
            "channels": 1,
        },
        {
            "name": "ResNet18",
            "model_class": ResNet18,
            "model_kwargs": {"in_channels": 3, "use_se": True},
            "paths": [f"resnet18_fold{i}.pth" for i in range(1, 9)],
            "channels": 3,
        },
        {
            "name": "ResNet34",
            "model_class": ResNet34,
            "model_kwargs": {"in_channels": 3, "use_se": True},
            "paths": [f"resnet34_fold{i}.pth" for i in range(1, 9)],
            "channels": 3,
        },
    ]
    
    # Create data loaders (cache them)
    print("\nPreparing data loaders...")
    loader_1ch = DataLoader(
        TestDataset1ch(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    loader_3ch = DataLoader(
        TestDataset3ch(TEST_CSV, TEST_AUDIO_DIR, CACHE_DIR),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    all_probs = []
    file_names = None
    total_models = 0
    
    for config in models_config:
        print(f"\n--- {config['name']} ---")
        
        # Check which model files exist
        existing_paths = [p for p in config["paths"] if Path(p).exists()]
        if not existing_paths:
            print(f"  No model files found, skipping...")
            continue
        
        print(f"  Found {len(existing_paths)}/{len(config['paths'])} models")
        
        # Select appropriate data loader
        loader = loader_1ch if config["channels"] == 1 else loader_3ch
        
        # Get predictions from each model
        for i, model_path in enumerate(existing_paths):
            print(f"  Loading {model_path}...", end=" ")
            
            model = config["model_class"](**config["model_kwargs"]).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            
            probs = []
            names = []
            with torch.no_grad():
                for x, fnames in loader:
                    x = x.to(DEVICE)
                    out = F.softmax(model(x), dim=1)
                    probs.append(out.cpu().numpy())
                    names.extend(fnames)
            
            all_probs.append(np.vstack(probs))
            total_models += 1
            print("✓")
            
            if file_names is None:
                file_names = names
    
    if not all_probs:
        print("Error: No models loaded!")
        return
    
    # Simple average (equal weight for all models)
    print(f"\n--- Ensemble ---")
    print(f"Total models: {total_models}")
    
    avg_probs = np.mean(all_probs, axis=0)
    predictions = [(name, int(np.argmax(avg_probs[i]))) for i, name in enumerate(file_names)]
    
    # Generate submission
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    submission_df.to_csv("mix.csv", index=False)
    
    print(f"\nSubmission saved to: mix.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df["TARGET"].value_counts().sort_index())
    
    return submission_df


if __name__ == "__main__":
    run_multi_ensemble()
