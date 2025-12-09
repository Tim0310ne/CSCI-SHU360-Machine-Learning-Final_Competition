# -*- coding: utf-8 -*-
"""
Urban Sound Classification with ResNet Architecture
Based on mlfa25_fc.py - keeps data loading/processing, replaces CNN with ResNet
"""

import os
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Data paths
BASE_DIR = Path(os.getenv("MLFA25_BASE_DIR", Path(__file__).parent / "Kaggle_Data"))

# Check if we need to unzip
if not BASE_DIR.exists():
    zip_path = BASE_DIR.parent / "Kaggle_Data.zip"
    if zip_path.exists():
        print(f"Found {zip_path}, extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR.parent)
        print("Extraction complete!")
    else:
        print(f"Warning: Dataset not found at {BASE_DIR} and no zip file at {zip_path}")

TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"
TEST_AUDIO_DIR = AUDIO_DIR / "test"

# Number of classes
NUM_CLASSES = 10

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default audio / feature configuration for Phase 1
AUDIO_CONFIG = {
    "sr": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "duration": 4.0,  # seconds, None means use full clip
    "fmax": 8000,
}

def audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048,
                             hop_length=512, duration=None, fmax=8000):
    """
    Convert audio file to Mel spectrogram (common CNN input format)

    Args:
        audio_path: Path to audio file
        sr: Sample rate (default 22050 Hz)
        n_mels: Number of Mel filter banks (default 128, corresponds to image height)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        duration: Fixed duration in seconds, if specified will truncate or pad

    Returns:
        mel_spec: Mel spectrogram (n_mels, time_frames)
        y: Audio signal
        sr: Sample rate
    """
    # Load audio
    y, original_sr = librosa.load(str(audio_path), sr=sr)

    # Truncate or pad if fixed duration is specified
    if duration is not None:
        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )

    # Convert to log scale (Log-Mel Spectrogram, more commonly used)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db, y, sr


def compute_log_mel_spectrogram(audio_path, config=None, normalize=True):
    """
    High-level helper:
    audio file path -> (optionally normalized) Log-Mel spectrogram

    Args:
        audio_path: Path to audio file
        config: dict with keys like AUDIO_CONFIG
        normalize: Whether to scale to [0, 1]

    Returns:
        mel: np.ndarray of shape (n_mels, time_frames), float32
        y: audio signal
        sr: sample rate
    """
    if config is None:
        config = AUDIO_CONFIG

    mel_spec_db, y, sr = audio_to_mel_spectrogram(
        audio_path=audio_path,
        sr=config.get("sr", 22050),
        n_mels=config.get("n_mels", 128),
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        duration=config.get("duration", None),
        fmax=config.get("fmax", 8000),
    )

    mel = mel_spec_db
    if normalize:
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    return mel.astype(np.float32), y, sr

class UrbanSoundDataset(Dataset):
    """
    PyTorch Dataset that reads kaggle_train.csv and returns
    (Log-Mel spectrogram tensor, classID).
    """

    def __init__(
        self,
        csv_path: Path = TRAIN_CSV,
        audio_dir: Path = AUDIO_DIR,
        folds=None,
        audio_config=None,
        cache_dir: Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(self.csv_path)

        if folds is not None:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        self.audio_config = audio_config or AUDIO_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, fold, file_name):
        if self.cache_dir is None:
            return None
        stem = Path(file_name).stem
        return self.cache_dir / f"mel_fold{fold}_{stem}.npy"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["slice_file_name"]
        fold = row["fold"]
        class_id = int(row["classID"])

        cache_path = self._cache_path(fold, file_name)
        if cache_path is not None and cache_path.exists():
            mel = np.load(cache_path)
        else:
            audio_path = self.audio_dir / f"fold{fold}" / file_name
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            mel, _, _ = compute_log_mel_spectrogram(
                audio_path,
                config=self.audio_config,
                normalize=True,
            )

            if cache_path is not None:
                np.save(cache_path, mel)

        # (n_mels, time) -> (1, n_mels, time) for CNN input
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, 128, T]
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        return mel_tensor, label_tensor


class UrbanSoundTestDataset(Dataset):
    """
    PyTorch Dataset for test set (no labels).
    Reads kaggle_test.csv and returns (Log-Mel spectrogram tensor, file_name).
    """

    def __init__(
        self,
        csv_path: Path = TEST_CSV,
        audio_dir: Path = TEST_AUDIO_DIR,
        audio_config=None,
        cache_dir: Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(self.csv_path)
        self.audio_config = audio_config or AUDIO_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, file_name):
        if self.cache_dir is None:
            return None
        stem = Path(file_name).stem
        return self.cache_dir / f"mel_test_{stem}.npy"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["slice_file_name"]

        cache_path = self._cache_path(file_name)
        if cache_path is not None and cache_path.exists():
            mel = np.load(cache_path)
        else:
            audio_path = self.audio_dir / file_name
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            mel, _, _ = compute_log_mel_spectrogram(
                audio_path,
                config=self.audio_config,
                normalize=True,
            )

            if cache_path is not None:
                np.save(cache_path, mel)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, 128, T]
        return mel_tensor, file_name


def create_train_val_dataloaders(
    train_folds=(1, 2, 3, 4, 5, 6, 7, 8),
    val_folds=(9,),
    batch_size=32,
    num_workers=0,
    cache_dir: Path | None = None,
):
    """
    Convenience function to build train/val DataLoaders for Phase 1.
    """
    train_dataset = UrbanSoundDataset(
        csv_path=TRAIN_CSV,
        audio_dir=AUDIO_DIR,
        folds=list(train_folds),
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
    )
    val_dataset = UrbanSoundDataset(
        csv_path=TRAIN_CSV,
        audio_dir=AUDIO_DIR,
        folds=list(val_folds),
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def create_test_dataloader(batch_size=32, num_workers=0, cache_dir: Path | None = None):
    """
    Create DataLoader for test set inference.
    """
    test_dataset = UrbanSoundTestDataset(
        csv_path=TEST_CSV,
        audio_dir=TEST_AUDIO_DIR,
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # Fixed: was self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AudioResNet(nn.Module):
    """
    ResNet for audio classification using Mel spectrograms.
    Input shape: (batch, 1, n_mels=128, time_frames)
    Output: (batch, num_classes=10)
    
    This is a ResNet-18 style architecture adapted for 1-channel audio spectrograms.
    Modified for audio: uses smaller initial kernel to preserve frequency information.
    Trained from scratch (no pretrained weights).
    """

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=NUM_CLASSES, 
                 in_channels=1, base_channels=64):
        super(AudioResNet, self).__init__()
        
        self.in_channels = base_channels
        
        # Initial convolution layer - MODIFIED for audio spectrograms:
        # Use 3x3 kernel instead of 7x7 to preserve more frequency detail
        # Use stride=1 initially to keep more spatial information
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # Use smaller pooling to preserve more information
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, 1, 128, time_frames)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18_audio(num_classes=NUM_CLASSES):
    """ResNet-18 for audio classification"""
    return AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_audio(num_classes=NUM_CLASSES):
    """ResNet-34 for audio classification"""
    return AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet_small_audio(num_classes=NUM_CLASSES):
    """Smaller ResNet for faster training - good for initial experiments"""
    return AudioResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, base_channels=32)


class AudioResNetLight(nn.Module):
    """
    Lighter ResNet specifically designed for audio mel spectrograms.
    Less aggressive downsampling, fewer parameters, better suited for this task.
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(AudioResNetLight, self).__init__()
        
        # Initial feature extraction (no aggressive downsampling)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with gradual downsampling
        self.layer1 = self._make_layer(32, 64, stride=2)   # 128 -> 64
        self.layer2 = self._make_layer(64, 128, stride=2)  # 64 -> 32
        self.layer3 = self._make_layer(128, 256, stride=2) # 32 -> 16
        self.layer4 = self._make_layer(256, 512, stride=2) # 16 -> 8
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Create a residual layer with 2 blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def resnet_light_audio(num_classes=NUM_CLASSES):
    """Light ResNet designed for audio - recommended for this task"""
    return AudioResNetLight(num_classes=num_classes)


# =============================================================================
# Training & Evaluation Functions
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.
    Returns loss, accuracy, macro-F1, and predictions.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    # Weighted score (80% accuracy + 20% macro-F1)
    weighted_score = 0.8 * epoch_acc + 0.2 * macro_f1

    return epoch_loss, epoch_acc, macro_f1, weighted_score, all_preds, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device=DEVICE,
    save_path="best_resnet_model.pth",
):
    """
    Full training loop with validation and model saving.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_score = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_score': []
    }

    print(f"\nTraining on {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_score, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_score'].append(val_score)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Score: {val_score:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"*** Best model saved (score: {best_score:.4f}) ***")

    print("\n" + "=" * 60)
    print(f"Training complete! Best score: {best_score:.4f}")
    print(f"Model saved to: {save_path}")

    return history


# =============================================================================
# Inference & Submission
# =============================================================================

def predict(model, test_loader, device=DEVICE):
    """
    Generate predictions for test set.
    Returns list of (file_name, predicted_class_id).
    """
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, file_names in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for fname, pred in zip(file_names, predicted.cpu().numpy()):
                predictions.append((fname, int(pred)))

    return predictions


def generate_submission(predictions, output_path="submission.csv"):
    """
    Generate Kaggle submission CSV file.
    
    Output format:
        ID,TARGET
        0,4
        1,5
        ...
    
    IDs are sequential (0 to N-1) based on the order from the test dataloader.
    
    Args:
        predictions: list of (file_name, predicted_class_id) in dataloader order
        output_path: path to save CSV file
    """
    # Simply use sequential IDs (0 to N-1) based on dataloader order
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    
    # Ensure integer types
    submission_df["ID"] = submission_df["ID"].astype(int)
    submission_df["TARGET"] = submission_df["TARGET"].astype(int)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 rows:")
    print(submission_df.head(10).to_string(index=False))
    print(f"\nClass distribution in predictions:")
    print(submission_df["TARGET"].value_counts().sort_index())
    
    return submission_df


# =============================================================================
# Main Training Script
# =============================================================================

def run_training(
    train_folds=(1, 2, 3, 4, 5, 6, 7),
    val_folds=(8,),
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    cache_dir=None,
    model_type="resnet18",
):
    """
    Complete training pipeline with ResNet.
    
    Args:
        model_type: "resnet18", "resnet34", or "resnet_small"
    """
    print("=" * 60)
    print("Urban Sound Classification - ResNet Training")
    print("=" * 60)

    # Setup cache directory
    if cache_dir is None:
        cache_dir = BASE_DIR / "mel_cache"

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        train_folds=train_folds,
        val_folds=val_folds,
        batch_size=batch_size,
        num_workers=0,
        cache_dir=cache_dir,
    )

    # Create model
    print(f"\nInitializing {model_type} model...")
    if model_type == "resnet18":
        model = resnet18_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet34":
        model = resnet34_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet_small":
        model = resnet_small_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet_light":
        model = resnet_light_audio(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=DEVICE,
        save_path="best_resnet_model.pth",
    )

    return model, history


def run_inference(model_path="best_resnet_model.pth", output_path="submission.csv", 
                  cache_dir=None, model_type="resnet18"):
    """
    Complete inference pipeline.
    """
    print("=" * 60)
    print("Urban Sound Classification - ResNet Inference")
    print("=" * 60)

    # Setup cache directory
    if cache_dir is None:
        cache_dir = BASE_DIR / "mel_cache"

    # Load model
    print("\nLoading model...")
    if model_type == "resnet18":
        model = resnet18_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet34":
        model = resnet34_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet_small":
        model = resnet_small_audio(num_classes=NUM_CLASSES)
    elif model_type == "resnet_light":
        model = resnet_light_audio(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Model loaded from: {model_path}")

    # Create test loader
    print("\nCreating test data loader...")
    test_loader = create_test_dataloader(
        batch_size=32,
        num_workers=0,
        cache_dir=cache_dir,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict(model, test_loader, device=DEVICE)

    # Save submission
    df = generate_submission(predictions, output_path=output_path)

    return df


# =============================================================================
# Entry Point for Training & Inference
# =============================================================================

def run_full_pipeline(model_type="resnet18", num_epochs=50):
    """
    Run the complete pipeline: training + inference.
    
    Args:
        model_type: "resnet18", "resnet34", or "resnet_small"
        num_epochs: Number of training epochs
    """
    # Check if data exists
    if not TRAIN_CSV.exists():
        print(f"Error: Training CSV not found at {TRAIN_CSV}")
        print("Please set MLFA25_BASE_DIR environment variable or update BASE_DIR in the code.")
        return

    # Training
    model, history = run_training(
        train_folds=(1, 2, 3, 4, 5, 6, 7),
        val_folds=(8,),
        batch_size=32,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        weight_decay=1e-4,
        model_type=model_type,
    )

    # Inference
    if TEST_CSV.exists():
        df = run_inference(
            model_path="best_resnet_model.pth",
            output_path="submission.csv",
            model_type=model_type,
        )
    else:
        print(f"\nWarning: Test CSV not found at {TEST_CSV}")
        print("Skipping inference step.")


if __name__ == "__main__":
    # Available models:
    # - "resnet_light": Recommended for audio (less aggressive downsampling)
    # - "resnet18": Standard ResNet-18 adapted for audio
    # - "resnet34": Deeper model
    # - "resnet_small": Smaller/faster model
    run_full_pipeline(model_type="resnet_light", num_epochs=30)

