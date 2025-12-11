# -*- coding: utf-8 -*-
"""
Hybrid audio classifier:
- Data loader & feature extraction are identical to mlfa25_resnet.py
  (UrbanSoundDataset/TestDataset, log-Mel, caching, class weights, etc.).
- Models draw from Final-Competition (custom ResNet / LSTM) without any
  pretrained backbones.
- Training strategy follows Final-Competition/src/main.py (SGD + momentum +
  weight decay + MultiStepLR, optional mixup). Validation / submission logic
  stays the same as mlfa25_resnet.py.
"""

from __future__ import annotations

import os
import zipfile
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    mode = "deterministic" if deterministic else "performance"
    print(f"Random seed set to: {seed} ({mode} mode)")


SEED = 42
set_seed(SEED, deterministic=False)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(os.getenv("MLFA25_BASE_DIR", Path(__file__).parent / "Kaggle_Data"))

if not BASE_DIR.exists():
    zip_path = BASE_DIR.parent / "Kaggle_Data.zip"
    if zip_path.exists():
        print(f"Found {zip_path}, extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR.parent)
        print("Extraction complete!")
    else:
        print(f"Warning: Dataset not found at {BASE_DIR} and no zip file at {zip_path}")

TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"
TEST_AUDIO_DIR = AUDIO_DIR / "test"

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_CONFIG = {
    "sr": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "duration": 4.0,
    "fmax": 8000,
}


# =============================================================================
# Data augmentation / mixup (same as mlfa25_resnet)
# =============================================================================

class SpecAugment:
    def __init__(self, freq_mask_param=10, time_mask_param=20, num_freq_masks=1, num_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mel_spec):
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)

        _, n_mels, time_steps = mel_spec.shape
        augmented = mel_spec.clone()

        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            augmented[:, f0:f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            augmented[:, :, t0:t0 + t] = 0

        return augmented.squeeze(0) if len(mel_spec.shape) == 2 else augmented


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Feature extraction (unchanged)
# =============================================================================

def audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512, duration=None, fmax=8000):
    y, original_sr = librosa.load(str(audio_path), sr=sr)

    if duration is not None:
        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, y, sr


def compute_log_mel_spectrogram(audio_path, config=None, normalize=True):
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


# =============================================================================
# Datasets / Dataloaders (unchanged)
# =============================================================================

class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path: Path = TRAIN_CSV, audio_dir: Path = AUDIO_DIR, folds=None, audio_config=None,
                 cache_dir: Path | None = None, augment: bool = False):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(self.csv_path)

        if folds is not None:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        self.audio_config = audio_config or AUDIO_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.augment = augment
        self.spec_augment = SpecAugment(
            freq_mask_param=8, time_mask_param=15, num_freq_masks=1, num_time_masks=1
        ) if augment else None

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

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, 128, T]
        if self.augment and self.spec_augment is not None:
            mel_tensor = self.spec_augment(mel_tensor)
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        return mel_tensor, label_tensor


class UrbanSoundTestDataset(Dataset):
    def __init__(self, csv_path: Path = TEST_CSV, audio_dir: Path = TEST_AUDIO_DIR,
                 audio_config=None, cache_dir: Path | None = None):
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


def create_train_val_dataloaders(train_folds=(1, 2, 3, 4, 5, 6, 7, 8), val_folds=(9,), batch_size=32,
                                 num_workers=4, cache_dir: Path | None = None, augment: bool = True):
    train_dataset = UrbanSoundDataset(
        csv_path=TRAIN_CSV,
        audio_dir=AUDIO_DIR,
        folds=list(train_folds),
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
        augment=augment,
    )
    val_dataset = UrbanSoundDataset(
        csv_path=TRAIN_CSV,
        audio_dir=AUDIO_DIR,
        folds=list(val_folds),
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def create_test_dataloader(batch_size=32, num_workers=0, cache_dir: Path | None = None):
    test_dataset = UrbanSoundTestDataset(
        csv_path=TEST_CSV,
        audio_dir=TEST_AUDIO_DIR,
        audio_config=AUDIO_CONFIG,
        cache_dir=cache_dir,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def compute_class_weights(folds, csv_path: Path = TRAIN_CSV):
    df = pd.read_csv(csv_path)
    df = df[df["fold"].isin(folds)]
    counts = df["classID"].value_counts().sort_index()
    class_counts = counts.reindex(range(NUM_CLASSES)).fillna(0) + 1e-6
    weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def get_class_names(csv_path: Path = TRAIN_CSV):
    df = pd.read_csv(csv_path)
    name_by_id = df.drop_duplicates("classID").set_index("classID")["class"]
    class_names = []
    for i in range(NUM_CLASSES):
        if i in name_by_id.index:
            class_names.append(str(name_by_id.loc[i]))
        else:
            class_names.append(str(i))
    return class_names


def plot_confusion_matrix(targets, preds, class_names, normalize=True, save_path="confusion_matrix.png"):
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# =============================================================================
# Models (from Final-Competition, no pretrained weights)
# =============================================================================

class BasicBlockFC(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class CustomResNetFC(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.layer1 = nn.Sequential(
            BasicBlockFC(64, 64, stride=1, downsample=False),
            BasicBlockFC(64, 64, stride=1, downsample=False),
        )
        self.layer2 = nn.Sequential(
            BasicBlockFC(64, 128, stride=2, downsample=True),
            BasicBlockFC(128, 128, stride=1, downsample=False),
        )
        self.layer3 = nn.Sequential(
            BasicBlockFC(128, 256, stride=2, downsample=True),
            BasicBlockFC(256, 256, stride=1, downsample=False),
        )
        self.layer4 = nn.Sequential(
            BasicBlockFC(256, 512, stride=2, downsample=True),
            BasicBlockFC(512, 512, stride=1, downsample=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, label_size, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.hidden2label = nn.Linear(hidden_size, label_size)

    def forward(self, seq):
        embed = seq
        if embed.dim() == 2:
            embed = self.embedding(seq)
        x = embed.transpose(0, 1)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        y = self.hidden2label(out[-1])
        return y


def create_model(model_type: str, in_channels: int = 1) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "custom_resnet":
        return CustomResNetFC(in_channels=in_channels, num_classes=NUM_CLASSES)
    elif model_type == "lstm":
        return LSTMClassifier(
            input_size=128,
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            label_size=NUM_CLASSES,
            bidirectional=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# Training (SGD strategy) / Evaluation / Submission
# =============================================================================

def train_one_epoch_sgd(model, loader, criterion, optimizer, device, mixup=False, mixup_alpha=0.2):
    model.train()
    avg_loss, steps = 0.0, 0
    for x, label in tqdm(loader, desc="Training"):
        x, label = x.to(device), label.to(device)
        if mixup:
            x, label1, label2, lam = mixup_data(x, label, mixup_alpha)

        logits = model(x)

        if mixup:
            loss = mixup_criterion(criterion, logits, label1, label2, lam)
        else:
            loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        steps += 1
    return avg_loss / max(1, steps)


def evaluate(model, val_loader, criterion, device):
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
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_score = 0.8 * epoch_acc + 0.2 * macro_f1

    return epoch_loss, epoch_acc, macro_f1, weighted_score, all_preds, all_targets


def train_model_sgd(
    model,
    train_loader,
    val_loader,
    num_epochs=30,
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    milestones: Tuple[int, int, int] | None = None,
    mixup=True,
    label_smoothing=0.05,
    class_weights=None,
    device=DEVICE,
    save_path="best_hybrid_model.pth",
    class_names=None,
    confusion_save_path=None,
):
    model = model.to(device)
    weight_tensor = class_weights.to(device) if class_weights is not None else None

    train_criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if milestones is None:
        milestones = (num_epochs // 4, num_epochs // 2, num_epochs * 3 // 4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_score = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_score": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch_sgd(
            model, train_loader, train_criterion, optimizer, device, mixup=mixup
        )
        val_loss, val_acc, val_f1, val_score, _, _ = evaluate(model, val_loader, val_criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_score"].append(val_score)

        print(
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | score={val_score:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"*** Best model saved to {save_path} (score={best_score:.4f}) ***")

    if confusion_save_path is not None and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, _, _, _, preds, targets = evaluate(model, val_loader, val_criterion, device)
        names = class_names or [str(i) for i in range(NUM_CLASSES)]
        plot_confusion_matrix(targets, preds, names, normalize=True, save_path=confusion_save_path)

    return history


def predict(model, test_loader, device=DEVICE):
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
    submission_df = pd.DataFrame({
        "ID": range(len(predictions)),
        "TARGET": [pred for _, pred in predictions]
    })
    submission_df["ID"] = submission_df["ID"].astype(int)
    submission_df["TARGET"] = submission_df["TARGET"].astype(int)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print("\nFirst 10 rows:")
    print(submission_df.head(10).to_string(index=False))
    print("\nClass distribution in predictions:")
    print(submission_df["TARGET"].value_counts().sort_index())
    return submission_df


# =============================================================================
# Pipelines
# =============================================================================

def run_training(
    train_folds=(1, 2, 3, 4, 5, 6, 7),
    val_folds=(8,),
    batch_size=48,
    num_epochs=30,
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    model_type="custom_resnet",
    mixup=True,
    label_smoothing=0.05,
    num_workers=4,
):
    print("=" * 60)
    print("Hybrid Audio Classification - Training (SGD)")
    print("=" * 60)

    cache_dir = BASE_DIR / "mel_cache"
    train_loader, val_loader = create_train_val_dataloaders(
        train_folds=train_folds,
        val_folds=val_folds,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        augment=True,
    )

    model = create_model(model_type, in_channels=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal params: {total_params:,} | Trainable: {trainable_params:,}")

    class_weights = compute_class_weights(train_folds)
    class_names = get_class_names()

    history = train_model_sgd(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        mixup=mixup,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        device=DEVICE,
        save_path="best_hybrid_model.pth",
        class_names=class_names,
        confusion_save_path="confmat_hybrid_val.png",
    )
    return model, history


def run_inference(model_path="best_hybrid_model.pth", output_path="submission.csv",
                  model_type="custom_resnet", batch_size=64, num_workers=4):
    print("=" * 60)
    print("Hybrid Audio Classification - Inference")
    print("=" * 60)

    cache_dir = BASE_DIR / "mel_cache"
    model = create_model(model_type, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Loaded model from {model_path}")

    test_loader = create_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    predictions = predict(model, test_loader, device=DEVICE)
    df = generate_submission(predictions, output_path=output_path)
    return df


def run_full_pipeline(model_type="custom_resnet", num_epochs=30):
    if not TRAIN_CSV.exists():
        print(f"Error: Training CSV not found at {TRAIN_CSV}")
        return

    model, _ = run_training(
        train_folds=(1, 2, 3, 4, 5, 6, 7),
        val_folds=(8,),
        batch_size=48,
        num_epochs=num_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        model_type=model_type,
        mixup=True,
        label_smoothing=0.05,
        num_workers=4,
    )

    if TEST_CSV.exists():
        run_inference(
            model_path="best_hybrid_model.pth",
            output_path="submission.csv",
            model_type=model_type,
        )
    else:
        print(f"Warning: Test CSV not found at {TEST_CSV}, skip inference.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid audio classifier (SGD)")
    parser.add_argument("--model_type", type=str, default="custom_resnet", choices=["custom_resnet", "lstm"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=bool, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    args = parser.parse_args()

    run_training(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        mixup=args.mixup,
        label_smoothing=args.label_smoothing,
    )
# -*- coding: utf-8 -*-
"""
Hybrid training script that reuses the data loader / feature extraction
from mlfa25_resnet.py, adopts custom models inspired by Final-Competition
(no pretrained weights), and borrows the SGD + MultiStepLR training strategy
from Final-Competition/src/main.py. Validation and submission flows remain
the same as mlfa25_resnet.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from mlfa25_resnet import (
    # base config / utils
    set_seed,
    BASE_DIR,
    TRAIN_CSV,
    TEST_CSV,
    AUDIO_DIR,
    TEST_AUDIO_DIR,
    NUM_CLASSES,
    DEVICE,
    AUDIO_CONFIG,
    # data + loaders
    UrbanSoundDataset,
    UrbanSoundTestDataset,
    create_train_val_dataloaders,
    create_test_dataloader,
    compute_class_weights,
    get_class_names,
    # augmentation utils
    mixup_data,
    mixup_criterion,
    # eval / submission (kept unchanged)
    evaluate,
    predict,
    generate_submission,
    plot_confusion_matrix,
)


# --------------------------------------------------------------------------- #
# Models (adapted from Final-Competition, without pretrained weights)
# --------------------------------------------------------------------------- #


class BasicBlock(nn.Module):
    """Residual block used in the custom ResNet (no pretrain)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class CustomResNet(nn.Module):
    """
    Lightweight ResNet from Final-Competition, adapted for audio mel inputs.
    Channels: 64->128->256->512 with residual blocks; no pretrained weights.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, downsample=False),
            BasicBlock(64, 64, stride=1, downsample=False),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),
            BasicBlock(128, 128, stride=1, downsample=False),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),
            BasicBlock(256, 256, stride=1, downsample=False),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),
            BasicBlock(512, 512, stride=1, downsample=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LSTMClassifier(nn.Module):
    """
    Sequence classifier from Final-Competition; here it expects
    an embedding-like tensor (e.g., flattened mel frames).
    Note: kept for completeness, default model is CustomResNet.
    """

    def __init__(
        self,
        input_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        label_size: int,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.hidden2label = nn.Linear(hidden_size, label_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq expected shape: [B, T] or embedding provided; we keep minimal compatibility.
        embed = seq
        if embed.dim() == 2:
            embed = self.embedding(seq)
        # reshape to [T, B, F] for LSTM
        x = embed.transpose(0, 1)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        y = self.hidden2label(out[-1])
        return y


def create_model(model_type: str, in_channels: int = 1) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "custom_resnet":
        return CustomResNet(in_channels=in_channels, num_classes=NUM_CLASSES)
    elif model_type == "lstm":
        # LSTM not default; dimensions here are illustrative.
        return LSTMClassifier(
            input_size=128,  # e.g., mel bins
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            label_size=NUM_CLASSES,
            bidirectional=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# --------------------------------------------------------------------------- #
# Training loop (SGD + MultiStepLR, mixup optional), validation unchanged
# --------------------------------------------------------------------------- #


def train_one_epoch_sgd(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> float:
    model.train()
    avg_loss = 0.0
    steps = 0
    for x, label in tqdm(loader, desc="Training"):
        x, label = x.to(device), label.to(device)
        if mixup:
            x, label1, label2, lam = mixup_data(x, label, mixup_alpha)

        logits = model(x)

        if mixup:
            loss = mixup_criterion(criterion, logits, label1, label2, lam)
        else:
            loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        steps += 1
    return avg_loss / max(1, steps)


def train_model_sgd(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    milestones: Tuple[int, int, int] | None = None,
    mixup: bool = False,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    device: torch.device = DEVICE,
    save_path: str = "best_hybrid_model.pth",
    class_names=None,
    confusion_save_path: str | None = None,
):
    model = model.to(device)
    weight_tensor = class_weights.to(device) if class_weights is not None else None

    train_criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )
    val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    if milestones is None:
        milestones = (num_epochs // 4, num_epochs // 2, num_epochs * 3 // 4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_score = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_score": [],
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch_sgd(
            model, train_loader, train_criterion, optimizer, device, mixup=mixup
        )
        # Validation (unchanged evaluate)
        val_loss, val_acc, val_f1, val_score, _, _ = evaluate(
            model, val_loader, val_criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_score"].append(val_score)

        print(
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | score={val_score:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"*** Best model saved to {save_path} (score={best_score:.4f}) ***")

    # Optional confusion matrix with the best checkpoint
    if confusion_save_path is not None and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, _, _, _, preds, targets = evaluate(model, val_loader, val_criterion, device)
        names = class_names or [str(i) for i in range(NUM_CLASSES)]
        plot_confusion_matrix(targets, preds, names, normalize=True, save_path=confusion_save_path)

    return history


# --------------------------------------------------------------------------- #
# Pipelines (training / inference) reusing loaders and submission from base
# --------------------------------------------------------------------------- #


def run_training(
    train_folds=(1, 2, 3, 4, 5, 6, 7),
    val_folds=(8,),
    batch_size: int = 48,
    num_epochs: int = 30,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    model_type: str = "custom_resnet",
    mixup: bool = True,
    label_smoothing: float = 0.05,
    num_workers: int = 4,
):
    print("=" * 60)
    print("Hybrid Audio Classification - Training (SGD strategy)")
    print("=" * 60)

    cache_dir = BASE_DIR / "mel_cache"
    train_loader, val_loader = create_train_val_dataloaders(
        train_folds=train_folds,
        val_folds=val_folds,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        augment=True,
    )

    model = create_model(model_type, in_channels=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal params: {total_params:,} | Trainable: {trainable_params:,}")

    class_weights = compute_class_weights(train_folds)
    class_names = get_class_names()

    history = train_model_sgd(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        mixup=mixup,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        device=DEVICE,
        save_path="best_hybrid_model.pth",
        class_names=class_names,
        confusion_save_path="confmat_hybrid_val.png",
    )
    return model, history


def run_inference(
    model_path: str = "best_hybrid_model.pth",
    output_path: str = "submission.csv",
    model_type: str = "custom_resnet",
    batch_size: int = 64,
    num_workers: int = 4,
):
    print("=" * 60)
    print("Hybrid Audio Classification - Inference")
    print("=" * 60)

    cache_dir = BASE_DIR / "mel_cache"
    model = create_model(model_type, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Loaded model from {model_path}")

    test_loader = create_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    predictions = predict(model, test_loader, device=DEVICE)
    df = generate_submission(predictions, output_path=output_path)
    return df


def run_full_pipeline(model_type: str = "custom_resnet", num_epochs: int = 30):
    if not TRAIN_CSV.exists():
        print(f"Error: Training CSV not found at {TRAIN_CSV}")
        return

    model, _ = run_training(
        train_folds=(1, 2, 3, 4, 5, 6, 7),
        val_folds=(8,),
        batch_size=48,
        num_epochs=num_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        model_type=model_type,
        mixup=True,
        label_smoothing=0.05,
        num_workers=4,
    )

    if TEST_CSV.exists():
        run_inference(
            model_path="best_hybrid_model.pth",
            output_path="submission.csv",
            model_type=model_type,
        )
    else:
        print(f"Warning: Test CSV not found at {TEST_CSV}, skip inference.")


if __name__ == "__main__":
    set_seed(42, deterministic=False)

    parser = argparse.ArgumentParser(description="Hybrid audio classifier (SGD)")
    parser.add_argument("--model_type", type=str, default="custom_resnet", choices=["custom_resnet", "lstm"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=bool, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    args = parser.parse_args()

    run_training(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        mixup=args.mixup,
        label_smoothing=args.label_smoothing,
    )

