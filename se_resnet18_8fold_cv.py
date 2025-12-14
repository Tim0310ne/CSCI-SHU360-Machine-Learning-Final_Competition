"""
SE-ResNet18 (from scratch) training pipeline for MLFA Urban Sound competition.

Strict requirements satisfied:
- ResNet18 initialized from scratch (no pretrained weights)
- 8-fold CV strictly by `fold` column (no reshuffle/resplit)
- Input: 3 channels = [Log-Mel, Delta, Delta-Delta]
  - n_fft=1024, hop_length=256
- Normalization:
  - waveform peak normalization (before spectrogram)
  - per-channel standardization (mean/std) on spectrogram channels
- Augmentations:
  - waveform: random time shift, gaussian noise, random gain
  - spectrogram: SpecAugment (freq + time masks)
  - mixup: alpha in [0.2, 0.4], mixes inputs + labels
- Loss:
  - weighted (inverse frequency) + label smoothing
- Final inference: soft ensemble of 8 fold models (avg probabilities)

Run (example):
  python MLFA/se_resnet18_8fold_cv.py --data_root MLFA/Kaggle_Data --epochs 35 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Audio / feature config
# ----------------------------

@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration_s: float = 4.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 8000.0


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # waveform: (1, T) or (C, T)
    peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return waveform / peak


def pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    # waveform: (C, T)
    t = waveform.shape[-1]
    if t == target_len:
        return waveform
    if t > target_len:
        return waveform[..., :target_len]
    pad = target_len - t
    return F.pad(waveform, (0, pad))


def random_time_shift(waveform: torch.Tensor, max_shift_pct: float = 0.1) -> torch.Tensor:
    # Roll in time dimension
    t = waveform.shape[-1]
    max_shift = int(t * max_shift_pct)
    if max_shift <= 0:
        return waveform
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(waveform, shifts=shift, dims=-1)


def random_gain(waveform: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2) -> torch.Tensor:
    gain = random.uniform(min_gain, max_gain)
    return waveform * gain


def add_gaussian_noise(waveform: torch.Tensor, noise_std: float = 0.005) -> torch.Tensor:
    # noise_std is relative to full-scale (since waveform is peak-normalized before aug)
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


def spec_augment(
    x: torch.Tensor,
    freq_mask_param: int = 18,
    time_mask_param: int = 28,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    Apply SpecAugment masks to multi-channel spectrogram.
    x: (C, H, W) where H=n_mels, W=time
    Masks are shared across channels to keep alignment.
    """
    c, h, w = x.shape
    out = x.clone()

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, h))
        if f == 0:
            continue
        f0 = random.randint(0, max(0, h - f))
        out[:, f0 : f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, w))
        if t == 0:
            continue
        t0 = random.randint(0, max(0, w - t))
        out[:, :, t0 : t0 + t] = 0.0

    return out


def compute_logmel_3ch(
    waveform: torch.Tensor,
    sr: int,
    cfg: AudioConfig,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    """
    waveform: (1, T) float32
    returns: (3, n_mels, time)
    """
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=cfg.sample_rate)

    target_len = int(cfg.sample_rate * cfg.duration_s)
    waveform = pad_or_trim(waveform, target_len)

    # Waveform peak normalization (required)
    waveform = peak_normalize(waveform)

    mel = mel_transform(waveform)  # (1, n_mels, time), power spectrogram
    mel = mel.squeeze(0)  # (n_mels, time)

    log_mel = torch.log(mel.clamp_min(1e-6))
    delta = torchaudio.functional.compute_deltas(log_mel)
    delta2 = torchaudio.functional.compute_deltas(delta)

    x = torch.stack([log_mel, delta, delta2], dim=0)  # (3, H, W)
    return x


def sample_mixup_lambda(alpha_low: float = 0.2, alpha_high: float = 0.4) -> float:
    alpha = random.uniform(alpha_low, alpha_high)
    lam = np.random.beta(alpha, alpha)
    # common trick: keep lam >= 0.5 so one sample dominates (more stable)
    lam = float(max(lam, 1.0 - lam))
    return lam


class AudioDataset(Dataset):
    """
    Dataset that:
    - loads waveform via torchaudio
    - applies waveform peak norm (required) before spectrogram conversion
    - builds 3-channel [log-mel, delta, delta-delta]
    - applies per-channel standardization (mean/std) if provided
    - optionally applies waveform aug + SpecAugment
    - optionally applies Mixup (returns y_a, y_b, lam)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        cfg: AudioConfig,
        num_classes: int,
        training: bool,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        enable_wave_aug: bool = True,
        enable_spec_aug: bool = True,
        enable_mixup: bool = True,
        mixup_prob: float = 1.0,
        mixup_alpha_low: float = 0.2,
        mixup_alpha_high: float = 0.4,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.cfg = cfg
        self.num_classes = num_classes
        self.training = training

        self.mean = mean  # (3,)
        self.std = std    # (3,)

        self.enable_wave_aug = enable_wave_aug and training
        self.enable_spec_aug = enable_spec_aug and training
        self.enable_mixup = enable_mixup and training
        self.mixup_prob = mixup_prob
        self.mixup_alpha_low = mixup_alpha_low
        self.mixup_alpha_high = mixup_alpha_high

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        fname = row["slice_file_name"]
        y = int(row["classID"]) if "classID" in row else -1

        # Training audio lives in audio/fold{fold}/file.wav
        if "fold" in row:
            fold = int(row["fold"])
            wav_path = self.audio_dir / f"fold{fold}" / fname
        else:
            # Test audio assumed in audio/test/
            wav_path = self.audio_dir / "test" / fname

        waveform, sr = torchaudio.load(str(wav_path))  # (C, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.enable_wave_aug:
            # Apply augmentations after base peak norm happens inside feature extractor.
            # We do them here by temporarily normalizing first for stable magnitudes.
            wf = peak_normalize(waveform)
            wf = random_time_shift(wf, max_shift_pct=0.1)
            wf = add_gaussian_noise(wf, noise_std=0.005)
            wf = random_gain(wf, min_gain=0.8, max_gain=1.2)
            waveform = wf

        x = compute_logmel_3ch(waveform, sr, self.cfg, self._mel)  # (3, H, W)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)

        if self.enable_spec_aug:
            x = spec_augment(x)

        return x, y

    def __getitem__(self, idx: int):
        x, y = self._load_item(idx)

        if self.enable_mixup and random.random() < self.mixup_prob:
            j = random.randint(0, len(self.df) - 1)
            x2, y2 = self._load_item(j)
            lam = sample_mixup_lambda(self.mixup_alpha_low, self.mixup_alpha_high)
            x = x * lam + x2 * (1.0 - lam)
            return x, int(y), int(y2), float(lam)

        return x, int(y), int(y), 1.0


# ----------------------------
# Model: SE-ResNet18 (from scratch)
# ----------------------------

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction=reduction)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SEResNet(nn.Module):
    def __init__(self, block: type, layers: Sequence[int], num_classes: int = 10) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc(x)
        return x


def se_resnet18(num_classes: int) -> nn.Module:
    # ResNet18 = [2,2,2,2]
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ----------------------------
# Loss: weighted + label smoothing, supports mixup
# ----------------------------

def weighted_label_smoothed_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    """
    logits: (N, C)
    target: (N,)
    class_weight: (C,) or None
    """
    n, c = logits.shape
    log_probs = F.log_softmax(logits, dim=1)

    nll = -log_probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1)  # (N,)

    if label_smoothing > 0.0:
        # mean log prob over incorrect classes (exact smoothing over C-1)
        lp_true = log_probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1)
        smooth = -(log_probs.sum(dim=1) - lp_true) / (c - 1)
        loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        loss = nll

    if class_weight is not None:
        w = class_weight.gather(dim=0, index=target)
        loss = loss * w

    return loss


def mixup_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: torch.Tensor,
    class_weight: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    """
    Mixup loss computed per-sample, then averaged.
    lam: (N,) in [0,1]
    """
    loss_a = weighted_label_smoothed_ce(logits, y_a, class_weight, label_smoothing)  # (N,)
    loss_b = weighted_label_smoothed_ce(logits, y_b, class_weight, label_smoothing)  # (N,)
    lam = lam.to(dtype=loss_a.dtype)
    loss = lam * loss_a + (1.0 - lam) * loss_b
    return loss.mean()


# ----------------------------
# Metrics
# ----------------------------

@torch.no_grad()
def eval_macro_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    correct = 0
    total = 0

    for x, y_a, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y_a.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    acc = correct / max(1, total)
    f1 = float(f1_score(all_targets, all_preds, average="macro"))
    return f1, acc


# ----------------------------
# Mean/Std (per channel) on training folds only
# ----------------------------

@torch.no_grad()
def compute_channel_mean_std(dataset: Dataset, device: torch.device, max_items: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-channel mean/std over all pixels.
    dataset __getitem__ must return (x, y_a, y_b, lam) with x: (3,H,W)
    """
    n_seen = 0
    sum_c = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_c = torch.zeros(3, device=device, dtype=torch.float64)
    count = 0

    for i in tqdm(range(len(dataset)), desc="mean/std", leave=False):
        x, _, _, _ = dataset[i]
        x = x.to(device=device, dtype=torch.float64)
        # (3,H,W) -> (3, HW)
        x2 = x.view(3, -1)
        sum_c += x2.sum(dim=1)
        sumsq_c += (x2 ** 2).sum(dim=1)
        count += x2.shape[1]
        n_seen += 1
        if max_items is not None and n_seen >= max_items:
            break

    mean = (sum_c / max(1, count)).to(dtype=torch.float32).cpu()
    var = (sumsq_c / max(1, count) - mean.to(device=device, dtype=torch.float64) ** 2).clamp_min(1e-12)
    std = torch.sqrt(var).to(dtype=torch.float32).cpu()
    return mean, std


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = train_df["classID"].value_counts().to_dict()
    freq = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freq = np.clip(freq, 1.0, None)
    inv = 1.0 / freq
    w = inv / inv.mean()  # normalize so mean weight ~1
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------
# Training
# ----------------------------

def train_one_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    audio_dir: Path,
    cfg: AudioConfig,
    num_classes: int,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    num_workers: int,
    amp: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean/std on training folds only (no aug, no mixup)
    meanstd_path = fold_dir / "meanstd.pt"
    if meanstd_path.exists():
        meanstd = torch.load(meanstd_path, map_location="cpu")
        mean = meanstd["mean"]
        std = meanstd["std"]
    else:
        base_ds = AudioDataset(
            df=train_df,
            audio_dir=audio_dir,
            cfg=cfg,
            num_classes=num_classes,
            training=False,  # disables aug/mixup
            mean=None,
            std=None,
        )
        mean, std = compute_channel_mean_std(base_ds, device=device, max_items=None)
        torch.save({"mean": mean, "std": std}, meanstd_path)

    # Datasets/loaders
    train_ds = AudioDataset(
        df=train_df,
        audio_dir=audio_dir,
        cfg=cfg,
        num_classes=num_classes,
        training=True,
        mean=mean,
        std=std,
        enable_wave_aug=True,
        enable_spec_aug=True,
        enable_mixup=True,
        mixup_prob=1.0,
        mixup_alpha_low=0.2,
        mixup_alpha_high=0.4,
    )
    val_ds = AudioDataset(
        df=val_df,
        audio_dir=audio_dir,
        cfg=cfg,
        num_classes=num_classes,
        training=False,
        mean=mean,
        std=std,
        enable_wave_aug=False,
        enable_spec_aug=False,
        enable_mixup=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # shuffling order is allowed; splits remain fold-pure
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model (from scratch)
    model = se_resnet18(num_classes=num_classes).to(device)

    # Class weights from training folds only (required)
    class_weight = compute_class_weights(train_df, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(5, epochs // 3), T_mult=2)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_f1 = -1.0
    best_path = fold_dir / "best.pt"

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"fold {fold_id} | epoch {epoch}/{epochs}", leave=False)
        for x, y_a, y_b, lam in pbar:
            x = x.to(device, non_blocking=True)
            y_a = y_a.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            lam_t = torch.tensor(lam, device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = mixup_loss(
                    logits=logits,
                    y_a=y_a,
                    y_b=y_b,
                    lam=lam_t,
                    class_weight=class_weight,
                    label_smoothing=label_smoothing,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(epoch - 1 + (n_batches / max(1, len(train_loader))))

            running += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=running / max(1, n_batches), lr=optimizer.param_groups[0]["lr"])

        val_f1, val_acc = eval_macro_f1(model, val_loader, device)
        entry = {"epoch": epoch, "val_macro_f1": val_f1, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(entry)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "cfg": cfg.__dict__,
                    "num_classes": num_classes,
                    "best_val_macro_f1": best_f1,
                    "fold": fold_id,
                },
                best_path,
            )

    with open(fold_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_path


# ----------------------------
# Inference: soft ensemble
# ----------------------------

@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs_all: List[np.ndarray] = []
    for x, _, _, _ in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    # Hard-coded HPC paths (same style as MLFA/resnet18.py)
    parser.add_argument("--data_root", type=str, default="/scratch/tl3735/MLFA/Kaggle_Data")
    parser.add_argument("--out_dir", type=str, default="/scratch/tl3735/MLFA/runs/se_resnet18_8fold")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--make_submission", action="store_true", help="After CV, ensemble 8 folds on test and write CSV.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    train_csv = data_root / "metadata" / "kaggle_train.csv"
    test_csv = data_root / "metadata" / "kaggle_test.csv"
    audio_dir = data_root / "audio"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    num_classes = int(df["classID"].nunique())
    assert num_classes == 10, f"Expected 10 classes, got {num_classes}"

    cfg = AudioConfig()

    best_paths: List[Path] = []
    folds = sorted(df["fold"].unique().tolist())
    assert len(folds) == 8, f"Expected 8 folds, got {folds}"

    for fold_id in folds:
        val_df = df[df["fold"] == fold_id].copy()
        train_df = df[df["fold"] != fold_id].copy()

        best_path = train_one_fold(
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            audio_dir=audio_dir,
            cfg=cfg,
            num_classes=num_classes,
            out_dir=out_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        best_paths.append(best_path)

    with open(out_dir / "best_models.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in best_paths], f, indent=2)

    if not args.make_submission:
        print(f"Done. Best model checkpoints saved under: {out_dir}")
        return

    # Build test dataset/loader using mean/std from each fold checkpoint, then ensemble probs.
    # We load per-fold mean/std from checkpoint to avoid leakage across folds.
    sample_sub = pd.read_csv(data_root / "metadata" / "kaggle_sample_submission.csv")
    # Ensure test_df ordering matches submission order by slice_file_name
    test_df = test_df.set_index("slice_file_name").loc[sample_sub["slice_file_name"]].reset_index()

    all_fold_probs: List[np.ndarray] = []
    for ckpt_path in best_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mean = ckpt["mean"]
        std = ckpt["std"]

        model = se_resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model"], strict=True)

        test_ds = AudioDataset(
            df=test_df,
            audio_dir=audio_dir,
            cfg=cfg,
            num_classes=num_classes,
            training=False,
            mean=mean,
            std=std,
            enable_wave_aug=False,
            enable_spec_aug=False,
            enable_mixup=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        probs = predict_proba(model, test_loader, device)
        all_fold_probs.append(probs)

    probs_ens = np.mean(np.stack(all_fold_probs, axis=0), axis=0)  # (N, C)
    preds = probs_ens.argmax(axis=1)

    sub = sample_sub.copy()
    sub["classID"] = preds

    sub_dir = Path(__file__).parent / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sub_dir / "submission_se_resnet18_8fold_soft_ens.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Submission saved to: {out_csv}")


if __name__ == "__main__":
    main()


