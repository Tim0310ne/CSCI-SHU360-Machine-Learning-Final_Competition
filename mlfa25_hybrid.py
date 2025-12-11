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

