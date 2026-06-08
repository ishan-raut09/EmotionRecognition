import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.model import EmotionResNet


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        return ((1 - p) ** self.gamma * ce_loss).mean()


class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape(48, 48)
        image = np.repeat(pixels[:, :, None], repeats=3, axis=2)
        label = int(row["emotion"])

        if self.transform:
            image = self.transform(image)

        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FER2013 emotion recognition model.")
    parser.add_argument("--csv-path", default="data/fer2013.csv")
    parser.add_argument("--save-path", default="models/emotion_model.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--no-pretrained", action="store_true", help="Train without ImageNet weights.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1))], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25)], p=0.6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.4, 2.5)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def split_fer2013(df):
    required = {"emotion", "pixels", "Usage"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    train_df = df[df["Usage"] == "Training"].copy()
    val_df = df[df["Usage"] == "PublicTest"].copy()
    test_df = df[df["Usage"] == "PrivateTest"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("FER2013 CSV must include Training and PublicTest rows.")

    return train_df, val_df, test_df


def make_balanced_sampler(df):
    labels = df["emotion"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
    class_counts = np.maximum(class_counts, 1)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_class_weights(df, device):
    labels = df["emotion"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
    class_counts = np.maximum(class_counts, 1)
    weights = np.sqrt(class_counts.sum() / (len(EMOTION_LABELS) * class_counts))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def set_backbone_trainable(model, trainable):
    for name, param in model.model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable
    for param in model.model.fc.parameters():
        param.requires_grad = True


def build_optimizer(model, lr, backbone_lr_scale, weight_decay):
    head_params = []
    backbone_params = []
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": lr * backbone_lr_scale})
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device, use_tta=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            if use_tta:
                outputs = (outputs + model(torch.flip(images, dims=[3]))) / 2.0

            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for target, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[target, pred] += 1

    per_class_recall = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(EMOTION_LABELS), dtype=np.float64),
        where=confusion.sum(axis=1) != 0,
    )
    return {
        "loss": running_loss / total,
        "acc": 100.0 * correct / total,
        "balanced_acc": 100.0 * per_class_recall.mean(),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def print_class_report(metrics):
    print("Per-class recall:")
    for label, recall in zip(EMOTION_LABELS, metrics["per_class_recall"]):
        print(f"  {label:8s}: {recall * 100:.2f}%")


def save_checkpoint(path, model, args, epoch, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "image_size": args.image_size,
            "temperature": args.temperature,
            "best_val_acc": metrics["acc"],
            "best_val_balanced_acc": metrics["balanced_acc"],
            "labels": EMOTION_LABELS,
        },
        path,
    )


def train():
    args = parse_args()
    seed_everything(args.seed)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Training on {device}. Mixed precision: {'on' if use_amp else 'off'}")

    df = pd.read_csv(args.csv_path)
    train_df, val_df, test_df = split_fer2013(df)
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = FERDataset(train_df, transform=train_transform)
    val_dataset = FERDataset(val_df, transform=eval_transform)
    test_dataset = FERDataset(test_df, transform=eval_transform) if not test_df.empty else None

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_df),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if test_dataset is not None
        else None
    )

    model = EmotionResNet(pretrained=not args.no_pretrained).to(device)
    set_backbone_trainable(model, trainable=args.freeze_epochs == 0)

    class_weights = make_class_weights(train_df, device)
    criterion = FocalLoss(weight=class_weights, gamma=1.5, label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_balanced_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreezing ResNet backbone for fine-tuning.")
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - epoch + 1, 1),
                eta_min=args.lr * 0.01,
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, eval_criterion, device, use_tta=True)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
            f"val loss {val_metrics['loss']:.4f}, val acc {val_metrics['acc']:.2f}%, "
            f"balanced {val_metrics['balanced_acc']:.2f}%"
        )

        if val_metrics["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = val_metrics["balanced_acc"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(args.save_path, model, args, epoch, val_metrics)
            print(f"Saved new best checkpoint to {args.save_path}")
            print_class_report(val_metrics)
        else:
            patience_counter += 1
            print(f"No balanced-accuracy improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    print(f"Best validation balanced accuracy: {best_balanced_acc:.2f}% at epoch {best_epoch}")

    if test_loader is not None and os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, eval_criterion, device, use_tta=True)
        print(
            f"PrivateTest | loss {test_metrics['loss']:.4f}, "
            f"acc {test_metrics['acc']:.2f}%, balanced {test_metrics['balanced_acc']:.2f}%"
        )
        print_class_report(test_metrics)


if __name__ == "__main__":
    train()
