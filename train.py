
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset

from fcifnet import FCIFNet


class NpzPatchDataset(Dataset):
    def __init__(self, archive: np.lib.npyio.NpzFile, split: str):
        required = [f"{split}_hsi", f"{split}_lidar", f"{split}_labels"]
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Missing NPZ keys: {missing}")
        self.hsi = archive[required[0]]
        self.lidar = archive[required[1]]
        self.labels = archive[required[2]].astype(np.int64, copy=False)
        if not (len(self.hsi) == len(self.lidar) == len(self.labels)):
            raise ValueError(f"Mismatched {split} sample counts")
        if self.labels.size and self.labels.min() < 0:
            raise ValueError("Labels must be zero-based non-negative integers")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.as_tensor(self.hsi[index], dtype=torch.float32),
            torch.as_tensor(self.lidar[index], dtype=torch.float32),
            torch.as_tensor(self.labels[index], dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the rebuttal-consistent FCIFNet")
    parser.add_argument("--data", type=Path, required=True, help="Prepared NPZ patch archive")
    parser.add_argument("--output", type=Path, default=Path("runs/fcifnet"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lambda-aux", type=float, default=0.0)
    parser.add_argument("--lambda-js", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=1e-5)
    parser.add_argument("--lambda-triplet", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=2.0)
    parser.add_argument("--triplet-start", type=int, default=15)
    parser.add_argument("--js-temperature", type=float, default=0.5)
    parser.add_argument("--tau2", type=float, default=0.9)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_channels(array: np.ndarray, name: str) -> int:
    if array.ndim == 4:  
        return int(array.shape[1])
    if array.ndim == 5:
        if array.shape[1] == 1:  
            return int(array.shape[2])
        if array.shape[2] == 1:  
            return int(array.shape[1])
    raise ValueError(f"Unsupported {name} shape: {array.shape}")


def make_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def compute_metrics(labels, predictions):
    return {
        "oa": float(accuracy_score(labels, predictions)),
        "aa": float(balanced_accuracy_score(labels, predictions)),
        "kappa": float(cohen_kappa_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for hsi, lidar, target in loader:
        output = model(hsi.to(device), lidar.to(device))["main_out"]
        labels.extend(target.numpy().tolist())
        predictions.extend(output.argmax(dim=1).cpu().numpy().tolist())
    return compute_metrics(labels, predictions)


def atomic_save(payload, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.data, allow_pickle=False)
    train_set = NpzPatchDataset(archive, "train")
    val_set = NpzPatchDataset(archive, "val")
    test_set = NpzPatchDataset(archive, "test")

    in_ch_hsi = infer_channels(train_set.hsi, "HSI")
    in_ch_lidar = infer_channels(train_set.lidar, "LiDAR")
    all_labels = np.concatenate([train_set.labels, val_set.labels, test_set.labels])
    num_classes = int(all_labels.max()) + 1

    train_loader = make_loader(train_set, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False, args.seed)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False, args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = FCIFNet(
        in_ch1=in_ch_hsi,
        in_ch2=in_ch_lidar,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        js_temperature=args.js_temperature,
        triplet_margin=args.triplet_margin,
        enable_triplet_after_epochs=args.triplet_start,
        hfce_residual_scale=0.7,
        disable_b1_gate=True,
        tau2=args.tau2,
        use_triplet_loss=args.lambda_triplet > 0,
        compute_decorr_mi_loss=args.lambda_sparse > 0,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_val_oa = -1.0
    best_path = args.output / "best_fcifnet.pt"

    for epoch in range(args.epochs):
        model.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for hsi, lidar, target in train_loader:
            hsi = hsi.to(device, non_blocking=True)
            lidar = lidar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                result = model(hsi, lidar, target)
                main_loss = criterion(result["main_out"], target)
                aux_loss = criterion(result["aux_out"], target)
                triplet = result["triplet_loss"]
                if triplet is None:
                    triplet = main_loss.new_zeros(())
                loss = (
                    main_loss
                    + args.lambda_aux * aux_loss
                    + args.lambda_js * result["js_diff"]
                    + args.lambda_sparse * result["loss_sparse"]
                    + args.lambda_triplet * triplet
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        if (epoch + 1) % 10 == 0:
            print(json.dumps(record, ensure_ascii=False))

        if val_metrics["oa"] > best_val_oa:
            best_val_oa = val_metrics["oa"]
            atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "in_ch_hsi": in_ch_hsi,
                    "in_ch_lidar": in_ch_lidar,
                    "num_classes": num_classes,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["val_metrics"],
        "test": test_metrics,
    }
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Best result:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
