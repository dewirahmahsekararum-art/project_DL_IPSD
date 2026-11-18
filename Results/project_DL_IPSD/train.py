"""Training SimpleCNN pada MSLD v2.0 dengan 5-fold cross validation.

Per fold:
    train data = Original Train + Augmented Train
    valid data = Original Valid  (dipakai untuk memilih checkpoint terbaik)
Test data TIDAK disentuh di sini (lihat test.py).

Contoh:
    python train.py --data_root /kaggle/input --epochs 60
    python train.py --folds 1 --epochs 2          # uji cepat
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from Models.CNN import SimpleCNN
from Utils.evaluate import plot_training
from Utils.getData import (build_samples, default_data_root, discover_folds, get_class_names,
                           getImageLabel)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """Satu epoch. optimizer=None -> mode evaluasi. Return (loss rata-rata, akurasi)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for src, trg in loader:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            pred = model(src)
            loss = loss_function(pred, trg)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(trg)
            correct += (pred.argmax(dim=1) == trg).sum().item()
            n += len(trg)
    return total_loss / n, correct / n


def train_fold(fold, fold_info, class_names, args, device):
    print(f"\n{'=' * 20} FOLD {fold} {'=' * 20}")
    set_seed(args.seed + fold)

    t0 = time.time()
    train_ds = getImageLabel(build_samples(fold_info, "train", class_names), img_size=args.img_size,
                             clahe=args.clahe, train=True, online_aug=args.online_aug)
    val_ds = getImageLabel(build_samples(fold_info, "valid", class_names), img_size=args.img_size,
                           clahe=args.clahe, mean=train_ds.mean, std=train_ds.std)
    print(f"train={len(train_ds)} (Original+Augmented)  valid={len(val_ds)}  "
          f"[data dimuat {time.time() - t0:.0f}s]")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SimpleCNN(input_c=3, output=len(class_names), img_size=args.img_size,
                      hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_function = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    ckpt_path = Path(args.ckpt_dir) / f"SimpleCNN_fold{fold}.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_function, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_function, device)
        scheduler.step(val_loss)

        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)
        print(f"fold {fold} | epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f}"
              f" | valid loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "class_names": class_names,
                "mean": train_ds.mean, "std": train_ds.std,
                "img_size": args.img_size, "clahe": args.clahe,
                "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping: valid loss tidak membaik selama {args.patience} epoch")
                break

    print(f"fold {fold}: checkpoint terbaik = epoch {best_epoch} (valid loss {best_val:.4f}) -> {ckpt_path}")
    history["best_epoch"] = best_epoch
    return history


def main():
    parser = argparse.ArgumentParser(description="Train SimpleCNN - MSLD v2.0, 5-fold CV")
    parser.add_argument("--data_root", default=default_data_root())
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15, help="early stopping (epoch)")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clahe", action="store_true", help="pre-processing: CLAHE pada kanal L (LAB)")
    parser.add_argument("--online_aug", action="store_true", help="augmentasi online tambahan pada data train")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", default="./Checkpoints", help="TIDAK di-commit ke GitHub")
    parser.add_argument("--out_dir", default="./Results")
    args = parser.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    all_folds = discover_folds(args.data_root)
    missing = [f for f in args.folds if f not in all_folds]
    if missing:
        raise SystemExit(f"Fold {missing} tidak ditemukan. Tersedia: {list(all_folds)}")
    class_names = get_class_names(all_folds[args.folds[0]])
    print(f"kelas ({len(class_names)}): {class_names}")

    histories = {}
    for fold in args.folds:
        histories[f"fold{fold}"] = train_fold(fold, all_folds[fold], class_names, args, device)
        plot_training({f"fold{fold}": histories[f"fold{fold}"]}, Path(args.out_dir) / f"training_fold{fold}.png",
                      title=f"SimpleCNN - fold {fold}")

    plot_training(histories, Path(args.out_dir) / "training.png", title="SimpleCNN - 5-fold cross validation")
    with open(Path(args.out_dir) / "training_history.json", "w") as f:
        json.dump({"args": vars(args), "class_names": class_names, "histories": histories}, f, indent=2)
    print(f"\nSelesai. Gambar training & riwayat disimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
