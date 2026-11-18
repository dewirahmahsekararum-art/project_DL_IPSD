"""Evaluasi SimpleCNN pada Original Test tiap fold (5-fold cross validation).

Metrik per fold: confusion matrix, accuracy, precision, recall, F1, ROC-AUC (OvR).
Nilai akhir = rerata (± std) dari semua fold.

Contoh:
    python test.py --data_root /kaggle/input
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from Models.CNN import SimpleCNN
from Utils.evaluate import METRIC_KEYS, compute_metrics, plot_confusion_matrix, plot_roc
from Utils.getData import build_samples, default_data_root, discover_folds, getImageLabel


@torch.no_grad()
def predict(model, loader, device):
    """Return (label asli, probabilitas softmax) untuk seluruh isi loader."""
    model.eval()
    labels, probs = [], []
    for src, trg in loader:
        logits = model(src.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(trg.numpy())
    return np.concatenate(labels), np.concatenate(probs).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Test SimpleCNN - MSLD v2.0, 5-fold CV")
    parser.add_argument("--data_root", default=default_data_root())
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt_dir", default="./Checkpoints")
    parser.add_argument("--out_dir", default="./Results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_folds = discover_folds(args.data_root)

    rows, pooled_true, pooled_prob = [], [], []
    class_names = None
    for fold in args.folds:
        ckpt_path = Path(args.ckpt_dir) / f"SimpleCNN_fold{fold}.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint {ckpt_path} tidak ada. Jalankan train.py dulu.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        class_names = ckpt["class_names"]

        model = SimpleCNN(input_c=3, output=len(class_names), img_size=ckpt["img_size"],
                          hidden_dim=ckpt["hidden_dim"], dropout=ckpt["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # hanya Original Test, dengan pre-processing & statistik normalisasi dari training
        test_ds = getImageLabel(build_samples(all_folds[fold], "test", class_names), img_size=ckpt["img_size"],
                                clahe=ckpt["clahe"], mean=ckpt["mean"], std=ckpt["std"])
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        y_true, y_prob = predict(model, loader, device)
        y_pred = y_prob.argmax(axis=1)

        metrics = compute_metrics(y_true, y_prob)
        rows.append({"fold": fold, "n_test": len(y_true), "best_epoch": ckpt["epoch"], **metrics})
        pooled_true.append(y_true)
        pooled_prob.append(y_prob)

        plot_confusion_matrix(y_true, y_pred, class_names, out_dir / f"confusion_matrix_fold{fold}.png",
                              title=f"Confusion Matrix - fold {fold} (Test)")
        print(f"\n--- fold {fold} (n={len(y_true)}) ---")
        print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                    target_names=class_names, zero_division=0, digits=4))
        print("confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=range(len(class_names))))
        print({k: round(v, 4) for k, v in metrics.items()})

    # ---- ringkasan: rerata dan std dari semua fold -------------------------------
    df = pd.DataFrame(rows).set_index("fold")
    summary = pd.DataFrame({"mean": df[METRIC_KEYS].mean(), "std": df[METRIC_KEYS].std(ddof=1) if len(df) > 1 else 0.0})
    df.to_csv(out_dir / "metrics_per_fold.csv", float_format="%.4f")
    summary.to_csv(out_dir / "metrics_summary.csv", float_format="%.4f")
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump({"folds": args.folds, "mean": summary["mean"].to_dict(), "std": summary["std"].to_dict()}, f, indent=2)

    # ---- gambar gabungan (prediksi semua fold digabung) --------------------------
    y_true_all, y_prob_all = np.concatenate(pooled_true), np.concatenate(pooled_prob)
    plot_confusion_matrix(y_true_all, y_prob_all.argmax(axis=1), class_names, out_dir / "confusion_matrix.png",
                          title=f"Confusion Matrix - gabungan {len(args.folds)} fold (Test)")
    plot_roc(y_true_all, y_prob_all, class_names, out_dir / "roc_curve.png",
             title=f"ROC Curve - gabungan {len(args.folds)} fold (Test)")

    print(f"\n{'=' * 14} RERATA {len(args.folds)}-FOLD (mean ± std) {'=' * 14}")
    for key in METRIC_KEYS:
        print(f"{key:<20} {summary.loc[key, 'mean']:.4f} ± {summary.loc[key, 'std']:.4f}")
    print(f"\nHasil disimpan di {out_dir}")


if __name__ == "__main__":
    main()
