"""Metrik evaluasi dan fungsi plot (confusion matrix, ROC, kurva training)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

METRIC_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_macro_ovr"]


def compute_metrics(y_true, y_prob):
    """Hitung semua metrik dari label asli (N,) dan probabilitas softmax (N, C)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    n_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)

    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # ROC-AUC one-vs-rest (macro). Jika ada kelas yang tidak muncul di y_true,
    # AUC dihitung hanya pada kelas yang ada.
    present = np.unique(y_true)
    if len(present) == n_classes:
        out["roc_auc_macro_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    else:
        y_bin = label_binarize(y_true, classes=range(n_classes))
        out["roc_auc_macro_ovr"] = float(np.mean([roc_auc_score(y_bin[:, c], y_prob[:, c]) for c in present]))
    return out


def plot_confusion_matrix(y_true, y_pred, class_names, path, title="Confusion Matrix"):
    """Heatmap confusion matrix: warna = proporsi per baris, angka = jumlah gambar."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2f})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporsi per kelas asli")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm


def plot_roc(y_true, y_prob, class_names, path, title="ROC Curve (one-vs-rest)"):
    """Kurva ROC satu-lawan-semua untuk tiap kelas."""
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n))
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_bin[:, c], y_prob[:, c]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(histories, path, title=None):
    """histories: {nama: {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}}"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], color=c, label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], color=c, linestyle="--", label=f"{name} valid")
        axes[1].plot(epochs, h["train_acc"], color=c, label=f"{name} train")
        axes[1].plot(epochs, h["val_acc"], color=c, linestyle="--", label=f"{name} valid")
    axes[0].set_ylabel("Loss (CrossEntropy)")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
