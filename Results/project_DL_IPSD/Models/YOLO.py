"""Model ke-3: YOLOv8-cls (Ultralytics) untuk klasifikasi, dilatih dari nol."""
import os
import shutil
from pathlib import Path

import numpy as np

from Utils.getData import build_samples

YOLO_VARIANTS = {"n": "yolov8n-cls.yaml", "s": "yolov8s-cls.yaml", "m": "yolov8m-cls.yaml"}


def _link(src, dst):
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_fold_dir(fold, fold_info, class_names, out_root):
    """<out_root>/fold<k>/{train,val,test}/<kelas>/ berisi symlink ke gambar asli."""
    fold_dir = Path(out_root) / f"fold{fold}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    for split, folder in (("train", "train"), ("valid", "val"), ("test", "test")):
        for cls in class_names:
            (fold_dir / folder / cls).mkdir(parents=True, exist_ok=True)
        for i, (path, label) in enumerate(build_samples(fold_info, split, class_names)):
            _link(path, fold_dir / folder / class_names[label] / f"{i:06d}_{Path(path).name}")
    return fold_dir


def build_yolo(variant="n"):
    from ultralytics import YOLO
    return YOLO(YOLO_VARIANTS[variant])


def train_yolo(fold, data_dir, args, project_dir):
    model = build_yolo(args.variant)
    model.train(
        data=str(data_dir), imgsz=args.img_size, epochs=args.epochs, batch=args.batch_size,
        patience=args.patience, optimizer="AdamW", lr0=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, seed=args.seed + fold, deterministic=True,
        pretrained=False,
        workers=args.num_workers, project=str(Path(project_dir).resolve()),  # harus absolut
        name=f"fold{fold}", exist_ok=True,
        plots=False, verbose=True,
    )
    return Path(project_dir) / f"fold{fold}" / "weights" / "best.pt"


def read_history(project_dir, fold):
    import pandas as pd
    df = pd.read_csv(Path(project_dir) / f"fold{fold}" / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    val_acc = df["metrics/accuracy_top1"].tolist()
    return {
        "train_loss": df["train/loss"].tolist(),
        "val_loss": df["val/loss"].tolist(),
        "train_acc": [float("nan")] * len(df),
        "val_acc": val_acc,
        "best_epoch": int(np.argmax(val_acc)) + 1,
    }


def predict_proba(weights, image_paths, class_names, img_size, batch=64, device=None):
    """Probabilitas softmax (N, C), urutan kolom = class_names."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    name_to_col = {n.lower(): i for i, n in enumerate(class_names)}
    col_of = [name_to_col[model.names[i].lower()] for i in range(len(model.names))]

    probs = np.zeros((len(image_paths), len(class_names)), dtype=np.float64)
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        for k, r in enumerate(model.predict(chunk, imgsz=img_size, verbose=False, device=device)):
            p = r.probs.data.cpu().numpy()
            for src_i, dst_i in enumerate(col_of):
                probs[start + k, dst_i] = p[src_i]
    return probs
