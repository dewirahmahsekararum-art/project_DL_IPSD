"""Utilitas data untuk Mpox Skin Lesion Dataset v2.0 (MSLD v2.0).

Struktur dataset yang diharapkan (nama folder dicari otomatis, huruf besar/kecil
dan spasi/underscore tidak berpengaruh):

    <data_root>/
        Original Images/.../fold1/{Train,Valid,Test}/<kelas>/*.jpg
        Original Images/.../fold2/...
        Augmented Images/.../fold1_AUG/Train/<kelas>/*.jpg
        ...

Aturan tugas yang ditegakkan di sini:
    * data TRAIN  = Original Train + Augmented Train
    * data VALID  = Original Valid saja
    * data TEST   = Original Test  saja
    * Valid/Test TIDAK PERNAH dipakai untuk training.

Cek struktur dataset tanpa memuat gambar:
    python -m Utils.getData --data_root /kaggle/input
"""
import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}
FOLD_PATTERN = re.compile(r"fold[\s_\-]*(\d+)(?:[\s_\-]*aug(?:mented)?)?", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Penemuan struktur folder
# --------------------------------------------------------------------------- #
def default_data_root():
    """Lokasi default dataset: /kaggle/input di Kaggle, ./Dataset di lokal."""
    return "/kaggle/input" if os.path.isdir("/kaggle/input") else "./Dataset"


def discover_folds(data_root):
    """Cari semua folder fold di bawah data_root.

    Return: {nomor_fold: {"orig": Path, "aug": Path}}
    ("aug" hanya ada kalau folder augmented ditemukan untuk fold tersebut).
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root tidak ditemukan: {data_root}")

    folds = {}
    for dirpath, dirnames, _ in os.walk(data_root, followlinks=True):
        if not FOLD_PATTERN.fullmatch(os.path.basename(dirpath)):
            continue
        if not any(d.lower() in SPLIT_ALIASES["train"] for d in dirnames):
            continue
        k = int(FOLD_PATTERN.fullmatch(os.path.basename(dirpath)).group(1))
        kind = "aug" if "aug" in dirpath.lower() else "orig"
        folds.setdefault(k, {})[kind] = Path(dirpath)

    folds = {k: v for k, v in folds.items() if "orig" in v}
    if not folds:
        top = sorted(p.name for p in data_root.iterdir())[:20]
        raise FileNotFoundError(
            f"Tidak menemukan folder 'fold<N>' berisi Train/Valid/Test di {data_root}.\n"
            f"Isi teratas: {top}\n"
            f"Jalankan 'python -m Utils.getData --data_root <path>' dan cek path dataset."
        )
    return dict(sorted(folds.items()))


def _find_split_dir(fold_dir, split):
    for child in Path(fold_dir).iterdir():
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES[split]:
            return child
    return None


def get_class_names(fold_info):
    """Nama kelas (urut abjad) diambil dari folder Original Train."""
    train_dir = _find_split_dir(fold_info["orig"], "train")
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def _list_split(split_dir, class_to_idx):
    samples = []
    for cls_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        key = cls_dir.name.lower()
        if key not in class_to_idx:
            raise KeyError(
                f"Kelas '{cls_dir.name}' di {split_dir} tidak ada di daftar kelas "
                f"{sorted(class_to_idx)}"
            )
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                samples.append((str(f), class_to_idx[key]))
    return samples


def build_samples(fold_info, split, class_names):
    """Daftar (path_gambar, label) untuk satu split pada satu fold.

    train -> Original Train + Augmented Train
    valid -> Original Valid
    test  -> Original Test
    """
    class_to_idx = {c.lower(): i for i, c in enumerate(class_names)}
    sources = [("orig", split)]
    if split == "train" and "aug" in fold_info:
        sources.append(("aug", "train"))

    samples = []
    for kind, sp in sources:
        split_dir = _find_split_dir(fold_info[kind], sp)
        if split_dir is None:
            raise FileNotFoundError(f"Folder '{sp}' tidak ada di {fold_info[kind]}")
        samples += _list_split(split_dir, class_to_idx)
    return samples


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def preprocess_image(path, img_size, clahe=False):
    """Baca gambar -> RGB -> (opsional) CLAHE pada kanal L -> resize. Return uint8 HxWx3."""
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if clahe:
        lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        lab[..., 0] = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        img = cv.cvtColor(lab, cv.COLOR_LAB2RGB)
    return cv.resize(img, (img_size, img_size), interpolation=cv.INTER_AREA)


def augment_image(img):
    """Augmentasi ringan online (hanya untuk data train bila --online_aug aktif)."""
    if random.random() < 0.5:
        img = cv.flip(img, 1)
    if random.random() < 0.5:
        img = cv.flip(img, 0)
    h, w = img.shape[:2]
    m = cv.getRotationMatrix2D((w / 2, h / 2), random.uniform(-25, 25), 1.0)
    img = cv.warpAffine(img, m, (w, h), borderMode=cv.BORDER_REFLECT_101)
    return cv.convertScaleAbs(img, alpha=random.uniform(0.85, 1.15), beta=random.uniform(-15, 15))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class getImageLabel(Dataset):
    """Dataset gambar + label kelas (integer) untuk satu split pada satu fold.

    Semua gambar di-resize dan disimpan sebagai uint8 di RAM supaya epoch cepat.
    mean/std: statistik normalisasi per kanal; jika None dihitung dari data ini
    (pakai statistik data TRAIN untuk valid/test).
    """

    def __init__(self, samples, img_size=128, clahe=False, mean=None, std=None,
                 train=False, online_aug=False):
        self.img_size = img_size
        self.train = train
        self.online_aug = online_aug

        paths = [p for p, _ in samples]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
            images = list(pool.map(lambda p: preprocess_image(p, img_size, clahe), paths))

        keep = [i for i, im in enumerate(images) if im is not None]
        if len(keep) != len(images):
            print(f"[getData] PERINGATAN: {len(images) - len(keep)} gambar tidak terbaca, dilewati.")
        self.images = np.stack([images[i] for i in keep]).astype(np.uint8)
        self.labels = np.array([samples[i][1] for i in keep], dtype=np.int64)

        if mean is None or std is None:
            x = self.images.astype(np.float32) / 255.0
            mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __getitem__(self, item):
        img = self.images[item]
        if self.train and self.online_aug:
            img = augment_image(img)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1), int(self.labels[item])

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- #
# Inspeksi struktur dataset (tanpa memuat gambar)
# --------------------------------------------------------------------------- #
def inspect(data_root):
    folds = discover_folds(data_root)
    class_names = get_class_names(next(iter(folds.values())))
    print(f"data_root : {data_root}")
    print(f"kelas ({len(class_names)}): {class_names}")
    for k, info in folds.items():
        print(f"\n== fold {k} ==")
        for kind, path in info.items():
            print(f"  [{kind}] {path}")
        for split in ("train", "valid", "test"):
            samples = build_samples(info, split, class_names)
            counts = np.bincount([y for _, y in samples], minlength=len(class_names))
            detail = ", ".join(f"{c}={n}" for c, n in zip(class_names, counts))
            print(f"  {split:<5} total={len(samples):<5} {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cek struktur dataset MSLD v2.0")
    parser.add_argument("--data_root", default=default_data_root())
    inspect(parser.parse_args().data_root)
