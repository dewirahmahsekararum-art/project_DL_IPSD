import argparse
import json
from pathlib import Path

from Models.YOLO import prepare_fold_dir, read_history, train_yolo
from Utils.evaluate import plot_training
from Utils.getData import default_data_root, discover_folds, get_class_names


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-cls - MSLD v2.0, 5-fold CV")
    parser.add_argument("--data_root", default=default_data_root())
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--variant", default="n", choices=["n", "s", "m"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="./Checkpoints/YOLO_data")
    parser.add_argument("--ckpt_dir", default="./Checkpoints/YOLO")
    parser.add_argument("--out_dir", default="./Results/YOLO")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds = discover_folds(args.data_root)
    missing = [f for f in args.folds if f not in all_folds]
    if missing:
        raise SystemExit(f"Fold {missing} tidak ditemukan. Tersedia: {list(all_folds)}")
    class_names = get_class_names(all_folds[args.folds[0]])
    print(f"kelas ({len(class_names)}): {class_names}")

    histories = {}
    for fold in args.folds:
        print(f"\n{'=' * 20} FOLD {fold} {'=' * 20}")
        data_dir = prepare_fold_dir(fold, all_folds[fold], class_names, args.work_dir)
        best = train_yolo(fold, data_dir, args, args.ckpt_dir)
        histories[f"fold{fold}"] = read_history(args.ckpt_dir, fold)
        print(f"fold {fold}: checkpoint terbaik -> {best}")
        plot_training({f"fold{fold}": histories[f"fold{fold}"]}, out_dir / f"training_fold{fold}.png",
                      title=f"YOLOv8{args.variant}-cls - fold {fold}")

    plot_training(histories, out_dir / "training.png", title=f"YOLOv8{args.variant}-cls - 5-fold cross validation")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"args": vars(args), "class_names": class_names, "histories": histories}, f, indent=2)
    print(f"\nSelesai. Gambar training & riwayat disimpan di {out_dir}")


if __name__ == "__main__":
    main()
