#!/usr/bin/env python3
import os, time, random, glob, json, yaml, argparse
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # hiệu năng tốt cho CNN (không cần tuyệt đối tái lập)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def resolve_dataset_root(yaml_path: Path, cfg: dict) -> Path:
    """
    YOLO (Ultralytics) resolve các path tương đối dựa trên VỊ TRÍ FILE YAML.
    Để đếm file chính xác, ta cũng phải resolve theo nguyên tắc đó.
    """
    base = yaml_path.parent
    rel = cfg.get("path", ".")
    return (base / rel).resolve()

def count_split(dataset_root: Path, cfg: dict, split: str):
    img_dir = dataset_root / cfg[split]
    lbl_dir = dataset_root / str(cfg[split]).replace("images", "labels")
    imgs = [p for p in glob.glob(str(img_dir / "**" / "*.*"), recursive=True)
            if p.lower().endswith(IMG_EXTS)]
    lbls = glob.glob(str(lbl_dir / "**" / "*.txt"), recursive=True)
    return {
        "split": split,
        "images": len(imgs),
        "labels": len(lbls),
        "img_dir": str(img_dir),
        "lbl_dir": str(lbl_dir),
    }

def pick_device(device_arg: str):
    """
    device_arg: "auto" | "cpu" | "0" | "0,1" ...
    """
    if device_arg.lower() == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_arg.lower() == "cpu":
        return "cpu"
    # cho phép truyền "0" hoặc "0,1"
    return device_arg

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train & Evaluate YOLO (Ultralytics) trên dataset YOLO format."
    )
    parser.add_argument("--yaml", type=str, required=True,
                        help="Đường dẫn tới dataset YAML (vd: /path/to/synth/data.yaml)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Checkpoint YOLO (vd: yolo11n.pt, yolo11s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Tỉ lệ dữ liệu dùng để train (0-1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help='GPU index "0" hoặc "0,1", hoặc "cpu", hoặc "auto"')
    parser.add_argument("--project", type=str, default="runs",
                        help="Thư mục gốc lưu kết quả (train/val/predict).")
    parser.add_argument("--name", type=str, default=None,
                        help="Tên run (mặc định: train_YYYYmmdd_HHMMSS)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold khi predict.")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Số lượng bbox tối đa mỗi ảnh khi predict.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience.")
    args = parser.parse_args()

    set_seed(args.seed)

    yaml_path = Path(args.yaml).expanduser().resolve()
    assert yaml_path.exists(), f"Không tìm thấy YAML: {yaml_path}"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_root = resolve_dataset_root(yaml_path, cfg)
    # Log nhanh số lượng ảnh/nhãn
    ds_stats = [count_split(dataset_root, cfg, s) for s in ("train", "val", "test")]
    print(json.dumps(ds_stats, indent=2, ensure_ascii=False))

    # Thông số Train
    run_name = args.name or f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    dev = pick_device(args.device)
    print(f"Training with: batch={args.batch}, epochs={args.epochs}, "
          f"fraction={args.fraction}, workers={args.workers}, imgsz={args.imgsz}, "
          f"device={dev}, run={run_name}")

    model = YOLO(args.model)

    results = model.train(
        data=str(yaml_path),     # để Ultralytics tự resolve path theo vị trí YAML
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=dev,
        batch=args.batch,
        workers=args.workers,
        cos_lr=True,
        fraction=args.fraction,
        plots=True,
        project=args.project,
        name=run_name,
        verbose=True,
        fliplr=0,               # vô hiệu hóa lật ngang
        cache=True,
        patience=args.patience,
        save_period=5,
        dropout=0.2,
        hsv_h=0.05,
        degrees=10,
        translate=0.1,
        shear=10,
        perspective=0.0005,
        mixup=0.2,
        cutmix=0.2
    )

    # Lưu đường dẫn run
    run_dir = Path(results.save_dir)
    print("✔ Train done. Save dir:", run_dir)

    best_w = run_dir / "weights" / "best.pt"
    last_w = run_dir / "weights" / "last.pt"
    assert best_w.exists() or last_w.exists(), "Không thấy best.pt/last.pt"
    ckpt = best_w if best_w.exists() else last_w
    print("run_dir:", run_dir)
    print("ckpt:", ckpt)

    # Đánh giá trên TEST
    print("⏳ Evaluating on TEST split...")
    model_best = YOLO(str(ckpt))
    tmetrics = model_best.val(
        device=dev,
        batch=max(8, min(32, int(args.batch))),
        split="test",
        plots=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
        data=str(yaml_path)  # đảm bảo dùng đúng YAML
    )

    # Log metrics
    print(f"mAP50-95: {float(tmetrics.box.map):.4f}")
    print(f"mAP50   : {float(tmetrics.box.map50):.4f}")
    print(f"mAP75   : {float(tmetrics.box.map75):.4f}")

    # Per-class mAP & names
    maps = getattr(getattr(tmetrics, "box", None), "maps", None)
    per_class = np.asarray(maps, dtype=float).ravel().tolist() if maps is not None else []

    names = getattr(getattr(model_best, "model", None), "names", None) or getattr(model_best, "names", None)
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    elif not isinstance(names, (list, tuple)):
        names = []

    # Dump JSON
    metrics_out = run_dir / "metrics_test.json"
    payload = {
        "mAP50-95": float(tmetrics.box.map),
        "mAP50": float(tmetrics.box.map50),
        "mAP75": float(tmetrics.box.map75),
        "per_class_mAP50-95": per_class,
        "names": names,
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("✔ Saved metrics ->", metrics_out)

    if per_class and names and len(per_class) == len(names):
        topk = sorted(zip(names, per_class), key=lambda x: x[1], reverse=True)[:10]
        print("Top-10 lớp theo mAP50-95:")
        for n, v in topk:
            print(f"  {n:>24s}: {v:.4f}")

    # Predict trên thư mục TEST và lưu kết quả
    test_dir = dataset_root / cfg["test"]
    all_imgs = list(test_dir.rglob("*"))
    # lọc file ảnh hợp lệ
    all_imgs = [p for p in all_imgs if p.suffix.lower() in IMG_EXTS]
    sample_imgs = random.sample(all_imgs, min(10, len(all_imgs)))

    pred_name = f"pred_test_best_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"⏳ Predict on {len(sample_imgs)} random TEST images...")

    pred = model_best.predict(
        source=[str(p) for p in sample_imgs],  # chỉ 10 ảnh
        imgsz=args.imgsz,
        device=dev,
        conf=args.conf,
        max_det=args.max_det,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=pred_name
    )

    # Cố gắng lấy thư mục lưu predict từ kết quả
    pred_dir = None
    try:
        if isinstance(pred, list) and pred:
            sd = getattr(pred[0], "save_dir", None)
            if sd:
                pred_dir = Path(sd)
    except Exception:
        pass
    if not pred_dir:
        # fallback (chuẩn Ultralytics: project/<task>/name)
        # task thường là 'detect'
        pred_dir = Path(args.project) / "detect" / pred_name

    print("✔ Pred saved to:", pred_dir)

if __name__ == "__main__":
    main()