#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, argparse, random
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# ================== Utils ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pick_device(device_arg):
    """
    Trả về 'cpu' hoặc 'cuda:{i}' theo đầu vào:
    - int -> 'cuda:{i}'
    - 'auto' -> 'cuda:0' nếu có CUDA, ngược lại 'cpu'
    - 'cpu' -> 'cpu'
    - '0' (chuỗi số) -> 'cuda:0'
    - 'cuda:1' -> giữ nguyên
    """
    if isinstance(device_arg, int):
        return f"cuda:{device_arg}"

    if device_arg is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    s = str(device_arg).strip().lower()

    if s == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if s == "cpu":
        return "cpu"
    if s.isdigit():                 # ví dụ "0", "1"
        return f"cuda:{s}"
    if s.startswith("cuda:"):       # ví dụ "cuda:2"
        return s

    # fallback: cho các giá trị mơ hồ như "cuda"/"gpu"
    if s in ("cuda", "gpu"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    return s  # trả nguyên trạng nếu không khớp case nào

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

# ================== Experiments ==================
def default_experiments():
    """
    Mỗi item là (tên_run, dict các override đầu vào cho model.train).
    Luôn set fliplr=0 theo yêu cầu, giữ cos_lr=True, cache=True, save_period=5.
    """
    exps = []

    # exps.append((
    #     "heavy_aug_adamw",
    #     dict(
    #         optimizer="AdamW",
    #         epochs=250,
    #         lr0=6e-4, lrf=0.01, weight_decay=0.012,
    #         hsv_h=0.07, hsv_s=0.6, hsv_v=0.6,
    #         degrees=15, translate=0.15, scale=0.60, shear=12,
    #         perspective=0.0010,
    #         mixup=0.35, cutmix=0.35,
    #         mosaic=0.9,
    #         dropout=0.2,
    #     )
    # ))

    # exps.append((
    #     "no_mosaic_sgd",
    #     dict(
    #         optimizer="SGD",
    #         epochs=250,
    #         lr0=0.005, lrf=0.01, momentum=0.9, weight_decay=5e-4,
    #         hsv_h=0.04, hsv_s=0.4, hsv_v=0.4,
    #         degrees=8, translate=0.08, scale=0.35, shear=8,
    #         perspective=0.0003,
    #         mixup=0.0, cutmix=0.0,
    #         mosaic=0.0,
    #         dropout=0.1,
    #     )
    # ))

    exps.append((
        "no_mosaic_sgd_ms",
        dict(
            optimizer="SGD",
            epochs=250,
            lr0=0.005, lrf=0.01, momentum=0.9, weight_decay=5e-4,
            hsv_h=0.04, hsv_s=0.4, hsv_v=0.4,
            degrees=8, translate=0.08, scale=0.35, shear=8,
            perspective=0.0003,
            mixup=0.0, cutmix=0.0,
            mosaic=0.0,
            dropout=0.1,
            multi_scale=True
        )
    ))
    # exps.append((
    #     "no_mosaic_sgd_ms_degrees",
    #     dict(
    #         optimizer="SGD",
    #         epochs=250,
    #         lr0=0.005, lrf=0.01, momentum=0.9, weight_decay=5e-4,
    #         hsv_h=0.04, hsv_s=0.4, hsv_v=0.4,
    #         degrees=30, translate=0.08, scale=0.35, shear=8,
    #         perspective=0.0003,
    #         mixup=0.0, cutmix=0.0,
    #         mosaic=0.0,
    #         dropout=0.1,
    #         multi_scale=True
    #     )
    # ))

    # exps.append((
    #     "base_ms",
    #     dict(
    #         optimizer="SGD",
    #         epochs=250,
    #         multi_scale=True
    #     )
    # ))
    
    return exps

# ================== Runner ==================
def run_one(model_path, yaml_path, project, base_overrides, exp_name, exp_overrides, patience):
    """
    base_overrides: các tham số giữ nguyên theo yêu cầu (device, batch, workers, imgsz, fraction, plots,...)
    exp_overrides: tham số thay đổi theo từng thí nghiệm
    """
    # Tên run gắn timestamp để không ghi đè
    run_name = f"{exp_name}_{now_ts()}"

    # Khởi tạo model mới mỗi lần để tránh “nhiễm” state
    model = YOLO(model_path)

    args = dict(
        data=str(yaml_path),
        project=project,
        name=run_name,
        cos_lr=True,
        cache=True,
        save_period=10,
        verbose=True,
        patience=patience,
        **base_overrides,
        **exp_overrides
    )

    print(f"\n===== TRAIN: {run_name} =====")
    print(json.dumps(args, indent=2))
    results = model.train(**args)

    run_dir = Path(results.save_dir)
    best_w = run_dir / "weights" / "best.pt"
    last_w = run_dir / "weights" / "last.pt"
    ckpt = best_w if best_w.exists() else last_w
    return run_dir, ckpt

def maybe_eval(ckpt_path, yaml_path, device, batch, split):
    if split not in {"val", "test"}:
        return None, None
    print(f"⏳ Evaluate {split} on: {ckpt_path}")
    model_best = YOLO(str(ckpt_path))
    tmetrics = model_best.val(
        data=str(yaml_path),
        split=split,
        device=device,
        batch=max(8, min(32, int(batch))),
        plots=True,
        save_txt=True,
        save_conf=True,
        verbose=True
    )
    return tmetrics, getattr(tmetrics, "speed", None)

# ================== Main ==================
def main():
    parser = argparse.ArgumentParser("YOLO sweep runner")
    parser.add_argument("--yaml", required=True, type=str, help="dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", type=str)
    parser.add_argument("--project", default="runs/nano", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--fraction", default=1.0, type=float)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patience", default=40, type=int)
    parser.add_argument("--only", default="", type=str,
                        help="Danh sách tên exp, phân tách bằng dấu phẩy, để chạy một phần (vd: base_sgd_100,heavy_aug_adamw_200)")
    parser.add_argument("--eval-split", default="test", type=str, choices=["none", "val", "test"])
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in cấu hình, không train")
    args = parser.parse_args()

    set_seed(args.seed)
    yaml_path = Path(args.yaml).expanduser().resolve()
    assert yaml_path.exists(), f"Không tìm thấy YAML: {yaml_path}"

    device = pick_device(args.device)
    base_overrides = dict(
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        fraction=args.fraction,
        plots=True,
        fliplr=0.0,   # đảm bảo không flip ở tất cả cấu hình
    )

    # Danh sách thí nghiệm mặc định
    exps = default_experiments()

    # Lọc theo --only nếu có
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    if only:
        exps = [e for e in exps if e[0] in only]
        assert exps, f"--only không khớp tên experiment nào."

    manifest = []
    print("\n=== Số thí nghiệm sẽ chạy:", len(exps))
    for name, ov in exps:
        item = {
            "name": name,
            "overrides": ov
        }
        if args.dry_run:
            print(json.dumps({"dry_run_preview": item}, indent=2))
            continue

        run_dir, ckpt = run_one(
            model_path=args.model,
            yaml_path=yaml_path,
            project=args.project,
            base_overrides=base_overrides,
            exp_name=name,
            exp_overrides=ov,
            patience=args.patience
        )

        metrics_summary = {}
        if ckpt and args.eval_split != "none":
            tmetrics, _ = maybe_eval(ckpt, yaml_path, device, args.batch, args.eval_split)
            if tmetrics and hasattr(tmetrics, "box"):
                metrics_summary = {
                    "mAP50-95": float(tmetrics.box.map),
                    "mAP50": float(tmetrics.box.map50),
                    "mAP75": float(tmetrics.box.map75)
                }

        item.update({
            "run_dir": str(run_dir),
            "ckpt": str(ckpt) if ckpt else None,
            "eval_split": args.eval_split,
            "metrics": metrics_summary
        })
        manifest.append(item)

    # Lưu manifest để tiện so sánh
    if not args.dry_run:
        out = Path(args.project) / f"sweep_manifest_{now_ts()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print("\n✔ Saved sweep manifest ->", out)

if __name__ == "__main__":
    main()