import os, shutil, argparse, re
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

def pick_device_arg():
    if torch.cuda.is_available(): return 0
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def reencode_to_mp4(src_path: Path, dst_path: Path):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    cap.release()
    out.release()
    if dst_path.exists():
        src_path.unlink(missing_ok=True)

def slugify(s: str) -> str:
    # Giữ chữ/số/_/./- ; thay còn lại bằng _
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

def f2token(x: float) -> str:
    # 0.3 -> '0p30', 0.45 -> '0p45'
    return str(x).replace('.', 'p')

def get_model_tag(weight_path: Path) -> str:
    # Nếu path dạng .../<exp>/weights/best.pt -> trả về <exp>
    try:
        if weight_path.parent.name == "weights":
            return slugify(weight_path.parent.parent.name)
    except Exception:
        pass
    # Fallback: dùng tên file .pt (ít khi dùng)
    return slugify(weight_path.stem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos_input')
    parser.add_argument('--output_root', default='test_video_result')
    parser.add_argument('--weight-file', required=True)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--force-mp4', action='store_true', help='Ép về .mp4 kể cả khi Ultralytics xuất .avi')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_root = Path(args.output_root)
    weight_file = Path(args.weight_file)
    CONF, NMS = float(args.conf), float(args.nms)

    ensure_dir(output_root)
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not video_files:
        print(f"Không tìm thấy video trong: {video_dir.resolve()}"); exit(0)

    model_name = get_model_tag(weight_file)
    print(f"Đang load model: {weight_file}")
    model = YOLO(str(weight_file))
    device_arg = pick_device_arg()

    for video_file in video_files:
        video_path = video_dir / video_file
        video_stem = slugify(Path(video_file).stem)
        video_output_dir = ensure_dir(output_root / video_stem)

        print(f"Đang xử lý: {video_file}")
        print(f"  Lưu kết quả vào: {video_output_dir}")

        gen = model.predict(
            source=str(video_path),
            conf=CONF,
            iou=NMS,
            device=device_arg,
            save=True,
            stream=True,
            project=str(video_output_dir),
            name="_tmp",
            exist_ok=True,
            verbose=False,
            show=False
        )

        last_save_dir = None
        for r in gen:
            if hasattr(r, "save_dir"):
                last_save_dir = Path(r.save_dir)

        save_dir = last_save_dir or (video_output_dir / "_tmp")

        # File Ultralytics đã ghi
        expected = save_dir / Path(video_path.name)
        if expected.exists():
            saved_video_path = expected
        else:
            candidates = [p for p in save_dir.glob("*.*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
            if not candidates:
                print("  ⚠️ Không thấy file video kết quả trong:", save_dir)
                continue
            saved_video_path = max(candidates, key=lambda p: p.stat().st_mtime)

        # Tên đích: <model>__<video>__confX_iouY
        base_name = f"{model_name}__{video_stem}__conf{f2token(CONF)}_iou{f2token(NMS)}"
        dest_mp4 = video_output_dir / f"{base_name}.mp4"

        # Xóa nếu đã tồn tại
        if dest_mp4.exists(): dest_mp4.unlink()

        if saved_video_path.suffix.lower() == ".mp4":
            shutil.move(str(saved_video_path), str(dest_mp4))
            final_path = dest_mp4
        else:
            if args.force_mp4:
                reencode_to_mp4(saved_video_path, dest_mp4)
                final_path = dest_mp4
            else:
                dest_same_ext = video_output_dir / f"{base_name}{saved_video_path.suffix.lower()}"
                if dest_same_ext.exists(): dest_same_ext.unlink()
                shutil.move(str(saved_video_path), str(dest_same_ext))
                final_path = dest_same_ext

        # Dọn _tmp (nếu trống)
        try:
            for p in sorted((video_output_dir / "_tmp").glob("*")): p.unlink()
            (video_output_dir / "_tmp").rmdir()
        except Exception:
            pass

        print(f"  ✅ OK: {final_path}")