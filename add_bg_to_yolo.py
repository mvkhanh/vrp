#!/usr/bin/env python3
import argparse, random, re, shutil, sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Add background-only images to YOLO train split")
    ap.add_argument("--backgrounds", required=True, type=Path, help="Folder chứa ảnh .webp nền")
    ap.add_argument("--dataset",     required=True, type=Path, help="Thư mục gốc dataset có cấu trúc YOLO")
    ap.add_argument("-n", "--num",   default=2500, type=int,   help="Số ảnh cần lấy ngẫu nhiên (mặc định 2500)")
    ap.add_argument("--split",       default="train", choices=["train","val","test"], help="Split đích (mặc định train)")
    ap.add_argument("--mode",        default="copy", choices=["copy","symlink","move"], help="Cách thêm ảnh vào images/train")
    ap.add_argument("--prefix",      default="bg", help="Tiền tố tên file nền (vd: bg_000001.webp)")
    ap.add_argument("--seed",        default=42, type=int, help="Seed lấy mẫu ngẫu nhiên")
    return ap.parse_args()

def ensure_dirs(ds_root: Path, split: str):
    img_dir = ds_root / "images" / split
    lbl_dir = ds_root / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"❌ Không tìm thấy cấu trúc {img_dir} và {lbl_dir}. Hãy tạo đúng cấu trúc dataset/{{images,labels}}/{{train,val,test}}", file=sys.stderr)
        sys.exit(1)
    return img_dir, lbl_dir

def next_start_index(img_dir: Path, prefix: str):
    # Tìm index lớn nhất sẵn có theo mẫu prefix_000001.webp
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.[A-Za-z0-9]+$")
    max_idx = -1
    for p in img_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx: max_idx = idx
                except: pass
    return max_idx + 1  # bắt đầu từ số tiếp theo

def link_copy_move(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        # tạo symlink tương đối cho gọn
        rel = src.resolve()
        try:
            dst.symlink_to(rel)
        except FileExistsError:
            dst.unlink()
            dst.symlink_to(rel)

def main():
    args = parse_args()
    random.seed(args.seed)

    if not args.backgrounds.exists():
        print(f"❌ backgrounds không tồn tại: {args.backgrounds}", file=sys.stderr); sys.exit(1)

    img_dir, lbl_dir = ensure_dirs(args.dataset, args.split)

    # Thu thập ảnh .webp (đệ quy)
    bg_list = sorted([p for p in args.backgrounds.rglob("*.webp") if p.is_file()])
    if len(bg_list) == 0:
        print("❌ Không tìm thấy ảnh .webp trong backgrounds/", file=sys.stderr); sys.exit(1)
    if len(bg_list) < args.num:
        print(f"⚠️  Chỉ có {len(bg_list)} ảnh .webp; sẽ lấy toàn bộ thay vì {args.num}.")
        args.num = len(bg_list)

    chosen = random.sample(bg_list, args.num)

    start_idx = next_start_index(img_dir, args.prefix)
    added = 0

    for i, src in enumerate(chosen, start=0):
        idx = start_idx + i
        stem = f"{args.prefix}_{idx:06d}"
        dst_img = img_dir / f"{stem}{src.suffix.lower()}"  # giữ .webp
        dst_lbl = lbl_dir / f"{stem}.txt"

        if dst_img.exists() or dst_lbl.exists():
            # rất hiếm khi trùng; an toàn thì nhảy tiếp
            k = 1
            while (img_dir / f"{stem}_{k}{src.suffix.lower()}").exists() or (lbl_dir / f"{stem}_{k}.txt").exists():
                k += 1
            dst_img = img_dir / f"{stem}_{k}{src.suffix.lower()}"
            dst_lbl = lbl_dir / f"{stem}_{k}.txt"

        link_copy_move(src, dst_img, args.mode)
        # Tạo label rỗng (no object)
        dst_lbl.touch()  # file 0 byte

        added += 1
        if added % 500 == 0:
            print(f"…đã thêm {added}/{args.num} ảnh")

    print(f"✅ Hoàn tất: đã thêm {added} ảnh nền → {img_dir}")
    print(f"   Nhãn rỗng tương ứng đã tạo tại → {lbl_dir}")
    print(f"   Chế độ: {args.mode} | Prefix: {args.prefix} | Split: {args.split}")

if __name__ == "__main__":
    main()