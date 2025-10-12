#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh ảnh tổng hợp (synthetic) cho YOLO từ các template RGBA (đã cắt BG)
- Mỗi ảnh đầu ra 640x640
- Biến đổi: scale, rotate, (tùy chọn) flip, jitter màu
- Bbox tính CHÍNH XÁC từ kênh alpha sau mọi biến đổi
- Ghi nhãn YOLO (class cx cy w h) chuẩn hóa [0,1]
- Hạn chế chồng lấn theo IoU + kiểm soát che khuất

YÊU CẦU MỚI:
- 80% ảnh MULTI-CLASS (trong một ảnh có thể có class trùng nhau)
- 20% ảnh SINGLE-CLASS (chỉ 1 class, nhiều instance)
- TỶ LỆ 80/20 ĐƯỢC ĐẢM BẢO TRONG TỪNG TẬP train/val/test
- CÂN BẰNG instance giữa các class trong MỖI TẬP (dùng lấy mẫu thiên về lớp đang thiếu)

Ghi chú: Việc cân bằng instance đạt gần như tuyệt đối nhờ chọn lớp theo số lượng \"đã đặt\" thấp nhất.
Nếu một vài instance bị loại do che khuất quá mạnh, vòng lặp sinh các ảnh tiếp theo sẽ tự kéo về trạng thái cân bằng.
"""
from __future__ import annotations
import re
import cv2
import math
import json
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Sequence

# ===================== CẤU HÌNH =====================
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

TEMPLATE_DIR = Path("../templates")      # template RGBA <class>_<idx>.png (BGRA)
OUT_DIR      = Path("dataset")                  # sẽ tạo images/{train,val,test}, labels/{train,val,test}
BG_DIR       = Path("../backgrounds")           # hoặc None -> nền màu trơn

IMG_SIZE = 640
N_IMAGES = 31250
OBJ_PER_IMG = (2, 8)                                 # (min, max)

SCALE_RANGE = (0.35, 0.95)
ANGLE_RANGE = (-25, 25)
ENABLE_FLIP_H = False
ENABLE_FLIP_V = False

COLOR_JITTER = True
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE   = (0.85, 1.15)

IOU_MAX = 0.05
MIN_BOX_W, MIN_BOX_H = 24, 24
MAX_PLACE_TRIES = 100                                # tăng nhẹ để giảm tỉ lệ thất bại đặt
TRY_ALT_TEMPLATES = 6                                # số lần thử template khác khi không đặt được

# Chống che khuất (occlusion)
LIMIT_OCCLUSION = True
MAX_COVER_PREV = 0.30
MAX_OVERLAP_NEW = 0.2
MIN_VISIBLE_RATIO = 0.60
ALPHA_THRESH = 1

SPLIT = {"train": 0.8, "val": 0.2, "test": 0}

# 80% multi / 20% single CHO TỪNG TẬP
MULTI_RATIO = 0.80

# Nền màu trơn khi không dùng BG_DIR
SOLID_BG_MODE = "random_light"   # "white" | "gray" | "random" | "random_light" | "random_dark"


# ===================== TIỆN ÍCH =====================
def sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", s)


def load_templates(root: Path) -> List[Path]:
    exts = {".png", ".webp"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files


def parse_class_from_stem(stem: str) -> str:
    m = re.match(r"(.+?)_\d+$", stem)
    if m:
        return sanitize(m.group(1))
    return sanitize(stem)


def list_classes(files: List[Path]) -> List[str]:
    names = sorted({parse_class_from_stem(p.stem) for p in files})
    return names


def group_templates_by_class(files: List[Path]) -> Dict[str, List[Path]]:
    buckets: Dict[str, List[Path]] = {}
    for p in files:
        cls = parse_class_from_stem(p.stem)
        buckets.setdefault(cls, []).append(p)
    return buckets


def rotate_bound(img: np.ndarray, angle_deg: float, border_value) -> np.ndarray:
    (h, w) = img.shape[:2]
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    new_w = int(abs(h * s) + abs(w * c))
    new_h = int(abs(h * c) + abs(w * s))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def compute_bbox_from_alpha(alpha: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    areaA = (ax2 - ax1 + 1) * (ay1 - ay1 + 1)  # bug fix below
    areaA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    areaB = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / float(areaA + areaB - inter)


def yolo_line_from_xyxy(cls_id: int, box: Tuple[int,int,int,int], W: int, H: int) -> str:
    x1, y1, x2, y2 = box
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{cls_id} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}"


def jitter_brightness_contrast(bgr: np.ndarray, b: float, c: float) -> np.ndarray:
    out = cv2.convertScaleAbs(bgr, alpha=c, beta=0)
    out = np.clip(out.astype(np.float32) * b, 0, 255).astype(np.uint8)
    return out


def read_rgba(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, a], axis=2)
    return img  # BGRA


def crop_to_alpha_tight(rgba: np.ndarray) -> np.ndarray | None:
    alpha = rgba[:, :, 3]
    bbox = compute_bbox_from_alpha(alpha)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return rgba[y1:y2+1, x1:x2+1, :]

# ====== THÊM MỚI: Bộ chọn background đồng đều ======
class BackgroundPicker:
    def __init__(self, bg_dir: Path | None, total_needed: int, img_size: int = 640):
        self.img_size = img_size
        self.paths: List[Path] = []
        if bg_dir and bg_dir.exists():
            self.paths = [p for p in bg_dir.rglob('*')
                          if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
        random.shuffle(self.paths)
        self.plan: List[Path] = self._build_plan(total_needed)
        self._i = 0  # con trỏ lấy kế tiếp

    def _build_plan(self, total_needed: int) -> List[Path]:
        if not self.paths or total_needed <= 0:
            return []
        n_bg = len(self.paths)
        if n_bg >= total_needed:
            # đủ hoặc dư → mỗi BG tối đa 1 lần
            return random.sample(self.paths, total_needed)
        # thiếu → lặp đều (round-robin), chênh lệch ≤ 1
        reps, rem = divmod(total_needed, n_bg)
        base = []
        for _ in range(reps):
            base.extend(self.paths)         # mỗi vòng: mỗi BG 1 lần
        if rem:
            base.extend(self.paths[:rem])   # phần dư: lấy từ đầu danh sách đã shuffle
        # để tránh 2 BG giống nhau đứng cạnh nhau quá nhiều, shuffle nhẹ theo block
        return base

    def next_image(self) -> np.ndarray | None:
        if not self.plan:
            return None
        if self._i >= len(self.plan):   # <-- thêm dòng này
            return None
        p = self.plan[self._i]
        self._i += 1
        bg = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bg is None:
            return self.next_image() if self._i < len(self.plan) else None
        return cv2.resize(bg, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    def has_any(self) -> bool:
        return len(self.plan) > 0

# ====== SỬA: make_background nhận bg_picker (nếu có) ======
def make_background(bg_picker: BackgroundPicker | None = None) -> np.ndarray:
    # ưu tiên kế hoạch BG đồng đều
    if bg_picker is not None and bg_picker.has_any():
        bg = bg_picker.next_image()
        if bg is not None:
            return bg

    # fallback: giữ logic nền màu trơn như cũ
    if SOLID_BG_MODE == "white":
        col = 255
        return np.full((IMG_SIZE, IMG_SIZE, 3), col, dtype=np.uint8)
    elif SOLID_BG_MODE == "gray":
        col = 180
        return np.full((IMG_SIZE, IMG_SIZE, 3), col, dtype=np.uint8)
    elif SOLID_BG_MODE == "random":
        col = np.random.randint(0, 256, size=(1,1,3), dtype=np.uint8)
        return np.full((IMG_SIZE, IMG_SIZE, 3), col, dtype=np.uint8)
    elif SOLID_BG_MODE == "random_light":
        col = np.random.randint(180, 256, size=(1,1,3), dtype=np.uint8)
        return np.full((IMG_SIZE, IMG_SIZE, 3), col, dtype=np.uint8)
    elif SOLID_BG_MODE == "random_dark":
        col = np.random.randint(0, 80, size=(1,1,3), dtype=np.uint8)
        return np.full((IMG_SIZE, IMG_SIZE, 3), col, dtype=np.uint8)
    else:
        return np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)


def place_one(canvas: np.ndarray, obj_rgba: np.ndarray, occ_mask: np.ndarray, per_obj_masks: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int,int,int,int], np.ndarray] | None:
    H, W = canvas.shape[:2]
    obj = crop_to_alpha_tight(obj_rgba)
    if obj is None:
        return None

    if ENABLE_FLIP_H and random.random() < 0.5:
        obj = cv2.flip(obj, 1)
    if ENABLE_FLIP_V and random.random() < 0.5:
        obj = cv2.flip(obj, 0)

    sh, sw = obj.shape[:2]
    scale = random.uniform(*SCALE_RANGE)
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    obj = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    angle = random.uniform(*ANGLE_RANGE)
    obj = rotate_bound(obj, angle, border_value=(0, 0, 0, 0))
    obj = crop_to_alpha_tight(obj)
    if obj is None:
        return None

    if COLOR_JITTER:
        b = random.uniform(*BRIGHTNESS_RANGE)
        c = random.uniform(*CONTRAST_RANGE)
        bgr = jitter_brightness_contrast(obj[:, :, :3], b, c)
        obj = np.dstack([bgr, obj[:, :, 3]])

    oh, ow = obj.shape[:2]
    if oh < MIN_BOX_H or ow < MIN_BOX_W:
        return None

    obj_bin = (obj[:, :, 3] >= ALPHA_THRESH).astype(np.uint8)

    for _ in range(MAX_PLACE_TRIES):
        max_x = W - ow
        max_y = H - oh
        if max_x < 1 or max_y < 1:
            return None
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        if obj_bin.max() == 0:
            continue

        if LIMIT_OCCLUSION:
            overlap_prev = (occ_mask[y:y+oh, x:x+ow] & obj_bin).sum()
            obj_area = int(obj_bin.sum())
            ratio_new = overlap_prev / max(1, obj_area)
            if ratio_new > MAX_OVERLAP_NEW:
                continue
            for prev_mask in per_obj_masks:
                overlapped = (prev_mask[y:y+oh, x:x+ow] & obj_bin).sum()
                prev_area = int(prev_mask.sum())
                if prev_area > 0 and (overlapped / prev_area) > MAX_COVER_PREV:
                    break
            else:
                pass
            if 'overlapped' in locals() and prev_area > 0 and (overlapped / prev_area) > MAX_COVER_PREV:
                overlapped = 0
                continue

        alpha = (obj[:, :, 3].astype(np.float32) / 255.0)[..., None]
        region = canvas[y:y+oh, x:x+ow, :]
        if region.shape[0] != oh or region.shape[1] != ow:
            continue

        blended = (alpha * obj[:, :, :3].astype(np.float32) + (1.0 - alpha) * region.astype(np.float32)).astype(np.uint8)
        box = (x, y, x + ow - 1, y + oh - 1)

        new_canvas = canvas.copy()
        new_canvas[y:y+oh, x:x+ow, :] = blended

        obj_mask_canvas = np.zeros((H, W), dtype=np.uint8)
        obj_mask_canvas[y:y+oh, x:x+ow] = obj_bin

        return new_canvas, box, obj_mask_canvas

    return None


# ========== HÀM MỚI: sinh theo KẾ HOẠCH LỚP cho từng object ==========
def synth_once_from_plan(templates_by_class: Dict[str, List[Path]], class_to_id: Dict[str, int], planned_classes: Sequence[str], bg_picker: BackgroundPicker | None = None) -> Tuple[np.ndarray, List[str]]:
    canvas = make_background(bg_picker)
    occ_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    per_obj_masks: List[np.ndarray] = []
    boxes: List[Tuple[int,int,int,int]] = []
    labels: List[str] = []

    for cls_name in planned_classes:
        # thử nhiều template cho cùng class nếu khó đặt
        for _try_tpl in range(TRY_ALT_TEMPLATES):
            p = random.choice(templates_by_class[cls_name])
            rgba = read_rgba(p)
            if rgba is None:
                continue
            # cố gắng đặt
            placed = place_one(canvas, rgba, occ_mask, per_obj_masks)
            if placed is None:
                continue
            cand_canvas, cand_box, cand_mask = placed
            # Kiểm tra IoU & kích thước
            ok = True
            for b in boxes:
                if iou(cand_box, b) > IOU_MAX:
                    ok = False
                    break
            if not ok:
                continue
            bw = cand_box[2] - cand_box[0] + 1
            bh = cand_box[3] - cand_box[1] + 1
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue

            # chấp nhận
            canvas = cand_canvas
            boxes.append(cand_box)
            occ_mask |= cand_mask
            per_obj_masks.append(cand_mask)
            cls_id = class_to_id[cls_name]
            labels.append(yolo_line_from_xyxy(cls_id, cand_box, IMG_SIZE, IMG_SIZE))
            break
        # nếu không đặt được sau TRY_ALT_TEMPLATES lần -> bỏ qua object này

    if LIMIT_OCCLUSION and labels:
        # loại các nhãn bị che khuất quá mức
        keep_labels: List[str] = []
        for i, lab in enumerate(labels):
            mask_i = per_obj_masks[i].copy()
            for j in range(i + 1, len(per_obj_masks)):
                mask_i[per_obj_masks[j] > 0] = 0
            vis_ratio = mask_i.sum() / max(1, per_obj_masks[i].sum())
            if vis_ratio >= MIN_VISIBLE_RATIO:
                keep_labels.append(lab)
        labels = keep_labels

    return canvas, labels


def ensure_dirs():
    for s in SPLIT.keys():
        (OUT_DIR/"images"/s).mkdir(parents=True, exist_ok=True)
        (OUT_DIR/"labels"/s).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(names: List[str]):
    data = {
        'path': str(OUT_DIR.resolve()),
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'names': names,
    }
    with open(OUT_DIR/"dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# ======= HÀM PHÂN PHỐI LỚP CÂN BẰNG CHO 1 ẢNH =======
def pick_class_min_count(counts: Dict[str, int], bottom_k: int = 8) -> str:
    # lấy ngẫu nhiên trong nhóm lớp đang THIẾU nhất (để tránh luôn dính 1 lớp khi hòa điểm)
    items = sorted(counts.items(), key=lambda kv: kv[1])
    cutoff_val = items[min(bottom_k-1, len(items)-1)][1]
    cands = [c for c, v in items if v <= cutoff_val]
    return random.choice(cands)


def build_plan_for_image(mode: str, k: int, names: List[str], counts: Dict[str, int]) -> List[str]:
    """
    mode: 'mixed' hoặc 'single'
    k: số object cần đặt
    counts: đếm instance đã sinh cho từng class (để cân bằng)
    """
    if mode == 'single' or len(names) == 1:
        cls = pick_class_min_count(counts)
        return [cls] * k
    # mixed: đảm bảo >= 2 lớp khác nhau
    plan: List[str] = []
    # ép 2 object đầu thuộc 2 lớp ít xuất hiện nhất (nếu có)
    c1 = pick_class_min_count(counts)
    # tạm tăng để lần chọn thứ hai không pick lại cùng lớp
    counts_temp = counts.copy()
    counts_temp[c1] += 1
    c2 = pick_class_min_count(counts_temp)
    plan.extend([c1, c2])
    # phần còn lại theo lớp thiếu nhất
    for _ in range(max(0, k - 2)):
        c = pick_class_min_count(counts)
        plan.append(c)
    return plan[:k]


def generate_split(split_name: str, n_images: int, names: List[str], class_to_id: Dict[str, int], templates_by_class: Dict[str, List[Path]], start_idx: int, bg_picker: BackgroundPicker | None = None,) -> int:
    """Sinh ảnh cho 1 tập (train/val/test) với 80% mixed, 20% single và cân bằng instance.
    Trả về chỉ số ảnh kế tiếp (để đặt tên file liên tục).
    """
    # chuẩn bị tỷ lệ 80/20
    n_mixed = int(round(n_images * MULTI_RATIO))
    n_single = n_images - n_mixed
    flags = ['mixed'] * n_mixed + ['single'] * n_single
    random.shuffle(flags)
    
    # bộ đếm instance theo class cho TỪNG TẬP -> giúp cân bằng trong tập đó
    inst_counts: Dict[str, int] = {n: 0 for n in names}

    pbar = tqdm(range(n_images), desc=f"{split_name}: synthesizing", unit="img")
    img_idx = start_idx
    for i in pbar:
        mode = flags[i]
        k = random.randint(OBJ_PER_IMG[0], OBJ_PER_IMG[1])
        plan_classes = build_plan_for_image(mode, k, names, inst_counts)

        img, lbls = synth_once_from_plan(templates_by_class, class_to_id, plan_classes, bg_picker)

        # cập nhật đếm theo NHÃN THỰC TẾ được giữ lại
        for line in lbls:
            cls_id = int(line.split()[0])
            inst_counts[names[cls_id]] += 1

        img_name = f"synth_{img_idx:06d}.jpg"
        lbl_name = f"synth_{img_idx:06d}.txt"
        cv2.imwrite(str(OUT_DIR/"images"/split_name/img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(OUT_DIR/"labels"/split_name/lbl_name, "w", encoding="utf-8") as f:
            f.write("\n".join(lbls))
        img_idx += 1

    # Thống kê nhanh cho tập này
    total = sum(inst_counts.values())
    if total > 0:
        min_c = min(inst_counts.values())
        max_c = max(inst_counts.values())
        print(f"[Stats:{split_name}] total inst={total}, min/class={min_c}, max/class={max_c}, diff={max_c-min_c}")
    else:
        print(f"[Stats:{split_name}] Không có nhãn nào – kiểm tra lại template hoặc ngưỡng che khuất.")

    return img_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dirs()

    templates = load_templates(TEMPLATE_DIR)
    assert templates, f"Không tìm thấy template trong {TEMPLATE_DIR}"
    bg_picker_global = BackgroundPicker(BG_DIR, N_IMAGES, img_size=IMG_SIZE)
    names = list_classes(templates)
    class_to_id = {n:i for i,n in enumerate(names)}
    id_to_class = {i:n for n,i in class_to_id.items()}

    templates_by_class = group_templates_by_class(templates)
    missing = [n for n in names if n not in templates_by_class or len(templates_by_class[n]) == 0]
    assert not missing, f"Thiếu template cho các lớp: {missing}"

    # lưu mapping để tái sử dụng
    with open(OUT_DIR/"classes.json", "w", encoding="utf-8") as f:
        json.dump({"names": names, "class_to_id": class_to_id}, f, ensure_ascii=False, indent=2)

    write_dataset_yaml(names)

    # Tính số ảnh cho từng tập
    n_train = int(round(N_IMAGES * SPLIT.get("train", 0)))
    n_val   = int(round(N_IMAGES * SPLIT.get("val", 0)))
    n_test  = N_IMAGES - n_train - n_val

    next_idx = 1
    next_idx = generate_split("train", n_train, names, class_to_id, templates_by_class, start_idx=next_idx, bg_picker=bg_picker_global)
    next_idx = generate_split("val",   n_val,   names, class_to_id, templates_by_class, start_idx=next_idx, bg_picker=bg_picker_global)
    next_idx = generate_split("test",  n_test,  names, class_to_id, templates_by_class, start_idx=next_idx, bg_picker=bg_picker_global)

    print(f"✅ Done. Out → {OUT_DIR.resolve()}")
    print(f"  - Viết {N_IMAGES} ảnh + nhãn YOLO")
    print(f"  - Classes: {len(names)} | lưu {OUT_DIR/'classes.json'} và {OUT_DIR/'dataset.yaml'}")


if __name__ == "__main__":
    main()