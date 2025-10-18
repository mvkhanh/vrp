# Vietnamese Retail Product Detection

> Build a **synthetic dataset** and train **lightweight object detectors** for **149 retail product classes** in Vietnamese supermarkets. The goal is **edge deployment** (e.g., Raspberry Pi) for **self‑checkout / automatic billing**.

![Demo](test_video_result/video9_demo.gif)

---

## ✨ Objectives
- Provide a complete **synthetic data generation** pipeline from single‑product photos + diverse backgrounds.
- Train and compare **YOLO11n** configurations (small, fast, edge‑friendly).
- Offer an easy demo for **image/video inference** and hints for **edge deployment**.

## 📦 Repository Layout (suggested)
```
.
├── backgrounds/                 # Diverse backgrounds (~5k images initially)
├── dataset/
│   ├── dataset.yaml             # YOLO data config
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── log/                         # Training/sweep logs
├── runs/                        # Ultralytics outputs
├── test_video_result/           # Inference outputs (contains the GIF demo)
│   └── video9_demo.gif
├── videos_input/                # Raw videos for testing
├── synthetic.py                 # Synthetic dataset generator
├── train.py                     # Train a single config
├── sweep_yolo_experiments.py    # Run multiple configs to find the best
├── test_with_video.py           # Video inference + save output
├── add_bg_to_yolo.py            # Utilities to blend/prepare backgrounds (optional)
├── train.sh                     # Example training script
└── README.md
```

## 🧱 Data Preparation
### 1) Product templates (PNG RGBA)
- Take **multiple viewpoints** per product (front, slight left/right yaw, optionally top/bottom).
- Remove background and export **PNG with alpha channel** (RGBA) using SAM 2 (Segment Anything Model)
- Naming suggestion per class, e.g. `COLGATE_XXX_01.png`, `WAKEUP247_YYY_02.png`…

### 2) Diverse backgrounds
- Collect indoor/outdoor scenes, counters, shelves, wood/stone tables, checkout desks, etc.
- Recommended **~5,000 images** for initial diversity.

### 3) Synthetic dataset (149 classes, ~25k train images)
`synthetic.py` will:
- Randomize the **number and types** of products per background.
- Apply **transformations**: position, scale, rotation, hue/saturation/value, brightness/contrast, light **occlusion**.
- Export images + **YOLO labels** (`.txt` with same basename).

> Arguments may differ — run:
```bash
python synthetic.py --help
```

## 🏋️ Training

Train a single config:
```bash
python train.py --yaml dataset/dataset.yaml --device 0
```
Run sweeps (example config names inside the script):
```bash
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 0 --only "heavy_aug_adamw" > log/heavy_aug_adamw.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 1 --only "no_mosaic_sgd"    > log/no_mosaic_sgd.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 2 --only "no_mosaic_sgd_ms" > log/no_mosaic_sgd_ms.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 3 --only "base_ms"          > log/base_ms.log 2>&1 &
```
> Tip: when running multiple GPUs, watch **CPU and I/O** (data loading, augmentations, disk speed). Tune `num_workers`, enable caching, or pre‑encode images if needed.

## ▶️ Inference (images/videos)
Images / folders:
```bash
yolo detect predict model=runs/detect/exp/weights/best.pt source=dataset/images/val device=0 save=True
```
Video with the repo script:
```bash
python test_with_video.py \
  --weights runs/detect/exp/weights/best.pt \
  --source videos_input/video_test6.mp4 \
  --save-dir test_video_result/video_test6
```

### Export GIF demo with FFmpeg
```bash
ffmpeg -i test_video_result/video_test6/output.mp4 \
  -vf "fps=10,scale=768:-1:flags=lanczos" -loop 0 \
  test_video_result/video9_demo.gif
```
Embed in README:
```md
![Demo](test_video_result/video9_demo.gif)
```

## 🚀 Edge Deployment (short hints)
- **Raspberry Pi 5**: run Ultralytics on CPU, or export **ONNX** and use ONNX Runtime / OpenVINO / TensorRT (where available).
- Export from Ultralytics:
```bash
yolo export model=runs/detect/exp/weights/best.pt format=onnx opset=12 dynamic=True
```
- Optimize camera I/O (GStreamer/V4L2). Use smaller input size (640/512) to boost FPS.
- Keep default NMS for simplicity; later consider fused post‑processing if you need more speed.

## 🧪 Evaluation
- Track **mAP@0.5:0.95**, Precision/Recall, and real‑time **FPS** on target hardware.
- In sweeps, log: augmentation recipe, optimizer, batch size, training time, and the best checkpoint (`best.pt`).

## 🗂️ Example YAML (149 classes)
```yaml
# dataset/dataset.yaml
train: images/train
val:   images/val
test:  images/test

names:
  0: CLASS_0
  1: CLASS_1
  # ...
  148: CLASS_148
```

## 🔥 Notes on synthetic quality
- Product cutouts must be **sharp with clean alpha** (avoid halos).
- **Balance classes** during generation (uniform or realistic long‑tail).
- Add **noise/blur/motion blur** to narrow the domain gap.
- Randomize **lighting** (gamma, brightness/contrast) and **camera tilt**.

---
