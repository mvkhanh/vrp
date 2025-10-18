nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 0 --only "heavy_aug_adamw" > log/heavy_aug_adamw.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 1 --only "no_mosaic_sgd" > log/no_mosaic_sgd.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 2 --only "no_mosaic_sgd_ms" > log/no_mosaic_sgd_ms.log 2>&1 &
nohup python sweep_yolo_experiments.py --yaml dataset/dataset.yaml --device 3 --only "base_ms" > log/base_ms.log 2>&1 &