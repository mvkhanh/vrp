# Vietnamese Retail Product Detection

Hệ thống **nhận diện sản phẩm bán lẻ Việt Nam** (Vietnamese Retail Product Detection) sử dụng **YOLOv11n** làm mô hình chính.  
Dữ liệu được **tạo tổng hợp (synthetic)** từ các template sản phẩm (đã tách nền RGBA) và các background chụp thực tế, giúp mô hình nhận diện tốt hơn trong môi trường bán lẻ thật.

---
1. Sinh dữ liệu tổng hợp:

Tạo ảnh tổng hợp theo chuẩn YOLOv11, có:
- Nhiều class sản phẩm (multi-class)
- Kiểm soát chồng lấn (IoU)
- Augmentation: scale, rotate, jitter màu, flip
- Ghi nhãn YOLO chuẩn hóa (class, cx, cy, w, h)
- ...

2. Huấn luyện và đánh giá mô hình: train.py

3. Thử nghiệm nhiều cấu hình hyperparameter tự động: sweep_yolo_experiments.py