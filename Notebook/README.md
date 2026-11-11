# Notebook

Notebook huấn luyện và đánh giá các mô hình được trình bày trong khóa luận (chạy trên Google Colab, 1 GPU). Năm seed cố định: 42, 123, 2026, 3407, 7777.

| Nhóm | Notebook | Nội dung |
|---|---|---|
| Mô hình triển khai | `DenseNet121/densenet121_yolo_combined_training_035_negv04.ipynb` | Huấn luyện 5 DenseNet121 và bộ kết hợp với YOLO26s (EXP-035) |
| Phân loại | `ResNet50/resnet50_transfer_5seed_031.ipynb` | ResNet50 học chuyển giao, 256 px, 5 seed |
| Phân loại | `ViT/vit_deit_small_transfer_llrd_016_256px.ipynb` | DeiT-S, giảm tốc độ học theo lớp (LLRD), 256 px, 5 seed |
| Kết hợp | `Hybrid/combined_3classifiers_yolo26s_v04_037.ipynb` | So sánh DenseNet121 / ResNet50 / DeiT-S khi kết hợp YOLO26s (F0 so với F3) |
| YOLO26s | `Yolov26/EXP-Y26S-NEG-V04-001.ipynb` | Phiên bản phát hiện cuối (bộ dữ liệu V04) |
| YOLO26s | `Yolov26/EXP-Y26S-LM-003_CLASS_WEIGHTED.ipynb`, `…LM-004_NO_CLASS_WEIGHT.ipynb` | Ảnh hưởng của trọng số lớp |
| YOLO26s | `Yolov26/EXP-Y26S-NEG-V02-001_…`, `…NEG-V03-001_…` | Bổ sung ảnh âm V02, V03 |
| YOLOv8s | `Yolov8/EXP-Y8S-NEG-V04-001.ipynb`, `…LM-003/004`, `…NEG-V02/V03-001` | Cùng các thí nghiệm trên với YOLOv8s |
| Đánh giá ảnh âm | `NEGATIVE_IMAGE_EVALUATION_V003_VS_NEG_V02.ipynb`, `…_VS_NEG_V03.ipynb` | Đếm ảnh âm bị báo bệnh nhầm trước và sau khi bổ sung ảnh âm |

Kết quả tương ứng nằm trong [../training_results/](../training_results/README.md) và `../EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/`.
