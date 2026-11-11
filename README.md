# KLTN — Nhận diện và phân loại bệnh trên lá cây điều

Khóa luận CNTT_KLCN183. Hệ thống nhận ảnh lá điều, phát hiện vùng nghi bệnh và phân loại 5 lớp: **Thán thư**, **Sâu vẽ bùa**, **Rỉ đỏ**, **Lá khỏe**, **Không phải lá điều**. Mô hình triển khai kết hợp **DenseNet121 ×5 seed** với **YOLO26s**, chạy trong một ứng dụng web FastAPI.

## Cấu trúc thư mục

| Thư mục / tệp | Nội dung |
|---|---|
| [Application/](Application/README.md) | Ứng dụng web (FastAPI + HTML/CSS/JS): chụp/tải ảnh, khoanh vùng, hướng dẫn chăm sóc; dùng được trên điện thoại |
| [EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/](EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/) | Mô hình kết hợp đang triển khai: cấu hình đóng băng, dự đoán, bảng kết quả, `pipeline_bundle/` |
| [Notebook/](Notebook/README.md) | Notebook huấn luyện và đánh giá các mô hình nêu trong báo cáo |
| [training_results/](training_results/README.md) | Kết quả thí nghiệm: phân loại (ResNet50, DeiT-S), phát hiện YOLO, mô hình kết hợp |
| [analysis_box_labels/](analysis_box_labels/) | Thống kê khung bao của bộ dữ liệu phát hiện |
| [DATASET_CHANGELOG.md](DATASET_CHANGELOG.md) | Lịch sử các phiên bản bộ dữ liệu |
| [CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md](CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md) | Quy tắc gán nhãn khung bao |
| [DISEASE_REFERENCE.md](DISEASE_REFERENCE.md) | Mô tả bệnh và ảnh tham chiếu |
| [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) | Giao thức Train / Validation / Test / ảnh thực địa |
| [STANDARD_STATISTICAL_ANALYSIS_FORMAT.md](STANDARD_STATISTICAL_ANALYSIS_FORMAT.md) | Quy chuẩn phân tích thống kê |

## Mô hình triển khai (EXP-035)

Quy trình: kiểm tra ảnh có phải lá điều (DenseNet121 + khoảng cách k láng giềng) → YOLO26s khoanh vùng bệnh → DenseNet121 kiểm tra từng vùng cắt → tổng hợp kết luận. Chi tiết luồng và ngưỡng ở [Application/README.md](Application/README.md).

Kết quả của bộ kết hợp DenseNet121 + YOLO26s (Stacker F3, bộ dữ liệu V13, Test 685 ảnh, ảnh thực địa 126 ảnh):

| Tập | Accuracy | Ghi chú |
|---|---|---|
| Validation sạch (1.361 ảnh) | 98,68% | Dùng để chọn cấu hình |
| Test (685 ảnh) | 97,08% (665/685) | Macro F1 96,77%; Wilson 95% từ 95,53% đến 98,10% |
| Ảnh thực địa (126 ảnh) | 94,44% (119/126) | **Mang tính mô tả**: tập này đã được tham khảo khi chọn mô hình và ngưỡng, không phải kiểm thử mù |

Nguồn: [`aggregate/combined_results.csv`](EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/aggregate/combined_results.csv), [`aggregate/combined_summary.json`](EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/aggregate/combined_summary.json).

## So sánh ba mô hình phân loại khi kết hợp YOLO26s

Cùng quy trình kết hợp, ba mô hình phân loại (trung bình ± SD qua 5 seed 42, 123, 2026, 3407, 7777). F0 là mô hình phân loại chạy một mình, F3 là mô hình kết hợp với YOLO26s. Chi tiết, khoảng tin cậy và kiểm định ở [EXP-COMBINED-3CLASSIFIERS-YOLO26S-V04-037/README.md](training_results/current_dataset/EXP-COMBINED-3CLASSIFIERS-YOLO26S-V04-037/README.md).

| Mô hình | Test F0 | Test F3 | Thực địa F0 | Thực địa F3 |
|---|---|---|---|---|
| DenseNet121 + YOLO26s | 96,06 ± 0,18 | 97,08 ± 0,18 | 81,27 ± 1,83 | 94,44 ± 0,56 |
| ResNet50 + YOLO26s | 95,59 ± 0,55 | 97,84 ± 0,19 | 76,03 ± 3,04 | 89,68 ± 0,97 |
| ViT (DeiT-S) + YOLO26s | 97,14 ± 0,59 | 97,55 ± 0,48 | 85,24 ± 2,14 | 91,90 ± 1,30 |

Nguồn thí nghiệm: DenseNet121 = EXP-035; ResNet50 = [EXP-RESNET50-TRANSFER-5SEEDS-031-256px](training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4/); DeiT-S = [EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px](training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-1192fd08/).

## Phát hiện vùng bệnh (YOLO)

| Nội dung | Thư mục |
|---|---|
| Trọng số lớp (class weight) YOLOv8s / YOLO26s | [training_results/yolo_leaf_miner_class_weight](training_results/yolo_leaf_miner_class_weight/README.md) |
| Bổ sung ảnh âm V02 | [training_results/yolo_negative_v02](training_results/yolo_negative_v02/README.md) |
| Bổ sung ảnh âm V03 | [training_results/yolo_negative_v03](training_results/yolo_negative_v03/README.md) |
| Phiên bản cuối V04 (YOLO26s, dùng trong mô hình kết hợp và web) | [training_results/yolo_negative_v04](training_results/yolo_negative_v04/EXP-Y26S-NEG-V04-001/README.md) |

Mọi chỉ số YOLO ở mục này là một seed, trừ khi README tương ứng nêu khác.

## Chạy ứng dụng web

Xem [Application/README.md](Application/README.md). Trọng số của mô hình triển khai (EXP-035) đã nằm trong Git (`EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/pipeline_bundle/`), nên clone xong là chạy được. **Bộ dữ liệu ảnh huấn luyện không nằm trong Git.**

## Giới hạn

- Tập ảnh thực địa nhỏ (126 ảnh) và đã được tham khảo khi lựa chọn mô hình, nên các con số trên tập này chỉ mang tính mô tả.
- Kết quả YOLO là một seed; chênh lệch dưới một điểm phần trăm chưa đủ để kết luận.
- Chưa có phép kiểm thử mù độc lập trên ảnh thực địa mới.
