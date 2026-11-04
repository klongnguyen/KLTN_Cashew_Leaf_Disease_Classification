# Tiến độ nghiên cứu — 04/11/2026

> Trang tổng hợp dựa trên kết quả, CSV và báo cáo đã lưu trong Git. **Đồng bộ tài liệu, không phải huấn luyện lại hay chạy Test mới.** [Danh mục toàn bộ 95 README](EXPERIMENT_INDEX.md).

## Đề tài và phương pháp

**CNTT_KLCN183 — Ứng dụng mô hình học sâu trong phân loại bệnh trên lá cây điều.** Classification 5 lớp: `anthracnose`, `healthy`, `leaf_miner`, `red_rust`, `not_cashew_leaf`. Detection YOLO khoanh 3 bệnh; Healthy/Not Cashew là ảnh 0 box. YOLO lesion → crop → classifier vẫn là pipeline nghiên cứu **chưa kiểm định end-to-end**.

## Các phiên bản dữ liệu

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| V08 | 4.790 | 1.430 | 686 |
| V09 | 4.914 | 1.445 | 686 |
| V10 | 4.920 | 1.445 | 686 |
| V11 | 4.916 | 1.447 | 685 |
| V12 | 4.975 | 1.452 | 685 |
| V13 | 4.972 | 1.452 | 685 |

V12→V13: Validation và Test giữ nguyên; Train giảm 3 Anthracnose. [Changelog](../DATASET_CHANGELOG.md) · [So sánh có kiểm toán](../training_results/current_dataset/DENSENET121_V12_V13_EXP027_028_COMPARISON.md).

## DenseNet121 V12/V13, 5 seeds

| Chỉ số | EXP-027 V12 | EXP-028 V13 |
|---|---:|---:|
| Val Accuracy | 97,245% | 97,287% |
| Val Macro F1 | 97,125% | 97,177% |
| Final Test Accuracy | **95,912%** | **95,912%** |
| Final Test Macro F1 | 95,510% | 95,523% |
| Holdout chung **118 ảnh cùng SHA256**, Accuracy | **80,17%** | **80,17%** |
| Healthy Recall, 11 ảnh giữ nguyên | 47,27% | 56,36% |

Seed chọn theo Validation Loss: 7777 ở EXP-027; 42 ở EXP-028. Healthy chỉ có 11 ảnh trên giao 118 nên chưa đủ kết luận tăng generalization. Hai file từng trùng SHA256 giữa Train V12 (nhãn Anthracnose) và Holdout09 (nhãn Healthy) cần kiểm nhãn chuyên môn; chúng đã bị loại khỏi Train V13. Holdout06/07/09 có khác biệt nguồn ảnh, không lấy số nguyên tập khác nhau làm controlled comparison. [Báo cáo chi tiết](../training_results/current_dataset/DENSENET121_V12_V13_EXP027_028_COMPARISON.md).

## ResNet50 targeted augmentation EXP-028/029/030

Tăng 0/15/35 ảnh augment từ 15 ảnh Healthy nguồn vào Train. Trên **7 ảnh real_healthy Validation**: Recall 42,86% → 60,00% → 77,14%, nhưng giữa EXP-028 và EXP-030 số bệnh→Healthy sai trên Holdout05 tăng **2,0 → 6,2 ảnh/seed**, Accuracy giảm **72,50% → 71,33%**. EXP-029 **không chạy Test/Holdout**. Không promote EXP-030 dựa theo Holdout lịch sử. [Báo cáo](../training_results/current_dataset/TARGETED_AUGMENTATION_RESNET50_EXP028_029_030_REPORT.md).

## YOLO NEG V03 — Validation-only

Train **3.497** (621 ảnh âm), Val **951** (58 âm), Test **412 giữ khóa**. Một seed, `cls_pw=0.25`, 640px: YOLOv8s mAP50-95 **38,00%**, báo bệnh nhầm **13/58** ảnh âm; YOLO26s mAP50-95 **39,81%**, báo nhầm **3/58** tại `conf=0.25`. So với NEG V02, YOLO26s vẫn báo nhầm 3/58 dù tăng negative Train. Kết quả không chứng minh Test hay thực địa. [Nguồn V03](../training_results/yolo_negative_v03/README.md) · [Take 01–07](../failure/YOLO26/README.md).

## Triển khai và việc cần làm

[WebApp](../Application/README.md) hiện dùng **DenseNet121 Transfer EXP-014 seed 123**, RGB 224×224, classification 5 lớp; binary `deployment_model.keras` **không có trong Git**, YOLO detection chưa tích hợp. Không tự thay deployment theo Test/Holdout đã xem.

Cần: kiểm tra near-duplicate cùng lá/cây/phiên, xác minh nhãn mâu thuẫn, fresh blind field holdout đủ 5 lớp, khóa detector/threshold theo Validation, đánh giá end-to-end và parity/inference latency, sau đó mới cân nhắc deployment.

[Protocol](RESEARCH_PROTOCOL.md) · [README chính](../README.md) · [Tiến độ 28/10 lưu trữ](RESEARCH_STATUS_2026-10-28.md).
