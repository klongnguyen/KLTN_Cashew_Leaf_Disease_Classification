# Tiến độ nghiên cứu — 28/10/2026

![V08–V11](assets/dataset_versions.svg)

## DenseNet121 V09–V11
EXP-018 huấn luyện trên V09; EXP-019, 020, 021 chỉ đánh giá/re-evaluate checkpoint 018. EXP-022 huấn luyện V10; EXP-023 khảo sát brightness trên Validation. EXP-024 huấn luyện V11 256px+brightness, EXP-025 chỉ Test/Holdout checkpoint 024; EXP-026 huấn luyện mới V11 224px không brightness.

| EXP | Validation Accuracy | Test Accuracy | Historical Holdout Accuracy |
|---|---:|---:|---:|
| EXP-024/025 | 97,26 ± 0,21% | 96,09 ± 0,42% | Holdout05 80,00 ± 1,56% |
| EXP-026 | 97,00 ± 0,14% | 96,23 ± 0,56% | Holdout05 74,83 ± 2,24% |

Cùng V11 nhưng 024/025 khác 026 cả về ảnh đầu vào lẫn brightness: không suy diễn một-fact ablation. Holdout05 chỉ có 4 supported classes và đã được kiểm tra nhiều lần. [Bảng và source](../training_results/current_dataset/DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md).

## ViT/DeiT, CBAM, YOLO
- ViT/XGBoost EXP-008 chọn Softmax A, seed 123 theo Validation D2H; EXP-009 inference only, kết quả B/C Test/Holdout lịch sử không được dùng để đổi lựa chọn. [Nguồn](../training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-009/README.md).
- CBAM V08: Δ Val Macro F1 ResNet50 +0,887 pp; DenseNet121 −1,384 pp; DeiT-Tiny −0,141 pp. Test/Holdout các ablation chưa chạy. [Nguồn](../training_results/current_dataset/CBAM_ABLATION_3MODELS_V08/README.md).
- YOLO26 Take01–07 chưa chốt detector, chưa kiểm định end-to-end pipeline. [Nguồn](../failure/YOLO26/README.md).

## Triển khai và giới hạn
- Application config.py trỏ tới DenseNet121 Transfer EXP-014 **seed 123**. Model binary không có trong Git; không tự đưa EXP-024/026 vào WebApp.
- Trước final cần xác minh nhóm ảnh near-duplicate, thực hiện **fresh blind holdout đủ 5 lớp**, kiểm thử inference/parity, phát hiện lesion và end-to-end YOLO crop → classifier.
