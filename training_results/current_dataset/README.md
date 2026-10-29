# Current Dataset — Danh mục và phân tích kết quả

![Dataset snapshots](../../docs/assets/dataset_versions.svg)

## Phân biệt dataset
| Snapshot | Train | Validation | Test | Nguồn |
|---|---:|---:|---:|---|
| V04 | 4.822 | 1.402 | 687 | [benchmark_v04.csv](benchmark_v04.csv) |
| V08 | 4.790 | 1.430 | 686 | [EXP-DENSENET121-TRANSFER-014](EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) |
| V09 | 4.914 | 1.445 | 686 | [V09–V11 timeline](DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) |
| V10 | 4.920 | 1.445 | 686 | [V09–V11 timeline](DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) |
| V11 | 4.916 | 1.447 | 685 | [EXP-024–026](DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) |

V05/V07 là mốc lịch sử riêng, có benchmark CSV và README tương ứng, **không giả định dùng cùng ảnh/nhãn với V04**. Đọc dataset manifest của từng experiment trước khi so sánh.

## Phân loại trên V08
![Benchmark V08](../../docs/assets/benchmark_v08_test_holdout.svg)

| Dòng | Val Acc | Test Acc | Historical Holdout Acc |
|---|---:|---:|---:|
| [DenseNet121 Transfer 014/016](EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md) | **97,10%** | 95,13% | 80,83% |
| [ResNet50 Transfer 009](EXP-RESNET50-TRANSFER-5SEEDS-009/README.md) | 95,23% | 92,89% | 75,83% |
| [ViT Softmax A / EXP-009](EXP-VIT-TRANSFER-XGBOOST-009/README.md) | 94,41% | **95,36%** | 74,58% |

Số trên là mean 5 seed. Test có 5 lớp; Holdout 48 ảnh chỉ 4 lớp, đã xem trong lịch sử. Seed WebApp DenseNet121 014 là **123**, chọn bằng Validation loss; không thay theo Test.

## DenseNet121 V09–V11
![V11 so sánh hồi cứu](../../docs/assets/v11_retrospective.svg)

[Chi tiết EXP-018–026](DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md). Đặc biệt: EXP-019/020/021/025 là các lần **evaluation-only** dùng checkpoint trước đó, không phải training mới. EXP-024+025 V11 256px+brightness Test **96,09%**, Holdout05 **80,00%**; EXP-026 V11 224px không brightness Test **96,23%**, Holdout05 **74,83%**. Khác hai yếu tố nên không chứng minh resize là nguyên nhân.

## CBAM V08
![CBAM hiệu quả](../../docs/assets/cbam_delta_validation.svg)

[CBAM paired ablation ba backbone](CBAM_ABLATION_3MODELS_V08/README.md) có 3 kiến trúc × 2 chế độ × 5 seed; **chỉ Validation** và chưa Test/Holdout. Cần phân biệt baseline_no_cbam cùng EXP với các baseline Transfer/Scratch khác.

## Nghiên cứu có thể tái lập
[Danh mục tất cả thư mục EXP](../../docs/EXPERIMENT_INDEX.md) · [Experiment progress](EXPERIMENT_PROGRESS_2026_10.md) · [Protocol](../../docs/RESEARCH_PROTOCOL.md). Các archive backfill chỉ có thể chứa trích xuất README/CSV; không mặc nhiên có checkpoint binary, figure hoặc nguồn ZIP đầy đủ.
