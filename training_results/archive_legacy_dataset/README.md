# Legacy Dataset Experiments
> **Đồng bộ điều hướng 28/10/2026.** ARCHIVE — có giá trị truy xuất, không phải benchmark hiện tại; giữ checkpoint cho reproducibility. [Tổng quan](../../README.md) · [Danh mục](../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../docs/RESEARCH_PROTOCOL.md).


Thư mục này lưu các experiment **cũ hơn benchmark V03/V04**, chủ yếu các lần train ban đầu trước khi protocol nhiều seed và dataset versioning được chuẩn hóa.

Hiện có:

```text
archive_legacy_dataset/
├── EXP-DENSENET121-SCRATCH-001/
└── EXP-VIT-SCRATCH-001/
```

Các kết quả trong đây chỉ dùng cho:
- lịch sử phát triển;
- failure/iteration analysis;
- tham khảo cấu hình cũ.

Không sử dụng chúng trong bảng benchmark hiện tại.

## Lưu ý về V03

Các experiment `*-002` sử dụng `Cashew_dataV03` vẫn được giữ tại `training_results/current_dataset/` để không làm hỏng các đường dẫn đã được dùng trước đây. Tuy nhiên kể từ khi `Cashew_dataV04` được khóa, chúng được xem là **historical experiments**, không phải current benchmark.

Benchmark hiện tại xem tại:

[`../current_dataset/README.md`](../current_dataset/README.md)

## Đối chiếu theo mốc 19/10/2026

- DenseNet121 001: 1 seed, CashewData_v01, Test Acc 95.31% trên **747 ảnh**, khác locked Test V05/V08.
- ViT 001: 1 seed, V01, có Validation nhưng chưa có Test metrics trong README nguồn.
- Các file README của từng thí nghiệm ghi rõ lịch sử, config, dataset, kết quả cùng các giới hạn so sánh.
- [Bảng tiến độ đầy đủ tại current_dataset](../current_dataset/EXPERIMENT_PROGRESS_2026_10.md).
