# Legacy Dataset Experiments

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
