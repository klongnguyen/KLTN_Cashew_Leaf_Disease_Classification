# Chín ảnh Real Test dự đoán sai — IMPROVED001

> **Chỉ sử dụng để phân tích hậu nghiệm và xác minh chuyên môn**. Không dùng Test để chọn mô hình, thay đổi siêu tham số hoặc sửa nhãn rồi đánh giá lại trên cùng Test như thể đây là dữ liệu độc lập.

| Ảnh Test | Ground truth | Dự đoán | Confidence |
|---|---|---|---:|
| `test/healthy/real_healthy_020.jpg` | healthy | leaf_miner | 94,42% |
| `test/red_rust/real_red_rust_029.jpg` | red_rust | anthracnose | 79,90% |
| `test/healthy/real_healthy_019.jpg` | healthy | leaf_miner | 69,60% |
| `test/red_rust/real_red_rust_026.jpg` | red_rust | not_cashew_leaf | 63,70% |
| `test/anthracnose/real_anthracnose_047.jpg` | anthracnose | red_rust | 59,70% |
| `test/anthracnose/real_anthracnose_049.jpg` | anthracnose | leaf_miner | 57,60% |
| `test/anthracnose/real_anthracnose_045.jpg` | anthracnose | not_cashew_leaf | 53,80% |
| `test/red_rust/real_red_rust_024.jpg` | red_rust | anthracnose | 48,00% |
| `test/healthy/real_healthy_017.jpg` | healthy | leaf_miner | 43,80% |

**Biểu mẫu review đề xuất:** `file`, `expert_label`, `multiple_diseases`, `symptom_visible`, `background_complex`, `image_quality`, `same_leaf_group`, `reviewer`, `review_notes`. Các cột cần review thủ công, không được tự điền bằng prediction.
