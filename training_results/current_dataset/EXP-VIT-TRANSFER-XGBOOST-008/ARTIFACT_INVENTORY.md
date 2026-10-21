# EXP-VIT-TRANSFER-XGBOOST-008 — Artifact inventory on GitHub

**Archive nguồn:** `EXP-VIT-TRANSFER-XGBOOST-008_ANALYSIS.zip` — **109 entries**, **4.541.180 bytes**, SHA256 `553AEC1C888DAFC3E6C1EDF9E65553A48DF082C68C8947E9B8EB4B9DB719EAD1`.

| Artifact | GitHub | Giới hạn |
|---|---|---|
| README, protocol review, status & inventory | ✅ | README biên tập để giải thích kết quả khoa học |
| 5-seed Validation metrics (15 rows) và Mean ± SD | ✅ | Bảng kết quả CSV với các metrics/denominators |
| 15 confusion matrices CSV | ✅ | 5 seeds × 3 classifiers, tổng từng class/1.430 ảnh đã được kiểm |
| Paired deltas, real-prefix Val subgroup | ✅ | Chỉ Validation |
| Classifier CV selection summary, training time, per-image timing, source parity | ✅ | CSV theo các artifact nguồn |
| Dataset audit, selection lock summary, selected seed metrics, metadata | ✅ summaries | Original full lock 46.131 bytes **chưa đưa nguyên bản** |
| EXP-007 checkpoint SHA inventory | ✅ | Chỉ SHA/size/đường dẫn, **không phải model weight** |
| SVG Accuracy & D2H summary | ✅ | Đồ thị tái tạo từ CSV, **không phải original PNG** |
| Original `notebook/vit_transfer_xgboost_008.ipynb` | ❌ | Source notebook **96.068 bytes**; SHA trong inventory nguồn |
| Original `*.png` figure/calibration panels | ❌ | Có trong ZIP nguồn, không được import nguyên bản |
| 15 `*_validation_predictions.csv` per image | ❌ | Chưa commit; không claim full per-image reproducibility |
| Trained `seed_*/xgboost.json` và Logistic packages | ❌ | Không commit executable model packages |
| Original ANALYSIS/FULL ZIP | ❌ | Cần canonical binary artifact storage |
| 5 binary DeiT EXP-007 `.keras` | ❌ | ~66,9MB/seed, không nằm trong EXP-008 source bundle; cấp riêng từ parent EXP-007 |
| V08 Final Test / Historical Holdout results for EXP-008 | **NOT_RUN** | Không suy diễn từ EXP-007 |
| Fresh blind field test | **NOT_RUN** | Cần trước khi claim cải thiện ứng dụng |

**Scientific provenance:** SHA của archive/checkpoints là của file nguồn. Không tuyên bố các file GitHub đã được viết lại có cùng checksum nếu không copy byte-for-byte. Cần binary checkpoints EXP-007 và model B/C để tái chạy đầy đủ.

**Ứng dụng:** Không thay web deployment, không chuyển classifier đang sử dụng.

[Experiment README](./README.md) · [Method audit](./METHOD_AND_LEAKAGE_REVIEW.md) · [Source file checksums](./source/SOURCE_ORIGINAL_CHECKSUMS.csv).
