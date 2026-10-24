# EXP-VIT-TRANSFER-XGBOOST-008 — DeiT-Tiny features + Softmax / Logistic Regression / XGBoost

**Trạng thái nguồn:** `COMPLETED_VALIDATION` (21/10/2026), classifier-only. **Final Test V08: NOT_RUN. Historical Holdout: NOT_RUN. Fresh blind field test: NOT_RUN.** Đây không phải lần huấn luyện lại ViT backbone: sử dụng đúng **5 checkpoint DeiT-Tiny đã khóa từ `EXP-VIT-TRANSFER-5SEEDS-007`**, trích embedding và fit các classifier B/C.

> **Kết luận cần giữ nguyên:** Logistic Regression (B) và XGBoost (C) có **Validation Accuracy cao hơn Softmax (A)**, nhưng cả B/C đều **tăng lỗi disease → Healthy trên deployment seed 123** (A **18/900**, B **21/900**, C **23/900**). Quy tắc đã khóa chọn **A / seed 123**, không chuyển sang XGBoost chỉ vì Accuracy. **Chưa có bằng chứng cải thiện trên ảnh thực địa mù mới**.

## 1. Nguồn dữ liệu và provenance

- Source archive người nghiên cứu cung cấp: `EXP-VIT-TRANSFER-XGBOOST-008_ANALYSIS.zip` (109 entries), SHA256: `553AEC1C888DAFC3E6C1EDF9E65553A48DF082C68C8947E9B8EB4B9DB719EAD1`.
- Dataset V08 `CashewData_Split (8).zip` SHA256 `75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7`; **Train 4.790 / Val 1.430 / Test 686 = 6.906** ảnh.
- Source experiment `EXP-VIT-TRANSFER-5SEEDS-007`, 5 seeds `42,123,2026,3407,7777`; seed deploy cố định **123** từ EXP-007, không đổi theo results EXP-008.
- Label order: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`. Validation true support lần lượt **308,216,264,328,314**, trong đó đúng **900 ảnh bệnh** thuộc Anthracnose/Leaf miner/Red rust (denominator D2H).
- EXP-008 ghi `SOURCE_MANIFEST_SHA256=C699A9ACA707DA25F143DB1FC3C9B5133EA2858BCDC468A158E5644E500EC715`; đây là **SHA của manifest nguồn riêng cho experiment**, không đồng nhất với các fingerprint manifest được tái tạo theo quy trình khác trong những EXP CNN trước. Dataset archive SHA vẫn như trên; không thay đổi V08.
- Source checkpoint selected seed 123 SHA256: `FD1DD6B775F29F400ED4D1E889B217A5C97B513E9C0066C07136B57B57E2BC89`. Các SHA của đủ 5 checkpoints nằm trong [checkpoint provenance](./source/exp007_checkpoint_inventory.csv).

## 2. Ba nhánh đối chứng, cùng feature extractor

| Nhánh | Mô hình | Cách thực hiện |
|---|---|---|
| **A: Softmax** | Classification head có sẵn từ EXP-007 | Dự đoán với pretrained + fine-tuned DeiT-Tiny checkpoint đã khóa; **không retrain** |
| **B: Logistic Regression** | `StandardScaler` + Logistic Regression | Extract **192-D** penultimate DeiT embedding, Train-only **3-fold CV** cho `C∈{0.01,0.1,1,10}`, fit cuối trên Train |
| **C: XGBoost** | `multi:softprob`, 5 lớp | Cùng 192-D embedding, **12 fixed candidate configs**, Train-only **3-fold CV**, fit cuối trên Train |

- Feature endpoint nguồn: `DeiTImageClassifier.output_dense.input`, **192 chiều**, input RGB float32 **0–255**, bilinear resize 224×224; augmentation ở `training=False`, preprocessing bên trong model **một lần**.
- Smoke / source parity report cho cả 5 seeds: **0 mismatch** trên 1.430 dự đoán Softmax Val so với EXP-007; source báo preprocess và checkpoint checks PASS. **Chưa chạy lại độc lập trên binary từ GitHub**.
- XGBoost/Logistic đều học từ embedding Train; Validation dùng để so sánh sau CV theo selection rule. **CV chỉ là classifier-level CV**: backbone nguồn đã học cả Train trước đó, nên không phải nested group-CV độc lập end-to-end.

## 3. Validation 5 seeds (mean ± sample SD)

| Metric | A — Softmax | B — Logistic Regression | C — XGBoost |
|---|---:|---:|---:|
| **Accuracy** | **94,41 ± 0,73%** | **96,56 ± 0,17%** | **95,85 ± 0,20%** |
| **Macro F1** | 94,29 ± 0,70% | **96,40 ± 0,18%** | 95,64 ± 0,20% |
| Log Loss ↓ | 0,1791 ± 0,0210 | **0,1084 ± 0,0041** | 0,1294 ± 0,0087 |
| **Disease → Healthy / 900** ↓ | 20,8 ± 4,44 | **18,6 ± 2,30** | 22,2 ± 2,05 |
| Healthy Recall | 98,24% | **98,80%** | 97,50% |
| Not cashew leaf Recall | 91,72% | **99,68%** | 99,55% |

**Lưu ý:** Trung bình nhiều seeds của B giảm D2H nhẹ so với A, nhưng **không cải thiện tại seed 123 đã khóa**; trên 5 seed D2H của XGBoost còn cao hơn Softmax. Chỉ số V08 **không đại diện test thực địa mới**.

### Chi tiết seed 123 (locked deployment)

| Metric | A: Softmax | B: Logistic | C: XGBoost |
|---|---:|---:|---:|
| Val đúng / 1.430 | 1.365 | **1.378** | 1.371 |
| Val Accuracy | 95,45% | **96,36%** | 95,87% |
| Val Macro F1 | 95,23% | **96,20%** | 95,69% |
| **Bệnh → Healthy / 900** ↓ | **18 (2,00%)** | 21 (2,33%) | 23 (2,56%) |
| Anthracnose Recall | **91,23%** | 89,61% | 87,99% |
| Healthy Recall | 97,22% | **99,54%** | 98,15% |
| Not cashew leaf Recall | 95,54% | **99,68%** | **99,68%** |

Tại seed 123, lỗi D2H của A xuất phát từ **18 Anthracnose→Healthy**; B có **20 Anthracnose + 1 Leaf miner→Healthy**; C có **23 Anthracnose→Healthy**. Trong classifier C, Anthracnose Recall giảm **3,25 pp** so A, vượt guardrail Recall đã định nghĩa.

Source tables: [per-seed metrics](./validation_metrics_5seeds_abc.csv), [mean ± SD](./validation_mean_std.csv), [per-seed D2H](./derived/disease_to_healthy_counts_5seeds.csv), [paired delta](./paired_deltas.csv) và [15 confusion matrix](./seed_123/A_confusion_matrix.csv).

## 4. Vì sao A được chọn, dù B có Val Accuracy cao nhất?

Quy tắc `selection/validation_selection_lock.json` của source: **minimum Disease→Healthy trước**, có guardrails per-class Recall / Macro F1 / Healthy Recall / Not-cashew Recall; nếu không có phương án cải thiện D2H cho **seed deploy 123**, fallback **A**. Source chọn `seed=123, classifier=A` và **không dùng Test/Holdout để đổi**.

- **A** D2H **18/900**, được giữ.
- **B** D2H **21/900**, dù Accuracy/Macro F1 cao hơn.
- **C** D2H **23/900**, Anthracnose Recall giảm mạnh.
- `ENABLE_HEALTHY_GATE=false`, **không có threshold postprocessing** trong EXP-008.
- Việc lựa chọn dựa trên Validation đã dùng nhiều lần vẫn có **selection bias**. Chỉ báo cáo là *selected by validation rule*, không phải *validated on new independent data*.

## 5. Khả năng áp dụng và những hạn chế

1. **Mục tiêu ưu tiên chưa đạt:** XGBoost tăng Accuracy nhưng tăng D2H; B cải thiện trung bình song seed deploy kém hơn baseline.
2. **Lớp Anthracnose:** B/C trade-off làm giảm recall, đặc biệt C. Cần kiểm tra ảnh `anthracnose_1480..1486` và nhóm lỗi lặp bằng nhãn chuyên môn, không sửa nhãn dựa trên model.
3. **Field generalization chưa đánh giá:** source chỉ có Validation; nhóm 35 ảnh Val có tiền tố `real_` là subgroup nhỏ, không phải blind field holdout. **Không được trích kết quả Test/Holdout EXP-007 thành kết quả EXP-008**.
4. **Data leakage:** audit nguồn cho **0 exact-byte duplicate**, **1 dHash candidate** `VISUAL_REVIEW_REQUIRED`, còn `GROUP_LEAKAGE_NOT_VERIFIED`. dHash khác pHash và không loại trừ ảnh gần trùng/cùng lá/cây/buổi chụp.
5. **CV leakage caveat:** backbone EXP-007 đã fit trên Train, do đó train-only folds B/C không mô phỏng hoàn toàn unseen backbone. Không tuyên bố end-to-end leakage-free OOF.
6. **Model package:** selected A/seed123 tham chiếu checkpoint EXP-007 bằng SHA. **FULL ZIP của experiment không tự chứa checkpoints**, không phải model đã sẵn sàng tải chạy từ repo.
7. **Không thay Application**; chưa đưa B/C hoặc threshold vào model đang phục vụ.

## 6. Hướng phát triển

- **P0:** audit Anthracnose bị nhận thành Healthy, ghi lỗi theo ảnh và theo nguồn/session; kiểm tra cặp dHash + candidates pHash phiên bản V08.
- **P1:** đối chứng cùng một **healthy decision threshold** trên A/B/C nhằm giảm D2H mà giữ Healthy Recall, Macro F1 và số cảnh báo sai trong giới hạn; lựa chọn bằng Validation, test trên **fresh blind field set có đủ 5 true classes**.
- **P1:** thử calibrated classifiers/threshold với group-aware Train CV và split theo leaf/session nếu có provenance. Không tối ưu theo Test V08/Holdout 48 ảnh đã bị xem trước.
- **P2:** khi có checkpoint và nhãn đủ tin cậy, thử YOLO crop ROI → DeiT embeddings; so sánh paired errors, inference latency, feature stability.

## 7. Trạng thái artifact GitHub

Đã import dạng **report + curated numeric/metadata artifacts**: per-seed metric CSV, 15 confusion CSV, D2H analysis, paired deltas, CV timing, real-prefix subgroup, dataset/source integrity summary, selection metadata snapshot, checkpoint SHA references, provenance. **Không commit** source ANALYSIS ZIP gốc 4,4 MB, notebook gốc, original PNG charts, 15 full per-image prediction CSV, trained XGBoost JSON binary/text rất lớn, FULL ZIP hoặc EXP-007 model checkpoints. Xem [inventory](./ARTIFACT_INVENTORY.md). Mọi checksum tham chiếu **bytes gốc từ archive nguồn**, không giả định JSON/CSV biên soạn có cùng hash.

[Technical review](./METHOD_AND_LEAKAGE_REVIEW.md) · [Experiment progress](../EXPERIMENT_PROGRESS_2026_10.md).
