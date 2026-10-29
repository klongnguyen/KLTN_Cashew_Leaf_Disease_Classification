# STANDARD TRAINING NOTEBOOK FORMAT — V2.0

> **Trạng thái:** Chuẩn đề xuất áp dụng cho thí nghiệm **mới**, cập nhật ngày 29/10/2026.  
> **Phạm vi:** Classification CNN, ResNet50, DenseNet121, ViT/DeiT và các biến thể Scratch / Transfer / Frozen / Fine-tuning / Attention (CBAM).  
> **Tài liệu đi kèm:** [`STANDARD_STATISTICAL_ANALYSIS_FORMAT.md`](STANDARD_STATISTICAL_ANALYSIS_FORMAT.md) — chuẩn phân tích liên mô hình, kiểm định thống kê và hình cho báo cáo.  
> **Tài liệu đối chiếu của dự án:** `docs/RESEARCH_PROTOCOL.md`, `docs/research_notes/AUDIT_NOTEBOOK_OUTPUTS_STATISTICAL_PLAN.md`, `DATASET_CHANGELOG.md`.

**Quy tắc chuyển tiếp:** Tài liệu này **thay thế** chuẩn notebook single-seed V1 đối với thí nghiệm mới. **Không sửa hồi cứu** selection lock, Test, dataset split, checkpoint hoặc metric của experiment lịch sử để trông nhất quán với V2. Các EXP cũ phải giữ provenance, ghi rõ khác biệt protocol và trạng thái `LEGACY` khi tổng hợp.

---

## 1. Mục đích và ranh giới công việc

Một notebook huấn luyện cần tạo được experiment **tái lập, kiểm toán được và so sánh được**, không chỉ train xong mô hình:

1. Cấu hình, seed, dataset checksum, class order, preprocessing và checkpoint rõ ràng.
2. Huấn luyện **5 seed** trên **cùng Train/Validation/Test đã khóa**; lưu kết quả từng seed.
3. Tạo đầy đủ **Learning Curves 2 hàng × 5 cột**, **Confusion Matrix 1 hàng × 5 cột** (Raw và Normalized), bảng per-class và phân tích lỗi.
4. Tính **mean ± sample SD** (`ddof=1`), kèm CI theo seed **nếu được áp dụng và ghi đúng ý nghĩa**.
5. Chọn phương án/seed/checkpoint **chỉ từ Validation**; Test và Holdout phải có cổng thực thi riêng.
6. Xuất `predictions.csv` có **toàn bộ xác suất 5 lớp**, đủ dữ liệu để notebook thống kê trung tâm chạy McNemar, bootstrap, calibration và so sánh liên mô hình.
7. Xuất `ANALYSIS.zip` nhẹ cho báo cáo và `FULL.zip` có checkpoints, không nhầm lẫn giữa file thực sự có và file chỉ được mô tả.

**Phân tầng trách nhiệm**:

| Tầng | Vai trò | Bắt buộc |
|---|---|---|
| **A — Training notebook** | Train/evaluate từng EXP; 5-seed metrics, panels, audit và atomic output | Có |
| **B — `STATISTICAL_ANALYSIS.ipynb`** | Đọc nhiều ANALYSIS ZIP, so sánh Scratch/Transfer, paired tests, bootstrap, CI, Holm, calibration và figures | Có khi làm benchmark liên mô hình |
| **C — Paper/Report** | Biểu đồ, bảng, giải thích hiệu ứng, giới hạn suy luận và khả năng triển khai | Có |

**Không** ép từng notebook training tự kiểm định tất cả các kiến trúc; tránh sao chép thống kê và sai lệch schema. Notebook phân tích trung tâm được mô tả ở tài liệu đi kèm.

---

## 2. Quy tắc khoa học không được vi phạm

### 2.1. Phân chia dữ liệu và cách sử dụng

| Tập | Mục đích | Không được làm |
|---|---|---|
| `train` | Học trọng số, train-only augmentation | Dùng ảnh/biến thể cùng lá từ Val/Test |
| `val` | Chọn checkpoint, seed, recipe, threshold và quyết định mô hình | Đánh giá như một external blind test |
| `test` **locked** | Đánh giá lựa chọn **đã khóa trước** | Tuning LR, augmentation, model, seed, threshold rồi gọi kết quả là blind |
| `historical_holdout` | Phân tích hồi cứu, lỗi thực địa, domain gap | Chọn model rồi quảng bá là fresh blind holdout |
| `fresh_blind_holdout` | Kiểm định độc lập cuối với nguồn thu thập mới | Cho training/tuning/selection tiếp xúc trước khi khóa |

- Các split phải giữ **nguyên cùng dataset version, manifest và source group** giữa 5 seed và các baseline cần so sánh. Không thay Test chỉ để tăng điểm.
- **Split theo nhóm** cùng lá/cây/phiên chụp (nếu metadata cho phép), kiểm tra exact hash, pHash/dHash candidates và đánh giá thủ công các cặp nghi vấn. `0 exact duplicate` **không chứng minh** không có near-duplicate/group leakage.
- Tiền tố `real_` **chỉ là quy tắc đặt tên**, không tự chứng minh nguồn thực địa độc lập. Phải lưu `source_type`, `leaf_id`, `tree_id`, `session_id` nếu xác minh được, nếu chưa thì `unknown`.
- Không đổi nhãn/chuyển split chỉ vì model dự đoán sai; chỉ cập nhật dataset bằng quy trình review chuyên môn rồi **tạo version/snapshot mới**.
- Dataset V11 hiện đã ghi nhận **4.916 train / 1.447 val / 685 test = 7.048 ảnh**; theo prefix `real_` tương ứng **238/54/28**. Đây là số liệu thuộc snapshot V11, **không** hard-code làm hằng số cho thí nghiệm/dataset mới.
- Holdout05 lịch sử: **120 ảnh thuộc 4 lớp có ground truth** (`anthracnose=46`, `healthy=13`, `leaf_miner=36`, `red_rust=25`, `not_cashew_leaf=0`), đã tiếp xúc và **không phải fresh blind 5-class**. Không so sánh trực tiếp `Macro F1` 4-class Holdout với `Macro F1` 5-class Test như thể là cùng định nghĩa.

### 2.2. Class label order và preprocessing

Năm lớp canonical (tên dùng cho CSV/report):

```text
anthracnose
healthy
leaf_miner
red_rust
not_cashew_leaf
```

**Lưu ý:** đây là **thứ tự canonical đề xuất** dựa trên một số EXP V08–V11; **thứ tự tensor thật phải đọc từ `class_names` / `class_to_idx` / `label_map.json` của chính EXP**. Không áp đặt thứ tự này lên checkpoint lịch sử có mapping khác. Luôn lưu `class_to_idx` và `idx_to_class` từ runtime cùng model package. Output probabilities phải được **map theo tên lớp**, không suy đoán theo index.

- Train augmentation chỉ trên Train; Val, Test và Holdout không augmentation ngẫu nhiên.
- Preprocess đúng backbone/version và ghi nơi áp dụng (`dataset` / `model` / `adapter`). Không vô tình nhân đôi `Rescaling(1/255)` và `preprocess_input`.
- Với Transfer, khai báo ImageNet weights, frozen layers, fine-tuned layers, BN behavior, classifier head, input size, trainable/non-trainable params. Với Scratch ghi khởi tạo trọng số.

### 2.3. Phân biệt 5 seed và 5 tập độc lập

- Seed cố định theo protocol dự án: **`[42, 123, 2026, 3407, 7777]`**. Nếu thí nghiệm có lý do thay seed phải khai báo khác biệt.
- Mỗi seed khởi tạo/training **độc lập**, dùng cùng split và cùng recipe. Không lấy 5 lần đánh giá lại **cùng một checkpoint** gọi là 5 lần train.
- Báo cáo `mean ± sample standard deviation (ddof=1)`; **SD không phải CI** và **5 seed trên một Test không tương đương 5 Test độc lập**.
- Thí nghiệm `EVAL_ONLY`/`VAL_ONLY` phải ghi rõ và kế thừa provenance/checkpoint từ `TRAINED` gốc.

---

## 3. Định danh thí nghiệm, cấu hình và trạng thái

### 3.1. Quy tắc ID

```text
EXP-{MODEL}-{TRAINING_MODE}-{5SEEDS}-{NUMBER}
# Ví dụ (theo nhánh lịch sử dự án)
EXP-RESNET50-TRANSFER-5SEEDS-009
EXP-DENSENET121-TRANSFER-5SEEDS-026
EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011
```

Không đổi ID experiment cũ; dùng `parent_experiment_id` nếu evaluation-only hoặc ablation dựa trên checkpoint cũ.

### 3.2. Các chế độ/trạng thái

| Trạng thái | Diễn giải |
|---|---|
| `PLANNED` | Chưa train |
| `TRAINING` | Có training đang thực hiện |
| `VAL_COMPLETE` | 5 seed và Validation hoàn tất; **chưa yêu cầu Test** |
| `SELECTION_LOCKED` | Chọn model/seed/threshold theo Val và lưu JSON lock |
| `TEST_COMPLETE` | Locked Test đã đánh giá theo protocol |
| `HISTORICAL_HOLDOUT_EVAL` | Holdout đã tiếp xúc được đánh giá hồi cứu |
| `FRESH_BLIND_EVAL` | Đánh giá bộ độc lập mới, chỉ sau protocol lock |
| `EVAL_ONLY` | Chỉ evaluate checkpoint từ EXP khác |
| `INCOMPLETE` | Thiếu output bắt buộc cho stage đã khai báo |

**Không** xem Test chưa chạy là lỗi nếu notebook đang chủ ý ở `VAL_COMPLETE`; không tuyên bố `TEST_COMPLETE` khi thiếu Test artifacts.

### 3.3. Config nằm ở một cell duy nhất

```python
EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-NEW"
PARENT_EXPERIMENT_ID = None
DATASET_VERSION = "Cashew_dataV11"   # Ví dụ: phải đối chiếu với snapshot thực tế
DATASET_ARCHIVE_SHA256 = "<verified sha256>"
DATASET_MANIFEST_SHA256 = "<verified manifest sha256>"
SEEDS = [42, 123, 2026, 3407, 7777]
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
TRAINING_MODE = "ImageNet_Transfer"
RUN_FINAL_TEST = False
RUN_HISTORICAL_HOLDOUT = False
RUN_FRESH_BLIND_HOLDOUT = False
VIEW_ONLY = False

# Phải được ghi trước mọi Test/Holdout:
SELECTION_PRIMARY = "val_macro_f1"  # Chuẩn đề xuất mới, không sửa EXP lịch sử
SELECTION_TIEBREAKER = "val_loss"
```

Các trường bắt buộc bổ sung: `model_name/backbone`, `weights`, `fine_tune_stages`, `optimizer`, `weight_decay`, `loss`, `augmentation`, `callbacks`, `framework/cuda`, `class_order`, `precision_policy`, `hardware`, `source_commit_sha`, `selection_rule`, `dataset_provenance`.

*Với EXP cũ đang dùng `val_loss` làm tiêu chí chính: giữ nguyên quy tắc cũ trong archive; tiêu chí `val_macro_f1` chỉ là mặc định đề xuất cho EXP mới. Khi thay criterion phải tạo EXP mới, so sánh bằng Validation.*

---

## 4. Trình tự notebook bắt buộc (chuẩn V2)

Các section dưới đây thay cho chuỗi 29 cell của V1. Có thể chia mỗi section thành nhiều code/markdown cell, nhưng giữ **thứ tự và output contract**.

| ID | Section | Output tối thiểu | Stage |
|---|---|---|---|
| 00 | EXPERIMENT OVERVIEW & HYPOTHESIS | Mục tiêu, ablation factor, mode, parent | Setup |
| 01 | IMPORT LIBRARIES | Phiên bản thư viện cần thiết | Setup |
| 02 | ENVIRONMENT / GPU | Python, TF/PyTorch, CUDA, GPU, platform | Setup |
| 03 | CONFIG / REPRODUCIBILITY | `experiment_config.json`, `SEEDS` | Setup |
| 04 | LOAD & VERIFY DATASET | Archive hash, manifest hash, split counts, label_map | Audit |
| 05 | DATASET QUALITY AUDIT | Class/split counts, corrupt/duplicate candidates, group leakage | Audit |
| 06 | DATASET DISTRIBUTION | Tổng ảnh theo lớp/split, `real_` theo split & class nếu có dữ liệu | Audit |
| 07 | PREPROCESSING & AUGMENTATION | Ghi rõ train-only, normalization, model contract | Setup |
| 08 | DATA PIPELINE | Shuffle, batch, prefetch, worker, performance | Setup |
| 09 | MODEL BUILD / SUMMARY | Weights, frozen layers, model summary, params | Setup |
| 10 | COMPILE / CALLBACKS | LR, loss, optimizer, checkpoint/ES/ReduceLR | Setup |
| 11 | SMOKE TEST | Verify batch, labels, output dim, save/load parity | Setup |
| 12 | TRAIN FIVE SEEDS | Model + history + logs từng seed | Train |
| 13 | PER-SEED VALIDATION | Metrics, report, predictions đầy đủ xác suất | Validation |
| 14 | VALIDATION PANELS | Curves **2×5**, CM raw & normalized **1×5** | Validation |
| 15 | ERROR / DOMAIN AUDIT | Confusions, confidence, real_ proxy, high-conf wrong | Validation |
| 16 | FIVE-SEED AGGREGATE | Mean, sample SD, seed CI nếu thích hợp, per-class summaries | Validation |
| 17 | SELECTION & LOCK | `validation_selection_lock.json` | Gate |
| 18 | LOCKED TEST (optional gated) | Test metrics/preds, CM, panels theo seed | Test |
| 19 | HISTORICAL HOLDOUT (optional gated) | 4-class-aware metrics, domain-gap, provenance | Holdout |
| 20 | FRESH BLIND HOLDOUT (separate, gated) | Blind field metrics, cohort/source manifest | Holdout |
| 21 | INFERENCE / DEPLOYMENT BENCHMARK | Latency mean/median/p95, FPS, size, params | Efficiency |
| 22 | GRAD-CAM / EXPLAINABILITY (conditional) | Heatmap + overlay + limitations | Optional |
| 23 | MANUAL IMAGE PREDICTION | Top-k, class mapping, preprocess parity | Utility |
| 24 | EXPERIMENT README & REGISTRY | `README.md`, registry row, status | Export |
| 25 | EXPORT ANALYSIS & FULL | `ANALYSIS.zip`, `FULL.zip`, checksums | Export |
| 26 | ARTIFACT AUDIT / COMPLETION | Manifest/checklist + stage status | Export |

### 4.1. Cell 04–06: Data audit và biểu đồ phân bố

- Thống kê theo **5 class × 3 split**: `class_counts_by_split.csv` và stacked bar chart.
- Thống kê `real_` theo **split** và **theo class × split** khi có manifest: `real_prefix_counts_by_split.csv`, `real_prefix_counts_by_split_class.csv`; nếu không có dữ liệu đủ chi tiết thì **để `UNKNOWN/PENDING`**, không suy đoán phân bố.
- Báo cáo `real_` chỉ là *filename proxy*, không phải proven field source; Holdout phải quản lý manifest riêng.
- Lưu `dataset_audit.json` với: checksum, corrupt paths, exact duplicate, near-duplicate candidates, group leakage review status, class distribution, source metadata availability.
- Các ảnh cùng nguồn/cùng lá phải được kiểm tra thủ công nếu có nghi vấn; không tự loại mẫu vì confidence thấp.

### 4.2. Cell 12: Huấn luyện thực sự 5 seed

```python
for seed in SEEDS:
    reset_session_and_seed(seed)
    model = build_fresh_model()        # Tuyệt đối không tái sử dụng weights seed trước
    history = train_one_seed(model, train_ds, val_ds, seed)
    export_seed_artifacts(model, history, seed)
```

Đây là **pseudocode định nghĩa contract**, không phải code chạy độc lập: các helper phải được triển khai theo framework. Nếu dùng 2-stage frozen/fine-tune, lưu `stage1`/`stage2` và nối history đúng thứ tự epoch, đánh dấu điểm chuyển stage. Checkpoint theo mỗi seed phải được lưu độc lập.

### 4.3. Cell 13–16: Quy tắc metrics

Tất cả **Validation / Test / Holdout** phải có `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `balanced_accuracy`, `loss` (nếu có), per-class `precision/recall/f1/support`, class-order, seed.

- `Macro F1` và `Balanced Accuracy` được ưu tiên cùng Accuracy.
- `sample SD`: `np.std(scores, ddof=1)`; báo cáo mean ± SD **từng metric**, n=5, số liệu theo seed.
- 95% CI qua seed (nếu tính): `mean ± t_{0.975,4} × SD/√5`, công bố giả định và ý nghĩa là **variation across random seeds on a fixed split**, không phải CI hiệu năng cho vườn cây mới.
- Per-class performance: báo `support` đúng tập và chỉ tính các lớp có ground-truth support theo quy ước đã công bố; không che việc thiếu lớp ở Holdout.
- Bảo toàn **raw counts** trước khi chuẩn hóa confusion matrix, không cộng các confusion normalized khác split rồi gọi là metric tổng.
- Lưu các lỗi trọng điểm: `anthracnose → healthy`, `healthy → disease`, `anthracnose ↔ leaf_miner`, `not_cashew_leaf → cashew_disease`.

### 4.4. Cell 17–20: Gating Test và Holdout

Trước bất kỳ Test/Holdout nào, phải có **selection lock được ghi đĩa**, tối thiểu:

```json
{
  "experiment_id": "EXP-...",
  "dataset_version": "Cashew_dataV11",
  "dataset_archive_sha256": "<verified>",
  "dataset_manifest_sha256": "<verified>",
  "selected_seed": 42,
  "selected_checkpoint_sha256": "<verified>",
  "selection_split": "validation",
  "primary_metric": "val_macro_f1",
  "tiebreaker": "val_loss",
  "selected_thresholds": null,
  "class_to_idx": {},
  "locked_at_utc": "<ISO 8601>",
  "protocol_version": "STANDARD_TRAINING_NOTEBOOK_FORMAT_V2"
}
```

- **Test** chỉ chạy khi `RUN_FINAL_TEST=True` và lock tồn tại/khớp checksum; tuyệt đối không chọn lại seed/recipe dựa vào Test.
- `RUN_HISTORICAL_HOLDOUT` chỉ chạy trên checkpoint đã khóa; ghi `EXPOSED/RETROSPECTIVE`, số ảnh, cohort, class support, khác biệt với Test và thời điểm ảnh đã được review.
- `RUN_FRESH_BLIND_HOLDOUT` chỉ chạy trên **tập ảnh mới độc lập**, kết quả chưa bị lộ trong quá trình phát triển, đóng băng tiêu chí từ trước.
- Không cần chạy Test cho mọi biến thể **ablation khám phá**; có thể ghi `VAL_ONLY` và chỉ final-evaluate ứng viên đã được chọn theo protocol. Tôn trọng việc một số EXP lịch sử đã được test trước đây.
- Không so chênh lệch V08/V09/V10/V11 hoặc Holdout02–05 là one-factor ablation khi dataset/GT đồng thời thay đổi.

### 4.5. Cell 21: Benchmark tốc độ có thể so sánh

Benchmark **cùng GPU/CPU, framework, dtype, batch=1, input size ghi rõ, số warm-up cố định, synchronize CUDA nếu dùng GPU**, đo rõ tính/không tính decode, resize, preprocessing, postprocessing. Xuất **mean, median, p95 (ms/image)**, FPS nếu có ý nghĩa, model size (MiB), total/trainable params, HW và benchmark protocol. Không so trực tiếp số ms đo khác phần cứng/runtime.

---

## 5. Bộ hình bắt buộc của từng training notebook

### 5.1. Learning Curves — chính xác **2 hàng × 5 cột**

```text
            seed 42   seed 123  seed 2026 seed 3407 seed 7777
Accuracy       [ ]       [ ]       [ ]       [ ]       [ ]
Loss           [ ]       [ ]       [ ]       [ ]       [ ]
```

- Mỗi subplot hiển thị **Train và Validation** rõ legend/axis.
- Cùng đơn vị cho các ô cùng hàng; chỉ rõ stage1/stage2 và best epoch nếu có.
- File: `figures/panels/training_curves_5seeds_2x5.png` (và `pdf` hoặc `svg` nếu tiện), 300 DPI cho báo cáo.
- Vẫn lưu `history.csv` mỗi seed và (nếu có) từng stage; không dùng duy nhất screenshot notebook.

### 5.2. Confusion Matrix — chính xác **1 hàng × 5 cột**, theo split

- Tạo **hai panel** cho mỗi split đã đánh giá: `raw` và `normalized` (normalize theo true-class rows), riêng cho Validation và Test. Holdout cũng có hai panel nếu thật sự chạy và đủ 5 seed.
- Không nhầm panel Validation với Test; cùng class order, same color scale (trong normalized panels), label tiếng Anh canonical rõ ràng.
- Lưu mỗi seed: `confusion_matrix_raw.csv/png`, `confusion_matrix_normalized.csv/png`; panel:
  - `figures/panels/validation_confusion_raw_1x5.png`
  - `figures/panels/validation_confusion_normalized_1x5.png`
  - `figures/panels/test_confusion_raw_1x5.png` *(chỉ nếu Test executed)*
  - `figures/panels/test_confusion_normalized_1x5.png` *(chỉ nếu Test executed)*

### 5.3. Hình/bảng bổ sung **bắt buộc khi hoàn tất Validation**

1. `class_distribution_by_split.png` — bar chart phân bố 5 class × 3 split.
2. `real_prefix_distribution_by_split.png` — 3 nhóm split; có ghi chú real-prefix chưa xác thực provenance.
3. `per_class_f1_heatmap_validation.png` — 5 lớp × 5 seed, kèm số support.
4. `seed_metric_errorbars_validation.png` — Accuracy, Macro Precision/Recall/F1, Balanced Acc **mean ± sample SD**, chú thích SD không phải CI.
5. `validation_error_montage_seed_selected.png` — 10–20 ảnh sai (và đúng low confidence), true/pred/confidence + filename; nếu chưa selected seed thì chọn theo **Validation** trước.
6. `validation_error_summary.csv` — số lượng và tỷ lệ lỗi từng cặp true→pred, false-healthy, healthy→disease, high-confidence wrong theo seed.

### 5.4. Khi chạy Test/Holdout

Xuất lại `per_class_f1_heatmap_{split}.png`, `seed_metric_errorbars_{split}.png`, confusion panels và error montage theo split. Hình `domain_gap_test_vs_holdout.png` chỉ mang ý nghĩa **khảo sát** nếu Holdout đã tiếp xúc hoặc khác class support; phải có chú thích rõ và không được dùng để chọn lại mô hình.

### 5.5. Hình nâng cao (có điều kiện; phần xử lý liên mô hình ở tài liệu riêng)

- Confidence histogram và `reliability_diagram_validation.png`, calibration table (`ECE`, `Brier`), ROC/PR one-vs-rest từ xác suất 5 lớp nếu protocol chấp thuận.
- Grad-CAM overlays cần **ảnh gốc + class mapping + vùng tổn thương tham chiếu**; attention heatmap thô chưa chứng minh model tập trung đúng lesion.
- Không bắt training notebook sinh forest plot, Holm table hoặc McNemar giữa các mô hình — các nội dung đó thuộc notebook thống kê B.

---

## 6. Chuẩn output CSV / JSON thống nhất

### 6.1. `predictions.csv` — một hàng / seed / ảnh / split

**Thứ tự và tên cột bắt buộc**:

```csv
experiment_id,architecture,training_mode,dataset_version,dataset_sha256,dataset_manifest_sha256,split,seed,image_id,relative_path,true_label,predicted_label,true_class_idx,predicted_class_idx,correct,confidence,prob_anthracnose,prob_healthy,prob_leaf_miner,prob_red_rust,prob_not_cashew_leaf
```

- `relative_path` thống nhất POSIX `/`; `image_id`/`source_group_id` nên đủ giúp ghép cặp ảnh khi so mô hình. Có thể bổ sung `leaf_id`, `tree_id`, `session_id`, `source_type`, `checkpoint_sha256`.
- `confidence = max(probabilities)` của **predicted_label**; mỗi xác suất hữu hạn trong `[0,1]`, tổng xác suất mỗi hàng xấp xỉ 1 (softmax).
- `correct` là bool; không chỉ lưu index, phải có **tên nhãn** và index riêng.
- `prob_{class}` phải được gán bằng **class mapping thật của checkpoint**. ViT/DeiT cũng phải xuất 5 class probabilities, không chỉ label/argmax.
- `shuffle=False` cho Val/Test/Holdout và kiểm tra alignment `y_true ↔ filename ↔ probability`.
- **Không impute** confidence/probabilities bị thiếu ở EXP lịch sử: đánh dấu `MISSING` hoặc chạy `EVAL_ONLY` với đúng checkpoint+input (không train lại/tune), lưu audit.

### 6.2. File metrics từng seed

```text
seed_42/validation/metrics.json
seed_42/validation/classification_report.csv
seed_42/validation/predictions.csv
seed_42/validation/confusion_matrix_raw.csv
seed_42/validation/confusion_matrix_normalized.csv
seed_42/validation/error_cases.csv
```

Tương tự với `seed_123`, `seed_2026`, `seed_3407`, `seed_7777`; Test/Holdout chỉ có khi executed. `metrics.json` lưu `experiment_id`, `dataset_hash`, `seed`, `split`, `support`, `class_names`, `accuracy`, `loss`, `macro_precision/recall/f1`, `balanced_accuracy`, `weighted_f1`, `runtime`, `checkpoint_sha256`, `evaluation_status`.

### 6.3. Aggregate 5 seed

```text
aggregate/validation_results_5seeds.csv
aggregate/validation_mean_std.csv
aggregate/per_class_validation_by_seed.csv
aggregate/per_class_validation_mean_std.csv
aggregate/seed_ci95_validation.csv          # nếu tính, phải nêu giả định
aggregate/training_time_mean_std.csv
aggregate/real_prefix_val_by_seed.csv
aggregate/test_results_5seeds.csv           # chỉ nếu Test executed
aggregate/test_mean_std.csv                 # chỉ nếu Test executed
aggregate/per_class_test_by_seed.csv        # chỉ nếu Test executed
aggregate/holdout_results_5seeds.csv        # nếu có Holdout
```

`EXPERIMENT_REGISTRY.csv` một hàng cho mỗi EXP **và có cột** `mode`, `dataset_hash`, `status`, `selection_lock`, `test_exposure_status`, `holdout_type`, `source_zip`. Không ghi `0` vào metric chưa chạy — dùng `null`/`PENDING`.

### 6.4. Chứng từ và kiểm toán artifact

- `experiment_config.json` và `dataset_audit.json` lưu full recipe, SHA256 và dataset provenance.
- `validation_selection_lock.json` ghi quyết định trước Test.
- `model_summary.txt`, `label_map.json`, `deployment_metadata.json` ghi đúng preprocess, class order và checkpoint.
- `analysis_manifest.json`: đường dẫn, checksum, kích thước, số hàng, trạng thái `PRESENT/MISSING/NOT_APPLICABLE/PENDING` của **mọi artifact bắt buộc**.
- `audit_required_artifacts.json`: đánh giá điều kiện hoàn thành và giải thích thiếu sót; không dùng `PENDING` để che file bị lỗi sau khi stage đã hoàn thành.
- `dataset/version`, `git_commit`, `framework`, `training_datetime`, `config_sha256`, `artifact_schema_version = 2` để giúp tái lập.

---

## 7. Cấu trúc thư mục xuất chuẩn

```text
EXP-<ID>/
├── README.md
├── experiment_config.json
├── dataset_audit.json
├── validation_selection_lock.json      # chỉ khi selection locked
├── label_map.json
├── model_summary.txt
├── analysis_manifest.json
├── audit_required_artifacts.json
├── seed_42/
│   ├── checkpoints/
│   │   ├── best.keras
│   │   └── last.keras
│   ├── stage1/history.csv               # nếu 2-stage
│   ├── stage2/history.csv               # nếu 2-stage
│   ├── history.csv
│   ├── validation/
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix_raw.csv
│   │   ├── confusion_matrix_normalized.csv
│   │   ├── confusion_matrix_raw.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── error_cases.csv
│   ├── test/                             # chỉ khi Test executed
│   └── historical_holdout/               # chỉ khi executed
├── seed_123/                             # cùng cấu trúc
├── seed_2026/
├── seed_3407/
├── seed_7777/
├── aggregate/
│   ├── validation_results_5seeds.csv
│   ├── validation_mean_std.csv
│   ├── per_class_validation_by_seed.csv
│   ├── per_class_validation_mean_std.csv
│   ├── training_time_mean_std.csv
│   └── ...                               # Test/Holdout khi executed
├── figures/
│   ├── panels/
│   │   ├── training_curves_5seeds_2x5.png
│   │   ├── validation_confusion_raw_1x5.png
│   │   └── validation_confusion_normalized_1x5.png
│   ├── class_distribution_by_split.png
│   ├── real_prefix_distribution_by_split.png
│   ├── per_class_f1_heatmap_validation.png
│   ├── seed_metric_errorbars_validation.png
│   └── validation_error_montage_seed_selected.png
├── benchmark/
│   └── inference_latency.csv
└── model_package/
    ├── deployment_metadata.json
    └── label_map.json
```

Tên `seed_42` chỉ là ví dụ; xuất đủ cả năm seed. File `ANALYSIS.zip` không chứa checkpoints nặng (trừ khi cố ý và đã ghi rõ); `FULL.zip` chứa đầy đủ checkpoints/model package **có thực**, chứ không chỉ lời hứa trong README.

---

## 8. Đóng gói `ANALYSIS.zip` và `FULL.zip`

| Tệp | Nội dung | Mục đích |
|---|---|---|
| `EXP-..._ANALYSIS.zip` | README, config, hashes, selection lock, CSV/JSON, panels PNG/SVG, label map, summary, manifest | Gửi nhóm, kiểm toán và statistical notebook |
| `EXP-..._FULL.zip` | Toàn bộ ANALYSIS + checkpoints, model package, notebook/script và log cần tái lập | Khôi phục model, inference/parity |

Phải kiểm tra trực tiếp ZIP sau đóng gói: số seed, số hàng predictions theo support, 5 probabilities, label order, ảnh panel không rỗng, checksum và sự tồn tại của khóa lựa chọn. **Không upload binary lên GitHub nếu vượt chính sách lưu trữ**; dùng storage ngoài Git và nêu đường dẫn, SHA256; không nói Git có checkpoint khi chỉ commit README/CSV.

---

## 9. Giao tiếp với notebook `STATISTICAL_ANALYSIS.ipynb`

Notebook thống kê trung tâm nhập từ **các ANALYSIS ZIP** đã chuẩn hóa, khớp `dataset_version`, `dataset_manifest_sha256`, `split`, `image_id`, `true_label`, `class_to_idx`, `task`, `protocol`, `seed`. Khi không khớp, phải cảnh báo và **không chạy paired test**.

**Training notebook cung cấp**:

- `master_metrics` inputs: one record per experiment/mode/seed/split.
- `master_predictions` inputs: one record per experiment/mode/seed/image với 5 xác suất.
- `master_per_class` inputs: per-class P/R/F1/support.
- `master_efficiency` inputs: params, size, latency; protocol rõ.

**Statistical notebook thực hiện** (chi tiết ở tài liệu companion):

1. Scratch baselines và Transfer baselines **mean ± SD qua seed**.
2. Same-architecture Scratch vs Transfer: paired difference/forest plot + bootstrap CI trên ảnh và McNemar khi cùng ảnh/nhãn.
3. Cross-architecture Macro-F1 heatmap, paired seed plots, McNemar + Holm (nếu thiết kế hợp lệ), Cochran's Q (trên >2 classifier cùng ảnh), effect size và CI.
4. Reliability Diagram, ECE/Brier, ROC/PR OVR, domain-gap Test/Holdout, Accuracy–Latency Pareto.
5. Không thực hiện selection/tuning dựa trên Test/Holdout đã tiếp xúc; nếu đã đánh giá lịch sử, ghi `retrospective/exploratory`.
6. Không so trực tiếp mAP detection với classification accuracy, hoặc YOLO 1-stage với two-stage nếu task/GT khác.

---

## 10. README đầu/cuối của từng EXP

### 10.1. Header Markdown mẫu ở đầu notebook

```markdown
# EXP-... — {Model} {Training Mode} — 5 seeds

- Research question / hypothesis:
- Parent experiment / ablation factor:
- Dataset version, archive SHA256, manifest SHA256:
- Train/Val/Test count, source group policy:
- Class order from runtime label map:
- Seeds: [42, 123, 2026, 3407, 7777]
- IMG_SIZE, batch, optimizer, LR, augmentation, stages:
- Checkpoint selection rule (Validation only):
- RUN_FINAL_TEST: false
- Historical holdout status / exposure:
- Hardware/framework:
- Planned figure panels / output schema version: 2
```

### 10.2. Result Summary mẫu cuối notebook

```markdown
# Experiment Result Summary — EXP-...

## Provenance and status
- Status: VAL_COMPLETE / SELECTION_LOCKED / TEST_COMPLETE / EVAL_ONLY
- Dataset snapshot SHA256:
- Selection lock filename and chosen seed:

## Five-seed Validation
| Metric | Mean | Sample SD | CI95 across seeds (if computed) |
|---|---:|---:|---|
| Accuracy | | | |
| Macro Precision | | | |
| Macro Recall | | | |
| Macro F1 | | | |
| Balanced Accuracy | | | |

## Test (only if gated and executed)
- Test status / exposure:
- Mean ± SD, per-class support, CM panels:

## Historical/Fresh Holdout (separate)
- Source / total / present classes / prior exposure:
- Results and limitations:

## Errors and efficiency
- Anthracnose→Healthy, Healthy→Disease, Leaf Miner↔Anthracnose:
- Inference mean/median/p95, device and timing scope:
- Model package parity status:

## Interpretation and limitations
- Did Validation improve? What does evidence support?
- Any one-factor confounding? Any leakage candidates?
- Next experiment must be selected without Test/Holdout tuning.
```

---

## 11. Definition of Done theo giai đoạn

### `VAL_COMPLETE` — đủ để tham gia exploratory/Validation benchmark

- [ ] Config+provenance, SHA256, runtime `label_map.json` và dataset audit được lưu.
- [ ] Đủ **5 seed thực sự train** cùng snapshot/recipe; checkpoint & history từng seed.
- [ ] Validation metrics, class report, predictions **5 probabilities** từng ảnh từng seed.
- [ ] `training_curves_5seeds_2x5.png` (Accuracy/Loss × 5 seed).
- [ ] `validation_confusion_raw_1x5.png` và `validation_confusion_normalized_1x5.png` + per-seed CSV/PNG.
- [ ] Per-class Validation F1 heatmap, Mean ± sample SD, class distribution, error analysis.
- [ ] `analysis_manifest.json`, `audit_required_artifacts.json`, README, ANALYSIS archive.
- [ ] Mọi thiếu sót đều được đánh dấu rõ; không khai sai trạng thái.

### `SELECTION_LOCKED`

- [ ] Seed/checkpoint/model/threshold được xác định **theo Validation**.
- [ ] `validation_selection_lock.json` có checksum và không bị sửa sau Test.

### `TEST_COMPLETE`

- [ ] Test đã chạy sau lock, không reselect theo Test.
- [ ] Đủ 5 seed Test metrics/predictions/CM panels theo **quy trình được định trước**, hoặc ghi rõ chỉ selected-seed nếu protocol quy định đánh giá seed được chọn; không trộn hai kiểu báo cáo.
- [ ] Nêu rõ Test là locked version nào và có từng bị xem/tune trong các thử nghiệm lịch sử không.

### `HISTORICAL_HOLDOUT_EVAL` hoặc `FRESH_BLIND_EVAL`

- [ ] Có manifest nguồn, split overlap audit, support từng lớp, lịch sử tiếp xúc và đúng loại Holdout.
- [ ] Không chọn model bằng Holdout; giải thích nếu thiếu lớp và khi so sánh domain gap.

### `FULL_ARCHIVED` / `DEPLOYMENT_READY` (nếu cần)

- [ ] FULL ZIP kiểm tra giải nén/load checkpoint/label mapping/preprocessing parity.
- [ ] Inference benchmark cùng protocol thiết bị, model package truy xuất được.
- [ ] Không đồng nhất `VAL_COMPLETE` với `DEPLOYMENT_READY`.

---

## 12. Checklist copy vào cuối mọi training notebook

```markdown
## STANDARD TRAINING NOTEBOOK V2 — COMPLETION CHECKLIST

### Dataset & reproducibility
- [ ] Experiment ID, parent, training_mode, five seeds
- [ ] Dataset ZIP+manifest SHA256, split sizes, class order from runtime
- [ ] Corrupt/duplicate/group audit, source/real_ provenance limitations
- [ ] No augmentation on Val/Test/Holdout

### Training / Validation (required)
- [ ] Five *independent* training seeds; per-seed checkpoints and history
- [ ] Per-seed Validation P/R/F1/support, Balanced Accuracy and predictions
- [ ] Full 5-class probabilities, correct/confidence/image_id/path
- [ ] Accuracy/Loss panel 2x5 and Validation CM raw/normalized panels 1x5
- [ ] Per-class F1 heatmap, Mean ± sample SD; seed CI labeled appropriately
- [ ] Error analysis including Anthracnose→Healthy and Healthy→Disease
- [ ] Selection rule based only on Validation and immutable lock JSON

### Gated Test / Holdout
- [ ] Test disabled by default; executed only after selection lock
- [ ] Test panels and per-seed outputs if Test executed
- [ ] Historical vs fresh blind holdout distinguished; class support explicit
- [ ] No tuning, seed/model switching based on Test/Holdout

### Reporting / export
- [ ] Benchmark device, warm-up, batch, latency mean/median/p95
- [ ] README with limitations and no unsupported claims
- [ ] ANALYSIS.zip + checksums; FULL.zip if checkpoints are packaged
- [ ] analysis_manifest.json and audit_required_artifacts.json passed
- [ ] Registry updated with status/provenance
```

---

## 13. Những thay đổi chính so với V1

| Mục V1 | V2 sửa thành | Lý do |
|---|---|---|
| Single `SEED = 42` | `SEEDS = [42,123,2026,3407,7777]` | Multi-seed, tính ổn định |
| Một accuracy curve và một loss curve | Panel Accuracy/Loss **2×5** | Quy cách đã thống nhất để so sánh 5 seed |
| Một Confusion Matrix | Raw + Normalized **1×5** cho từng evaluated split | Dễ đọc/so sánh, không trộn split |
| `predictions.csv` chỉ confidence | 5 per-class probabilities + ID + provenance | McNemar/bootstrap/calibration/ensemble |
| Test ngay sau Train | Validation → aggregate → **selection lock** → gated Test | Chống Test leakage |
| Không phân biệt real holdout | Historical vs fresh blind, support và exposure rõ | Khoa học về domain shift |
| Không tổng hợp seed | Mean ± sample SD và CI seed ghi rõ ý nghĩa | Thống kê đáng tin cậy |
| Một kết quả ZIP | `ANALYSIS.zip` và `FULL.zip` cùng manifest audit | Gửi nhẹ, bảo toàn artifact |
| Một notebook ôm mọi việc | Training atomic outputs + statistical analysis riêng | Chuẩn hóa so sánh liên kiến trúc |
| DoD buộc Test luôn phải chạy | DoD **theo stage**, `VAL_COMPLETE` hợp lệ | Hạn chế Test bị xem lặp lại |

---

## 14. Phụ lục — Khôi phục YOLO Failure checkpoints và ghép pipeline CNN/ViT

Phần này quy định **interface** khi sử dụng YOLO26 đã huấn luyện; **không thay thế protocol riêng của detection**. Các experiment trong [YOLO archive](failure/YOLO26/README.md) vẫn giữ trạng thái nghiên cứu, chưa được chốt detector final.

### 14.1. Có thể tái sử dụng checkpoint của Take Failure không?

**Có**, với điều kiện có thật **checkpoint YOLO đã fine-tune** (thường là weights/best.pt hoặc bản export tương đương) và khôi phục đúng kiến trúc, cấu hình và label mapping. Thư mục Failure không vô hiệu hóa trọng số; có thể dùng để **smoke test/inference** cùng classifier baseline mà không train lại YOLO hoặc CNN.

**Không** nhầm:
- weights/best.pt đã fine-tune trên bệnh lá điều với yolo26s.pt **pretrained gốc**.
- results.csv, mAP, confusion matrix, các ảnh box với file weights có thể infer lại.
- "GitHub có README/metrics" với "GitHub có checkpoint binary". Báo cáo Take 005 ghi rõ .pt và FULL ZIP **không commit vào Git**.
- Take 006 có tên archive 006 nhưng nội bộ config, metrics và tên checkpoint còn ghi 005; phải xác minh hash, nội dung và nguồn checkpoint trước khi sử dụng.
- Take 006 mAP50=0,7298 và Take 007 mAP50=0,6993 nhưng **Test/GT khác nhau** nên không dùng chênh lệch này để chọn detector cuối.

### 14.2. Bộ artifact tối thiểu cho YOLO tái sử dụng

| File | Chức năng | Ưu tiên |
|---|---|---|
| weights/best.pt hoặc mô hình export đã thử load | Dự đoán bbox trên ảnh mới | **Bắt buộc** để infer ảnh mới |
| weights/last.pt | Tiếp tục huấn luyện (không thay thế best khi infer) | Tùy chọn |
| experiment_config.json + args.yaml | Kiến trúc, imgsz, training recipe, seed, package version | Bắt buộc để xác thực |
| data.yaml / names / class_id mapping | Đối chiếu anthracnose, leaf_miner, red_rust | Bắt buộc |
| ZIP SHA256, checkpoint SHA256, dataset/split/GT manifest | Truy nguyên checkpoint và benchmark | Bắt buộc khi báo cáo |
| results.csv, per_class_metrics.csv, confusion matrices | Phân tích lịch sử huấn luyện | Không thể thay checkpoint |
| detections.csv/json có image ID, bbox, class, confidence | Crop/phân loại offline trên **chính ảnh đã được infer** nếu không còn .pt | Có điều kiện |

Nếu chỉ có số liệu tổng hợp và hình, **không thể** tái tạo prediction bbox trên ảnh mới. Nếu đã lưu tọa độ bbox dự đoán gắn với image ID, có thể thử phần crop→classifier offline trên đúng những ảnh cũ nhưng không thể chạy ảnh mới.

### 14.3. Kiểm tra ZIP trước khi định dùng checkpoint

~~~python
from pathlib import Path
from zipfile import ZipFile

archive = Path("EXP-Y26S-SMALL-006.zip")
with ZipFile(archive) as z:
    for name in z.namelist():
        if name.lower().endswith((".pt", ".onnx", "args.yaml", "data.yaml", "experiment_config.json")):
            print(name)
~~~

Nếu có .pt và đã giải nén/xác minh được nguồn, kiểm tra nạp và dự đoán:

~~~python
from ultralytics import YOLO

detector = YOLO("/path/to/verified/weights/best.pt")
print(detector.names)
results = detector.predict("example.jpg", imgsz=640, conf=0.25, verbose=False)
for result in results:
    for box in result.boxes:
        print(box.xyxy[0].tolist(), int(box.cls[0]), float(box.conf[0]))
~~~

**Lưu ý:** conf=0.25 ở đây chỉ là smoke test; threshold cuối phải chọn từ Validation và khóa trước khi đánh giá Test.

### 14.4. Contract YOLO → Crop → Classifier

~~~text
Original RGB image
  → verified YOLO26 fine-tuned checkpoint
  → bbox + lesion class + detector confidence
  → crop trên ẢNH GỐC (không crop từ ảnh đã vẽ bounding box)
  → optional padding + resize + đúng preprocess classifier
  → verified CNN/ResNet50/DenseNet121/ViT checkpoint
  → probabilities của 5 lớp + mapping kết quả cấp ảnh
~~~

- Detector hiện nhận **3 loại tổn thương**; classifier có **5 lớp** (thêm healthy và not_cashew_leaf). Thiết kế từ Validation cách xử lý **no-box**, multi-box/multi-disease, non-cashew và conflict giữa nhãn YOLO và CNN. **No-box không đồng nghĩa Healthy**.
- Vì CNN được train chủ yếu bằng ảnh nguyên lá, crop nhỏ có thể làm giảm chất lượng; phải đánh giá **full image → GT crop → predicted YOLO crop** và chỉ kết luận lợi ích khi thực sự đo end-to-end.
- Đầu ra pipeline cần có image_id, source group, detector EXP/weights hash, tọa độ bbox, detection class/confidence, classifier EXP/weights hash, 5 xác suất, crop/padding policy, latency, các trường hợp no detection và flags lỗi.
- Không so trực tiếp **detection mAP** với **classification Accuracy**. Chỉ đối chiếu khi chuẩn hóa về cùng task, cùng ảnh và ground truth.
- Không dùng Test/Holdout đã quan sát để chọn lại detector, threshold, seed hoặc crop policy rồi gọi là fresh blind test.

**Definition of Done cho reuse:** có checkpoint YOLO nạp được; xác minh ID/checksum/mapping; có checkpoint classifier và preprocess contract; chạy inference một ảnh mới; export crop + predictions; kiểm định pipeline theo protocol trên tập phù hợp; lưu runtime/README. Không bắt buộc train lại hai mô hình chỉ để thử tích hợp.

---

## 15. Nguồn quyết định và giới hạn phạm vi

1. Quy ước 5 seed, test lock, hash/manifest và cảnh báo Holdout theo **`docs/RESEARCH_PROTOCOL.md`**.
2. Schema predictions, package FULL/ANALYSIS, hình so sánh, McNemar/bootstrap/Holm/CI theo **`docs/research_notes/AUDIT_NOTEBOOK_OUTPUTS_STATISTICAL_PLAN.md`** (bản lịch sử 27/10/2026).
3. Số liệu riêng V11 / Holdout05 theo **`DATASET_CHANGELOG.md`** và notebook `DenseNet121` EXP-025; không phải dữ liệu mặc định cho mọi EXP.
4. Tiêu chí `val_macro_f1` cho experiment mới, cách đặt tên hình chuẩn V2, manifest audit và DoD theo stage là **quy định thiết kế được đề xuất để thống nhất notebook từ đây trở đi**; không khẳng định mọi experiment cũ đều đã đáp ứng.
5. Tài liệu này mô tả classification. YOLO/detection có các metric, nhãn và đối tượng prediction khác, phải dùng **protocol detection riêng**. Thống kê liên task chỉ thực hiện khi định nghĩa lại cùng đơn vị đánh giá và ground truth.

**Mục tiêu cuối:** dữ liệu có nguồn rõ → multi-seed Validation đáng tin → chọn theo Val → locked Test/Holdout đúng quy tắc → một kho artifact thống nhất → notebook thống kê trung tâm và báo cáo khóa luận tái lập được.