# IMPLEMENTATION SPEC — EXP-VIT-TRANSFER-XGBOOST-008

> **Dành cho:** Codex / Antigravity / coding agent xây dựng notebook Google Colab (ưu tiên GPU), có thể chạy lại trên Kaggle hoặc môi trường Linux được cấu hình tương đương.
>
> **Trạng thái:** IMPLEMENTATION SPEC — **chưa triển khai, chưa huấn luyện XGBoost, chưa có kết quả EXP-008**.
>
> **Mục tiêu nghiên cứu chính:** **giảm lỗi lá bệnh bị dự đoán thành `healthy`** (false-healthy / missed disease) khi áp dụng ViT cho ảnh lá điều thực tế, **không hy sinh quá mức khả năng nhận diện lá khỏe, các lớp bệnh khác và ảnh `not_cashew_leaf`**.
>
> **Quy chuẩn bắt buộc:** `STANDARD_TRAINING_NOTEBOOK_FORMAT.md`; **nguồn kỹ thuật:** `EXP-VIT-TRANSFER-5SEEDS-007_ANALYSIS.zip`, đặc biệt notebook `notebook/vit_transfer_5seed_007.ipynb`, `experiment_config.json`, `validation_selection_lock.json`, `dataset_manifest.csv` và per-seed predictions.

---

## 0. Yêu cầu tối quan trọng — agent phải đọc trước khi code

**Đây là thí nghiệm thay CLASSIFIER, không phải một lần train ViT mới.**

1. **Không retrain/fine-tune DeiT-Tiny.** Load **đúng 5 checkpoint đã chọn của EXP-007**, đóng băng hoàn toàn và trích xuất đặc trưng (embedding) từ ViT. Không tái khởi tạo pretrained backbone rồi coi đó là checkpoint đã huấn luyện trên lá điều.
2. **Ba đối chứng tối thiểu** cho **mỗi seed**: **A = head/logits gốc của EXP-007 (Softmax baseline)**, **B = Logistic Regression trên ViT embedding**, **C = XGBoost trên CÙNG embedding**. Phương án D bổ sung `healthy` decision threshold là **tùy chọn, báo cáo riêng**.
3. **Chỉ dùng Train để fit Logistic Regression/XGBoost, tính scaler, search CV và mọi phép biến đổi học được**. Validation chỉ dùng để **chọn phương án đã định nghĩa trước và khóa lựa chọn**. **Không dùng V08 Test hoặc 48 ảnh Real Holdout lịch sử để tune/chọn model/threshold/seed/feature extractor.**
4. Việc XGBoost fit trên Train embedding được tạo ra bởi ViT đã được fit trên cùng Train là **cách học classifier thông thường**, không tự nó là leakage giữa Train/Val. Nhưng CV trên embedding này **không phải cross-validation độc lập cho toàn pipeline học ViT**; không được phát biểu ngược lại. Nếu chuyển sang stacking logits/probabilities làm meta-features thì phải có **out-of-fold base-model training đúng nghĩa**, ngoài phạm vi mặc định.
5. **Giữ nguyên dataset V08, labels, thứ tự lớp, 5 seeds và manifest/checkpoint hashes theo EXP-007**. Không thay split, ảnh hoặc nhãn để cải thiện số liệu. Mọi bất đồng metadata phải dừng lại và báo cáo.
6. **Không chế tạo metrics.** Notebook phải có chế độ smoke test và chạy đầy đủ; phần chưa chạy ghi `NOT_RUN`. Xử lý lỗi thiếu checkpoint bằng **fail-fast**, không âm thầm lấy seed khác hoặc thay model.
7. **Trước khi thực hiện bất kỳ lần đánh giá Test/Holdout mới nào phải tạo selection lock bất biến**; mặc định tắt historical evaluation do tập 48 ảnh đã bị lộ trong các thí nghiệm trước. Cần **fresh blind field set** từ ảnh/vườn/phiên chụp độc lập để có bằng chứng thật về cải thiện thực địa.
8. **Không sửa / ghi đè EXP-007**, các experiment ResNet50/DenseNet121 hoặc model web production. EXP-008 là thí nghiệm độc lập với namespace riêng.

### Đầu ra cần bàn giao

- Notebook có thể chạy: `Notebook/ViT/vit_transfer_xgboost_5seed_008.ipynb` (nếu repo đang dùng thư mục `Notebook/VIT/`, agent kiểm tra và sử dụng **đúng đường dẫn thực tế**, không tạo thư mục trùng do khác chữ hoa/thường).
- Đặc tả trong repo: `Notebook/ViT/EXP-VIT-TRANSFER-XGBOOST-008_IMPLEMENTATION_SPEC.md` (hoặc nhánh `Notebook/VIT/` thực tế).
- Kết quả riêng trong `training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-008/` trên Google Drive; bao gồm báo cáo, `experiment_config.json`, provenance lock, metric CSV/JSON, prediction CSV, confusion matrices, so sánh 5 seeds, model packages, figure PNG và ZIP export.
- Một `README.md` và `MODEL_REPORT.md` mô tả **so sánh có kiểm soát** A/B/C; nếu chưa có field set độc lập thì tuyên bố rõ **chưa chứng minh cải thiện ngoài thực địa**.

---

## 1. Nguồn gốc nghiên cứu (đã kiểm tra từ EXP-007 gốc)

### 1.1. Model và kỹ thuật đang kế thừa

| Thuộc tính | EXP-007 gốc — phải giữ/đối chiếu |
|---|---|
| ID gốc | `EXP-VIT-TRANSFER-5SEEDS-007` |
| Architecture | KerasHub `DeiTImageClassifier` pretrained **DeiT-Tiny Distilled** |
| Preset | `kaggle://keras/deit/keras/deit_tiny_distilled_patch16_224_imagenet/1` |
| Framework trong notebook | `keras-hub==0.32.0` (phiên bản gốc; lock thêm TensorFlow/Keras runtime thực tế) |
| Input | RGB `float32` pixel **0–255**, shape `(224,224,3)` |
| Preprocessing | `classifier.preprocessor(...)` **đã được nhúng vào model** gốc, không `/255` bên ngoài, không gọi thêm preset preprocessor một lần nữa |
| Output gốc | **5 logits** (`activation=None`), không phải probability; baseline lấy `softmax(logits)` |
| Train gốc | Stage 1 head/frozen backbone 8 epoch, Stage 2 full fine-tuning tối đa 12 epoch |
| Seeds | `[42, 123, 2026, 3407, 7777]` |
| Dataset V08 | Train **4.790**, Validation **1.430**, Test **686** |
| Thứ tự nhãn **cố định** | `['anthracnose','healthy','leaf_miner','red_rust','not_cashew_leaf']` |
| Checkpoint từng seed | `.../EXP-VIT-TRANSFER-5SEEDS-007/seed_{SEED}/best_model.keras` |
| Metric chọn checkpoint trong seed | **Minimum Validation Loss** |
| Quy tắc chọn deployment seed ở EXP-007 | **Highest Validation Macro F1**, tie-break **Validation Loss** → **seed 123** |
| Baseline EXP-007 Validation Accuracy | **94,41 ± 0,73%** / 5 seeds |
| Baseline EXP-007 Validation Macro F1 | **94,29 ± 0,70%** / 5 seeds |
| Baseline EXP-007 Final Test Accuracy | **95,36 ± 0,49%** / 5 seeds — **đã mở ở EXP-007**, không còn mù với quá trình phát triển sau đó |
| Baseline EXP-007 Historical Holdout Accuracy | **74,58 ± 6,14%** / 5 seeds; 48 ảnh, **chỉ có 4 lớp ground-truth** |

**Quy tắc khoa học:** Không được tuyên bố EXP-008 đạt những số liệu nêu trên. Đây chỉ là **baseline EXP-007** để agent kiểm tra parity và xây đối chứng.

### 1.2. Các SHA/provenance KHÔNG được bỏ qua

- Dataset ZIP V08 SHA256: `75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7`.
- `dataset_manifest_sha256` **do notebook EXP-007 ghi trong lock/config:** `C699A9ACA707DA25F143DB1FC3C9B5133EA2858BCDC468A158E5644E500EC715`.
- **Lưu ý hash quan trọng:** ở một số mô hình CNN trên cùng V08, dự án dùng manifest digest `1810D7E6...` theo **phương pháp canonical khác**. **Không mặc định hai digest phải bằng nhau hoặc kết luận dataset khác.** Agent phải đọc hàm tạo fingerprint của EXP-007 và tái tạo **đúng thuật toán source**. Kiểm tra dataset-level ZIP SHA256 + từng path/label/SHA256 trước khi so sánh theo mô hình khác. SHA256 của **bytes của CSV xuất từ EXP-007** cũng là một khái niệm riêng.
- Lấy chính `validation_selection_lock.json` trong EXP-007 làm **nguồn chân lý checkpoint SHA**, **không chỉ dựa vào đường dẫn**:

| Seed | SHA256 checkpoint gốc (từ `validation_selection_lock.json`) |
|---:|---|
| 42 | `B8DFD61FF3D5D84698125E82B429359A9916C1B58DDEE1CB6C5B593C7F2B3E0A` |
| 123 | `FD1DD6B775F29F400ED4D1E889B217A5C97B513E9C0066C07136B57B57E2BC89` |
| 2026 | `551126D758F761F484E1FB5BAC1C4BE56E87321EBB93930BC6C54C8B6340193C` |
| 3407 | `526C4AC9FA5DD696AF2B6576E647EB8FD2919BBA0966F7638909346F34B3861E` |
| 7777 | `FE5BAD5406783A5BD27251376EA10855FE5A96BC6467957BEEDF5339625E411F` |

**Agent không được giả định checkpoint binary có trong ANALYSIS ZIP.** Theo source, notebook đã xuất đường dẫn Drive và checkpoint SHA; nếu chỉ có `_ANALYSIS.zip` mà **không có `best_model.keras`/FULL ZIP**, dừng trước feature extraction, báo thiếu artifact chính xác, không tải pretrained ImageNet để thay thế.

### 1.3. Baseline lỗi bệnh → Healthy phải tái lập

Định nghĩa **DISEASE** là 3 nhãn: `anthracnose`, `leaf_miner`, `red_rust`; không gộp `not_cashew_leaf` vào disease. **Healthy** đúng bằng label index `1`.

Trên **Validation 1.430 ảnh**, số ảnh bệnh là **308 + 264 + 328 = 900**. Theo predictions nguồn của EXP-007:

| Seed | Disease → Healthy sai | Mẫu disease | Healthy Recall | `real_` Val đúng/tổng |
|---:|---:|---:|---:|---:|
| 42 | 21 | 900 | 213/216 | 29/35 |
| **123 (deploy EXP-007)** | **18** | **900** | **210/216** | **25/35** |
| 2026 | 26 | 900 | 215/216 | 24/35 |
| 3407 | 24 | 900 | 211/216 | 29/35 |
| 7777 | 15 | 900 | 212/216 | 28/35 |

**Gating/parity:** với seed 123, **baseline argmax phải tái tạo 18 false-healthy / 900 disease trên Validation**. Với các seed khác cũng phải khớp per-sample source predictions, không chỉ khớp aggregate accuracy. Không phát biểu rằng **18/900 là lỗi trên ảnh thực địa**: đó là chỉ số Validation tổng; subset `real_` 35 ảnh quá nhỏ.

### 1.4. Dataset và ảnh thực địa

- Validation V08 gồm **35 ảnh có tiền tố `real_`** (Anthracnose 9, Healthy 3, Leaf miner 18, Red rust 5); kiểm tra lại bằng manifest. Chỉ dùng làm **subgroup chẩn đoán thứ cấp**; tiền tố tên file không tự chứng minh độc lập vườn/cây/phiên chụp.
- Historical Holdout gốc có 49 ảnh nhưng **1 ảnh trùng byte với V08 Test** đã loại, còn **48 ảnh**: Anthracnose 13, Healthy 5, Leaf miner 24, Red rust 6; **0 ground truth `not_cashew_leaf`**. Dataset này đã được xem trong EXP-007/ResNet50/DenseNet trước đây, **không dùng để phát triển/tune EXP-008**.
- V08 đã được kiểm tra trùng byte; vẫn cần **audit ảnh gần trùng/cùng lá/cùng buổi**. Nếu phát hiện vấn đề, lưu báo cáo/audit và đề xuất dataset version mới; **không tự ý đổi split V08 trong notebook này**.

---

## 2. Câu hỏi nghiên cứu và giả thuyết cần kiểm chứng

**RQ1:** Với **đúng một embedding DeiT-Tiny cố định**, XGBoost có giảm số lượng `anthracnose`/`leaf_miner`/`red_rust` bị phân thành `healthy` so với Softmax head gốc và Logistic Regression hay không?

**RQ2:** Cải thiện (nếu có) có giữ được **Healthy Recall**, **Macro F1** và hiệu năng `not_cashew_leaf` trên Validation hay không?

**RQ3:** Kết quả đó có **ổn định trên 5 ViT seeds** và có được xác nhận bởi **bộ field test chưa từng dùng** hay không?

**Giả thuyết nghiên cứu, không phải kết luận:** XGBoost có thể khai thác tương tác phi tuyến trong embedding tốt hơn classifier tuyến tính, nhưng không khắc phục được domain shift nếu embedding chứa thông tin nền/ánh sáng gây nhiễu.

### Tiêu chí thành công định trước (để tránh thay mục tiêu sau khi xem kết quả)

- **Primary — Validation disease-to-healthy rate (D2H):** `FN_healthy_from_disease / n_disease`, tính trên **900 ảnh disease** của Val V08, và **raw count** đi kèm. **Càng thấp càng tốt**.
- **Safety guardrail 1:** Validation `macro_f1 >= macro_f1(Softmax baseline cùng seed) - 0.01` (không giảm quá **1 điểm phần trăm**).
- **Safety guardrail 2:** Validation `healthy_recall >= healthy_recall(Softmax baseline cùng seed) - 0.02` (không giảm quá **2 điểm phần trăm**; hạn chế "cứ dự đoán tất cả là bệnh").
- **Safety guardrail 3:** Validation `recall(not_cashew_leaf) >= recall_baseline - 0.02`; kiểm tra disease per-class Recall, không được che khuất bởi Accuracy tổng.
- **Guardrail bổ sung:** Không làm giảm **Recall của từng lớp bệnh quá 2 điểm %** so với baseline cùng seed, trừ khi báo cáo rõ việc phá guardrail và **KHÔNG** coi là ứng viên triển khai.
- **Secondary:** Val Macro F1, disease macro recall, per-class Recall, Balanced Accuracy, Log Loss, Brier/multiclass calibration, `real_` Validation 35 ảnh; **không chọn model/threshold theo 35 ảnh này** vì cỡ mẫu ít và đã quan sát nhiều lần.
- **Final deployment claim:** Chỉ báo "giảm missed disease trên ảnh thực địa" khi **đúng checkpoint + classifier + threshold khóa trước** đạt cải thiện trên **fresh blind field test đủ 5 lớp**, kiểm toán source independence.

> **Lưu ý thống kê:** Gần như mọi đổi một quyết định trên 900 ảnh Val gây chênh `1/900 = 0,111 điểm %`. Trên 48 ảnh Holdout, **1 ảnh = 2,083 điểm %**. Không diễn giải thay đổi 1–2 ảnh như khác biệt chắc chắn trên tổng thể tương lai.

---

## 3. Thiết kế thí nghiệm bắt buộc

### 3.1. Bảng các nhánh so sánh

| Nhánh | Backbone/embedding | Classifier | Được fit trên | Mục đích |
|---|---|---|---|---|
| **A — Baseline** | Checkpoint DeiT-Tiny EXP-007 từng seed | **Head gốc + Softmax(logits)** | **Không fit thêm** | Đối chứng đã khóa; bắt buộc tái lập predictions gốc |
| **B — Linear probe** | **Cùng checkpoint/cùng embedding** | **StandardScaler + multinomial Logistic Regression** | Train embedding của seed đó | Phân biệt lợi ích của classifier mới với độ phi tuyến của XGBoost |
| **C — XGBoost** | **Cùng checkpoint/cùng embedding** | **XGBClassifier `multi:softprob`** | Train embedding của seed đó | Kiểm tra classifier cây boosting |
| **D — tùy chọn** | Giữ nguyên A/B/C | **Healthy abstention/threshold rule** | Chỉ chọn policy bằng Validation sau khi fit classifier | Đánh giá trade-off bỏ sót bệnh vs cảnh báo nhầm; báo cáo tách riêng |

**Điều kiện công bằng:** A/B/C cùng ảnh, cùng thứ tự nhãn, cùng feature extractor và cùng dataset. Không được cấp thêm Grad-CAM/supervised crop cho XGBoost trong khi baseline chỉ nhận ảnh gốc; nếu có phải ghi là thí nghiệm khác.

### 3.2. Ma trận huấn luyện 5 seeds

Chạy đúng seed ViT: **42, 123, 2026, 3407, 7777**. Với mỗi seed:

1. Xác minh checksum checkpoint `.keras` và load `compile=False, safe_mode=True` đúng phiên bản Keras/KerasHub.
2. Tạo `FeatureExtractor` chỉ từ checkpoint này; **no backbone training**, BatchNorm/Dropout/augmentation luôn inference mode.
3. Trích **Train 4.790 embedding** và **Val 1.430 embedding**.
4. Lấy logits cùng checkpoint cho **A** và kiểm tra parity với source predictions.
5. Huấn luyện **B/C trên Train embedding**; cross-validation trên Train embedding để chọn hyperparameters (không dùng Val cho tuning CV).
6. Đánh giá A/B/C trên **Val** và tạo per-seed report.
7. Chỉ **sau khi đủ cả 5 seeds Validation + mọi kiểm thử** mới áp dụng **model/decision selection rule đã cố định** và tạo selection lock.

### 3.3. Quy tắc chọn deploy model — không nhìn Holdout

- **Giữ cố định deployment ViT seed 123 của EXP-007 cho nhánh deployment chính**, bảo đảm phép thử classifier không lén đổi feature extractor theo Test/Holdout. Cả 5 seeds còn lại vẫn báo cáo mean ± sample SD để đánh giá biến thiên.
- Với **seed 123**, chọn giữa **A/B/C** và (nếu có) các policy D đã định nghĩa trước bằng Validation như sau:
  1. Loại mọi candidate **vi phạm guardrail** ở mục 2.
  2. Trong số candidate còn lại, chọn **D2H count thấp nhất**.
  3. Tie-break: **Val Macro F1 cao hơn**, sau đó **Val Log Loss thấp hơn** (nếu cùng hiệu chuẩn/evaluation protocol), sau đó chọn mô hình **đơn giản hơn** theo thứ tự `A`, `B`, `C`.
  4. Nếu mọi phương án B/C đều phá guardrail hoặc không giảm D2H, chọn **A baseline**; ghi nhận **thí nghiệm không chứng minh cải thiện**.
- **Không reselect seed 3407/7777** chỉ vì Holdout lịch sử từng có điểm cao hơn. Muốn nghiên cứu chọn seed bằng metric mới phải tạo **thí nghiệm khác có protocol preregistered**, không gọi đó là ablation classifier-only ở đây.
- Ghi **cả kết quả** (a) raw classifiers A/B/C (argmax), (b) optional post-hoc threshold policy với classifier đó. **Không trộn chúng trong cùng hàng**.

---

## 4. Đặc tả trích xuất ViT embedding — điểm dễ làm sai nhất

### 4.1. Nguồn tính toán và vị trí embedding

EXP-007 dùng (từ notebook gốc):

```python
classifier = keras_hub.models.DeiTImageClassifier.from_preset(
    PRESET,
    num_classes=5,
    activation=None,
    head_dtype="float32",
)
inputs = keras.Input(shape=(224, 224, 3), dtype="float32")
processed = classifier.preprocessor(augmentation(inputs))
outputs = classifier(processed)  # logits (batch, 5)
model = keras.Model(inputs, outputs)
```

**Trong EXP-008 phải load checkpoint `best_model.keras` đã train**, không gọi đoạn `from_preset(...)` phía trên để tạo feature extractor mới rồi quên load weights.

**Yêu cầu feature API:**

1. Load model gốc đã save, tìm chính xác nested `DeiTImageClassifier` / backbone / classification head và đọc **layer graph/config**.
2. Trích **biểu diễn ngay trước classification head** (pre-classifier embedding) tương ứng **cùng đường forward** với logits. Không lấy 5 logits hoặc 5 Softmax scores làm XGB input mặc định.
3. DeiT-Tiny thường có hidden dimension **192**, nhưng DeiT-**Distilled** có thể có **CLS token, distillation token, mean/token fusion hoặc nhánh head riêng tùy phiên bản KerasHub**. **Không cứng mã số chiều 192, token index 0/1 hoặc một thao tác trung bình** trước khi introspection. Agent phải ghi `observed_feature_shape`, `representation_source_layer`, `token_policy`, `backbone_config`, số chiều thực tế. Nếu head nhận embedding tổng hợp khác (ví dụ 384), sử dụng biểu diễn **thực tế** và giải thích.
4. Trong mọi trường hợp, lưu extractor dạng Keras model/package được reload độc lập, **nhưng không nhân đôi preprocessing**. Inference input `RGB 0..255`, resize bilinear giống gốc. **Không gọi cả wrapper preprocess và standalone preprocessor liên tiếp**.
5. Nếu không lấy được pre-head tensor ổn định từ saved model bằng public API, **FAIL-FAST** và báo cần sửa interface/export feature endpoint của notebook 007. **Không chọn một layer bất kỳ chỉ để notebook chạy được.**

### 4.2. Các smoke tests bắt buộc

- `checkpoint_sha256 == validation_selection_lock[seed]` cho từng seed.
- `source_predictions_parity`: cùng thứ tự `relative_path`, labels, dự đoán từng ảnh giống `seed_{seed}/validation_predictions.csv` gốc (không chỉ Accuracy); top-1 exact match. Nếu không khớp, dừng lại để debug preprocessing/order/version.
- `saved_model_reload`: logits trước/sau reload gần như bằng nhau (`atol=1e-5`, `rtol=1e-4` hoặc ghi tolerance phù hợp và giải trình).
- `feature_extractor_determinism`: cùng một ảnh qua 2 lần inference có features giống nhau (`np.allclose`, không dropout).
- `feature_extractor_batch_invariance`: so embedding của ảnh khi infer batch1 vs batchN trong tolerance (chú ý sai khác số học GPU ở mức rất nhỏ).
- `feature_no_nan_inf` và `feature_dim > 5`, shape `(n_images, d)` nhất quán với n Train/Val.
- `non_trainable_model`: 0 weight trainable **trong pha trích feature**; Verify không có optimizer step cập nhật DeiT.
- `single_preprocessing`: test against original EXP-007 inference, tránh double normalization.
- Các input ảnh/caption mapping xáo trộn đều phải phát hiện: định danh mỗi hàng `relative_path`, `image_sha256`, `label`, `seed`, `feature_checkpoint_sha256`.

### 4.3. Cache embedding có provenance

- Lưu riêng `seed_{seed}/embeddings/train_features.npy`, `val_features.npy` (float32), `train_meta.csv`, `val_meta.csv`, `feature_config.json`, `feature_stats.json`.
- Metadata CSV thứ tự chính xác: `row_id, split, relative_path, image_sha256, true_label, true_class, seed, checkpoint_sha256`.
- Giữ **cùng thứ tự đường dẫn** cho A/B/C, không shuffle Validation trong bất kỳ phép đo nào.
- Fingerprint cache bằng: `dataset_zip_sha256 + dataset_manifest_source_hash + seed + checkpoint_sha256 + extractor_method_version + token_policy + preprocessing_version + ordered_paths_digest`. Cache stale phải invalidate.
- Kiểm tra `n_train=4790`, `n_val=1430`, không có NaN/Inf, train/val không trùng SHA byte và không trùng `relative_path`.
- Không đưa `val_features` vào `fit`, `fit_transform`, chọn scaler, feature PCA hoặc training early stopping cho classifier. Nếu muốn dimensionality reduction (PCA) phải fit trên **Train fold** và chỉ transform Val; **không cần PCA trong bản đầu**.

---

## 5. Classifier A — Original Softmax head baseline

- Với mỗi seed, load source checkpoint, suy luận Val để nhận **logits (N,5)**, chuyển xác suất bằng **`softmax(logits, axis=-1)` đúng một lần**. Không làm Softmax lần nữa trên xác suất.
- Predicted label = `argmax(probability)`. So từng dòng với original EXP-007 source per-seed `validation_predictions.csv`.
- Per-seed metrics A phải gần/khớp chính xác các file source `aggregate/validation_results_5seeds.csv`, ngoài sai khác float nhỏ. **Quan trọng nhất: D2H raw count theo bảng 1.3**.
- Baseline Test/Holdout numbers của EXP-007 chỉ **được dẫn nguồn như reference**, không tự dùng làm EXP-008 test.
- Lưu `source_baseline_parity.json`, `seed_{seed}/softmax/validation_predictions.csv`, `classification_report.csv`, `confusion_matrix.csv`, `metrics.json`.

---

## 6. Classifier B — Logistic Regression (linear-probe baseline)

**Phương pháp:** Xây `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(...))])`.

- `X_train = embedding_train` của seed tương ứng; `y_train` từ metadata V08, 5 lớp, **không đưa class string làm feature**.
- `max_iter >= 3000` hoặc dùng solver phù hợp với scikit-learn version; **ghi chính xác version và optimizer status**.
- Candidate `C` định trước: `[0.01, 0.1, 1.0, 10.0]`, mặc định `class_weight=None`. Nếu thêm class weight phải đăng ký trước và coi là một nhánh bổ sung, không tự thử nhiều lần dựa trên Val.
- CV: `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)` chỉ trên **Train embedding**, scaler fit **trong từng CV training fold** (không fit trên train+Val; không fit toàn bộ Train trước khi chạy CV), select bằng `macro_f1` mean; tie-break CV D2H then `C` nhỏ hơn.
- Refit pipeline tốt nhất trên **4.790 train embeddings** của seed; đo Val bằng pipeline.predict/predict_proba.
- Lưu train-CV folds, từng trial, convergence diagnostics, chosen C, model `joblib` (chỉ load artifact do chính project tạo; tránh untrusted pickle), `requirements.lock`.
- Không dùng `validation_metrics` để chọn vô hạn lần tham số Logistic Regression; chỉ áp dụng **quy tắc chọn classifier A/B/C** đã định trước ở mục 3.3.

---

## 7. Classifier C — ViT embedding + XGBoost (nhánh nghiên cứu chính)

### 7.1. Tham số và API đề xuất

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    tree_method="hist",
    eval_metric="mlogloss",
    n_estimators=300,          # candidate, không phải kết quả đã chọn
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    reg_lambda=3.0,
    random_state=seed,
    n_jobs=4,
)
```

- Dùng **embedding `float32`** trên CPU; không cần GPU XGBoost vì ~4.790 × số chiều embedding nhỏ. Không thử `gpu_hist` deprecated.
- Dùng `predict_proba`, kiểm tra output **(N,5)**, cột tương ứng index labels `[0,1,2,3,4]`. **Không coi raw tree margin là probability** nếu chưa transform.
- Tránh dừng sớm (`early_stopping_rounds`) trực tiếp trên **Validation** rồi tiếp tục báo cáo chính Validation đó như đánh giá hoàn toàn độc lập của cấu hình. Trong bản đầu, **số trees và tham số lựa chọn bằng 3-fold CV trên Train**; refit toàn bộ Train, Validation chỉ dùng lần chọn A/B/C.
- Set seed cho `numpy`, `random`, framework và XGBoost; giới hạn threads để tái lập gần đúng. Lưu môi trường và XGB package version.

### 7.2. Tuning budget hữu hạn, không oversearch

- Dùng **danh sách tối đa 12–16 cấu hình cố định từ trước**, không grid hàng trăm cấu hình và không tối ưu theo 48 ảnh Holdout. Đề xuất `ParameterSampler` với random_state cố định hoặc hardcode 12 configs:
  - `n_estimators` ∈ `{150, 300, 500}`
  - `max_depth` ∈ `{2, 3, 4}`
  - `learning_rate` ∈ `{0.03, 0.05, 0.08}`
  - `subsample` ∈ `{0.8, 1.0}`
  - `colsample_bytree` ∈ `{0.8, 1.0}`
  - `min_child_weight` ∈ `{1, 3, 5}`
  - `reg_lambda` ∈ `{1.0, 3.0, 10.0}`
- Trong mỗi seed ViT, chạy **3-fold Stratified CV chỉ Train embeddings**. Cùng một candidate list giữa tất cả seeds (không bổ sung theo kết quả riêng từng seed).
- Chọn XGB hyperparameter bằng **CV Macro F1 trung bình**; tie-break **CV disease→Healthy count thấp hơn** rồi mô hình nông/ít trees hơn. **Không dùng Holdout**. Nếu thay tiêu chí selection thành cost-sensitive objective, phải làm rõ và đăng ký như một EXP khác.
- Có thể thử `sample_weight` cho disease **chỉ ở một ablation đăng ký trước**, vì việc tăng weight disease khiến mọi lỗi disease bị phạt mạnh hơn, **không tối ưu riêng hướng disease→Healthy**. Ghi rõ giả thuyết/đối chứng riêng và không kết luận từ softmax confidence.
- Tập Val chỉ chấm điểm A/B/C theo metric đã đăng ký; không tiếp tục lặp tuning vì thấy số liệu không đẹp.

### 7.3. Khi nào có thể nói XGBoost hữu ích?

Chỉ khi **C raw** (chưa thêm threshold trick) giảm D2H so với A/B **và không phá guardrail** trên Validation, cùng **xu hướng tương đối nhất quán trên 5 seeds**, sau đó được xác nhận trên **fresh field test**. Nếu chỉ thay threshold làm giảm D2H nhưng XGBoost raw không tốt hơn, cần kết luận **decision policy** cải thiện, không phải XGBoost feature classifier cải thiện.

---

## 8. Tùy chọn D — Healthy decision threshold (tách riêng khỏi hiệu quả XGBoost)

**Ý nghĩa:** tăng mức bằng chứng cần thiết trước khi mô hình được phép kết luận ảnh **Healthy**, nhằm hạn chế bỏ sót lá bệnh. Trade-off: có thể tăng false alarm trên lá khỏe.

Một policy đơn giản (chỉ áp dụng sau khi đã có xác suất 5 lớp, **không sửa model weights**):

```python
import numpy as np
HEALTHY_IDX = 1

def predict_with_healthy_gate(proba: np.ndarray, tau: float) -> np.ndarray:
    raw = np.argmax(proba, axis=1)
    gated = raw.copy()
    healthy_raw = raw == HEALTHY_IDX
    insufficient = healthy_raw & (proba[:, HEALTHY_IDX] < tau)
    non_healthy_scores = proba.copy()
    non_healthy_scores[:, HEALTHY_IDX] = -np.inf
    gated[insufficient] = np.argmax(non_healthy_scores[insufficient], axis=1)
    return gated
```

- `tau` candidate list **đăng ký trước**: `[0.50, 0.60, 0.70, 0.80, 0.90]`; `raw argmax` là đối chứng riêng. Trên xác suất chưa calibration, ngưỡng này là một *decision rule heuristic*, không phải risk đảm bảo.
- Nếu thực hiện threshold selection bằng Validation, phải dùng **cùng protocol/tuning budget cho A/B/C** để công bằng và thực hiện **chỉ sau khi cross-validation fit classifier hoàn tất**.
- Ưu tiên giữ **Healthy Recall**, **Not Cashew Recall** và Macro F1 guardrails. Nếu không đạt, fallback raw classifier.
- Do threshold được chọn bằng Validation, **điểm Validation sau threshold là selection-biased**; không mô tả là unbiased performance. Phải xác nhận trên bộ blind mới.
- Có thể chọn **tắt hẳn D** ở EXP-008 v1 để giữ phép so sánh classifier thuần. Nếu bật, báo cáo thành bảng hai tầng `raw` vs `gated` và khóa policy.

---

## 9. Leakage audit và quy trình bảo vệ Test/Holdout

### 9.1. Những phép kiểm phải được chạy TRƯỚC Training classifier

- Recreate ordered V08 manifest từ source EXP-007, xác minh 6.906 ảnh, 5 classes, 3 splits và checksum dataset ZIP.
- **Fail** nếu xuất hiện overlap cùng SHA-256 byte khác split, khác nhãn, hoặc cùng path; lưu `dataset_audit/exact_duplicates.csv`.
- Audit perceptual similarity: dùng pHash/SSIM **làm shortlist**, xuất `dataset_audit/near_duplicate_candidates.csv` và contact sheet để nhóm kiểm tra. **Không coi pHash match là leakage được chứng minh**, cũng không dùng "0 byte duplicate" để kết luận leakage-free ở cấp ảnh cùng lá/cây/phiên chụp.
- Nếu không có leaf/session/source IDs, ghi `GROUP_LEAKAGE_NOT_VERIFIED`. Notebook có thể chạy dạng **exploratory**, nhưng final scientific claims phải gắn hạn chế và ưu tiên chuẩn bị grouped split mới.
- Check Train/Val/Test image preprocessing giống EXP-007; augmentation tắt trong embedding extraction của **mọi split**.
- Thiết kế feature cache tách Train/Val và phần audit, không bao giờ đọc features của Test/Holdout trong bước fit/tune.
- Kiểm tra mọi CV fold train/val đều chỉ nằm trong **Train split V08**; scaler fitted per fold. Nếu có group IDs ở Train, dùng `StratifiedGroupKFold` phù hợp thay cho stratified CV thường.

### 9.2. Validation selection lock và Test gates

- Tạo `validation_selection_lock.json` ngay sau khi quyết định (mục 3.3), bao gồm:
  - EXP ID, date UTC, notebook source SHA256, config snapshot SHA256, dataset ZIP SHA256, manifest source/protocol fingerprint, sorted path digest.
  - SHA256 **của cả 5 DeiT checkpoints**, embedding extractor descriptor/version.
  - Danh sách hyperparameter candidate CV cố định, per-seed B/C chosen hyperparameters và model file hashes.
  - Per-seed Validation A/B/C results, selected deployment `seed=123`, selected classifier `A|B|C`, optional `threshold`, tie-breaks/guardrails, reason, raw/gated flag.
  - `historical_holdout_not_for_selection=true`, `v08_test_previously_exposed=true`.
- Không cho chỉnh lock sau khi Test/Holdout trừ khi tạo **experiment ID mới + changelog**.
- Flags mặc định: `RUN_V08_TEST=False`, `RUN_HISTORICAL_HOLDOUT=False`, `RUN_FRESH_BLIND_TEST=False`. Nếu user bật `RUN_V08_TEST` hoặc historical: **require lock/checksum**, đánh giá **đúng model đã chọn** và các model comparator như diagnostics, gắn nhãn **RETROSPECTIVE / PREVIOUSLY EXPOSED**; không dùng làm selection/tune.
- `RUN_FRESH_BLIND_TEST=True` chỉ khi có **manifest + data-source provenance mới**, kiểm toán không overlap bytes/near duplicates và có đủ **5 classes**, giữ nguyên selected classifier/seed/threshold. Nếu có retake cùng lá/buổi chụp khác, cần đánh dấu và tách; không gọi là blind.
- Lưu `test_run_intent.json` và `evaluation_lock.json`, không auto-download holdout nếu flags false.

### 9.3. Trường hợp dữ liệu bổ sung trong tương lai

- **Không đưa 48 ảnh Holdout lịch sử** vào Train/Val của EXP-008 chỉ vì kết quả thấp. Không sử dụng class distribution/error image list Holdout để tối ưu classifier/threshold.
- Nếu thêm ảnh vườn mới vào Train thì tạo **dataset snapshot mới, split theo nhóm lá/cây/session trước mọi augmentation**, **new EXP ID** và field blind test mới khác nguồn.
- Không chọn/đổi seed thành 3407 chỉ vì trên historical Holdout từng cao hơn seed 123.

---

## 10. Metrics bắt buộc, công thức và kiểm tra

Với `y_true` theo thứ tự lớp gốc, `y_pred` cùng thứ tự ảnh:

```python
DISEASE_IDS = {0, 2, 3}  # anthracnose, leaf_miner, red_rust
HEALTHY_ID = 1
OTHER_ID = 4

disease_mask = np.isin(y_true, list(DISEASE_IDS))
false_healthy_count = int(np.sum(disease_mask & (y_pred == HEALTHY_ID)))
false_healthy_rate = false_healthy_count / int(np.sum(disease_mask))
healthy_recall = np.mean(y_pred[y_true == HEALTHY_ID] == HEALTHY_ID)
```

**Bắt buộc ghi count và mẫu số**, không chỉ %:

- `disease_to_healthy_count`, `disease_total`, `disease_to_healthy_rate` và breakout từng bệnh → healthy (`anthracnose_to_healthy`, `leaf_miner_to_healthy`, `red_rust_to_healthy`).
- `healthy_true_count`, `healthy_correct`, `healthy_recall`, `healthy_predicted_count`, `healthy_precision`.
- `not_cashew_leaf_recall` và `not_cashew_leaf_precision`; `disease_macro_recall` trên đúng **3 lớp bệnh**, không gộp Healthy hoặc Not Cashew.
- `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `balanced_accuracy`, per-class precision/recall/F1/support, NLL/log_loss (`labels=[0,1,2,3,4]`), confusion matrix **5×5**.
- Reliability curve/Expected Calibration Error hoặc multiclass Brier **diagnostic**. Nếu confidence chưa calibrate, ghi rõ; không nói `p=0.9` là 90% sự thật.
- `real_` subgroup: accuracy và D2H raw counts mỗi class với **n nhỏ**, không dùng để chọn model/threshold.
- Hiệu năng: `feature_extraction_ms/image`, `classifier_predict_ms/image`, **`end_to_end_ms/image`**, RAM, train tuning time (CPU), XGBoost/LogisticRegression model size. Không so tốc độ XGBoost-only với ViT end-to-end.
- `paired_delta` từng seed so với baseline: chênh D2H count, macro F1, Healthy Recall, Not Cashew Recall, per-image wrong→right và right→wrong; báo cáo **mean ± sample SD (ddof=1)** và từng seed đầy đủ, không xem 5 seeds là 5 tập ảnh độc lập.

**Metric sanity checks:** 5 classes đúng thứ tự; confusion sum đúng số ảnh; diagonal sum / N = Accuracy; disease total Validation = **900**; true Healthy Validation = **216**; true Not Cashew Validation = **314**; all row path unique; probabilities sum1 (tolerance), no NaN/Inf; output order mapping đúng.

### 10.1. Significance/uncertainty (optional nhưng khuyến nghị)

- Thêm **McNemar exact test / paired bootstrap CI** cho sai-đúng của A/B/C trên **cùng** ảnh Validation, ghi rõ multiple comparisons và selection-on-Val caveat. Không dùng bootstrap Validation sau khi chọn để giả vờ có confirmation độc lập.
- Đối với fresh field set, ưu tiên **bootstrap theo nhóm cây/phiên chụp**, không bootstrap từng ảnh nếu nhóm phụ thuộc nhau.
- Không coi *mean ± SD 5 seeds* là confidence interval về performance trên ảnh vườn khác.

---

## 11. Cấu trúc notebook theo STANDARD_TRAINING_NOTEBOOK_FORMAT.md

**Giữ luồng đánh số CELL 0–29 của chuẩn dự án**, nhưng điều chỉnh rõ **Train = fit classifier từ embedding**, thay vì fine-tune DeiT. Không tạo learning curve epoch giả cho frozen DeiT/XGBoost.

| Notebook cell | Tiêu đề / tác vụ cụ thể | Outputs tối thiểu |
|---:|---|---|
| **0** | EXPERIMENT OVERVIEW — mục tiêu D2H, A/B/C, dữ liệu V08, 5 seeds, trạng thái no-backbone-training | Markdown overview |
| **1** | IMPORT LIBRARIES / install package pinned, GPU check, environment | `environment.json`, `requirements.lock` |
| **2** | CONFIG — tất cả params, flags, seed list, classifier and threshold candidate lists | `experiment_config.json` |
| **3** | REPRODUCIBILITY / `set_seed`, deterministic options, hardware | `runtime.json` |
| **4** | DRIVE / local / Kaggle PATH RESOLUTION; read-only EXP-007 sources | resolved path manifest |
| **5** | DATASET ZIP SHA + V08 manifest & image audit | `dataset_audit_summary.json` |
| **6** | LABEL INDEX / counts / subgroup `real_` | class distribution tables & plot |
| **7** | READ/DECODE/RESIZE exactly as original; feature pipeline no aug | input smoke |
| **8** | SOURCE EXP-007 CHECKPOINT INDEX / SHA verify 5 seeds | `checkpoint_inventory.csv` |
| **9** | RESTORE DeiT seed checkpoint + inspect nested architecture/feature endpoint | `feature_extractor_config.json` |
| **10** | EMBEDDING AND SOFTMAX PARITY SMOKE (2 images → full Val) | `pipeline_smoke.json`, parity CSV |
| **11** | EMBEDDING EXTRACTOR, batch inference, caching Train and Val per seed | `embeddings/*.npy`, metadata |
| **12** | BASELINE A original Softmax metrics and predictions | per-seed A reports |
| **13** | CV configuration, held Train-only `StratifiedKFold` | CV fold manifest |
| **14** | LINEAR PROBE B — fit/cross-validate/refit & save | `logreg.joblib`, CV CSV |
| **15** | XGBOOST C — fit/cross-validate/refit & save | `xgb.json`, CV CSV |
| **16** | VAL EVALUATION A/B/C per seed, raw argmax | metrics, predictions, confusion |
| **17** | OPTIONAL HEALTHY GATE D (switchable) | separate policy comparison |
| **18** | AGGREGATE 5-SEED RESULTS + paired deltas | summary CSV/MD |
| **19** | PER-CLASS ERRORS / D2H, subgroup real_, calibration | error tables/plots |
| **20** | VALIDATION SELECTION + IMMUTABLE LOCK | `validation_selection_lock.json` |
| **21** | METHOD-APPROPRIATE CURVES — CV scores vs params/trees, not DeiT training curves | CV learning/validation plots |
| **22** | CONFUSION MATRICES 5 SEEDS × 3 CLASSIFIERS, raw + optional gated separate | 5×5 panel PNG/CSV |
| **23** | SAVE FINAL VALIDATION METRICS, consistency checks | JSON, full CSV |
| **24** | INFERENCE SPEED (feature + classifier) / resource consumption | timing CSV |
| **25** | FAILURE ANALYSIS: recurring D2H, wrong→right, right→wrong, high-confidence wrong | traceable image list + panels |
| **26** | LOCK-GATED V08 TEST / HISTORICAL HOLDOUT evaluation only | optional marked retrospective results |
| **27** | MANUAL INFERENCE IMAGE — full pipeline, label order, optional gate | inference demonstration |
| **28** | AUTOMATED README + MODEL REPORT + checkpoints provenance | `README.md`, `MODEL_REPORT.md` |
| **29** | FULL / ANALYSIS ZIP export, checksums, result registry, final checklist | 2 archives, manifests |

**Diễn giải chuẩn:** nếu một cell mục tiêu không phù hợp vì ViT đã huấn luyện xong (ví dụ `Callbacks`, `Train Learning Curves`), thay bằng **CV/search diagnostics** và ghi rõ rationale. Giữ chức năng cơ bản của chuẩn: import/config/data/model/reproducibility/metrics/plots/error analysis/deployment package/export/readme.

---

## 12. Thư mục kết quả bắt buộc

```text
EXP-VIT-TRANSFER-XGBOOST-008/
├── README.md
├── MODEL_REPORT.md
├── experiment_config.json
├── environment.json
├── requirements.lock
├── execution_status.json
├── pipeline_smoke.json
├── dataset_audit/
│   ├── dataset_audit_summary.json
│   ├── exact_duplicates.csv
│   ├── near_duplicate_candidates.csv
│   └── audit_contact_sheets/                 # nếu có candidates
├── source/
│   ├── exp007_validation_selection_lock.json # source copy, không chỉnh
│   ├── exp007_checkpoint_inventory.csv
│   ├── exp007_config_sha256.json
│   └── provenance.json
├── aggregate/
│   ├── validation_metrics_5seeds_by_classifier.csv
│   ├── validation_mean_std_by_classifier.csv
│   ├── disease_to_healthy_by_seed_and_classifier.csv
│   ├── per_class_validation_by_seed.csv
│   ├── paired_deltas_vs_softmax.csv
│   ├── train_cv_best_params_by_seed.csv
│   ├── real_filename_val_subgroup_by_seed.csv
│   ├── deployment_selection_candidates.csv
│   ├── model_size_latency_comparison.csv
│   └── validation_summary.md
├── seed_42/
│   ├── embeddings/
│   │   ├── train_features.npy
│   │   ├── val_features.npy
│   │   ├── train_meta.csv
│   │   ├── val_meta.csv
│   │   ├── feature_config.json
│   │   └── feature_stats.json
│   ├── softmax/
│   │   ├── validation_predictions.csv
│   │   ├── confusion_matrix.csv
│   │   ├── classification_report.csv
│   │   └── metrics.json
│   ├── logistic_regression/
│   │   ├── cv_trials.csv
│   │   ├── cv_fold_metrics.csv
│   │   ├── model.joblib
│   │   ├── validation_predictions.csv
│   │   ├── confusion_matrix.csv
│   │   └── metrics.json
│   ├── xgboost/
│   │   ├── cv_trials.csv
│   │   ├── cv_fold_metrics.csv
│   │   ├── model.json
│   │   ├── validation_predictions.csv
│   │   ├── confusion_matrix.csv
│   │   └── metrics.json
│   ├── optional_healthy_gate/
│   │   └── gated_validation_comparison.csv
│   └── error_analysis/
│       ├── disease_to_healthy.csv
│       ├── recurrent_errors.csv
│       └── wrong_to_right_vs_softmax.csv
├── seed_123/                               # tương tự, deployment seed source
├── seed_2026/                              # tương tự
├── seed_3407/                              # tương tự
├── seed_7777/                              # tương tự
├── selection/
│   ├── validation_selection_lock.json      # khóa trước Test/Holdout
│   ├── selected_classifier_metrics.json
│   └── selection_rationale.md
├── model_package/
│   ├── selected_deit_checkpoint.sha256
│   ├── classifier_model.json               # XGB nếu chọn C
│   ├── logistic_regression.joblib           # nếu chọn B
│   ├── deployment_manifest.json
│   ├── label_map.json
│   ├── preprocess_contract.json
│   └── load_and_inference_test.json
├── figures/
│   ├── val_confusion_matrices_5seeds_abc.png
│   ├── per_class_recall_abc.png
│   ├── d2h_counts_5seeds.png
│   ├── cv_model_selection.png
│   ├── paired_deltas.png
│   ├── calibration_curves.png
│   └── hard_case_contact_sheet.png
├── evaluation/                             # CHỈ xuất khi flags bật & lock hợp lệ
│   ├── retrospective_v08_test/
│   ├── historical_holdout_48/
│   └── fresh_blind_field_test/
├── archives/
│   ├── EXP-VIT-TRANSFER-XGBOOST-008_ANALYSIS.zip
│   └── EXP-VIT-TRANSFER-XGBOOST-008_FULL.zip
└── artifact_sha256_manifest.csv
```

`seed_42` chỉ là ví dụ cây thư mục; **cả 5 seed bắt buộc có outputs riêng**. Nếu classifier cuối cùng là A (Softmax), model package phải trỏ đến EXP-007 source artifact và không tự tạo XGBoost deployment package giả.

### FULL vs ANALYSIS ZIP

- **FULL ZIP:** notebook output-free (hoặc notebook đầu ra theo định dạng chuẩn dự án), đầy đủ source code/config/metadata/models/classifier artifacts/embeddings/figures/reports, nhưng **KHÔNG đưa dataset ảnh có bản quyền vào ZIP** trừ khi quyền dữ liệu cho phép. Với 5 ViT checkpoints gốc, có thể lưu *verified artifact references* thay vì sao chép checkpoint nếu lớn; tuy nhiên khi đó FULL ZIP **không self-contained**, phải nói rõ.
- **ANALYSIS ZIP:** config/protocol/hashes/validation reports, compact CSV, confusion/per-class, 5-seed figures, lỗi điển hình, representative inference, notebook code; không chứa binaries quá lớn và toàn bộ embeddings nếu cần giảm kích thước.
- Kiểm tra `sha256_manifest`, số file/byte và việc mở được ZIP; tránh tạo `.zip` chứa nhiều tệp `NOT_RUN` nhưng README ghi COMPLETED.

---

## 13. Giao thức deployment thử nghiệm

**Inference pipeline:** ảnh RGB bất kỳ → resize bilinear 224×224 và float32 0–255 → **DeiT original preprocessing đúng một lần** → verified extractor frozen seed 123 → feature vector → classifier đã chọn (B scaler+LR; hoặc C XGB) → `predict_proba` → optional Healthy gate → label theo class map gốc.

- Khởi tạo/load model **một lần mỗi process**, không load checkpoint mỗi ảnh.
- Nếu `not_cashew_leaf` có xác suất cao, không tự diễn giải thành bệnh điều; render nhãn đúng.
- Nếu confidence gần nhau hoặc model dự đoán Healthy thiếu chắc chắn, có thể đề xuất UI cảnh báo "cần chụp lại/kiểm tra thủ công" **chỉ sau khi nghiên cứu policy và calibration riêng**, không tuyên bố đó là kết quả đã được validated.
- Lưu `model_type`, `experiment_id`, `seed`, `checkpoint_sha256`, `classifier_sha256`, `token_policy`, `preprocessing`, `class_names`, `threshold` và full test recipe trong deployment manifest.
- **Không đẩy model mới vào Application/web production** trong nhiệm vụ notebook; chỉ cung cấp adapter/documentation.

---

## 14. Definition of Done / checklist agent phải tự đánh dấu

### Nguồn và reproducibility

- [ ] Đã đọc notebook EXP-007 thực tế + `STANDARD_TRAINING_NOTEBOOK_FORMAT.md`, không phỏng đoán KerasHub token API.
- [ ] Đã tải và xác minh **5 checkpoint `.keras` SHA256** theo source selection lock. Thiếu checkpoint → fail rõ ràng.
- [ ] Dataset ZIP V08 SHA đúng; 6.906 ảnh đúng split 4.790/1.430/686 và 5 lớp đúng thứ tự.
- [ ] Hash manifest tái tạo bằng thuật toán notebook EXP-007; khác định nghĩa canonical khác thì giải thích, không lờ đi.
- [ ] Thư viện được pin/log, source code/notebook và model binaries có SHA256.

### Đúng thí nghiệm classifier-only

- [ ] KerasHub DeiT **không được fit/không đổi weights** trong EXP-008; feature extractor đúng checkpoint EXP-007 theo seed.
- [ ] Trích pre-head features thật, ghi observed shape/token policy và đường đi preprocessing; **không dùng logits làm features** ở nhánh B/C.
- [ ] Train embeddings 4.790, Val 1.430, đồng nhất ảnh và order cho A/B/C.
- [ ] Full Validation source predictions parity cho **cả 5 seeds**; seed 123 D2H **18/900**, Healthy Recall **210/216**.
- [ ] LogisticRegression scaler fit trong từng CV Train fold; XGBoost CV trên Train-only, candidate list cố định, không sờ Test/Holdout.
- [ ] 5-seed A/B/C đã chạy, thống kê mean ± SD cùng per-seed raw metrics.
- [ ] Optional threshold có raw-vs-gated report và Healthy Recall guardrails; không giấu false alarms.

### Metrics và bảo vệ khoa học

- [ ] Bảng D2H có **count/mẫu số** theo seed/classifier/lớp bệnh; Healthy Recall, Not Cashew Recall, disease macro recall, Macro F1, Log Loss đầy đủ.
- [ ] Confusion matrix theo từng seed/classifier = số ảnh Validation; class order đúng.
- [ ] Không có exact byte duplicates xuyên split; near-duplicate/group-leakage audit status được công khai (có thể chưa fully resolved → không claim leakage-free).
- [ ] Không tune/chọn model theo 48 Holdout hay Test V08 đã mở; `validation_selection_lock.json` tạo **TRƯỚC** mọi evaluation gated.
- [ ] Deploy ViT seed **123** cố định; classifier thắng theo pre-registered D2H + guardrails, hoặc fallback A nếu không chứng minh cải thiện.
- [ ] Nếu thiếu fresh blind field test: báo **NOT_RUN**, không viết kết luận "đã cải thiện ảnh thực địa".

### Trình bày, lưu trữ, xuất bản

- [ ] README/MODEL_REPORT giải thích EXP-007 baseline, mục tiêu, cấu hình, 5-seed comparisons, strengths/limitations, future work, non-blind limitations.
- [ ] Có source config, CV trial logs, per-seed predictions, confusion/class reports, D2H/error tables, plots, runtime/latency end-to-end.
- [ ] Model package và example inference có self-test sau reload.
- [ ] Tạo và verify `ANALYSIS.zip`, `FULL.zip` cùng checksum/manifest; ghi rõ những artifacts không được đóng gói.
- [ ] Không sửa EXP-007, dataset V08, source checkpoints, hoặc Application deployment.

---

## 15. Hướng dẫn hành động để agent bắt đầu

1. **Đọc** notebook `vit_transfer_5seed_007.ipynb` trong ANALYSIS ZIP, `experiment_config.json`, `validation_selection_lock.json`, per-seed Validation predictions, rồi đọc `STANDARD_TRAINING_NOTEBOOK_FORMAT.md`.
2. **Tạo outline notebook** theo CELL 0–29 ở mục 11, xuất mã source trước, **không chạy CV/đánh giá ngay**.
3. **Chạy smoke tests** cho dataset, 5 checkpoint, source predictions parity, embedding endpoint/token policy/single preprocessing. Nếu bất kỳ test không pass → báo blocker và dừng.
4. **Trích cache Train/Val embedding** cho đúng 5 seeds với hash/provenance đầy đủ. So A với predictions EXP-007.
5. **Chạy B/C train-only CV + refit**, đánh giá A/B/C trên Validation. Sinh metrics, confusion matrices, paired delta, D2H and safety guardrails.
6. **Chọn classifier cho frozen ViT seed 123 theo protocol** đã đăng ký, tạo selection lock, export model package, check load/inference.
7. **Không chạy Test/Historical Holdout theo mặc định.** Nếu được yêu cầu đánh giá hồi cứu, chỉ làm sau lock và gắn nhãn đã lộ; đề xuất fresh independent field test như bước xác nhận cuối.
8. Kết thúc báo cáo chính xác số liệu đã có, đường dẫn output, hash, trạng thái theo checklist. **Không tự push GitHub trừ khi được yêu cầu**; nếu được yêu cầu, sử dụng nhánh/PR và tránh commit binary lớn hoặc dữ liệu cá nhân.

### Prompt ngắn dùng kèm tài liệu

> Hãy xây dựng notebook `EXP-VIT-TRANSFER-XGBOOST-008` đúng theo implementation spec đính kèm và `STANDARD_TRAINING_NOTEBOOK_FORMAT.md`. Kế thừa 5 checkpoint EXP-007 đã khóa; không retrain backbone. Thực hiện đối chứng Softmax, Logistic Regression và XGBoost trên cùng pre-head DeiT embedding và cùng V08; tối ưu **giảm disease→Healthy** trên Validation có Macro F1/Healthy/Not-Cashew Recall guardrails. Chạy kiểm thử checkpoint/preprocessing/per-image baseline parity trước CV, không dùng Test/Historical Holdout để chọn model. Khóa seed 123 và classifier đã chọn, lưu metrics, predictions, figures, model package, README và ZIP. Nếu thiếu checkpoint hoặc không tìm được feature endpoint đúng, dừng và báo blocker thay vì tạo kết quả giả.

---

*Implementation specification prepared on 2026-10-21. Các con số EXP-007 được dùng làm baseline đã quan sát từ gói nguồn; **EXP-008 chưa chạy**.*
