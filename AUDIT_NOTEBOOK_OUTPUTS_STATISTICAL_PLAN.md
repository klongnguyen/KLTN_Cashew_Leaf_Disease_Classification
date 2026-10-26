# Kiểm tra output notebook và kế hoạch so sánh thống kê — Cashew Leaf Disease

**Ngày kiểm tra:** 27/10/2026  
**Phạm vi:** Kiểm tra mã nguồn, cấu hình và output nhúng trong 4 notebook đã tải lên. **Chưa kiểm tra trực tiếp nội dung bytes của các ZIP được sinh ra**, vì người dùng chưa đính kèm ZIP. Các notebook Scratch chưa được cung cấp trong lượt này.

## 1. Notebook và khả năng so sánh

| Notebook | Task | Dataset / seeds | Trạng thái Test trong notebook | Trạng thái ZIP |
|---|---|---|---|---|
| `resnet50_transfer_5seed_009.ipynb` | Classification 5 lớp | `CashewData_Split (8).zip`, 5 seeds | Đã chạy 5 seed | FULL + ANALYSIS, output ghi đã xác minh |
| `densenet121_transfer_5seed_014.ipynb` | Classification 5 lớp | Cùng split và seeds | `RUN_FINAL_TEST=False`, đang PENDING | FULL + ANALYSIS, output ghi đã xác minh |
| `vit_transfer_5seed_007.ipynb` | Classification 5 lớp (DeiT-tiny) | Cùng split và seeds | Đã chạy 5 seed | FULL + ANALYSIS, output ghi đã tạo |
| `train-yolov26.ipynb` | Detection 3 bệnh | Bộ Roboflow riêng, một seed 42, 640 px | Đã chạy Test, output 39 ảnh, 1.335 instances | Một ZIP riêng, không có cặp FULL/ANALYSIS |

Cùng split Classification: Train 4.790, Val 1.430, Test 686; class `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`; seeds `[42,123,2026,3407,7777]`.

## 2. Kiểm tra tính đồng nhất của output

| Thành phần | ResNet50 Transfer | DenseNet121 Transfer | ViT Transfer | YOLO26s |
|---|---|---|---|---|
| Lịch sử train dạng CSV | `seed_*/stage{1,2}/history.csv` | Như ResNet | `seed_*/history.csv` + `aggregate/history_5seeds.csv` | `train_run/results.csv` do Ultralytics |
| Accuracy/Loss panel 5 seed | `figures/panels/training_curves_5seeds_panel.png` | Giống ResNet | `aggregate/training_curves_5seeds.png` | `train_run/results.png`, chỉ 1 seed |
| Validation confusion raw PNG | Có mỗi seed + panel | Có mỗi seed + panel | Có mỗi seed + panel | Có ở `train_run/confusion_matrix.png` sau khi notebook ghi kết quả Validation vào thư mục train; chưa có `validation_run/` riêng |
| Validation confusion normalized PNG | Có mỗi seed + panel | Có mỗi seed + panel | Có mỗi seed | Có qua `train_run/confusion_matrix_normalized.png`; chưa tách `validation_run/` |
| Validation confusion raw CSV | Có mỗi seed | Có mỗi seed | Có mỗi seed | Có qua `train_run/confusion_matrix.csv` sau khi ghi kết quả Validation |
| Validation confusion normalized CSV | Có mỗi seed | Có mỗi seed | **Thiếu CSV** (có PNG) | Có qua `train_run/confusion_matrix_normalized.csv` |
| Test confusion PNG | **Panel 5 seed** (không thấy per-seed Test PNG) | **Chưa chạy Test** | Per-seed raw/normalized và panel | `test_run/confusion_matrix*.png` có nếu được sinh bởi `val()` |
| Test confusion CSV | Per-seed raw/normalized | Chưa chạy | Per-seed raw/normalized | `test_run/confusion_matrix*.csv` từ custom render |
| Classification report | `seed_*/validation/report.csv` với tên thực `classification_report.csv` | Như ResNet | `seed_*/validation_classification_report.csv` | `per_class_metrics.csv`, không cùng task |
| Per-image predictions | `seed_*/validation/predictions.csv` và tương tự Test | Như ResNet | `seed_*/validation_predictions.csv`, tương tự Test | Chỉ file custom tổng hợp theo ảnh; không lưu đầy đủ box/prediction/probability |
| Xác suất từng class | Có `prob_{class}` cho mỗi ảnh | Có `prob_{class}` | **Thiếu**; chỉ true/pred index | Confidence từng box chưa được export thành bảng |
| `confidence`, `correct` per image | Có | Có | **Thiếu** trong predictions CSV | Không có định dạng ảnh classification |
| Aggregate `validation_results_5seeds.csv` | Có | Có | Có | Không áp dụng (1 seed) |
| Aggregate `test_results_5seeds.csv` | Có | **Chưa có** | Có | Không áp dụng |
| Aggregate Mean ± SD | Có: `validation_mean_std_summary.csv` | Có: `validation_mean_std_summary.csv` | Có: `validation_mean_std.csv` | Không có multi-run mean ± SD |
| Archive | `FULL.zip`, `ANALYSIS.zip` | `FULL.zip`, `ANALYSIS.zip` | `FULL.zip`, `ANALYSIS.zip` | `EXP-Y26S-SMALL-007.zip` |

**Điểm nhấn:** ViT có Confusion Matrix nhưng không hoàn toàn cùng *tên file/đường dẫn/schema*. ViT chưa xuất xác suất 5 lớp; do đó các Reliability Diagram, ECE, Brier Score, ROC/PR one-vs-rest, phân tích confidence và ensemble sẽ không đủ thông tin nếu chỉ dùng ANALYSIS ZIP. Có thể load mô hình từ FULL ZIP và chạy inference lại trên tập đã khóa **mà không train lại**, chỉ nhằm bổ sung output dự đoán, không tuning.

**Điểm nhấn YOLO:** `train-yolov26.ipynb` có hàm xuất confusion cho Validation và Test. Notebook lưu ma trận Validation ở `runs/validation/...` **đồng thời ghi một bản vào `train_run/`**, nên biểu đồ và CSV Validation vẫn được mang vào ZIP thông qua `train_run/`. Tuy nhiên ZIP không có thư mục `validation_run/` minh bạch, tên file dễ bị hiểu sai là ma trận Train. Nên xuất riêng `validation_run/` và giữ `train_run/` dành cho đồ thị quá trình train.

**Việc cần kiểm tra trực tiếp sau này:** Đối chiếu danh sách file thật trong ANALYSIS ZIP bằng script manifest checker: file có tồn tại, không rỗng, đúng 5 seed, đủ số hàng prediction, class/order và dataset fingerprint, chuẩn hóa path.

## 3. Sửa notebook theo ưu tiên

### P0 (trước khi thu thập/so sánh final)
1. **Đồng bộ `predictions.csv`**: các trường `experiment_id, architecture, training_mode, dataset_version, dataset_sha256, split, seed, image_id, relative_path, true_label, predicted_label, true_class_idx, predicted_class_idx, correct, confidence, prob_anthracnose, prob_healthy, prob_leaf_miner, prob_red_rust, prob_not_cashew_leaf`.
2. Thêm `confidence`, `correct`, 5 class probabilities cho ViT; thống nhất `true_label/predicted_label` thành tên lớp (string) và giữ index riêng. Bổ sung probability **không đòi hỏi train lại** nếu còn checkpoint.
3. Với DenseNet transfer: khi cấu hình và checkpoint đã được khóa bởi Validation, chạy final Test **một lần**; xuất Test 5 seed + panels. Không dùng Test để đổi recipe.
4. Đồng bộ cách chọn checkpoint/seed theo **cùng tiêu chí chính** (đề xuất `val_macro_f1`; `val_loss` để tie-break) cho các lần thử nghiệm mới; ResNet và Dense hiện ưu tiên `val_loss`, ViT ưu tiên `val_macro_f1`. Không sửa hậu nghiệm những thí nghiệm đã khóa để 'đẹp điểm' — ghi rõ nguyên tắc cũ khi báo cáo; nếu thay đổi phải tạo experiment mới, lựa chọn chỉ từ Validation.
5. Kiểm tra cùng dataset bằng SHA-256/manifest, không chỉ so sánh tên dataset. Phải tránh ảnh của cùng lá/cây xuất hiện giữa Train, Validation và Test.
6. Với YOLO: phải xác định và khóa dataset detection riêng, giữ `test` dành cho final; không so sánh trực tiếp `mAP` với Accuracy/F1 classification.

### P1 (đồng nhất hình, bảng, CSV)
7. Cùng convention tên `seed_*/validation|test/metrics.json`, `predictions.csv`, `classification_report.csv`, `confusion_matrix_raw.csv/png`, `confusion_matrix_normalized.csv/png`, `error_cases.csv`, `history.csv`.
8. Thống nhất trong `aggregate/`: `validation_results_5seeds.csv`, `validation_mean_std.csv`, `test_results_5seeds.csv`, `test_mean_std.csv`, `per_class_{split}_by_seed.csv`, `per_class_{split}_mean_std.csv`, `training_time_mean_std.csv`.
9. ViT nên bổ sung Validation normalized CM CSV; ResNet/Dense nên bổ sung mỗi seed Test CM PNG bên cạnh panel.
10. Có `analysis_manifest.json` và `audit_required_artifacts.json` kiểm tra file bắt buộc, trạng thái `PENDING` được đánh dấu minh bạch. `ANALYSIS.zip` chỉ text/CSV/JSON/PNG; `FULL.zip` thêm checkpoints. `archive_version`: ví dụ `v1`.
11. Cần chuẩn hóa thời gian suy luận trên **cùng thiết bị, cùng số warm-up, batch 1, đồng bộ CUDA, preprocessing+postprocessing rõ ràng**, rồi ghi mean, median, p95, kích thước model, tham số.
12. Sửa mô tả `NOT_EXECUTED` trong đầu ResNet50/ViT vì chính notebook có output training và export đã chạy; ghi trạng thái/phiên bản chính xác.

### P2 (phục vụ thống kê nâng cao)
13. Phân phối confidence, Reliability Diagram/ECE, Brier Score, ROC-AUC OVR (classification), Grad-CAM nếu có metadata vùng bệnh.
14. YOLO: lưu đầy đủ `detections.json`, `ground_truth_annotations.json`, class, score và normalized/absolute bbox; IoU matched pairs; AP_small/medium/large theo quy ước chính thức; `seed`, GPU, latency có warm-up và measurement methodology.
15. `real_holdout` có 49 ảnh nguồn, 1 ảnh trùng với Test nội bộ đã bị loại → còn 48 ảnh hợp lệ; chỉ 4 lớp bệnh/healthy, **không có `not_cashew_leaf`**. Báo cáo rõ số mẫu và metric tương ứng. Không đánh đồng với Test 5 lớp 686 ảnh. `real_` filename trong Val chỉ là proxy nguồn ảnh, không thay cho metadata đã kiểm chứng.

## 4. Đề xuất kiến trúc phân tích trung tâm

### Tầng A — Notebook train (`atomic outputs`)
Từng notebook chỉ train/evaluate và export artifact tiêu chuẩn trên Validation, Test (sau khi lock). Không tạo biểu đồ so sánh nhiều mô hình tại đây.

### Tầng B — Statistical analysis notebook (gom kết quả)
Đọc các ANALYSIS ZIP của CNN/ResNet/DenseNet/ViT Scratch và Transfer. Chuẩn hóa schema; verify `image_id`, `true_label`, dataset hash, `split`, seed. Tạo `master_metrics.csv` (1 row/model/mode/seed/split), `master_predictions.parquet` (1 row/model/mode/seed/image), `master_per_class.csv`, `master_efficiency.csv`. Mỗi khác biệt phải được ghi trong provenance và không impute thiếu dữ liệu.

### Tầng C — Paper/report figures
**Giai đoạn 1 — Theo chế độ Scratch:** Bar/point chart Accuracy/Macro-Recall/Macro-F1/Balanced Acc có ±sample SD 5 seed; nếu cần, 95% CI bằng t với df=4 (mang nghĩa bất định qua seed, không phản ánh bất định do sampling test); per-class F1 heatmap; Val confusion panels; error overlap/Venn/UpSet.

**Giai đoạn 2 — So sánh trong cùng kiến trúc:** ResNet Scratch vs Transfer; DenseNet Scratch vs Transfer; ViT Scratch vs Transfer. Sử dụng paired difference plot, paired bootstrap CI ở cấp ảnh cho Accuracy/Macro-F1, McNemar (binary correct/incorrect) cho model đã chọn bằng Validation. Wilcoxon / paired t-test qua 5 seed chỉ dùng thận trọng nếu thiết kế thực sự ghép cặp hợp lệ, tránh diễn giải p-value quá mức.

**Giai đoạn 3 — So sánh giữa kiến trúc:** Model × technique heatmap của Macro-F1; forest plot của chênh lệch so với baseline; cùng test, cùng input-output; Cochran's Q nếu >2 classifier trên cùng ảnh, McNemar post hoc với Holm; paired bootstrap cho hiệu số Macro-F1. Trên Val để khám phá/chọn model, trên Test khóa để xác nhận một lần.

**Giai đoạn 4 — Kiểm tra độ tin cậy/triển khai:** Reliability Diagram/ECE (tính từ per-class probabilities), đồ thị Accuracy vs latency (Pareto frontier), tham số, dung lượng, p95 latency, domain gap từ public vs real holdout, `false healthy` và OOD `not_cashew_leaf` theo class.

**Giai đoạn 5 — YOLO + classifier:** Với Detection so sánh mAP50, mAP50–95, per-class AP và recall lesion nhỏ; đối chiếu pipeline 1-stage vs 2-stage chỉ khi các hệ thống được chuẩn hóa về **cùng task, cùng ground truth, cùng ảnh**. Ví dụ Image-level `healthy`/`not cashew`/`disease` được quy định quy tắc trước khi áp dụng McNemar image-level, hoặc so sánh bbox detections bằng mAP ở cùng 3 lớp bệnh.

## 5. Phân biệt hai loại độ bất định

- **Biến động qua 5 seed:** mô tả `mean ± SD`; seed CI = `mean ± t_(0.975,4) × SD / sqrt(5)` nếu t-based assumptions được xem là chấp nhận được. Không gọi `mean ± SD` là CI.
- **Bootstrap CI trên ảnh/đơn vị lấy mẫu:** lấy bootstrap image hoặc **cluster bootstrap** theo `leaf_id/tree_id` nếu có nhiều ảnh liên quan; dùng **cùng chỉ số ảnh resample cho hai mô hình** để tính `Δ Macro-F1` hoặc `Δ Accuracy` và CI95%. Không coi 5 seed cùng test là 5 bộ test độc lập.
- **Multiple comparisons:** Holm correction cho tập kiểm định được đặt trước; luôn báo cáo effect size, 95% CI, p-value hiệu chỉnh, support lớp, không chỉ câu "p < 0.05".
- **Khóa Test:** Không dùng Test/Real Holdout để chọn seed/hyperparameter, điều chỉnh threshold hay sàng lọc model. Kết quả Test đánh giá hiệu năng lựa chọn đã được khóa từ Val.

## 6. Bộ hình ưu tiên xuất từ notebook Statistical Analysis

01. `A_scratch_baseline_metrics_with_seed_sd.png` (Accuracy/Recall/Precision/F1, rõ `SD` không phải CI).
02. `B_transfer_baseline_metrics_with_seed_sd.png`.
03. `C_architecture_scratch_vs_transfer_forest_delta_CI95.png`.
04. `D_per_class_f1_heatmap_validation.png`.
05. `E_pairwise_mcnemar_holm_table.csv`.
06. `F_seed_stability_pairplot.png`.
07. `G_reliability_diagram_val.png` và `G_calibration_ece.csv` (thiết lập/hiệu chỉnh chỉ qua Val).
08. `H_accuracy_latency_pareto.png`.
09. `I_domain_gap_real_holdout.png`.
10. `J_YOLO_AP_small_medium_large.png` (sau khi có detection annotation phù hợp).

## 7. Quy trình chốt kết quả

1. Audit và sửa lệch output schema **không thay đổi thí nghiệm/không tuning theo Test**.
2. Gom ANALYSIS ZIP và dựng bảng Validation cùng định nghĩa metric.
3. So sánh Scratch cùng nhóm, Transfer cùng nhóm; sau đó so sánh cùng kiến trúc theo kỹ thuật; cuối cùng xếp hạng các kiến trúc theo Val Macro-F1, class-specific recall, thời gian/size và mức biến động seed.
4. Khóa phương án đề cử (và dự phòng), threshold, checkpoint, version dataset.
5. Thực hiện final Test một lần cho mô hình/config đã định trước; nếu đã test một số mô hình, ghi rõ đây là các test đã thực hiện, không sử dụng các kết quả này để sửa/tuning mô hình. Real Holdout được báo cáo riêng.
6. Đánh giá khả năng triển khai cuối cùng theo Pareto (quality, robustness, latency, model size), không mặc định model có Test Accuracy cao nhất là tốt nhất.

**Tình trạng:** Mới xác minh từ notebook và log output bên trong, **chưa kiểm tra byte-by-byte file ZIP thực tế**.
