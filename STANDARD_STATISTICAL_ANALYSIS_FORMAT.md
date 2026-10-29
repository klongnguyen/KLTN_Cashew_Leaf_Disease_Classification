# STANDARD STATISTICAL ANALYSIS FORMAT — V1.0

> **Áp dụng:** `STATISTICAL_ANALYSIS.ipynb`, dùng kết quả chuẩn V2 từ CNN, ResNet50, DenseNet121, ViT/DeiT.  
> **Ngày chốt:** 29/10/2026.  
> **Tài liệu ghép cặp:** [`STANDARD_TRAINING_NOTEBOOK_FORMAT.md`](STANDARD_TRAINING_NOTEBOOK_FORMAT.md).  
> **Căn cứ dự án:** `docs/research_notes/AUDIT_NOTEBOOK_OUTPUTS_STATISTICAL_PLAN.md` (27/10/2026), `docs/RESEARCH_PROTOCOL.md`.

## 1. Mục đích và nguyên tắc

Notebook này **không train mô hình**, mà xác thực provenance và tổng hợp kết quả nhiều architecture/training mode/seeds thành bảng và hình nghiên cứu. Nó không quyết định chọn model theo Test/Holdout đã quan sát. Mọi phép so sánh phải ghi rõ `EXPLORATORY_VALIDATION`, `LOCKED_TEST_CONFIRMATORY` (nếu thật sự được thiết kế trước), hoặc `RETROSPECTIVE_EXPOSED`.

Các giới hạn bắt buộc:

- Chỉ ghép cặp dự đoán khi cùng **dataset hash/manifest, tập ảnh, nhãn thật, task, class order sau chuẩn hóa**.
- Các kiến trúc có thể dùng preprocessing khác nhau hợp lệ; phải lưu riêng preprocessing/recipe và kiểm toán inference parity.
- Không dùng số liệu V08/V09/V10/V11 hoặc Holdout02–05 làm controlled one-factor benchmark nếu snapshot và cohort khác nhau.
- Không coi 5 seed là 5 tập ảnh độc lập; không diễn giải CI qua seed là CI qua vườn hoặc phiên chụp.
- Không sử dụng Test/Holdout đã xem để tuning/hyperparameter/threshold, hoặc chọn lại seed/model.
- Với Real Holdout05 chỉ **4 lớp được gán nhãn**, phải báo class support và phương pháp tính Macro-F1 riêng; không gọi là blind field 5-class.

## 2. Input contract và kiểm tra đầu vào

Nhập từng `EXP-..._ANALYSIS.zip`, kiểm tra `analysis_manifest.json`, `experiment_config.json`, `validation_selection_lock.json` (nếu đã Test), `aggregate/*.csv`, `seed_*/validation/predictions.csv`, Test/Holdout khi có và per-class reports.

`predictions.csv` **bắt buộc** có `experiment_id,architecture,training_mode,dataset_version,dataset_sha256,dataset_manifest_sha256,split,seed,image_id,relative_path,true_label,predicted_label,true_class_idx,predicted_class_idx,correct,confidence` và **5 cột xác suất** (`prob_anthracnose`, `prob_healthy`, `prob_leaf_miner`, `prob_red_rust`, `prob_not_cashew_leaf`).

Trước khi phân tích:

1. Kiểm tra số row bằng support từng split/seed; unique `(experiment_id,seed,split,image_id)`.
2. Kiểm tra xác suất hữu hạn, trong `[0,1]`, tổng ≈1; predicted label khớp argmax + label map.
3. Kiểm tra hash/ground truth và alignment ảnh giữa mô hình; báo `NOT_COMPARABLE` thay vì cố ghép nhầm.
4. Ghi rõ model được **TRAINED** hay **EVAL_ONLY**; ngăn đánh giá lại checkpoint được mô tả là training mới.
5. Không tự điền zeros cho metrics/probabilities thiếu; đánh dấu `MISSING/PENDING`, nếu cần bổ sung bằng inference-only có provenance.

Master outputs:

```text
statistical_analysis/
├── provenance_audit.csv
├── master_metrics.csv                   # 1 row/model/mode/seed/split
├── master_predictions.parquet          # 1 row/model/mode/seed/image
├── master_per_class.csv
├── master_efficiency.csv
├── statistical_test_registry.csv       # so sánh dự kiến / multiplicity families
├── analyses/
│   ├── scratch_baselines.csv
│   ├── transfer_baselines.csv
│   ├── scratch_vs_transfer.csv
│   ├── cross_architecture.csv
│   ├── paired_image_bootstrap.csv
│   ├── mcnemar_holm.csv
│   ├── calibration_ece_brier.csv
│   └── domain_gap.csv
├── figures/
├── README.md
└── ANALYSIS_STATISTICAL_REPORT.zip
```

## 3. Thống kê mô tả 5 seed

Bắt buộc báo mean, sample SD (`ddof=1`), seed values, min/max và `n=5` cho Accuracy/Macro Precision/Macro Recall/Macro F1/Balanced Accuracy, Training time.

$$\mathrm{SD}_{\mathrm{sample}}=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar x)^2}{n-1}}$$

CI theo seed có thể dùng `mean ± t(0.975, df=4)·SD/√5` với giả định rõ. CI này phản ánh **biến động do seed trên một split cố định** chứ không phản ánh lấy mẫu ảnh/vườn độc lập. Không gọi `mean ± SD` là confidence interval.

## 4. Kế hoạch kiểm định và giả thuyết

| Cấp so sánh | Phép thử/CI phù hợp | Chú thích bắt buộc |
|---|---|---|
| Cùng kiến trúc +/− attention hoặc Scratch–Transfer (5 paired seeds) | Paired differences, paired t-test hoặc Wilcoxon khi phù hợp; effect size, CI delta | Cỡ mẫu **5 seed rất nhỏ**, report thăm dò; seed ghép cặp không bảo đảm độc lập hoàn toàn |
| Hai classifier **đã chọn bằng Validation** trên cùng ảnh Test | McNemar cho correctness 0/1 | Chỉ khi cùng ảnh, nhãn và lựa chọn đã được khóa trước; nếu Test từng dùng chọn model phải ghi retrospective |
| Hiệu số Accuracy/Macro F1 trên cùng ảnh | **Paired image bootstrap** 95% CI | Resample cùng image IDs cho A/B; nếu ảnh cùng lá/cây thì cluster bootstrap theo group nếu có metadata |
| >2 classifier trên cùng ảnh | Cochran's Q, post-hoc McNemar với Holm | Định trước family tests; không tự tạo hàng chục test rồi chỉ lấy p tốt nhất |
| Per-class performance | F1/Recall/support; bootstrap phù hợp nếu đủ support | Lớp thiếu trong Holdout không được coi như lớp có ground truth |
| Classification calibration | ECE, Reliability Diagram, Brier Score, ROC-AUC/PR-AUC OVR | Cần full probabilities và protocol calibration, dùng Validation để fit nếu calibration/tuning |
| Robustness/domain shift | Public Test vs field holdout plot + cohort table | Là quan sát phân phối, không nhân quả; đánh dấu exposed vs blind, khác số lớp |
| Runtime/deploy | Accuracy–latency Pareto | Cùng thiết bị, batch, warm-up, sync, preprocessing scope |

- Báo `effect size` phù hợp: ví dụ mean paired delta (điểm phần trăm) và standardized paired effect size nếu được định nghĩa; **không** sử dụng một công thức Cohen's d chung cho mọi thiết kế.
- Báo 95% CI, p-value (nếu có), p adjusted Holm theo **family đã khai báo**, support và giới hạn giả định. `p<0.05` không tự động nghĩa có cải tiến triển khai.
- Nếu giả định paired test không được đáp ứng, ghi `NOT_APPLICABLE` và chỉ mô tả delta/CI hợp lệ; không ép chạy phép thử.

## 5. Bộ hình/bảng chuẩn theo thứ tự báo cáo

| Key | Tên output | Dữ liệu |
|---|---|---|
| A | `A_scratch_baseline_metrics_with_seed_sd.png` | Scratch Accuracy / Macro P / Macro R / Macro F1 / Balanced Acc ± **sample SD** |
| B | `B_transfer_baseline_metrics_with_seed_sd.png` | Transfer metrics như trên |
| C | `C_architecture_scratch_vs_transfer_forest_delta_CI95.png` | Δ + CI95% (nêu unit of uncertainty) |
| D | `D_per_class_f1_heatmap_validation.png` | Model × Class Macro-/Per-Class F1 trên Validation |
| E | `E_pairwise_mcnemar_holm_table.csv` | Exact/approx McNemar, p, adjusted p, discordant counts |
| F | `F_seed_stability_pairplot.png` | Metric từng seed/architecture/mode; paired lines |
| G | `G_reliability_diagram_val.png` + `G_calibration_ece.csv` | Calibration/ECE/Brier trên Val |
| H | `H_accuracy_latency_pareto.png` | Accuracy/Macro F1 vs latency + size |
| I | `I_domain_gap_real_holdout.png` | Test vs real holdout, notes class support/exposure |
| J | `J_YOLO_AP_small_medium_large.png` | **Có điều kiện:** chỉ detection benchmark với GT chuẩn và size definition nhất quán |

Các hình có caption: dataset snapshot/hash, split, task, n seed, số ảnh, class support, errorbar nghĩa SD hay CI, selection status, legend và metric unit. Lưu PNG 300 dpi, tùy thêm PDF/SVG và CSV nguồn.

## 6. Thứ tự phân tích tránh test-selection leakage

1. Audit schema, provenance, cùng ảnh/nhãn; không dùng Test ở bước chọn.
2. **Scratch-vs-Scratch** trên Validation, cùng dataset version. Bảng mean ± SD và stability.
3. **Transfer-vs-Transfer** trên Validation, cùng dataset version.
4. **Trong cùng kiến trúc**: Scratch–Transfer, baseline–attention, one-factor ablation trên **cùng split**, với phân biệt confounders.
5. **Cross-architecture**: per-class heatmap, paired differences; chọn theo **Validation Macro F1, disease→healthy, latency/model size** theo rule ghi trước.
6. Lưu model selection lock và test protocol; chỉ sau đó thực hiện locked Test (nếu còn chưa bị lộ). Không chạy lại/tune nếu kết quả thấp.
7. Báo Test/Holdout đã exposed là **retrospective**, sử dụng để mô tả lỗi/domain gap không phải chọn lại model.
8. Triển khai Pareto và phân tích inference parity, không mặc định highest Test Acc là deployment winner.

## 7. YOLO và pipeline hai giai đoạn: so sánh đúng cùng task

Đối với **YOLO direct disease detection**, lưu bbox class/score/absolute+normalized coords, GT boxes, matching IoU, per-class P/R/F1/AP, mAP50, mAP50–95, AP small/medium/large nếu có quy ước/GT phù hợp và latency. Không so mAP bbox với image classification accuracy.

Đối với **YOLO → crop → classifier**, trước tiên so classifier trên full image vs GT crop vs predicted crop để định lượng domain shift input; sau đó so end-to-end với 1-stage **chỉ khi quy đổi cùng task và ground truth**. Phải mô tả cách xử lý no detection, multi-lesion, healthy, not_cashew_leaf. Test/threshold chọn trên Validation.

## 8. Definition of Done

- [ ] Tất cả source experiments có provenance audit và status.
- [ ] Dataset hash, image IDs, class order, support khớp trước paired comparison.
- [ ] 5-seed mean±sample SD được báo rõ; CI seed không nhầm CI ảnh.
- [ ] Các paired tests/bootstraps có mô tả unit, assumptions, effect size và multiplicity.
- [ ] McNemar/Cochran's Q/Holm chỉ chạy khi đúng design, không p-hack Test.
- [ ] Đủ hình A–I **khi dữ liệu thực sự hỗ trợ**, nếu không có `PENDING/NOT_APPLICABLE` với lý do.
- [ ] Calibration không bị thiếu probabilities mà vẫn giả vờ tính ECE/Brier.
- [ ] Holdout có class-support/exposure và không được gọi nhầm fresh blind.
- [ ] Figures + source CSV + README + analysis ZIP đã kiểm tra tồn tại.

---

**Ghi chú phạm vi:** Đây là chuẩn phương pháp/đầu ra được thống nhất để áp dụng từ các thí nghiệm mới. Nó **không** tuyên bố mọi EXP trên GitHub hiện đã có đủ các artifact hoặc đủ điều kiện kiểm định.