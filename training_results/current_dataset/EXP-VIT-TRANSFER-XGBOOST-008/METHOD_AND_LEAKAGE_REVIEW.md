# Protocol review — EXP-VIT-TRANSFER-XGBOOST-008

## Dựa trên archive gốc

Source `EXP-VIT-TRANSFER-XGBOOST-008_ANALYSIS.zip` có **109 entries**, SHA256 `553AEC1C888DAFC3E6C1EDF9E65553A48DF082C68C8947E9B8EB4B9DB719EAD1`. Đây là classifier-only ViT study, **không fine-tune lại DeiT backbone** trong 008. Những kết luận sau chỉ dựa trên artifacts đã cung cấp, không tự chạy lại checkpoint binaries.

- Config cho phép **5 seeds EXP-007** fixed; selected source seed **123**.
- Classifier A **original softmax head**, B **LogisticRegression + fold-specific StandardScaler**, C **XGBoost multi:softprob**. XGBoost có 12 candidates; B `C={0.01, 0.1, 1, 10}`; classifier-level **3-fold Train CV**.
- Embedding là endpoint `DeiTImageClassifier.output_dense.input`, **192 features**. Input RGB 224, preprocess **một lần**.
- `source/validation_parity_5seeds.csv` ghi **0** Softmax mismatches trên 1430 ảnh Val trong cả 5 seeds. Model files gốc không nằm trong repo, nên test parity là theo báo cáo source.
- Group-based independence **chưa chứng minh**; kiểm toán pHash trước đó từ các EXP CNN tìm nhiều candidates hơn dHash của riêng archive này. Không kết luận archive ít dHash candidate hơn nghĩa là sạch leakage.
- Classifier CV được thực hiện Train-only nhưng embedding từ DeiT đã được học trên toàn bộ Train, bao gồm mọi CV fold. **Do đó CV không tương đương nested/group-out-of-fold training toàn pipeline**, chỉ dùng để chọn hyperparameters classifier.
- Validation sau selection mang thiên lệch; Final Test, Historical Holdout và fresh field test trong EXP-008 **đều NOT_RUN**.

## Chính sách quyết định & nguy cơ

Seed 123 được khóa trước experiment. Trong 900 ảnh bệnh Val, Softmax A bỏ sót **18**, Logistic B **21**, XGBoost C **23** ảnh vì dự đoán Healthy. Dù Accuracy lần lượt **95,45 / 96,36 / 95,87%**, tiêu chí D2H được ưu tiên nên **selected A**, threshold gate disabled. B/C thường nâng mạnh Recall `not_cashew_leaf` nên Accuracy cải thiện, nhưng không giải quyết lỗi bệnh→Healthy trên seed deploy.

**Không lựa chọn lại seed** sang 42/2026/3407 dựa trên D2H của từng classifier; điều đó thay đổi protocol deployment. **Không gọi đây là kết quả Test**.

## Gợi ý nghiên cứu tiếp

- Audit nguồn và nhãn ảnh Anthracnose→Healthy (một số ảnh 1480–1486 lặp trên nhiều mô hình).
- Ablation separate `classifier effect` vs `threshold/policy effect`: cùng validation selection/guardrail, fresh field verification.
- Báo cáo recall Anthracnose/Leaf miner/Red rust, D2H count/rate, Healthy Precision/Recall, Not Cashew Recall, Macro F1, calibration, inference time, paired fixes/regressions cùng seed.
- Kiểm định nhóm lá/buổi chụp và thu thập ảnh fresh blind đủ 5 true classes; holdout 48 ảnh lịch sử không dùng để tuning.

**Dữ liệu cốt lõi từ nguồn**: `validation_metrics_5seeds_abc.csv`, `validation_mean_std.csv`, `selection/validation_selection_lock.json`, `dataset_audit/dataset_audit_summary.json`. Snapshot trên GitHub chỉ bao gồm phần chọn lọc; xem `ARTIFACT_INVENTORY.md`.
