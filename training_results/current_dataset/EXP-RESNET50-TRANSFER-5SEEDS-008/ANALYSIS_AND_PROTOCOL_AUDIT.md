# EXP-RESNET50-TRANSFER-5SEEDS-008 — Independent review notes

**Nguồn:** `EXP-RESNET50-TRANSFER-5SEEDS-008_ANALYSIS.zip` + `resnet50_transfer_5seed_008.ipynb` (20/10/2026). Báo cáo này phân biệt **số liệu nguồn** với **suy luận nghiên cứu**; không sửa nhãn hoặc tuyên bố đã chạy lại model binary.

## Các kết luận chắc chắn dựa trên source

- **Pretrained/ImageNet:** `weights="imagenet"` (khác Scratch-008). `pipeline_smoke.json` có `imagenet_initialization=PASS` và `preprocess_max_abs_error=0.0`.
- **Both stages backbone frozen:** `stage1_backbone_trainable_params=0`, `stage2_backbone_trainable_params=0`; Stage 2 chỉ cập nhật head. Từ khóa "transfer" hợp lý, **nhưng "partial fine-tuning" không chính xác cho bản này**.
- **Không biết nhãn từ đầu:** pre-fit validation range **9.23–27.06%**; Accuracy ở batch đầu thấp. End of first epoch Val ~89–92% đến sau ~300 optimization steps, dùng pretrained representations và đầu phân loại mới học.
- **Stage 2 không có lợi ích lớn:** 4/5 seed selected stage2 nhưng best stage2 epoch trong [1,4]; seed 3407 chọn **stage1 epoch10** (`val_loss=0.142581`). Trong 4 seed, last-stage2 val loss rebound từ best 0.0011–0.0037; chưa phải nghiêm trọng.
- **Clean Train Probe:** 500 ảnh cố định, 97.08 ± 0.46% so với Val 95.09 ± 0.22% → gap 1.99 ± 0.32 pp. Không khẳng định không có overfitting/domain shift chỉ từ gap nhỏ.
- **pHash:** 51 cross-split/cross-label candidates; 10 Val paths nghi similar Train. 10/10 đúng mỗi seed; bỏ chúng Val seed3407 còn 1.354/1.420 = 95.35%. Không chứng minh 51 cặp là near-duplicate thật hay không có leakage; phải review ảnh và group by original leaf/session.
- **Domain shift:** Val 95.09%; 35 real_ Val 71.43%; historical external Holdout 77.08%; Test V08 93.24%. Đây là **gap đáng điều tra**, không phải chứng minh lỗi cụ thể do background/lighting khi chưa có xác nhận provenance.
- **Historical Holdout:** 49 ảnh trong archive, 1 ảnh trùng byte Test bị loại → 48 chấm, chỉ 4 true classes; đã được nghiên cứu trước EXP-008 → kết quả là historical descriptive, không phải blind inference. Source `evaluation_lock.json` khóa seed3407 bằng val loss; không chọn lại theo Holdout.

## Tính truy xuất dữ liệu

| Trường / File nguồn | SHA256 |
|---|---|
| ANALYSIS ZIP nguyên bản | `a408af9b540208045f2214c71cb6072b75e8b1442efacd12d88ae510107bc24f` |
| `experiment_config.json` nguyên bản | `167a748a76308806849a5c7dfe9f64e903c766751ca03cd22b9cea3a084debcc` |
| `validation_selection_lock.json` nguyên bản | `aabd917cb021e6591b45350083ff763b6cf26234451d87a7034b28322a1daece` |
| `aggregate/validation_results_5seeds.csv` | `a67f6cc93735b60a6cfba72d84d78cd2fb834c2d671c6f021a63a4af0e5137b0` |
| `aggregate/test_results_5seeds.csv` | `1219d5ba776e9462d6908541360ced0d1dc51565b7041b2098a2dcf2474e6748` |
| `real_holdout/aggregate/real_holdout_results_5seeds.csv` | `49157397b860a8160557263bd5b93d67fb7b77908aa4369f2d2b9a859d619b76` |

**Không nhầm bản gốc với snapshot trên GitHub:** Các CSV chính được nhập lại từ source cho việc lưu báo cáo. Checksum trong bảng là SHA256 **của file trong ZIP gốc**, không phải lời cam kết checksum snapshot đã nhập lại bằng tay phải giống hệt byte.

## Lưu ý phương pháp và sử dụng trong luận văn

- Các số **mean ± sample SD 5 seed** đo biến thiên do seed trên dữ liệu cố định, không phải 95% confidence interval về accuracy deployment ngoài thực địa.
- Accuracy nhóm 35 `real_` được tính theo **tên file**, không xác minh 35 ảnh có thật sự từ vườn/phiên chụp khác với Train.
- Giữ thật rõ: **EXP-RESNET50-SCRATCH-5SEEDS-008** = Scratch V07 / Validation 89.20%; **EXP-RESNET50-TRANSFER-5SEEDS-008** = pretrained ImageNet frozen V08 / Validation 95.09%, Test 93.24%; **đây là các thí nghiệm khác nhau**.
- So sánh ResNet50-008 vs DenseNet121-014 cùng V08 **không cô lập kiến trúc** do DenseNet đã fine-tune `conv5_*`, ResNet vẫn frozen.
- Không thay checkpoint/ảnh nhãn/test hoặc chọn seed sau khi nhìn Holdout. Muốn ablation thực sự cần pre-register đổi duy nhất trainable layers, còn lại giữ theo protocol.
- Khi viết báo cáo cuối, lưu riêng `EXP-008 source archive` và `notebook` vào kho chính thức, chỉ làm claim tái lập binary sau khi hash model thật từ FULL ZIP.

[Experiment report](./README.md) · [Dataset audit snapshot](./dataset_audit_summary.json) · [Preprocessing smoke](./pipeline_smoke.json).

