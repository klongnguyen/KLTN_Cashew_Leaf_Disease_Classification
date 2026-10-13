# KẾ HOẠCH ĐIỀU TRA CHẤT LƯỢNG DATASET VÀ KHẢ NĂNG TỔNG QUÁT HÓA THỰC ĐỊA

**Mã kế hoạch:** `DATASET-INVESTIGATION-001`  
**Ngày lập:** 13/10/2026  
**Trạng thái:** **PLANNED — chưa thực hiện các thí nghiệm đề xuất**  
**Phạm vi:** `CashewData_base` (Kaggle, 4 lớp) ↔ `CashewData_Split (7)` (5 lớp) — nghiên cứu chuyên sâu nguyên nhân suy giảm trên ảnh `real_`.  
**Liên quan:** [Báo cáo IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md) · [Review protocol](./REVIEW_GUIDELINE.md).

> **Quy tắc phân biệt:** Các số liệu hiện có là **kết quả đã đo**, đặc điểm ảnh người dùng quan sát là **bằng chứng quan sát**, còn tác động của background, resize, nhãn chưa chuẩn và domain shift mới là **giả thuyết cần kiểm chứng**. Không kết luận nguyên nhân chỉ từ ảnh dự đoán sai.

## 0. Vấn đề nghiên cứu và hiện trạng

**Mục tiêu tổng quát:** (1) kiểm tra việc làm sạch/bổ sung dữ liệu có khắc phục các lỗi cụ thể của dataset Kaggle hay không; (2) xác định tác nhân có thể giải thích sự giảm chất lượng phân loại trên ảnh thực địa; (3) chứng minh hiệu quả cải tiến bằng đánh giá độc lập và có đối chứng.

### Hiện trạng đóng băng từ các experiment đã chạy

| Chỉ số | BASE001 (Kaggle, 4 lớp) | IMPROVED001 (Split 7, 5 lớp) |
|---|---:|---:|
| Ảnh hợp lệ | 5.360 | 6.907 |
| Test Accuracy | 94,50% | 93,01% |
| Macro F1 | 94,53% | 92,50% |
| Nhóm trùng file SHA256 | 2 | 0 |
| Nhóm trùng file có xung đột nhãn | 1 | 0 |
| Nghi gần trùng khác split (IMPROVED001) | Chưa đưa vào cột so sánh tương đương | 39 cặp pHash |
| Real Train / Validation / Test | Không có phân tích riêng theo `real_` | 170 / 0 / 28 |
| Real Test Accuracy | Không đo riêng | **67,86% = 19/28** |
| Non-real Test Accuracy | Không đo riêng | **94,08% = 620/659** |

**Nguyên tắc giải thích:** Hai Test Set khác nhau về số lớp, dữ liệu và phép chia → **không được lấy chênh lệch Accuracy tổng làm bằng chứng dataset cải tiến kém/tốt hơn**. Các ảnh mang tiền tố `real_` là chỉ báo nguồn dữ liệu dựa trên tên file, chưa thay thế provenance metadata.

**Bối cảnh ảnh thực địa đã được người thu thập mô tả:** ảnh điện thoại thường 3024×4032, có đất, cành cây, lá khác và ánh sáng không đồng đều. Kích thước gốc lớn và có nền thực tế **không phải tự nó là lỗi dữ liệu**. Giả thuyết cần kiểm chứng là cách xử lý và mức độ đại diện trong huấn luyện ảnh hưởng ra sao.

**Ranh giới version:** `CashewData_Split (7)` lần này **6.907 ảnh**; repository đang có `Cashew_dataV05` **6.911 ảnh**. Khác snapshot, không được tráo file/split vào thí nghiệm hiện tại hoặc ghi đè kết quả V05.

## 1. Câu hỏi nghiên cứu và giả thuyết

| ID | Câu hỏi / giả thuyết | Loại bằng chứng cần có |
|---|---|---|
| RQ1 | Bộ cải tiến đã khắc phục trùng ảnh và xung đột nhãn của Kaggle chưa? | Exact hash, pHash review, reviewer độc lập, đối chiếu tỷ lệ |
| RQ2 | Ảnh Real lỗi nhiều hơn do **nhãn** hay do mô hình? | Blind label review + phân tích đúng/sai ở 28 ảnh |
| H1 | Model nhạy với **background**, học tín hiệu nền thay vì triệu chứng | Controlled background-only ablation trên Validation, kiểm chứng với Grad-CAM |
| H2 | Resize 3024×4032 → 224×224 làm méo lá/giảm chi tiết lesion nhỏ | Direct resize vs letterbox/ROI khi giữ điều kiện khác cố định |
| H3 | Thiếu đại diện Real Train/Validation gây hạn chế generalization/checkpoint selection | Real-Val mới, per-class Recall/Macro F1, đối chứng |
| H4 | Model thiếu ổn định do random seed / overfitting | Lặp nhiều seed cùng split, learning curves, mean±SD |
| RQ3 | Improved train data có nâng hiệu quả khi so trên Real Holdout chung hay không? | Paired evaluation 4-class, cùng tập độc lập, không leak |

## 2. Nguyên tắc thí nghiệm bắt buộc

1. **Đóng băng** cấu hình, manifest, checkpoint và chỉ số `BASE001`, `IMPROVED001`. Ghi checksum ZIP, ID, seed, số lớp và Train/Val/Test. Không sửa lại số liệu đã công bố.
2. **Không dùng Test Set để tuning/chọn seed/chọn crop/chọn model.** Cụ thể, 9 ảnh Test đã thấy lỗi chỉ để phân tích hậu nghiệm; mọi thiết kế sửa đổi phải được đánh giá/chọn trên **Train/Validation hoặc ảnh mới**.
3. **Giữ nguyên 687 ảnh Test hiện có**, bao gồm 28 ảnh `real_`. Nếu xác nhận có leakage, lập version mới và benchmark mới, **không âm thầm thay Test**.
4. **Cùng lá/cây/phiên chụp là một nhóm** khi xây dựng tập mới; ưu tiên group split từ `leaf_id`, `tree_id`, `session_id`. Nếu metadata không có, ghi `unknown`, không tuyên bố sạch leakage.
5. Các ablation so sánh **một thay đổi một lần**; cố định architecture, seed, optimizer, LR, batch, epochs, callbacks, split, augmentation và cách tính metric; mọi ngoại lệ phải khai báo.
6. Cố định metric chính **Macro F1 trên Real Validation**; metric phụ: Balanced Accuracy, Accuracy, per-class Recall, Healthy false positives, Not-Cashew false rejection và khoảng tin cậy nếu đủ nhóm độc lập.
7. Khi Test 4-class cuối, dùng cùng **Real Holdout hoàn toàn mới** cho mô hình Base-4class và Improved-4class. Không chỉ loại `not_cashew_leaf` khỏi Test của mô hình 5 lớp rồi gọi là công bằng.
8. Luôn phân loại phát hiện theo 3 mức: **ĐÃ XÁC NHẬN / NGHI VẤN CẦN KIỂM CHỨNG / CHƯA ĐỦ BẰNG CHỨNG**.

## 3. Luồng công việc

```text
BASE001 + IMPROVED001 (giữ nguyên benchmark đã chạy)
             |
             v
[WP0] Freeze provenance, manifest, checkpoint, protocol
             |
             v
[WP1] Review 28 Real Test (9 sai + 19 đúng) / gắn nhãn nguyên nhân
             |             \
             |              \> [WP2] Review nghi gần trùng & leakage
             v
[WP3] Tạo Real Validation độc lập từ Train/ảnh mới, group-aware
             |
             v
[WP4] Thí nghiệm có đối chứng: resize / ROI / background
             |
             v
[WP5] Bổ sung real-train đúng bệnh/đa dạng + 3–5 seeds
             |
             v
[WP6] FINAL: chung 4 lớp + Real Holdout mới, độc lập hoàn toàn
             |
             v
[WP7] Báo cáo nguyên nhân, độ chắc chắn và giới hạn
```

### WP0 — Khóa đối tượng và bộ bằng chứng (P0)

**Làm thế nào:** Lưu riêng `raw_zip_sha256`, `experiment_config.json`, `split_manifest.csv` (ảnh/nhãn/split/`real`/group metadata), checkpoint, lịch sử fit; đánh dấu nguồn gốc từng file, kể cả trường hợp không xác định. Đối chiếu `CashewData_Split (7)` với `Cashew_dataV05` trước mọi chuyển đổi.

**Mục tiêu:** Có chứng cứ để tái lập, xác định chính xác ảnh nào được đánh giá, tránh so sánh nhầm version.

**Phải kết luận:** Đã khóa và định danh **những gì**? Có đủ metadata nhóm lá/cây/phiên chụp không? **Tiêu chí hoàn tất:** Manifest có đủ 4.825/1.395/687 và checksum được lưu; không ghi đè bản gốc.

**Đầu ra:** `protocol_lock.md`, `dataset_manifest.csv` (metadata, không chứa ảnh gốc trên GitHub).

### WP1 — Điều tra 28 ảnh Real Test (P0)

**Làm thế nào:**
- Review **cả 9 ảnh sai và 19 ảnh đúng**. Reviewer kiểm nhãn **trước khi** xem dự đoán/confidence; reviewer thứ hai (nếu có) kiểm độc lập; bất đồng chuyển cho chuyên môn bệnh cây.
- Quan sát từng ảnh ở (A) bản gốc 3024×4032; (B) bản đầu vào model sau **direct resize** 224×224; (C) bản chỉ đánh dấu ROI lá để quan sát, **không** dùng ảnh Test ROI để chọn preprocessing.
- Ghi: xác minh nhãn, nhiều bệnh hay không, mức tổn thương, tỷ lệ diện tích lá chính, phức tạp background, ánh sáng, blur, chi tiết còn sau resize; gán tối đa nhiều nhãn **nguyên nhân nghi vấn** thay vì ép một nguyên nhân.
- So sánh phân bố thuộc tính của **nhóm đúng và nhóm sai** trên toàn 28 ảnh. Nếu cả hai nhóm đều nền phức tạp, không quy lỗi trực tiếp cho nền.

**Mục tiêu:** Phân biệt khả năng nhãn không nhất quán, triệu chứng quá nhỏ/mơ hồ, lỗi từ chối và ảnh hưởng nghi vấn của nền/resize.

**Phải kết luận:** Bao nhiêu ảnh **được chuyên gia xác nhận nhãn đúng/sai/không chắc**; bao nhiêu ảnh có nền phức tạp nhưng đúng/sai; bao nhiêu ảnh mất khả năng quan sát lesion sau resize; xác định giả thuyết kiểm chứng tiếp. Không sửa dữ liệu Test đã khóa để báo điểm mới.

**Đầu ra:** [Protocol review](./REVIEW_GUIDELINE.md), [mẫu CSV review ảnh](./templates/real_image_review.csv), `real_review_summary.md`; chú ý các lỗi Healthy→Leaf Miner, Red Rust→Anthracnose và Real→Not Cashew.

### WP2 — Review 39 cặp nghi gần trùng xuyên split (P0)

**Làm thế nào:** Từ `near_duplicates_cross_split_review.csv`, hiển thị từng cặp side-by-side, kiểm SHA256, pHash, hình dạng lá, vân lá, vết bệnh, góc chụp, nền, EXIF/source nếu có. Đánh dấu `same_file_variant` / `same_leaf_or_session` / `different_subject` / `uncertain`; ưu tiên 15 cặp khác nhãn và 16 nhóm similarity liên-split (các tập này có thể chồng lên nhau, không cộng thành số cặp độc lập).

**Mục tiêu:** Xác định dấu hiệu leakage và nhãn xung đột **thật**, không coi pHash≤4 là chứng minh trùng.

**Phải kết luận:** Tổng cặp đã kiểm, số xác nhận cùng lá/biến thể, số không trùng, số không chắc; các split bị ảnh hưởng. Nếu có leakage, tạo `dataset_vNext` và benchmark mới sau khi group split lại từ dữ liệu phát triển, giữ rõ lịch sử benchmark cũ.

**Đầu ra:** [CSV review các cặp](./templates/duplicate_pair_review.csv) và `duplicate_audit_report.md`.

### WP3 — Tạo Real Validation độc lập (P1, sau WP1–WP2)

**Làm thế nào:** Chỉ chọn từ **170 Real Train** hiện tại hoặc thu thêm ảnh mới **chưa ở Test**, group-aware theo lá/cây/session; giữ 28 Real Test và 687 Test không đổi. Phân tầng 4 lớp, ưu tiên bổ sung Healthy (Real Train 14), Red Rust (23), Anthracnose (44); báo thiếu mẫu nếu không thể có nhóm độc lập. Phiên bản dataset mới cần manifest/hash riêng.

**Mục tiêu:** Có Validation đại diện domain thật để so checkpoint/preprocessing, vì Real Validation trong IMPROVED001 = **0**.

**Phải kết luận:** Mỗi lớp có bao nhiêu ảnh và **bao nhiêu nhóm độc lập** ở Real Train/Real Val; phân bố điều kiện ánh sáng/nền; class imbalance còn mức nào. Chọn checkpoint bằng Val đã định trước, không dựa Test.

**Đầu ra:** `real_val_manifest.csv`, `split_audit.json`, `dataset_vNext_changelog.md`.

### WP4 — Kiểm chứng Resize, ROI và Background (P1)

**Thiết kế thí nghiệm chính (cố định Train + Validation sau WP3):**

| ID đề xuất | Nhánh | Một thay đổi kiểm chứng | Dữ liệu chọn phương án |
|---|---|---|---|
| A0 | Control | Resize thẳng mọi ảnh về 224×224 (baseline) | Real Val |
| A1 | Ratio | Giữ tỷ lệ, letterbox/padding thành 224×224 | Real Val |
| A2 | ROI | Crop lá chính bằng ROI chuẩn hóa, sau đó resize | Real Val |
| B1 | Background-only | Giữ nguyên lá chính và scale; thay/làm mờ nền, tránh thay đổi lesion | Real Val |

**Làm thế nào:** Preprocessing phải nhất quán giữa Train và Validation của từng nhánh; kiểm tra ước lượng crop/mask không dùng nhãn dự đoán của Test. Với A1, ảnh tỷ lệ 3:4 khi giữ nguyên sẽ hiển thị gần **168×224** và padding đối xứng đến 224×224. Với B1, mask lá chính cần chất lượng đủ tốt, cùng vị trí/kích thước; nên báo thêm tác dụng phụ của viền mask. Grad-CAM chỉ là **giải thích hỗ trợ**, không đủ chứng minh phụ thuộc vào nền.

**Mục tiêu:** Tách dần tác động biến dạng/thu nhỏ lesion khỏi tác động nền ảnh; không quy công cải thiện ROI hoàn toàn cho việc bỏ nền vì crop còn thay scale.

**Phải kết luận:** Thay đổi Macro F1, Balanced Acc, per-class Recall trên **Real Val**, mức biến thiên giữa các seed, và ảnh minh họa cùng mẫu; có đủ cơ sở gọi background là một yếu tố đóng góp hay chưa.

**Tiêu chí quyết định đặt trước khi chạy:** Ưu tiên Macro F1 trên Real Val; không chọn phương án chỉ nhờ tăng một lớp nhưng làm giảm nghiêm trọng lớp khác. **Không có một ngưỡng phần trăm nào được chứng minh sẵn**; nếu đặt ngưỡng kỹ thuật (ví dụ ≥3 điểm phần trăm) cần công bố trước, kèm CI và kích thước tập.

**Đầu ra:** `ablation_results.csv`, `ablation_comparison.png`, `gradcam_examples/`, `preprocessing_protocol.md`.

### WP5 — Tăng độ đa dạng Real Train và đánh giá nhiều seeds (P1–P2)

**Làm thế nào:** Thu thập thêm **lá/cây/phiên chụp mới**, nhiều ánh sáng, nền đất/cành, mật độ lá, mức bệnh, giai đoạn bệnh và thiết bị; ưu tiên Healthy/Red Rust/Anthracnose thay vì tăng thêm nhiều ảnh tương tự cùng lá. Xác minh nhãn trước khi train. Sau khi freeze protocol, chạy 3–5 seed (gợi ý cùng seed framework: 42, 123, 2026, 3407, 7777) và giữ fixed split.

**Mục tiêu:** Xem đa dạng đối tượng thực địa và cân bằng lớp có cải thiện Real Validation/generalization hay không và kết quả có ổn định giữa seed hay không.

**Phải kết luận:** Có cải thiện nhất quán ở Macro F1 và Recall/nhóm bệnh trên Real Val? Mức dao động mean±SD? Mức độ tăng số **nhóm lá/cây độc lập** chứ không chỉ số ảnh?

**Đầu ra:** `real_collection_log.csv`, `seed_summary.csv`, learning curves, class metrics; không thay đổi locked Test để thử nghiệm vòng lặp.

### WP6 — So sánh cuối cùng có kiểm soát bằng Real Holdout chung (P2)

**Làm thế nào:**
1. Thiết lập **một Real Holdout mới độc lập** (lá/cây/phiên mới), không xuất hiện trong Train/Val của **bất kỳ** nhánh Base hoặc Improved. Tập 28 ảnh Real Test đã dùng để phân tích lỗi **không được lấy làm tiêu chuẩn chọn bản cải tiến**.
2. Huấn luyện `BASE-4CLASS` và `IMPROVED-4CLASS` riêng, thống nhất bốn lớp `anthracnose`/`healthy`/`leaf_miner`/`red_rust`, DenseNet121 Scratch và protocol phù hợp. Một mô hình 5 lớp không tự tương đương mô hình 4 lớp chỉ bằng bỏ cột trong báo cáo.
3. Cố định cấu hình/seed/selection rule theo Validation, sau đó **đánh giá một lần cuối** trên cùng Real Holdout, tính paired accuracy/Macro F1, per-class Recall và confusion matrix, ghi CI (ưu tiên bootstrap theo nhóm lá/cây nếu có metadata).
4. Lớp `not_cashew_leaf` là **bài toán mở rộng riêng**, dùng một tập out-of-scope độc lập và báo false rejection; không trộn làm thay đổi bài toán so sánh bốn lớp.

**Mục tiêu:** Trả lời liệu dataset cải tiến thật sự nâng khả năng tổng quát hóa trên ảnh vườn; tránh kết luận từ hai Test Set/số lớp khác nhau.

**Phải kết luận:** Chênh lệch trên **cùng một nhóm ảnh**, độ bất định, nhóm bệnh có lợi/hại, độ ổn định theo seeds. Nếu CI rộng hoặc ít nhóm độc lập → **chưa đủ bằng chứng**, không viết cải thiện chắc chắn.

**Đầu ra:** `final_real_holdout_manifest.csv`, `paired_predictions.csv`, `final_comparison.md`.

### WP7 — Tổng hợp bằng chứng thành phần báo cáo khóa luận (P2)

**Phải có:** Bảng trước/sau làm sạch, quy trình split/leakage, confusion matrix từng model, 28 real review (chỉ thống kê aggregate khi công bố), ablation controlled, paired Holdout comparison, mean±SD, giới hạn, đề xuất còn tồn tại.

**Mẫu diễn đạt kết luận:**
- **Đã xác nhận:** `IMPROVED001` có 0 nhóm file trùng byte SHA256 trong audit; trên Real Test 28 ảnh accuracy 67,86%; Real Val ban đầu không có ảnh.
- **Nghi vấn:** ảnh nền phức tạp và resize có thể tác động; cần chứng minh bằng A1/A2/B1 và phân tích trên Val.
- **Chưa kết luận:** bộ Improved hiệu quả hơn bộ Kaggle về generalization cho tới khi WP6 hoàn tất.

**Đầu ra:** `DATASET_INVESTIGATION_FINAL_REPORT.md` và section báo cáo “Dataset quality, Domain Shift and Generalization”.

## 4. Thứ tự, phụ thuộc, ưu tiên và Definition of Done

| WP | Ưu tiên | Đầu vào bắt buộc | Hoàn thành khi |
|---|---|---|---|
| WP0 | P0 | 2 source archive, metadata, split | Có manifest/checksum, protocol khóa |
| WP1 | P0 | 28 ảnh Real + predictions | **28/28** được review; nhãn và mức bất định rõ |
| WP2 | P0 | 39 pHash cross-split pairs | **39/39** cặp đã được xác minh/phân loại |
| WP3 | P1 | WP0, WP2, Real Train/ảnh mới | Real Val độc lập, report counts/groups |
| WP4 | P1 | WP3 | Có A0/A1/A2/B1, so sánh Real Val |
| WP5 | P1–P2 | WP3, WP4 và ảnh bổ sung | Có collection log + mean±SD nhiều seed |
| WP6 | P2 | WP4/WP5 freeze, Holdout mới | Hai nhánh 4 lớp chấm cùng Holdout |
| WP7 | P2 | WP0–WP6 | Có báo cáo kết luận kèm mức độ bằng chứng |

**Phạm vi thực thi hiện tại:** mới **lập kế hoạch**; chưa review chuyên môn 28 ảnh, chưa xác minh 39 cặp, chưa tạo Real Val và chưa chạy các ablation trong tài liệu này.

## 5. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Tác động | Kiểm soát |
|---|---|---|
| Tuning theo các ảnh Real Test đã xem | Test optimism / đánh giá thiếu khách quan | Chỉ forensic review; mọi lựa chọn trên Real Val hoặc dữ liệu mới |
| Cùng lá xuất hiện nhiều góc/phiên | Leakage và CI quá lạc quan | Group split; group-aware bootstrap; ghi metadata nguồn |
| Không đủ Real Healthy/Red Rust | Macro F1 Real Val dao động mạnh | Bổ sung nhiều cây/lá mới, nêu cỡ mẫu và CI |
| Mask/crop biến đổi cả kích thước lesion | Quy nguyên nhân background sai | Thêm background-only intervention với scale lá giữ nguyên |
| Nhãn đa bệnh nhưng classification một nhãn | Model có thể “sai” do định nghĩa bài toán | Hội đồng review, báo ambiguous/multi-label, không sửa Test tùy tiện |
| Nhầm dataset 6.907 với V05 6.911 | So sánh sai version | Hash/manifests, không ghi đè benchmark chính |
| pHash≤4 sinh false positive | Gọi nhầm leakage | Review hình và nguồn; trạng thái uncertain |

## 6. Tài liệu và biểu mẫu

- [Báo cáo phát hiện hiện tại của IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md)
- [Quy trình review thủ công, định nghĩa các mức đánh giá](./REVIEW_GUIDELINE.md)
- [Biểu mẫu CSV review 28 Real Test](./templates/real_image_review.csv)
- [Biểu mẫu CSV review 39 cặp pHash](./templates/duplicate_pair_review.csv)
- [9 Real Test đã dự đoán sai (để truy vết, không để tune)](./EXP-DENSENET121-SCRATCH-IMPROVED-001/REAL_TEST_ERRORS.md)

**Nguyên tắc cuối:** “Dataset sạch hơn” (ít duplicate/sai nhãn) và “mô hình tổng quát hóa tốt hơn” là **hai luận điểm khác nhau**. Phải đánh giá từng luận điểm bằng dữ liệu và protocol thích hợp.
