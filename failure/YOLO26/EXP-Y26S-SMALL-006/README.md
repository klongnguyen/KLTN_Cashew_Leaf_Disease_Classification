# EXP-Y26S-SMALL-006 — YOLO26s Selective Clear-Lesion Annotation

> **Source archive:** `EXP-Y26S-SMALL-006.zip`  
> **Model:** YOLO26s  
> **Task:** Direct disease object detection  
> **Classes:** `anthracnose`, `leaf_miner`, `red_rust`

## 1. Mục tiêu của Take 006

Take 006 thay đổi annotation policy so với Take 005:

- **không cố bounding mọi chấm bệnh quá nhỏ**;
- ưu tiên lesion **rõ ràng, đủ lớn và có ý nghĩa thị giác**;
- với Anthracnose, giảm số box cho các chấm đen rất nhỏ/khó học;
- với Red Rust, tiếp tục không cố annotation exhaustively toàn bộ hàng loạt chấm nhỏ;
- giữ YOLO26s, input `640×640` và các hyperparameter chính gần như Take 005.

Mục tiêu là kiểm tra liệu supervision sạch hơn, ít object cực nhỏ hơn có giúp detector ổn định hơn hay không.

> ⚠️ **Traceability warning:** archive bên ngoài là `EXP-Y26S-SMALL-006`, nhưng `experiment_config.json`, `metrics_summary.json` và tên checkpoint bên trong vẫn ghi `EXP-Y26S-SMALL-005`. Kết quả trong thư mục này được quản lý là **Take 006** theo tên archive. Cần sửa `EXPERIMENT_ID` trước lần train tiếp theo.

## 2. Dataset statistics

| Split | Images | Images with boxes | Boxes | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|---:|
| Train | 441 | 441 | 12,781 | 3,169 | 643 | 8,969 |
| Validation | 42 | 40 | 1,108 | 236 | 65 | 807 |
| Test | 20 | 20 | 491 | 79 | 25 | 387 |

Annotation density: Train **28.98**, Validation **26.38**, Test **24.55 boxes/image**. So với Take 005, Train giảm từ **32.63 → 28.98 boxes/image**.

Train box distribution: Anthracnose **24.8%**, Leaf Miner **5.0%**, Red Rust **70.2%**. Class imbalance vẫn rất lớn: Red Rust có khoảng **14 lần** số box của Leaf Miner.

## 3. Kết quả Test tổng thể

| Metric | Take 005 | Take 006 | Delta |
|---|---:|---:|---:|
| Precision | 0.6929 | **0.7473** | **+0.0544** |
| Recall | 0.6506 | **0.6648** | **+0.0142** |
| F1 | 0.6711 | **0.7036** | **+0.0325** |
| mAP@0.50 | 0.6609 | **0.7298** | **+0.0689** |
| mAP@0.75 | 0.2793 | **0.3577** | **+0.0784** |
| mAP@0.50:0.95 | 0.3405 | **0.3877** | **+0.0472** |
| Mean IoU (matched TP) | **0.7629** | 0.7603 | -0.0026 |
| Median IoU (matched TP) | **0.7741** | 0.7720 | -0.0021 |

**Take 006 cải thiện rõ Precision, F1 và mAP trong khi IoU của các detection đúng gần như giữ nguyên.** Đây là tín hiệu tích cực cho annotation policy mới: giảm bớt object quá nhỏ/khó học không làm localization của TP xấu đi, nhưng giúp chất lượng detection tổng thể tăng.

> ⚠️ Take 005 và Take 006 **không phải ablation hoàn toàn sạch** vì số ảnh và số box ở từng split cũng thay đổi. Vì vậy không được quy toàn bộ mức tăng chỉ cho annotation policy.

## 4. Custom IoU evaluation

Tại `confidence=0.25`, `IoU≥0.50`:

| Metric | Take 005 | Take 006 | Delta |
|---|---:|---:|---:|
| TP | 357 | 356 | -1 |
| FP | 246 | **230** | -16 |
| FN | 174 | **135** | -39 |
| Precision | 0.5920 | **0.6075** | +0.0155 |
| Recall | 0.6723 | **0.7251** | +0.0527 |
| F1 | 0.6296 | **0.6611** | +0.0315 |
| Mean IoU TP | **0.7629** | 0.7603 | -0.0026 |

Take 006 giảm cả FP và FN trong custom matching. Tuy nhiên số GT Test cũng thay đổi (`531 → 491 boxes`), nên raw TP/FP/FN chỉ dùng để theo dõi tiến trình chứ không phải so sánh tuyệt đối.

## 5. Kết quả theo từng class

| Class | Metric | Take 005 | Take 006 | Delta |
|---|---|---:|---:|---:|
| Anthracnose | Precision | **0.7233** | 0.6934 | -0.0299 |
|  | Recall | 0.5425 | **0.5949** | +0.0524 |
|  | F1 | 0.6200 | **0.6404** | +0.0204 |
|  | mAP50 | 0.6203 | **0.6666** | +0.0463 |
|  | mAP50-95 | 0.2636 | **0.2847** | +0.0211 |
| Leaf Miner | Precision | 0.8229 | **0.8571** | +0.0342 |
|  | Recall | 0.6000 | **0.7198** | +0.1198 |
|  | F1 | 0.6940 | **0.7825** | +0.0885 |
|  | mAP50 | 0.6584 | **0.7839** | +0.1255 |
|  | mAP50-95 | 0.4012 | **0.5108** | +0.1096 |
| Red Rust | Precision | 0.5326 | **0.6916** | **+0.1590** |
|  | Recall | **0.8094** | 0.6796 | **-0.1298** |
|  | F1 | 0.6424 | **0.6855** | +0.0431 |
|  | mAP50 | 0.7040 | **0.7388** | +0.0348 |
|  | mAP50-95 | 0.3566 | **0.3675** | +0.0109 |

### Diễn giải

**Anthracnose:** giảm annotation cho các chấm đen cực nhỏ tạo trade-off hợp lý: Precision giảm nhẹ nhưng Recall, F1 và mAP tăng.

**Leaf Miner:** là class cải thiện mạnh nhất, nhưng Train tăng từ `519 → 643 boxes` và Test chỉ có **25 GT boxes**, nên chưa thể quy toàn bộ mức tăng cho annotation policy.

**Red Rust:** Precision tăng mạnh `0.5326 → 0.6916`, Recall giảm `0.8094 → 0.6796`, nhưng F1 và mAP vẫn tăng. Đây là **precision–recall trade-off**: prediction sạch hơn nhưng bỏ sót nhiều lesion hơn.

## 6. Confusion matrix và background FP

Normalized confusion matrix của Take 006 cho thấy:

- True Anthracnose: khoảng **86%** được nhận thành Anthracnose;
- True Leaf Miner: khoảng **68%** được nhận đúng; khoảng **28%** bị nhầm thành Anthracnose;
- True Red Rust: khoảng **93%** được nhận thành Red Rust;
- trong cột background, Red Rust chiếm phần lớn background false-positive entries (~`0.60`), tiếp theo Anthracnose (~`0.29`).

Do đó **background FP vẫn là bottleneck**, đặc biệt với Red Rust.

## 7. F1–Confidence analysis

Test F1 curve đạt xấp xỉ **F1 ≈ 0.70 tại confidence ≈ 0.372**. Không được dùng Test để chọn threshold deployment; threshold cuối phải được chọn trên Validation, khóa lại, rồi mới đánh giá Test.

## 8. Training behavior

Training chạy đủ **50 epochs**. Validation best `mAP50 ≈ 0.6895` tại epoch **44** và best `mAP50-95 ≈ 0.3455` tại epoch **46**. Test `mAP50 = 0.7298`, cao hơn validation peak; cần giữ nguyên split ở các take tiếp theo để so sánh công bằng.

## 9. Điểm cải thiện chính so với Take 005

1. Precision tăng `0.6929 → 0.7473`.
2. F1 tăng `0.6711 → 0.7036`.
3. mAP50 tăng `0.6609 → 0.7298`.
4. mAP50-95 tăng `0.3405 → 0.3877`.
5. Custom Recall tăng và FN giảm.
6. IoU gần như không đổi quanh `0.76`.
7. Annotation density giảm, phù hợp hơn với mục tiêu “lesion đủ rõ và đủ lớn để detector học”.

Take 006 hiện là **kết quả YOLO tốt hơn Take 005 về metric tổng thể**.

## 10. Hạn chế còn lại

- **Class imbalance:** Red Rust 8,969 boxes (**70.2%**), Anthracnose 3,169 (**24.8%**), Leaf Miner 643 (**5.0%**).
- **Thiếu negative images trong Train:** `441/441` ảnh Train đều có box; Validation chỉ có 2 ảnh không box; Test `20/20` đều có box.
- **Red Rust mất Recall:** giảm gần 13 điểm phần trăm.
- **Annotation policy vẫn cần định lượng:** cần quy định khi nào gộp cluster, kích thước lesion tối thiểu, cách xử lý lesion sát nhau và mức độ exhaustive nhất quán.
- **Split thay đổi giữa Take 005/006:** đây là progress comparison, chưa phải controlled ablation.
- **Traceability metadata sai ID:** các file nội bộ vẫn ghi `EXP-Y26S-SMALL-005`.

## 11. Kết luận

Take 006 cho thấy chiến lược **ưu tiên lesion rõ, đủ lớn và có giá trị học thay vì bounding mọi chấm nhỏ** đang phù hợp hơn Take 005.

Kết quả tổng thể: `F1 = 0.7036`, `mAP50 = 0.7298`, `mAP50-95 = 0.3877`, Mean IoU ≈ `0.76`.

Chưa nên xem là final detector vì Red Rust vẫn chiếm phần lớn annotation, Leaf Miner còn ít dữ liệu, Train chưa có negative images, Red Rust Recall giảm, split chưa cố định và threshold chưa được chọn bằng Validation.

**Đề xuất Take 007:** giữ annotation policy của Take 006, khóa split, bổ sung negative images, tăng dữ liệu Leaf Miner, giữ YOLO26s + 640 trước khi thử tăng resolution/model size.
