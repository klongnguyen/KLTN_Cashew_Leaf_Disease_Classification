# Đánh giá kết quả phân loại mới nhất trên Cashew_dataV05

Ngày rà soát: 2026-10-05. Nguồn: [thư mục kết quả trên Google Drive](https://drive.google.com/drive/folders/1DGKTnzsvVc60RIEn5lQgflR3s-loM24G). Báo cáo này đánh giá ba lần chạy scratch mới nhất trong thư mục: [DenseNet121-005](https://drive.google.com/drive/folders/1L7eSB4LGdn5-BuP8TJ9na2hR1sPKgzV1), [ResNet50-005](https://drive.google.com/drive/folders/13XL5dy2-q7Wqf4O1TNPeyJuHBYAOsmcP) và [ViT-004](https://drive.google.com/drive/folders/19yy8LGAT0HY2C27fjZbA1JlgIHgS0NFf). Đây là kết quả **Validation tại checkpoint có `val_loss` thấp nhất**, chưa phải kết quả Test.

## 1. Khả năng so sánh và phạm vi dữ liệu

- Cả ba cấu hình ghi `Cashew_dataV05`, ảnh đầu vào 224 × 224, batch 32, năm seed `42, 123, 2026, 3407, 7777`, học từ đầu và không dùng pretrained weights. Tập dữ liệu có 4.822 Train, 1.402 Validation, 687 Test; thứ tự lớp là `anthracnose`, `healthy`, `leaf_miner`, `not_cashew_leaf`, `red_rust`. Nguồn: [cấu hình DenseNet](https://drive.google.com/file/d/1HN_Pxxn-Tucd0vXhNsskEoqbjA5IlOV5/view), [ResNet](https://drive.google.com/file/d/1tHW97bO9D21_owJs-M-LxPCUsxgG5-Js/view), [ViT](https://drive.google.com/file/d/1BEW2gacLypV3KVHeaZrrhYW4h0enmazF/view).
- Đã so sánh cả 6.911 dòng `dataset_manifest.csv` của ba lần chạy. Sau khi bỏ phần tiền tố đường dẫn Colab khác nhau, các bộ `(split, class, label, đường dẫn tương đối)` trùng nhau theo từng dòng; không có dòng lặp trong mỗi manifest. Kiểm tra này xác nhận **thành viên split được ghi nhận** giống nhau, chưa xác minh độc lập byte ảnh hay ảnh trùng nội dung dưới tên khác. Nguồn: [manifest DenseNet](https://drive.google.com/file/d/1BNwpb_sr301J6JKarRvt1RhsU8Z7dO1T/view), [ResNet](https://drive.google.com/file/d/1sInnPq8uFmEVzyEryFJoaVmK-YuR3Piz/view), [ViT](https://drive.google.com/file/d/1BzZnr4FIaIvl5Dj7Bv-iOe5Gy3VNMoAY/view).
- DenseNet121 và ResNet50 dùng Adam, LR ban đầu `0,001`, tối đa 50 epoch; ViT dùng AdamW, LR `0,0003`, weight decay `0,0001`, tối đa 60 epoch. Số tham số lần lượt khoảng 7,57 triệu; 24,64 triệu; 0,348 triệu. Vì kiến trúc, số tham số và công thức huấn luyện khác nhau, bảng dưới đây mô tả hiệu quả của **từng cấu hình chạy**, không chứng minh riêng hiệu ứng kiến trúc.
- Ba cấu hình đều đặt `run_final_test=false`; DenseNet và ViT còn ghi `final_test_completed=false`. ResNet ghi `Status: INCOMPLETE` và `test_result: null` trong các file metrics. Không có số Test để kết luận khả năng tổng quát hóa cuối cùng. Nguồn: [README ResNet](https://drive.google.com/file/d/1562o5E5p0G5v_ygEpkQ3v63cVImB1Th9/view), [metrics ResNet seed 42](https://drive.google.com/file/d/1mdb3XdDTXhbvbOnwYjIpwvlnfewvK0at/view).

Lưu ý về nguồn gốc: thư mục local `training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-004` trong repo ghi **Cashew_dataV04**, còn [Drive ViT-004](https://drive.google.com/file/d/1BEW2gacLypV3KVHeaZrrhYW4h0enmazF/view) ghi **Cashew_dataV05**. Hai bộ này trùng experiment ID nhưng khác phiên bản dữ liệu và số liệu. Không ghép hoặc ghi đè chúng chỉ theo ID.

## 2. Kết quả Validation của năm seed

Trung bình ± độ lệch chuẩn mẫu (`ddof=1`), tính từ checkpoint được lưu theo `min(val_loss)`:

| Mô hình / experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy | Khoảng Macro F1 theo seed |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet121 scratch / `005` | **94,58 ± 0,95%** | **94,71 ± 0,81%** | **94,24 ± 1,02%** | **94,43 ± 0,93%** | **94,24 ± 1,02%** | 92,88–95,25% |
| ResNet50 scratch / `005` | 92,43 ± 1,94% | 92,76 ± 1,99% | 92,09 ± 1,95% | 92,30 ± 1,98% | 92,09 ± 1,95% | 89,73–94,29% |
| Compact ViT scratch / `004` | 88,33 ± 1,44% | 88,41 ± 1,55% | 87,53 ± 1,50% | 87,81 ± 1,52% | 87,53 ± 1,50% | 86,19–90,16% |

Nguồn bảng tổng hợp: [DenseNet](https://drive.google.com/file/d/1F0BJ5NvBXSIpmrdTC7uzxdejnNkstVaP/view), [ResNet](https://drive.google.com/file/d/1mb5AriAODvrv56uMl9N3RQG9BmBF2G4c/view), [ViT](https://drive.google.com/file/d/1zlAZ5JT1Up-FH4rTUyGuRXMHrCowVY1A/view). Khoảng Macro F1 được tính lại từ [năm seed DenseNet](https://drive.google.com/file/d/1eAkp47ILkn9gKy7WJhDsz9q8np7SKwQN/view), [ResNet](https://drive.google.com/file/d/1xzOe_d3eicPOgZadxPG1Zm5e-vdWHcKI/view) và [ViT](https://drive.google.com/file/d/1MJzlufx14VcqaEo3GzwjATH-ssS2zVYK/view).

DenseNet đứng đầu cả Accuracy và Macro F1: hơn ResNet lần lượt **2,15** và **2,13 điểm phần trăm (pp)**; hơn ViT **6,25** và **6,62 pp**. Thứ tự DenseNet > ResNet > ViT lặp lại ở cả năm seed khi so Macro F1 theo cặp cùng seed. Tuy nhiên, năm seed cùng dùng một tập Validation 1.402 ảnh, nên đây không phải năm tập kiểm thử độc lập; chênh lệch là mô tả trên split hiện tại, chưa phải bằng chứng về độ hơn kém trên dữ liệu triển khai.

| Seed | DenseNet Acc / F1 | ResNet Acc / F1 | ViT Acc / F1 |
|---:|---:|---:|---:|
| 42 | 94,44 / 94,28% | 94,22 / 94,20% | 88,94 / 88,35% |
| 123 | 95,01 / 94,81% | 92,30 / 92,19% | 86,80 / 86,19% |
| 2026 | 95,44 / 95,25% | 89,87 / 89,73% | 87,80 / 87,31% |
| 3407 | 95,01 / 94,94% | 94,44 / 94,29% | 90,51 / 90,16% |
| 7777 | 93,01 / 92,88% | 91,30 / 91,11% | 87,59 / 87,05% |

DenseNet có độ lệch chuẩn Macro F1 nhỏ nhất (**0,93 pp**). ResNet dao động nhiều nhất (**1,98 pp**), đặc biệt seed `2026` thấp hơn seed `3407` **4,56 pp** Macro F1. Nếu chọn một seed để đóng gói theo Validation Macro F1, DenseNet chọn `2026`, ViT chọn `3407`; ResNet sẽ chọn `3407` **nếu áp dụng cùng tiêu chí**, vì cấu hình ResNet không ghi quy tắc chọn deployment seed riêng. Không dùng Test để chọn seed.

## 3. Hiệu quả theo lớp và kiểu nhầm lẫn

Macro F1 theo lớp dưới đây là trung bình ± độ lệch chuẩn của **năm F1 theo seed**, tính lại từ từng `classification_report.csv`; đây không phải F1 tính trên một confusion matrix gộp.

| Lớp | DenseNet F1 | ResNet F1 | ViT F1 |
|---|---:|---:|---:|
| `anthracnose` | 90,75 ± 2,49% | 86,99 ± 3,19% | 81,50 ± 3,03% |
| `healthy` | 93,73 ± 0,81% | 90,87 ± 4,28% | 81,92 ± 3,70% |
| `leaf_miner` | 93,81 ± 0,83% | 92,79 ± 1,39% | 89,04 ± 0,70% |
| `not_cashew_leaf` | 96,32 ± 1,02% | 94,61 ± 1,25% | 93,84 ± 1,87% |
| `red_rust` | 97,55 ± 0,48% | 96,24 ± 1,62% | 92,76 ± 1,32% |

`anthracnose` là lớp yếu nhất của cả ba cấu hình. Ở DenseNet, recall trung bình của lớp này là **91,63%**, thấp hơn `red_rust` **99,31%**; ở ViT, recall `anthracnose` chỉ **82,59%** và `healthy` **77,42%**. DenseNet cải thiện mạnh `healthy` so với ViT trên Validation, nhưng lỗi `healthy → anthracnose` vẫn còn.

Các cặp lỗi nhiều nhất khi cộng confusion matrix từ năm seed:

| Mô hình | Nhầm lẫn nổi bật | Tổng lượt trong 5 lần đánh giá |
|---|---|---:|
| DenseNet | `healthy → anthracnose`; `leaf_miner → anthracnose`; `anthracnose → red_rust` | 63; 61; 54 |
| ResNet | `healthy → anthracnose`; `leaf_miner → anthracnose`; `anthracnose → red_rust` | 104; 80; 61 |
| ViT | `healthy → anthracnose`; `anthracnose → red_rust`; `leaf_miner → anthracnose` | 146; 117; 104 |

Đây là **lượt dự đoán** trên cùng 1.402 ảnh Validation được đánh giá lại năm lần, không phải số ảnh sai khác nhau. Các cặp lỗi chỉ ra nơi cần đọc ảnh lỗi và rà quy tắc gán nhãn, nhất là ranh giới triệu chứng `anthracnose` với `healthy`, `leaf_miner` và `red_rust`. Chưa thể quy lỗi cụ thể cho nhãn sai hoặc chất lượng ảnh nếu chưa xem ảnh gốc. Nguồn: các `classification_report.csv` và `confusion_matrix.csv` trong [seed DenseNet](https://drive.google.com/drive/folders/1L7eSB4LGdn5-BuP8TJ9na2hR1sPKgzV1), [seed ResNet](https://drive.google.com/drive/folders/13XL5dy2-q7Wqf4O1TNPeyJuHBYAOsmcP), [seed ViT](https://drive.google.com/drive/folders/19yy8LGAT0HY2C27fjZbA1JlgIHgS0NFf).

## 4. Đường học, checkpoint và chi phí

| Mô hình | Epoch checkpoint tốt nhất (khoảng) | Epoch thực chạy (khoảng) | Train − Val Accuracy tại checkpoint, trung bình | Tại epoch cuối, trung bình | Thời gian huấn luyện/seed, trung bình ± SD |
|---|---:|---:|---:|---:|---:|
| DenseNet | 18–26 | 26–34 | +1,97 pp | +4,04 pp | 33,37 ± 3,27 phút |
| ResNet | 14–29 | 22–37 | +2,77 pp | +6,11 pp | 25,45 ± 5,76 phút |
| ViT | 21–37 | 31–47 | −1,06 pp | +2,96 pp | 6,20 ± 1,06 phút |

Thời gian của ba lần chạy scratch đã được đo quanh `model.fit` và lưu ở `training_time_seconds` theo từng seed trong `training_summary_5seeds.csv`. Số đo gồm cả Validation mỗi epoch và callback/lưu checkpoint; không gồm giải nén dữ liệu, dựng model hay đánh giá cuối sau `fit`. Độ lệch chuẩn thời gian trong bảng được tính trên năm seed với `ddof=1`.

Ở cả 15 seed, `val_loss` cuối cao hơn mức thấp nhất đã dùng để lưu checkpoint; do đó dùng checkpoint `min(val_loss)` là cần thiết nếu muốn báo cáo đúng mô hình được lưu. Ở ResNet, `val_loss` tăng trung bình **0,057** từ minimum tới epoch cuối, đi cùng khoảng cách Train–Val tăng từ **2,77** lên **6,11 pp**. Đây là tín hiệu cần theo dõi về khả năng khái quát hóa sau epoch tốt nhất, chưa đủ để xác định nguyên nhân. Với ViT, Train Accuracy tại checkpoint có thể thấp hơn Validation vì ảnh Train được augmentation; không suy ra rò rỉ dữ liệu từ dấu âm này. Cần đo `train_clean_accuracy` tại checkpoint, trên Train không augmentation, nếu muốn lượng hóa khoảng cách tổng quát hóa chính xác hơn.

ResNet có một bẫy đọc số: `EXPERIMENT_REGISTRY.csv` ghi `Best Val Acc` là **đỉnh Accuracy trong lịch sử**; nó có thể khác Accuracy của checkpoint `min(val_loss)`. Ví dụ seed `42`: **94,58%** ở đỉnh lịch sử nhưng **94,22%** tại checkpoint; seed `2026`: **90,51%** so với **89,87%**. Bảng so sánh ở báo cáo này dùng giá trị tại checkpoint, trùng với `validation_results_5seeds.csv`. Nguồn: [registry](https://drive.google.com/file/d/1l9iLykKXmGCIe9_zHvOuGkvv1YESXvGK/view), [training summary ResNet](https://drive.google.com/file/d/1X3kA1YyuTIyEsfMNpkngQqeIHpxeI5YV/view).

Tổng thời gian ghi nhận cho năm seed lần lượt khoảng **166,86 phút** (DenseNet), **127,24 phút** (ResNet), **30,98 phút** (ViT). ViT có 347.717 tham số, DenseNet 7.566.917, ResNet 24.641.413. Đây là thời gian của các phiên huấn luyện đã ghi, chưa phải phép đo inference/FPS chuẩn hóa; không suy tốc độ triển khai từ số tham số hoặc thời gian Train. Nguồn: [training summary DenseNet](https://drive.google.com/file/d/1f6rkLVcxVHXqBjp1Y0I6-vxk7mNOQN9Y/view), [ResNet](https://drive.google.com/file/d/1X3kA1YyuTIyEsfMNpkngQqeIHpxeI5YV/view), [ViT](https://drive.google.com/file/d/1z0G5OelU5znjSY-2dXCKs6jDTCHlvNup/view).

## 5. Đánh giá và bước tiếp theo

1. **Ứng viên tốt nhất trên Validation hiện tại: DenseNet121-005.** Nó đứng đầu Accuracy, Macro F1 và có độ dao động theo seed thấp nhất. Giữ ResNet50-005 và ViT-004 làm đối chứng cùng V05; chưa gọi DenseNet là mô hình tốt nhất trên Test hay ngoài thực tế.
2. **Đóng băng cấu hình và năm checkpoint**, sau đó chạy đúng tập Test khóa một lần cho cả ba cấu hình theo cùng quy trình. Lưu prediction, classification report, confusion matrix từng seed và trung bình ± độ lệch chuẩn mẫu. Chỉ dùng Validation để chọn checkpoint/seed; báo cáo Test sau khi lựa chọn đã cố định.
3. **Rà ảnh lỗi Validation theo các cặp trên**, ưu tiên `healthy → anthracnose`, `leaf_miner → anthracnose`, `anthracnose → red_rust`. Ghi nhận nhóm triệu chứng và chất lượng ảnh; nếu sửa dữ liệu hoặc hyperparameter sau khi xem Validation, tạo experiment ID mới và giữ nguyên lần chạy này.
4. **Bổ sung đo lường còn thiếu**: `train_clean_accuracy` tại checkpoint cho phân tích gap, thời gian inference và kích thước deployment model trong một môi trường chung. Không dùng thời gian Train hiện có để xếp hạng tốc độ sản phẩm.
5. **Sửa định danh thí nghiệm trước khi đồng bộ vào repo**: ViT `004` trên Drive là V05, còn ViT `004` hiện có trong repo là V04. Nên lưu theo khóa gồm cả dataset version và experiment ID, hoặc cấp ID mới rõ ràng, để báo cáo và checkpoint không bị lẫn.

### Cách tính

Tất cả tỷ lệ phần trăm ở bảng chính là `100 × metric`; chênh lệch dùng điểm phần trăm. Độ lệch chuẩn lấy trên năm seed với `ddof=1`, theo file tổng hợp gốc. Các thống kê lớp và cặp nhầm lẫn được tính lại từ 15 báo cáo phân loại và 15 confusion matrix Validation. Khoảng Train–Val lấy từ `history.csv` tại epoch có `val_loss` nhỏ nhất và tại epoch cuối; Train Accuracy trong log là trên luồng huấn luyện có augmentation. Báo cáo không dùng Test để chọn hay so sánh mô hình.
