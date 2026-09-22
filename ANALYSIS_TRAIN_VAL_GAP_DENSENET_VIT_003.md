# Phân tích khoảng cách Train Accuracy và Validation Accuracy

## Phạm vi

Phân tích này dựa trên hai experiment mới nhất:

- `EXP-DENSENET121-SCRATCH-5SEEDS-003`;
- `EXP-VIT-SCRATCH-5SEEDS-003`.

Hai mô hình dùng cùng `Cashew_dataV04`, cùng 4.822 ảnh train, 1.402 ảnh validation,
687 ảnh test và cùng năm seed `[42, 123, 2026, 3407, 7777]`. Việc đề xuất thay đổi
chỉ được đánh giá trên train/validation; không thay đổi hoặc dùng test set để chọn cấu hình.

Nguồn được dùng:

- [DenseNet121 config](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/experiment_config.json)
- [DenseNet121 summary](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/README.md)
- [DenseNet121 learning curves](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg)
- [DenseNet121 executed notebook](Notebook/DenseNet121/densenet121_5seed.ipynb)
- [ViT config](training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/experiment_config.json)
- [ViT summary](training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/README.md)
- [ViT learning curves](training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg)
- [ViT executed notebook](Notebook/ViT/VIT_5seed.ipynb)

Các package kết quả 003 trong repository không chứa `history.csv` theo từng seed. Vì vậy,
các số theo epoch bên dưới được khôi phục từ training log nằm trong output của notebook đã chạy;
figure SVG và bảng tổng hợp được dùng để kiểm tra chéo. Output của ViT seed 2026 bị thiếu một số
dòng tiến trình trung gian, nên không suy diễn số liệu không còn được lưu.

## Kết luận chính

Validation Accuracy ở cuối quá trình huấn luyện **đã tương đối ổn định**, nhưng ổn định ở một
mặt bằng thấp hơn Train Accuracy. Trung bình độ lệch chuẩn của Validation Accuracy trong năm epoch
cuối chỉ khoảng **0,45 điểm phần trăm** ở cả DenseNet121 và ViT. Vấn đề chính vì thế là
**generalization gap**, không còn chủ yếu là dao động ở cuối quá trình.

Khoảng cách trung bình tại epoch cuối:

| Model | Train Acc | Val Acc | Train − Val |
|---|---:|---:|---:|
| DenseNet121 | 97,32% | 89,34% | **7,98 điểm %** |
| ViT | 91,26% | 83,87% | **7,39 điểm %** |

Trong năm epoch cuối:

| Model | Mean Train Acc | Mean Val Acc | Mean gap | Mean độ lệch chuẩn Val Acc |
|---|---:|---:|---:|---:|
| DenseNet121 | 97,09% | 89,13% | **7,97 điểm %** | **0,45 điểm %** |
| ViT | 90,87% | 83,90% | **6,96 điểm %** | **0,44 điểm %** |

Điều này cho thấy giảm learning rate đã giúp đường validation bớt dao động, nhưng không làm
validation tiếp tục tiến gần train. Train vẫn tăng hoặc giữ mức cao trong khi validation đạt trần.

## DenseNet121 003

![DenseNet121 training curves](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg)

| Seed | Best epoch theo val_loss | Stop epoch | Gap tại best epoch | Gap epoch cuối | Gap trung bình 5 epoch cuối |
|---:|---:|---:|---:|---:|---:|
| 42 | 26 | 34 | 8,29 điểm % | 9,46 điểm % | 9,23 điểm % |
| 123 | 23 | 31 | 7,11 điểm % | 8,41 điểm % | 8,21 điểm % |
| 2026 | 30 | 38 | 7,41 điểm % | 8,46 điểm % | 8,13 điểm % |
| 3407 | 19 | 27 | 6,15 điểm % | 7,31 điểm % | 7,81 điểm % |
| 7777 | 25 | 33 | 3,54 điểm % | 6,27 điểm % | 6,46 điểm % |
| **Trung bình** | **24,6** | **32,6** | **6,50 điểm %** | **7,98 điểm %** | **7,97 điểm %** |

DenseNet121 đã overfit rõ ràng. Checkpoint tốt nhất xuất hiện sớm hơn điểm dừng đúng bằng khoảng
patience 8 epoch. Trong khoảng chờ đó, train tiếp tục tăng còn validation không cải thiện tương ứng,
làm gap trung bình tăng từ 6,50 lên 7,98 điểm phần trăm.

Seed 42 cũng cho thấy learning rate ban đầu `1e-3` khá mạnh đối với DenseNet121 train từ scratch:
Validation Accuracy dao động từ 69,47% ở epoch 5 xuống 32,67% ở epoch 8; Validation Loss tăng lên
6,7572. Sau các lần ReduceLROnPlateau, validation ổn định hơn, nhưng lúc này train đã tiến gần 97%.

Điểm tích cực là `EarlyStopping(restore_best_weights=True)` đã khôi phục trọng số tại best epoch.
Vì vậy checkpoint dùng để đánh giá tốt hơn trạng thái ở epoch cuối, dù biểu đồ vẫn hiển thị phần
huấn luyện sau best epoch.

## ViT 003

![ViT training curves](training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg)

| Seed | Best epoch theo val_loss | Stop epoch | Gap epoch cuối | Gap trung bình 5 epoch cuối |
|---:|---:|---:|---:|---:|
| 42 | 20 | 30 | 8,71 điểm % | 8,45 điểm % |
| 123 | 32 | 42 | 5,92 điểm % | 5,92 điểm % |
| 2026 | 25 | 35 | 5,83 điểm % | 4,65 điểm % |
| 3407 | 35 | 45 | 8,98 điểm % | 9,41 điểm % |
| 7777 | 28 | 38 | 7,52 điểm % | 6,38 điểm % |
| **Trung bình** | **28,0** | **38,0** | **7,39 điểm %** | **6,96 điểm %** |

ViT cũng có dấu hiệu overfit muộn rất rõ. Ví dụ:

- seed 42: gap tại best epoch chỉ 2,49 điểm %, nhưng tăng thành 8,71 điểm % ở epoch cuối;
- seed 7777: gap tại best epoch chỉ 0,66 điểm %, nhưng tăng thành 7,52 điểm % ở epoch cuối;
- seed 3407: năm epoch cuối giữ gap trung bình 9,41 điểm %.

Patience 10 làm mỗi seed tiếp tục chạy thêm 10 epoch sau điểm có Validation Loss tốt nhất. Khoảng
này không giúp validation tốt hơn nhưng tạo thêm thời gian cho mô hình khớp train. Đây là nguyên
nhân trực tiếp khiến hai đường tách xa ở đoạn cuối. Việc rút ngắn patience sẽ dừng hình sớm hơn,
nhưng chỉ ngăn gap tăng thêm; nó không tự cải thiện năng lực tổng quát hóa.

## Những yếu tố tạo ra khoảng cách

### 1. Huấn luyện from scratch với tập train tương đối nhỏ

DenseNet121 có 7.566.917 tham số và được học hoàn toàn từ đầu trên 4.822 ảnh. Đây là điều kiện dễ
ghi nhớ đặc điểm riêng của train. DenseNet121 đạt khoảng 97% Train Accuracy, trong khi Validation
Accuracy dừng quanh 89%.

ViT chỉ có 347.717 tham số nhưng số tham số thấp không bảo đảm generalization tốt. ViT train từ
scratch cần nhiều dữ liệu hơn CNN vì thiếu inductive bias về tính cục bộ và tính bất biến dịch chuyển.
Với patch `16×16`, tám transformer block và chỉ 4.822 ảnh, mô hình vẫn học được các pattern riêng
của train mà không chuyển hoàn toàn sang ảnh validation.

### 2. Augmentation hiện tại còn nhẹ và giống nhau cho hai kiến trúc

Cả hai mô hình chỉ dùng:

- horizontal flip;
- rotation factor 0,04 (xấp xỉ ±14,4°);
- zoom 5%;
- translation 3%;
- contrast 8%.

Không có brightness variation, color jitter có kiểm soát, random erasing, MixUp hoặc CutMix.
Augmentation nhẹ giúp bảo toàn triệu chứng bệnh nhưng chưa mô phỏng đủ thay đổi về ánh sáng,
background, kích thước lesion và vị trí crop. Việc dùng đúng một cấu hình augmentation cho CNN và
ViT cũng bỏ qua nhu cầu regularization khác nhau của hai kiến trúc.

### 3. Learning-rate schedule đang phản ứng sau khi validation đã dao động

DenseNet121 bắt đầu với Adam `1e-3`. Các lần giảm LR đưa mô hình về vùng ổn định, nhưng chỉ sau
nhiều epoch có Validation Loss dao động mạnh. Train đã học rất nhanh trước khi validation bắt kịp.

ViT dùng AdamW `3e-4`, weight decay `1e-4`, không có warm-up và dùng ReduceLROnPlateau. Với
Transformer train từ scratch, warm-up thường hữu ích vì attention và position embedding còn chưa
ổn định ở đầu quá trình. Weight decay `1e-4` có thể chưa đủ cho tập dữ liệu này.

### 4. Patience dài làm phần cuối biểu đồ thể hiện overfitting rõ hơn

- DenseNet121: patience 8;
- ViT: patience 10.

Mọi seed đều dừng đúng 8 hoặc 10 epoch sau best epoch. Checkpoint cuối cùng dùng best weights nên
không bị mất hoàn toàn chất lượng, nhưng các epoch chờ khiến Train Accuracy tiếp tục tăng và làm hai
đường cách xa hơn. Cần phân biệt **gap tại best checkpoint** và **gap tại epoch dừng**.

### 5. Train Accuracy và Validation Accuracy hiện chưa được đo trong cùng điều kiện

Train Accuracy do `model.fit()` báo cáo được tích lũy trong khi:

- augmentation đang bật;
- dropout đang bật;
- trọng số thay đổi sau từng batch.

Validation Accuracy được đo sau epoch với augmentation và dropout tắt, bằng một bộ trọng số cố định.
Hai đại lượng vì thế không hoàn toàn tương đương. Trong trường hợp này train vẫn cao hơn validation,
nên measurement mismatch không giải thích hết gap. Nếu đánh giá lại toàn bộ train set bằng
`model.evaluate(train_eval_ds, training=False)` sau mỗi epoch, clean-train accuracy có thể còn cao
hơn và gap tổng quát hóa thật có thể lớn hơn con số đang thấy.

### 6. DenseNet121 có nhiều Batch Normalization và batch thực tế trên mỗi GPU nhỏ

Global batch size là 32 trên hai GPU, tương đương khoảng 16 mẫu mỗi replica. DenseNet121 dùng rất
nhiều Batch Normalization. Nếu statistics được cập nhật theo replica, batch nhỏ có thể làm running
statistics nhiễu và khiến validation biến động, đặc biệt ở giai đoạn learning rate cao. Đây là giả
thuyết cần kiểm tra bằng thí nghiệm một GPU hoặc synchronized BatchNorm; chưa đủ bằng chứng để kết
luận đây là nguyên nhân duy nhất.

### 7. Một số lớp khó tổng quát hóa hơn rõ rệt

`anthracnose` là lớp yếu nhất:

- DenseNet121 Test F1 trung bình: 81,30 ± 2,00%;
- ViT Test F1 trung bình: 70,43 ± 4,44%.

ViT cũng có `healthy` F1 chỉ 79,72 ± 2,22%. Điều này gợi ý ảnh có triệu chứng nhẹ, pattern tương tự,
background/ánh sáng hoặc label khó phân biệt đang giới hạn validation. Chênh lệch số lượng lớp chỉ ở
mức vừa phải (806–1.101 ảnh train/lớp, tỷ lệ 1,37 lần), nên class imbalance không có vẻ là nguyên
nhân chính. Test chỉ được dùng ở đây để mô tả kết quả đã khóa, không dùng để chọn thay đổi tiếp theo.

### 8. Validation có thể hơi khó hơn train

DenseNet121 có Test Accuracy 91,82% nhưng Validation Accuracy 89,87%; ViT gần như bằng nhau
(85,30% test và 85,28% validation). DenseNet có thể gặp distribution hoặc độ khó hơi khác ở
validation. Cần audit nguồn ảnh, điều kiện chụp, độ nét và mức độ bệnh theo split. Không nên đổi test
set hoặc di chuyển ảnh giữa các split chỉ để làm hai đường đẹp hơn.

## Mục tiêu nên tối ưu

Không nên cố làm hai đường trùng nhau bằng cách làm Train Accuracy giảm mạnh. Hai đường gần nhau
nhưng cùng thấp là underfitting. Mục tiêu phù hợp hơn là:

1. Validation Accuracy/Macro-F1 bằng hoặc cao hơn baseline 003.
2. Gap tại **best checkpoint** giảm, ưu tiên dưới 5 điểm phần trăm.
3. Validation Accuracy trong năm epoch cuối có độ lệch chuẩn dưới 1 điểm phần trăm ở từng seed.
4. Validation Loss không tăng liên tục trong khi Train Loss tiếp tục giảm.
5. Kết luận dựa trên mean ± sample SD của năm seed; không chọn seed hoặc hyperparameter bằng test.

Tiêu chí ổn định đã gần đạt ở 003; tiêu chí cần cải thiện nhất là validation plateau và gap tại best
checkpoint.

## Kế hoạch thí nghiệm đề xuất

### Bước 1 — Đo đúng gap trước khi đổi kiến trúc

Thêm một `train_eval_ds` không shuffle, không augmentation và chạy inference mode sau mỗi epoch.
Lưu ba đường:

- `train_aug_accuracy`: metric trong `fit()`;
- `train_clean_accuracy`: evaluate train với augmentation/dropout tắt;
- `val_accuracy`.

Đánh dấu best epoch theo `val_loss` trên figure và báo cáo cả `gap_at_best_epoch` lẫn
`gap_at_stop_epoch`. Đây là thay đổi ưu tiên cao nhất vì nó tách measurement gap khỏi generalization
gap thật.

### Bước 2 — Điều chỉnh learning rate trước

Đây là thí nghiệm ít làm thay đổi bài toán nhất.

**DenseNet121:**

- thử initial LR `3e-4` thay cho `1e-3`;
- warm-up 3–5 epoch rồi cosine decay, hoặc giữ ReduceLROnPlateau nhưng bắt đầu thấp hơn;
- giữ nguyên split và mọi augmentation trong lần thử LR để đo đúng tác động.

**ViT:**

- warm-up 5 epoch;
- cosine decay từ `3e-4` xuống `1e-6`;
- không để ReduceLROnPlateau là cơ chế duy nhất điều khiển LR.

### Bước 3 — Tăng regularization có kiểm soát

Chỉ thay một nhóm yếu tố mỗi experiment:

- label smoothing `0,05`, sau đó mới cân nhắc `0,10`;
- DenseNet121: chuyển Adam sang AdamW với weight decay `1e-4` hoặc thêm L2 nhỏ;
- ViT: thử weight decay `5e-4`, rồi `1e-3` nếu validation không giảm;
- ViT: thử stochastic depth/drop-path `0,05–0,10` hoặc tăng transformer dropout từ `0,10`
  lên `0,15`; không thay đồng thời tất cả;
- giữ head dropout `0,40` ở thí nghiệm đầu để xác định tác động riêng của weight decay/schedule.

### Bước 4 — Augmentation theo đặc điểm lá điều

Thêm dần:

- brightness khoảng ±10%;
- crop/translation vừa phải để mô phỏng crop từ YOLO;
- random erasing nhẹ;
- MixUp hoặc CutMix ở mức thấp nếu pipeline chuyển sang soft labels.

Không nên tăng hue/saturation mạnh vì màu tổn thương là tín hiệu phân loại. Mọi augmentation mới cần
kiểm tra bằng preview để bảo đảm không biến đổi nhãn bệnh về mặt ngữ nghĩa.

### Bước 5 — Dừng sớm hơn sau khi đã cải thiện regularization

- DenseNet121: thử patience 5–6;
- ViT: thử patience 6–7.

Điều này ngăn phần cuối tiếp tục mở rộng gap và tiết kiệm thời gian. Nó không thay thế regularization
hay schedule. Luôn dùng `restore_best_weights=True`.

### Bước 6 — Kiểm tra BatchNorm của DenseNet121

Chạy một thí nghiệm đối chứng với:

- một GPU và batch 32; hoặc
- hai GPU nhưng batch lớn hơn nếu bộ nhớ cho phép; hoặc
- synchronized BatchNorm nếu phiên bản TensorFlow/Keras và model builder hỗ trợ ổn định.

Chỉ giữ thay đổi này nếu giảm dao động validation qua nhiều seed. Không kết luận từ một seed.

### Bước 7 — Pretrained là hướng có khả năng cải thiện mạnh nhất

Nếu mục tiêu cuối là accuracy và độ ổn định thay vì bắt buộc scratch:

- DenseNet121 ImageNet pretrained, freeze backbone trước rồi fine-tune LR nhỏ;
- ViT/DeiT pretrained rồi fine-tune, thay vì học attention hoàn toàn từ 4.822 ảnh.

Đây phải là experiment ID và training mode mới, không ghi đè hoặc so như cùng một cấu hình scratch.

## Ma trận thử nghiệm tối thiểu

| Experiment | Thay đổi duy nhất | Mục đích |
|---|---|---|
| 004-A | Thêm clean-train evaluation, không đổi training | Đo gap đúng |
| 004-B | LR thấp hơn + warm-up/cosine | Giảm dao động và học quá nhanh |
| 004-C | Label smoothing 0,05 + weight decay đã chọn | Giảm memorization |
| 004-D | Augmentation tăng nhẹ | Tăng độ đa dạng hợp lệ |
| 004-E | Pretrained, experiment family riêng | Giảm data hunger |

Có thể sàng lọc A–D trên validation với một tập seed phát triển được định trước, sau đó chạy đủ năm
seed cho cấu hình cuối. Chỉ mở khóa test khi toàn bộ lựa chọn đã chốt.

## Ưu tiên thực hiện

1. Thêm clean-train evaluation và `gap_at_best_epoch`.
2. DenseNet121: hạ LR xuống `3e-4`; ViT: thêm warm-up + cosine decay.
3. Thử label smoothing và weight decay bằng experiment riêng.
4. Rút patience sau khi schedule/regularization đã ổn.
5. Audit trực quan `anthracnose` và `healthy` trên train/validation, không thay đổi test set.
6. Nếu không bắt buộc scratch, chuyển sang pretrained trong một nhánh experiment mới.

Kỳ vọng hợp lý không phải là Train Accuracy và Validation Accuracy bằng nhau tuyệt đối. Với dữ liệu
hiện tại, giảm gap tại best checkpoint xuống khoảng 3–5 điểm phần trăm trong khi giữ hoặc tăng
Validation Macro-F1 sẽ là cải thiện có ý nghĩa và đáng tin cậy hơn một biểu đồ chỉ trông sát nhau.
