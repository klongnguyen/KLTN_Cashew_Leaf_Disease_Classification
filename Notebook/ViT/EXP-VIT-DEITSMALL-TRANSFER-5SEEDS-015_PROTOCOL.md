# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015 — protocol và trạng thái

**Trạng thái:** notebook đã dựng và kiểm tra cục bộ, **chưa huấn luyện**. Chưa có Accuracy, F1, checkpoint hay kết quả Test/Holdout nào cho EXP-015.

Notebook: `Notebook/ViT/vit_deit_small_transfer_5seeds_015_256px.ipynb`. Script dựng: `Notebook/ViT/build_vit_transfer_015.py` (đọc `ResNet50/resnet50_transfer_5seed_031.ipynb`, giữ nguyên mọi cell dùng chung, chỉ thay các cell phụ thuộc kiến trúc).

## Mục tiêu

So sánh ViT với DenseNet121 và ResNet50 **dưới cùng một công thức Transfer**, để chênh lệch kết quả chỉ do kiến trúc. Công thức chuẩn là DenseNet121 `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05`; ResNet50 `EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4` đã được dựng theo cùng công thức đó, và notebook này làm tương tự.

## Phần giống hệt (được kiểm bằng "fairness contract" ở cell 03a, dừng trước khi huấn luyện nếu lệch)

| Hạng mục | Giá trị |
|---|---|
| Dataset | `Cashew_dataV13` / `CashewData_Split (13).zip`, Train 4.972 / Val 1.452 / Test 685, SHA256 `e1647b72…4e04`, manifest `2cee0fb4…07` |
| Holdout | `RL_Cashew_holdout (9).zip`, 126 ảnh, 4 lớp có nhãn (descriptive audit, không phải blind benchmark) |
| Thứ tự lớp | anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf |
| Seeds | 42, 123, 2026, 3407, 7777 |
| Đầu vào | 256×256, RGB 0–255, bilinear resize, batch 16 |
| Optimizer / loss | AdamW (weight decay 1e-4), SparseCategoricalCrossentropy |
| Head | Dropout 0.40 |
| Stage 1 | LR 3e-4, tối đa 12 epoch, EarlyStopping patience 4, ReduceLROnPlateau patience 2 (min 1e-6, hệ số 0.5) |
| Stage 2 | LR 1e-5, tối đa 30 epoch, EarlyStopping patience 8, ReduceLROnPlateau patience 4 (min 1e-7, hệ số 0.5), optimizer mới |
| Augmentation (chỉ Train) | HorizontalFlip, Rotation 0.08, Zoom 0.10, Contrast 0.25, Brightness 0.25, HSV jitter (saturation 0.6–1.4, hue 0.03), random grayscale 0.10; không oversample |
| Chọn checkpoint | Validation loss thấp nhất, tie-break Macro-F1 cao hơn, rồi Stage 1; seed triển khai theo cùng luật |
| Test / Holdout | luôn bật, chỉ chạy sau `validation_selection_lock.json`, không dùng để chọn gì |
| Định dạng đầu ra | STANDARD_TRAINING_NOTEBOOK_FORMAT V2 (FULL + ANALYSIS ZIP, upload Drive) |

## Phần khác nhau do kiến trúc (khai báo trong `experiment_config.json`)

| Hạng mục | DenseNet121 (029) | ResNet50 (031) | DeiT-Small (015) |
|---|---|---|---|
| Backbone | `keras.applications` ImageNet | `keras.applications` ImageNet | KerasHub `deit_small_distilled_patch16_224_imagenet` (dự phòng `hf://facebook/deit-small-distilled-patch16-224`) |
| Tham số backbone | 7.037.504 | 23.587.712 | 21.689.472 (256 px) |
| Preprocessing | `/255` + ImageNet mean/std | Caffe BGR | `/255` + ImageNet mean/std (giống DenseNet121; kiểm bằng `keras.applications.densenet.preprocess_input`) |
| Đầu phân loại | Dropout + Dense 5 softmax | Dropout + Dense 5 softmax | Hai đầu (CLS, DIST), mỗi đầu Dropout 0.40 + Dense 5, trung bình logit rồi softmax |
| Stage 2 mở | `conv5_` không BatchNorm: 2.129.920 (30,3%) | `conv5_block1–3`: 14.953.472 (63,4%) | 4/12 khối encoder cuối + LayerNorm cuối: 7.098.624 (32,7%) |
| Chuẩn hóa | BatchNorm đóng băng | BatchNorm đóng băng | LayerNorm, không có thống kê chạy |

### Hai quyết định thiết kế cần người dùng biết

1. **4 khối cuối ở Stage 2.** ViT không có "stage" như CNN. Chọn 4/12 khối vì tỉ lệ tham số mở (32,7%) gần nhất DenseNet121 chuẩn (30,3%); ResNet50 mở 63,4%. Các thí nghiệm ViT cũ (EXP-011/014) mở 2 khối. Giá trị là một dòng cấu hình (`STAGE2_UNFREEZE_LAST_BLOCKS`); đổi giá trị thì phải dùng EXP ID mới.
2. **256 px với preset 224 px.** KerasHub DeiT chỉ dựng cho một kích thước ảnh cố định và không tự nội suy position embedding. Notebook dựng backbone 256 px rồi chép trọng số ImageNet 224 px vào; chỉ position embedding đổi kích thước: hai hàng CLS và DIST giữ nguyên, lưới 14×14 được nội suy bicubic (`tf.image.resize`) sang 16×16. Giống ý đồ `interpolate_pos_encoding` của EXP-014 nhưng dùng TensorFlow, nên khác nhau chút ít về hệ số bicubic.

## Kiểm tra đã làm cục bộ (TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.32.0, CPU)

- Toàn bộ cell code biên dịch được; chỉ còn hai tên chưa định nghĩa (`DRIVE_DATASET_ARCHIVE`, `DRIVE_PROJECT_DIR`) nằm trong nhánh Colab không bao giờ được chạy và **có sẵn từ notebook 031**.
- Cell smoke chạy PASS với trọng số "pretrained" ngẫu nhiên: sao chép 198 tensor đúng từng phần tử, 1 tensor (position embedding) nội suy, hàng CLS/DIST không đổi, nội suy cùng kích thước là phép đồng nhất; backbone 21.689.472 tham số; Stage 1 mở 0, Stage 2 mở 7.098.624; embedding và khối 1 đóng băng, khối 12 cập nhật; không có BatchNorm; lưu/nạp lại `.weights.h5` và `.keras`; augmentation đồng nhất khi suy luận và ngẫu nhiên khi train; cổng phủ saturation PASS.
- Chạy thu nhỏ đầu-cuối trên CPU (1 seed, 1+1 epoch, 20 ảnh ngẫu nhiên, trọng số giả): khóa recipe, huấn luyện hai stage, chọn stage theo Validation, Validation lock, Final Test, xuất `deployment_model.keras` cùng hai model đặc trưng, và đo độ trễ batch-1 đều chạy xong không lỗi. Các con số Accuracy trong lần chạy này vô nghĩa (dữ liệu giả) và không được dùng.

## Chưa kiểm tra được (cần Colab GPU)

- Sandbox không truy cập được Kaggle/Hugging Face, nên **chưa tải trọng số ImageNet thật**. Các khẳng định về preset (cấu hình backbone, 21.666.432 tham số ở 224 px, `layer_norm_epsilon = 1e-12`) được notebook tự kiểm bằng `assert` ở lần chạy đầu tiên và dừng với thông báo rõ nếu lệch.
- So khớp preprocessing với converter của preset chỉ chạy được khi có preset; nếu không tải được converter thì ghi `UNVERIFIED`, còn so khớp với `keras.applications.densenet.preprocess_input` đã PASS.
- Thời gian huấn luyện thực tế, Accuracy/F1, và hành vi trên GPU (tính tất định chỉ ở mức best effort).

## Cách chạy

1. Mở notebook trên Colab, chọn GPU runtime, chạy tuần tự từ trên xuống. Cell 01a cài `keras-hub==0.32.0`; nếu Colab báo phải khởi động lại runtime thì khởi động lại và chạy lại từ đầu.
2. Dataset và Holdout được tải bằng Drive file ID đã ghim và kiểm SHA256 trước khi giải nén (cùng cơ chế với notebook 031).
3. Kết quả ghi vào `training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-256px-<revision>/` và hai gói `_FULL.zip`, `_ANALYSIS.zip`.
4. Không dùng Test/Holdout để chọn cấu hình; muốn đổi bất kỳ giá trị nào ở bảng trên, tạo EXP ID mới.
