# STANDARD TRAINING NOTEBOOK FORMAT
## Chuẩn chung cho notebook huấn luyện mô hình phân loại ảnh

> Áp dụng cho các notebook: CNN, ResNet50, DenseNet121, Vision Transformer và các mô hình classification khác trong khóa luận.

---

# 1. Mục đích

Tài liệu này quy định một **format notebook thống nhất** để tất cả thành viên trong nhóm:

- huấn luyện mô hình theo cùng một trình tự;
- lưu đầy đủ hyperparameter;
- đảm bảo khả năng tái lập thí nghiệm;
- lưu kết quả theo cùng cấu trúc;
- dễ so sánh giữa CNN, ResNet50, DenseNet121, ViT;
- tránh thiếu Confusion Matrix, Classification Report, history hoặc checkpoint;
- hỗ trợ tổng hợp kết quả vào báo cáo khóa luận;
- thuận tiện kiểm tra lại một experiment sau nhiều tuần.

Format được xây dựng dựa trên notebook mẫu `resnet50-v3.ipynb`, giữ lại luồng chính:

```text
Import
→ Config
→ Load Dataset
→ Preprocess
→ Augmentation
→ Build Model
→ Compile
→ Callbacks
→ Train
→ Save History
→ Learning Curves
→ Evaluate Test
→ Predict Test
→ Confusion Matrix
→ Classification Report
→ Save Metrics
→ Export Results
→ Manual Inference Test
```

và bổ sung một số phần cần thiết để quản lý experiment khoa học tốt hơn.

---

# 2. Quy tắc chung

## 2.1. Một notebook = một experiment chính

Không nên train nhiều cấu hình không liên quan trong cùng một notebook.

Ví dụ:

```text
resnet50_frozen_v1.ipynb
resnet50_finetune_v1.ipynb
densenet121_frozen_v1.ipynb
vit_base_v1.ipynb
```

Không nên:

```text
all_models_final_new_v2_last.ipynb
```

---

## 2.2. Mỗi experiment phải có ID

Format:

```text
EXP-{MODEL}-{MODE}-{VERSION}
```

Ví dụ:

```text
EXP-RESNET50-FROZEN-001
EXP-RESNET50-FINETUNE-001
EXP-DENSENET121-FROZEN-001
EXP-VIT-BASE-001
EXP-CNN-BASELINE-001
```

---

## 2.3. Mỗi notebook phải lưu kết quả vào một folder riêng

Ví dụ:

```text
/kaggle/working/
└── EXP-RESNET50-FROZEN-001/
    ├── best.keras
    ├── last.keras
    ├── history.csv
    ├── metrics.json
    ├── classification_report.csv
    ├── confusion_matrix.csv
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── predictions.csv
    ├── model_summary.txt
    ├── experiment_config.json
    ├── sample_predictions/
    └── README.md
```

---

# 3. Cấu trúc notebook chuẩn

Notebook nên được chia theo các section sau.

---

# CELL 0 — EXPERIMENT OVERVIEW

## Markdown cell

Mỗi notebook bắt đầu bằng thông tin experiment.

Template:

```markdown
# EXPERIMENT: EXP-RESNET50-FROZEN-001

## Model
ResNet50

## Training Mode
ImageNet pretrained + Frozen Backbone

## Task
5-class image classification

## Dataset
Cashew Disease Dataset vX.X

## Classes
- anthracnose
- leaf_miner
- red_rust
- healthy
- not_cashew_leaf

## Objective
Đánh giá hiệu quả của ResNet50 pretrained ImageNet khi đóng băng backbone.

## Main Output
- best model
- training history
- learning curves
- confusion matrix
- classification report
- metrics.json
- predictions.csv
- experiment archive
```

---

# CELL 1 — IMPORT LIBRARIES

## Mục tiêu

Import toàn bộ thư viện tại một vị trí duy nhất.

Không import rải rác trong notebook trừ trường hợp thư viện chỉ dùng cho optional demo.

## Nhóm thư viện cần có

```text
OS / System
Numerical
Data Processing
Visualization
Image Processing
Deep Learning Framework
Callbacks
Model Architecture
Evaluation Metrics
Timing
```

Template:

```python
# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import os
import json
import random
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
```

Nếu model sử dụng kiến trúc pretrained:

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
```

Thay tương ứng đối với:

```text
DenseNet121
EfficientNet
MobileNet
...
```

---

# CELL 2 — ENVIRONMENT INFORMATION

## Mục tiêu

Ghi lại môi trường huấn luyện để kết quả có thể tái lập.

Bắt buộc hiển thị:

```text
Python version
TensorFlow/PyTorch version
GPU
CUDA nếu có
Ngày chạy experiment
```

Template output mong muốn:

```text
Experiment ID : EXP-RESNET50-FROZEN-001
Date          : 2026-09-xx
TensorFlow    : 2.xx
GPU           : Tesla T4
GPU Count     : 2
```

Nếu sử dụng Kaggle/Colab, nên ghi rõ:

```text
Platform: Kaggle
Accelerator: 2 x Tesla T4
```

---

# CELL 3 — CONFIGURATION

## Đây là cell quan trọng nhất của notebook

Toàn bộ hyperparameter phải đặt ở đây.

Không hard-code hyperparameter ở nhiều cell.

Template:

```python
# ============================================
# 3. CONFIG
# ============================================

EXPERIMENT_ID = "EXP-RESNET50-FROZEN-001"

SEED = 42

DATASET_VERSION = "cashew_dataset_v1"
DATASET_PATH = "/path/to/dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 1e-3

MODEL_NAME = "ResNet50"
TRAINING_MODE = "ImageNet_Frozen"

OUT_DIR = f"/kaggle/working/{EXPERIMENT_ID}"
```

---

## 3.1. Reproducibility

Bắt buộc:

```python
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Nếu có thể, bật deterministic operations.

---

## 3.2. Các config phải được lưu

Ít nhất:

```text
experiment_id
dataset_version
dataset_path
model_name
training_mode
seed
img_size
batch_size
epochs
learning_rate
optimizer
loss
augmentation
pretrained_weights
fine_tune
```

---

# CELL 4 — LOAD DATASET

## Cấu trúc folder chuẩn

```text
dataset/
├── train/
│   ├── anthracnose/
│   ├── leaf_miner/
│   ├── red_rust/
│   ├── healthy/
│   └── not_cashew_leaf/
│
├── val/
│   ├── ...
│
└── test/
    ├── ...
```

---

## Quy tắc load

### Train

```text
shuffle = True
```

### Validation

```text
shuffle = False
```

### Test

```text
shuffle = False
```

Lý do:

`shuffle=False` cho Test giúp giữ đúng thứ tự giữa:

```text
y_true
y_pred
filename
```

---

## Thông tin bắt buộc phải in

```text
Number of train images
Number of validation images
Number of test images

Class names
Number of classes
```

Ví dụ:

```text
Train images : 6890
Val images   : 1475
Test images  : 1482

Classes:
0 - anthracnose
1 - leaf_miner
2 - red_rust
3 - healthy
4 - not_cashew_leaf
```

---

# CELL 5 — DATASET STATISTICS

> Đây là phần bổ sung so với notebook mẫu và nên có trong mọi experiment.

## Mục tiêu

Kiểm tra dataset trước khi train.

Phải thống kê:

```text
class
train_count
val_count
test_count
total
```

Bảng mong muốn:

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Anthracnose | | | | |
| Leaf Miner | | | | |
| Red Rust | | | | |
| Healthy | | | | |
| Not Cashew Leaf | | | | |
| **Total** | | | | |

---

## Biểu đồ nên có

```text
Class Distribution
```

Không bắt buộc lưu trong mọi experiment nếu dataset version không thay đổi, nhưng ít nhất phải có ở notebook kiểm tra dataset.

---

# CELL 6 — PREPROCESSING

## Mục tiêu

Áp dụng preprocessing đúng với model.

Ví dụ ResNet50 ImageNet:

```python
preprocess_input(...)
```

Không sử dụng đồng thời:

```text
Rescaling(1/255)
+
preprocess_input
```

nếu kiến trúc pretrained không yêu cầu như vậy.

---

## Bắt buộc ghi rõ trong Markdown

Ví dụ:

```markdown
## Preprocessing

Model: ResNet50

Input preprocessing:
`tf.keras.applications.resnet50.preprocess_input`

Image normalization:
Theo chuẩn ImageNet của ResNet50.

Không sử dụng `Rescaling(1./255)`.
```

---

# CELL 7 — DATA AUGMENTATION

## Nguyên tắc

Augmentation chỉ áp dụng cho:

```text
TRAIN
```

Không augmentation:

```text
Validation
Test
Real Holdout Test
```

---

## Augmentation phải được ghi rõ

Ví dụ:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.03, 0.03),
])
```

---

## Trong báo cáo experiment phải lưu

```text
Horizontal Flip       : Yes
Rotation              : No
Zoom                  : 0.05
Translation           : 0.03
Brightness            : No
Contrast              : No
```

Không ghi chung chung:

```text
có augmentation
```

---

# CELL 8 — DATA PIPELINE OPTIMIZATION

Nếu dùng `tf.data`, nên:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)
```

Nếu sử dụng cache phải cân nhắc RAM.

---

# CELL 9 — BUILD MODEL

## Quy tắc

Cell này chỉ dùng để:

1. khởi tạo backbone;
2. cấu hình frozen/fine-tuning;
3. xây classifier head;
4. tạo model;
5. in `model.summary()`.

---

## Ví dụ cấu trúc

```text
Input
↓
Data Augmentation
↓
Pretrained Backbone
↓
Global Average Pooling
↓
Dense
↓
Dropout
↓
Dense
↓
Dropout
↓
Softmax
```

---

## Với transfer learning

Phải ghi rõ:

```text
Backbone:
Weights:
include_top:
Backbone trainable:
Number of frozen layers:
Number of trainable layers:
```

Ví dụ:

```text
Backbone          : ResNet50
Weights           : ImageNet
Include Top       : False
Backbone Trainable: False
Mode              : Frozen
```

---

# CELL 10 — MODEL SUMMARY

`model.summary()` phải được lưu ra file:

```text
model_summary.txt
```

Nên ghi:

```text
Total parameters
Trainable parameters
Non-trainable parameters
```

Các giá trị này cần dùng trong phần so sánh hiệu năng giữa model.

---

# CELL 11 — COMPILE MODEL

## Mỗi notebook phải ghi rõ

```text
Optimizer
Initial Learning Rate
Loss Function
Training Metrics
```

Ví dụ:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## Markdown trước cell

```markdown
## Compile Configuration

- Optimizer: Adam
- Initial Learning Rate: 1e-3
- Loss: Sparse Categorical Crossentropy
- Training Metric: Accuracy
```

---

# CELL 12 — CALLBACKS

Notebook mẫu sử dụng ba callback chính và đây nên là chuẩn mặc định.

---

## ModelCheckpoint

Lưu model tốt nhất.

Cần ghi:

```text
monitor
mode
save_best_only
filepath
```

---

## EarlyStopping

Cần ghi:

```text
monitor
patience
restore_best_weights
```

---

## ReduceLROnPlateau

Cần ghi:

```text
monitor
factor
patience
min_lr
```

---

## Template báo cáo

```text
ModelCheckpoint
- monitor: val_accuracy
- save_best_only: True

EarlyStopping
- monitor: val_loss
- patience: 8
- restore_best_weights: True

ReduceLROnPlateau
- monitor: val_loss
- factor: 0.3
- patience: 2
- min_lr: 1e-6
```

---

# CELL 13 — SAVE EXPERIMENT CONFIG

> Nên lưu config trước khi train.

File:

```text
experiment_config.json
```

Ví dụ:

```json
{
    "experiment_id": "EXP-RESNET50-FROZEN-001",
    "dataset_version": "cashew_dataset_v1",
    "model": "ResNet50",
    "weights": "imagenet",
    "fine_tune": false,
    "img_size": 224,
    "batch_size": 32,
    "max_epochs": 50,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "sparse_categorical_crossentropy",
    "seed": 42
}
```

---

# CELL 14 — TRAIN MODEL

## Train

```python
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

training_time = time.time() - start_time
```

---

## Sau khi train phải ghi

```text
Maximum Epochs
Actual Epochs
Best Epoch
Best Validation Accuracy
Minimum Validation Loss
Training Time
```

Ví dụ:

```text
Max epochs       : 50
Actual epochs    : 18
Best epoch       : 12
Best val accuracy: 0.9712
Min val loss     : 0.0814
Training time    : 17m 42s
```

---

# CELL 15 — SAVE TRAINING HISTORY

Bắt buộc lưu:

```text
history.csv
```

Các cột có thể gồm:

```text
epoch
accuracy
loss
val_accuracy
val_loss
learning_rate
```

Không chỉ lưu ảnh biểu đồ.

CSV cần thiết để sau này tổng hợp biểu đồ giữa nhiều model.

---

# CELL 16 — LEARNING CURVES

## Bắt buộc có 2 biểu đồ

### Accuracy Curve

```text
Train Accuracy
Validation Accuracy
```

File:

```text
accuracy_curve.png
```

---

### Loss Curve

```text
Train Loss
Validation Loss
```

File:

```text
loss_curve.png
```

---

## Nội dung cần phân tích

Không chỉ đưa hình.

Phải ghi nhận:

```text
Best validation epoch?
Có overfitting không?
Train/Val gap lớn không?
Validation loss có tăng trở lại không?
ReduceLR được kích hoạt khi nào?
EarlyStopping dừng ở epoch nào?
```

---

# CELL 17 — LOAD BEST MODEL BEFORE TEST

> Khuyến nghị bổ sung.

Đảm bảo đánh giá đúng checkpoint tốt nhất.

Ví dụ:

```python
best_model = tf.keras.models.load_model(
    f"{OUT_DIR}/best.keras"
)
```

Không nên mặc định model cuối cùng là model tốt nhất.

---

# CELL 18 — EVALUATE TEST SET

## Test chỉ chạy sau khi model/config đã được chọn

```python
test_loss, test_accuracy = model.evaluate(test_ds)
```

Bắt buộc lưu:

```text
test_loss
test_accuracy
```

---

## Quy tắc khoa học

Không dùng Test Set để:

- chọn learning rate;
- chọn augmentation;
- chọn epoch;
- chọn model architecture;
- chọn dropout;
- chọn threshold.

Những quyết định trên phải dựa vào Validation Set.

---

# CELL 19 — PREDICT TEST SET

Cần thu thập:

```text
y_true
y_pred
prediction probability
```

Nên bổ sung:

```text
filename
confidence
correct
```

Sau đó lưu:

```text
predictions.csv
```

Format đề xuất:

| filename | true_label | predicted_label | confidence | correct |
|---|---|---|---:|---|
| image001.jpg | anthracnose | anthracnose | 0.972 | True |
| image002.jpg | red_rust | anthracnose | 0.618 | False |

---

# CELL 20 — CONFUSION MATRIX

Bắt buộc có:

```text
Raw Confusion Matrix
```

Nên bổ sung:

```text
Normalized Confusion Matrix
```

Lưu:

```text
confusion_matrix.png
confusion_matrix_normalized.png
confusion_matrix.csv
```

---

## Phân tích bắt buộc

Trả lời:

```text
Class nào bị nhầm nhiều nhất?
Class A thường bị nhầm thành class nào?
Healthy có bị nhầm thành disease không?
Not Cashew Leaf có bị nhầm thành cashew disease không?
```

---

# CELL 21 — CLASSIFICATION REPORT

Sử dụng:

```text
Precision
Recall
F1-score
Support
```

cho từng class.

Bắt buộc lưu:

```text
classification_report.csv
```

---

## Metrics tổng hợp phải lấy

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted Precision
Weighted Recall
Weighted F1
Balanced Accuracy
```

Trong khóa luận nên ưu tiên:

```text
Macro F1
Balanced Accuracy
```

bên cạnh Accuracy.

---

# CELL 22 — PER-CLASS METRIC TABLE

Tạo bảng:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Anthracnose | | | | |
| Leaf Miner | | | | |
| Red Rust | | | | |
| Healthy | | | | |
| Not Cashew Leaf | | | | |

Điều này giúp phân tích class khó thay vì chỉ nhìn Accuracy tổng.

---

# CELL 23 — SAVE FINAL METRICS

Bắt buộc tạo:

```text
metrics.json
```

Format chuẩn:

```json
{
    "experiment_id": "",
    "dataset_version": "",
    "model_name": "",
    "backbone": "",
    "weights": "",
    "training_mode": "",
    "fine_tune": false,

    "seed": 42,
    "img_size": 224,
    "batch_size": 32,
    "max_epochs": 50,
    "actual_epochs": 0,
    "best_epoch": 0,

    "optimizer": "",
    "initial_learning_rate": 0.0,
    "loss_function": "",

    "best_val_accuracy": 0.0,
    "best_val_loss": 0.0,

    "test_loss": 0.0,
    "test_accuracy": 0.0,

    "macro_precision": 0.0,
    "macro_recall": 0.0,
    "macro_f1": 0.0,
    "balanced_accuracy": 0.0,

    "weighted_precision": 0.0,
    "weighted_recall": 0.0,
    "weighted_f1": 0.0,

    "training_time_seconds": 0.0,
    "inference_time_ms_per_image": 0.0,

    "total_parameters": 0,
    "trainable_parameters": 0,
    "non_trainable_parameters": 0,

    "classes": []
}
```

---

# CELL 24 — INFERENCE SPEED

> Bổ sung để phục vụ phần so sánh khả năng triển khai Web.

Đo thời gian inference trung bình.

Nên đo sau warm-up.

Lưu:

```text
mean inference time / image
FPS
```

Ví dụ:

```text
Inference time: 18.4 ms/image
FPS           : 54.35
```

---

# CELL 25 — ERROR ANALYSIS

## Đây là phần nên có để notebook mang tính nghiên cứu hơn

Tìm các ảnh:

```text
False Positive
False Negative / Misclassification
Lowest Confidence Correct Prediction
Highest Confidence Wrong Prediction
```

---

## Nên hiển thị ít nhất

```text
10–20 ảnh dự đoán sai
```

Mỗi ảnh ghi:

```text
True:
Predicted:
Confidence:
```

---

## Cần phân tích

Ví dụ:

```text
- lesion quá nhỏ;
- background phức tạp;
- ánh sáng yếu;
- hai bệnh có pattern tương tự;
- ảnh mờ;
- bệnh giai đoạn nhẹ;
- nhiều bệnh trên một lá.
```

---

# CELL 26 — SAMPLE TEST PREDICTIONS

Hiển thị một số prediction đúng và sai.

Nên chọn:

```text
5 correct
5 incorrect
```

hoặc một số mẫu đại diện cho từng class.

---

# CELL 27 — MANUAL IMAGE INFERENCE

Giữ lại chức năng giống notebook mẫu:

```text
Upload image
→ Resize
→ Preprocess
→ Predict
→ Display class
→ Display confidence
→ Display all probabilities
```

Output:

```text
Prediction : Anthracnose
Confidence : 0.9321

All probabilities:
anthracnose     0.9321
leaf_miner      0.0218
red_rust        0.0381
healthy         0.0043
not_cashew_leaf 0.0037
```

---

# CELL 28 — SAVE README RESULT SUMMARY

Mỗi experiment nên tự tạo:

```text
README.md
```

Template:

```markdown
# EXP-RESNET50-FROZEN-001

## Dataset
Cashew Dataset v1

## Model
ResNet50 ImageNet Frozen

## Configuration
- Image size: 224
- Batch size: 32
- Max epochs: 50
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: Sparse Categorical Crossentropy
- Seed: 42

## Training
- Actual epochs:
- Best epoch:
- Training time:

## Validation
- Best validation accuracy:
- Best validation loss:

## Test Results
- Accuracy:
- Macro Precision:
- Macro Recall:
- Macro F1:
- Balanced Accuracy:

## Efficiency
- Parameters:
- Model size:
- Inference time:
- FPS:

## Main Errors
- ...
- ...

## Conclusion
...
```

---

# CELL 29 — ZIP RESULTS

Cuối notebook:

```text
experiment folder
↓
ZIP
```

Ví dụ:

```text
EXP-RESNET50-FROZEN-001.zip
```

Bắt buộc kiểm tra file zip chứa:

```text
best model
history
metrics
classification report
confusion matrix
learning curves
config
README
```

---

# 4. Folder output chuẩn

Tất cả model classification nên dùng cấu trúc giống nhau:

```text
EXP-{ID}/
│
├── checkpoints/
│   ├── best.keras
│   └── last.keras
│
├── logs/
│   ├── history.csv
│   └── training_log.txt
│
├── metrics/
│   ├── metrics.json
│   ├── classification_report.csv
│   └── confusion_matrix.csv
│
├── figures/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── confusion_matrix_normalized.png
│
├── predictions/
│   ├── predictions.csv
│   └── error_cases/
│
├── model_summary.txt
├── experiment_config.json
└── README.md
```

Nếu muốn đơn giản hơn trong giai đoạn đầu, có thể để tất cả file trong một folder nhưng tên file phải thống nhất.

---

# 5. Quy tắc đặt tên notebook

Format:

```text
{model}_{training_mode}_{dataset_version}_{experiment_no}.ipynb
```

Ví dụ:

```text
cnn_baseline_dataset_v1_001.ipynb
resnet50_frozen_dataset_v1_001.ipynb
resnet50_finetune_dataset_v1_001.ipynb
densenet121_frozen_dataset_v1_001.ipynb
vit_base_dataset_v1_001.ipynb
```

---

# 6. Quy tắc đặt tên model

Không dùng:

```text
model_final.keras
model_final2.keras
best_new.keras
last_final.keras
```

Dùng:

```text
EXP-RESNET50-FROZEN-001_best.keras
EXP-RESNET50-FROZEN-001_last.keras
```

---

# 7. Experiment Registry

Ngoài từng notebook, nhóm cần một file tổng:

```text
EXPERIMENT_REGISTRY.csv
```

Cấu trúc:

| Experiment ID | Model | Dataset | IMG Size | Batch | LR | Mode | Best Val Acc | Test Acc | Macro F1 | Status |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| EXP-CNN-001 | CNN | v1 | 224 | 32 | 1e-3 | Scratch | | | | Completed |
| EXP-R50-001 | ResNet50 | v1 | 224 | 32 | 1e-3 | Frozen | | | | Completed |
| EXP-D121-001 | DenseNet121 | v1 | 224 | 32 | 1e-3 | Frozen | | | | Running |
| EXP-VIT-001 | ViT | v1 | 224 | 32 | 1e-4 | Finetune | | | | Planned |

---

# 8. Bảng kết quả chuẩn cho báo cáo khóa luận

Mọi notebook classification cuối cùng phải cung cấp đủ dữ liệu để điền bảng sau.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Acc | Params | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | | | | | | | |
| ResNet50 | | | | | | | |
| DenseNet121 | | | | | | | |
| ViT | | | | | | | |

---

# 9. Những thông tin KHÔNG được thiếu

## Dataset

- [ ] Dataset Version
- [ ] Train count
- [ ] Validation count
- [ ] Test count
- [ ] Class names
- [ ] Class distribution

## Reproducibility

- [ ] Experiment ID
- [ ] Random Seed
- [ ] Framework Version
- [ ] GPU
- [ ] Model Version

## Training

- [ ] Image Size
- [ ] Batch Size
- [ ] Maximum Epochs
- [ ] Actual Epochs
- [ ] Optimizer
- [ ] Learning Rate
- [ ] Loss
- [ ] Augmentation
- [ ] Callbacks
- [ ] Training Time

## Model

- [ ] Pretrained Weights
- [ ] Frozen/Fine-tuning
- [ ] Model Summary
- [ ] Total Parameters
- [ ] Trainable Parameters

## Evaluation

- [ ] Test Accuracy
- [ ] Macro Precision
- [ ] Macro Recall
- [ ] Macro F1
- [ ] Balanced Accuracy
- [ ] Per-class Metrics
- [ ] Confusion Matrix
- [ ] Classification Report

## Visualization

- [ ] Accuracy Curve
- [ ] Loss Curve
- [ ] Confusion Matrix
- [ ] Error Samples

## Deployment

- [ ] Model Size
- [ ] Inference Time
- [ ] FPS
- [ ] Manual Prediction Test

## Export

- [ ] Best Checkpoint
- [ ] History CSV
- [ ] Config JSON
- [ ] Metrics JSON
- [ ] Predictions CSV
- [ ] README
- [ ] ZIP Results

---

# 10. Markdown template đặt ở đầu mỗi notebook

Có thể copy nguyên block sau sang đầu notebook mới.

```markdown
# MODEL TRAINING EXPERIMENT

## 1. Experiment Information

| Field | Value |
|---|---|
| Experiment ID | |
| Date | |
| Member | |
| Model | |
| Training Mode | |
| Dataset Version | |
| Framework | |
| GPU | |

## 2. Research Objective

**Question:**  
Experiment này nhằm kiểm tra điều gì?

**Hypothesis:**  
Kỳ vọng trước khi train là gì?

## 3. Dataset

| Split | Number of Images |
|---|---:|
| Train | |
| Validation | |
| Test | |

Classes:

1. Anthracnose
2. Leaf Miner
3. Red Rust
4. Healthy
5. Not Cashew Leaf

## 4. Configuration

| Parameter | Value |
|---|---|
| Seed | |
| Image Size | |
| Batch Size | |
| Max Epochs | |
| Optimizer | |
| Learning Rate | |
| Loss | |
| Pretrained Weights | |
| Frozen/Fine-tune | |

## 5. Augmentation

| Augmentation | Value |
|---|---|
| Horizontal Flip | |
| Rotation | |
| Zoom | |
| Translation | |
| Brightness | |
| Contrast | |

## 6. Callbacks

| Callback | Configuration |
|---|---|
| ModelCheckpoint | |
| EarlyStopping | |
| ReduceLROnPlateau | |

## 7. Expected Outputs

- best model
- training history
- accuracy curve
- loss curve
- confusion matrix
- classification report
- metrics JSON
- predictions CSV
- experiment README
```

---

# 11. Markdown template đặt ở cuối notebook

```markdown
# EXPERIMENT RESULT SUMMARY

## 1. Training Summary

| Metric | Result |
|---|---:|
| Max Epochs | |
| Actual Epochs | |
| Best Epoch | |
| Best Val Accuracy | |
| Best Val Loss | |
| Training Time | |

## 2. Test Results

| Metric | Result |
|---|---:|
| Test Loss | |
| Test Accuracy | |
| Macro Precision | |
| Macro Recall | |
| Macro F1 | |
| Balanced Accuracy | |
| Weighted F1 | |

## 3. Efficiency

| Metric | Result |
|---|---:|
| Total Parameters | |
| Trainable Parameters | |
| Model Size | |
| Inference Time / Image | |
| FPS | |

## 4. Main Confusions

- Class ... thường bị nhầm với ...
- Class ... có Recall thấp nhất.
- Class ... có Precision thấp nhất.

## 5. Failure Analysis

Các trường hợp model thường dự đoán sai:

1. ...
2. ...
3. ...

## 6. Conclusion

Experiment cho thấy:

- ...
- ...
- ...

## 7. Decision

- [ ] Giữ model làm baseline.
- [ ] Tiếp tục fine-tuning.
- [ ] Thử augmentation khác.
- [ ] Thử learning rate khác.
- [ ] Loại model khỏi vòng tiếp theo.

## 8. Next Experiment

Experiment tiếp theo:

`EXP-...`
```

---

# 12. Format code comment chuẩn

Mỗi section nên có header:

```python
# ============================================================
# 01. IMPORT LIBRARIES
# ============================================================
```

Tiếp tục:

```text
01. IMPORT LIBRARIES
02. ENVIRONMENT
03. CONFIG
04. LOAD DATASET
05. DATASET STATISTICS
06. PREPROCESSING
07. AUGMENTATION
08. DATA PIPELINE
09. BUILD MODEL
10. MODEL SUMMARY
11. COMPILE
12. CALLBACKS
13. SAVE CONFIG
14. TRAIN
15. SAVE HISTORY
16. LEARNING CURVES
17. LOAD BEST MODEL
18. TEST EVALUATION
19. TEST PREDICTION
20. CONFUSION MATRIX
21. CLASSIFICATION REPORT
22. PER-CLASS METRICS
23. SAVE FINAL METRICS
24. INFERENCE SPEED
25. ERROR ANALYSIS
26. SAMPLE PREDICTIONS
27. MANUAL INFERENCE
28. SAVE README
29. ZIP RESULTS
```

---

# 13. Format `metrics.json` chuẩn của cả nhóm

Đây nên là schema chung cho tất cả classifier.

```json
{
    "experiment": {
        "experiment_id": "",
        "date": "",
        "member": "",
        "dataset_version": ""
    },

    "model": {
        "model_name": "",
        "backbone": "",
        "weights": "",
        "training_mode": "",
        "fine_tune": false,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "non_trainable_parameters": 0
    },

    "training_config": {
        "seed": 42,
        "img_size": 224,
        "batch_size": 32,
        "max_epochs": 50,
        "optimizer": "",
        "initial_learning_rate": 0.0,
        "loss": ""
    },

    "training_result": {
        "actual_epochs": 0,
        "best_epoch": 0,
        "best_val_accuracy": 0.0,
        "best_val_loss": 0.0,
        "training_time_seconds": 0.0
    },

    "test_result": {
        "test_loss": 0.0,
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "weighted_f1": 0.0
    },

    "efficiency": {
        "model_size_mb": 0.0,
        "inference_time_ms_per_image": 0.0,
        "fps": 0.0
    },

    "classes": []
}
```

---

# 14. Những điểm từ notebook mẫu nên giữ nguyên

Notebook mẫu hiện có luồng khá tốt và nên tiếp tục duy trì các phần sau:

1. Import rõ ràng.
2. Một cell config riêng.
3. Đặt random seed.
4. Load Train/Validation/Test riêng biệt.
5. `shuffle=False` cho Validation và Test.
6. Preprocessing riêng theo ResNet50.
7. Augmentation nhẹ.
8. Frozen ImageNet backbone.
9. ModelCheckpoint.
10. EarlyStopping.
11. ReduceLROnPlateau.
12. Lưu `history.csv`.
13. Accuracy curve.
14. Loss curve.
15. Evaluate trên Test.
16. Sinh `y_true` và `y_pred`.
17. Confusion Matrix.
18. Classification Report.
19. Lưu `metrics.json`.
20. ZIP kết quả.
21. Test thủ công bằng ảnh upload.

---

# 15. Những phần nên bổ sung so với notebook mẫu

Để phù hợp hơn với khóa luận và so sánh khoa học giữa nhiều model, nên bổ sung:

1. **Experiment ID.**
2. **Dataset Version.**
3. **Thông tin GPU/framework.**
4. **Dataset class statistics.**
5. **Lưu experiment config.**
6. **Lưu model summary.**
7. **Actual epochs / best epoch.**
8. **Macro Precision.**
9. **Macro Recall.**
10. **Macro F1.**
11. **Balanced Accuracy.**
12. **Normalized Confusion Matrix.**
13. **Predictions CSV.**
14. **Inference time.**
15. **FPS.**
16. **Model size.**
17. **Error analysis.**
18. **Sample incorrect predictions.**
19. **README summary cho từng experiment.**
20. **Experiment Registry dùng chung cả nhóm.**

---

# 16. Quy trình notebook cuối cùng

```text
┌───────────────────────────────┐
│  EXPERIMENT OVERVIEW          │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  IMPORT + ENVIRONMENT         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  CONFIG + RANDOM SEED         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  LOAD DATASET                 │
│  + DATASET STATISTICS         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  PREPROCESS + AUGMENTATION    │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  BUILD + COMPILE MODEL        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  CALLBACKS + SAVE CONFIG      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  TRAIN                        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  SAVE HISTORY                 │
│  + LEARNING CURVES            │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  LOAD BEST CHECKPOINT         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  TEST EVALUATION              │
│  + PREDICTIONS                │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  CONFUSION MATRIX             │
│  CLASSIFICATION REPORT        │
│  MACRO F1 / BALANCED ACC      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  EFFICIENCY                   │
│  TIME / FPS / MODEL SIZE      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  ERROR ANALYSIS               │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  MANUAL TEST                  │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  SAVE README + ZIP RESULTS    │
└───────────────────────────────┘
```

---

# 17. Definition of Done cho một notebook

Một experiment chỉ được xem là **COMPLETED** khi:

```text
Notebook chạy từ đầu đến cuối không lỗi
+
Best checkpoint đã lưu
+
History đã lưu
+
Config đã lưu
+
Metrics đã lưu
+
Confusion Matrix đã lưu
+
Classification Report đã lưu
+
Predictions đã lưu
+
Learning curves đã lưu
+
Error analysis đã thực hiện
+
Inference speed đã đo
+
README đã hoàn thành
+
Result folder đã ZIP
```

Nếu thiếu một trong các phần quan trọng trên:

```text
Status = INCOMPLETE
```

---

# 18. Checklist copy vào cuối mỗi notebook

```markdown
## Experiment Completion Checklist

### Reproducibility
- [ ] Experiment ID
- [ ] Dataset Version
- [ ] Seed
- [ ] Environment
- [ ] GPU
- [ ] Config saved

### Dataset
- [ ] Train count
- [ ] Validation count
- [ ] Test count
- [ ] Classes verified
- [ ] Class distribution checked

### Training
- [ ] Model summary saved
- [ ] Best checkpoint saved
- [ ] History CSV saved
- [ ] Actual epochs recorded
- [ ] Training time recorded

### Evaluation
- [ ] Test loss
- [ ] Test accuracy
- [ ] Macro Precision
- [ ] Macro Recall
- [ ] Macro F1
- [ ] Balanced Accuracy
- [ ] Per-class metrics
- [ ] Confusion Matrix
- [ ] Classification Report

### Analysis
- [ ] Accuracy curve
- [ ] Loss curve
- [ ] Error cases
- [ ] Sample predictions
- [ ] Failure analysis

### Efficiency
- [ ] Parameter count
- [ ] Model size
- [ ] Inference time
- [ ] FPS

### Export
- [ ] metrics.json
- [ ] predictions.csv
- [ ] experiment_config.json
- [ ] README.md
- [ ] ZIP result
```

---

# 19. Nguyên tắc áp dụng cho cả nhóm

1. Tất cả classifier phải sử dụng cùng format notebook.
2. Không thay đổi Test Set giữa các model.
3. Không dùng Test Set để chọn hyperparameter.
4. Cùng dataset version khi so sánh model.
5. Cùng preprocessing hợp lệ cho từng backbone.
6. Mọi augmentation phải được ghi rõ.
7. Mọi experiment phải có seed.
8. Mọi kết quả phải lưu ra file, không chỉ nằm trong output cell.
9. Accuracy không phải metric duy nhất.
10. Phải báo cáo Macro-F1 và Balanced Accuracy.
11. Phải lưu per-class performance.
12. Phải phân tích prediction sai.
13. Phải đo inference time nếu model được cân nhắc triển khai Web.
14. Không ghi đè kết quả experiment cũ.
15. Mỗi experiment phải có ID duy nhất.
16. File `metrics.json` của tất cả classifier phải cùng schema.
17. Mọi notebook cuối cùng phải chạy được từ đầu đến cuối.
18. Chỉ model đạt Definition of Done mới được đưa vào bảng so sánh cuối khóa luận.

---

# 20. Kết luận

Format chuẩn được chốt theo cấu trúc:

```text
Experiment Metadata
→ Dataset
→ Preprocessing
→ Augmentation
→ Model
→ Training
→ Validation
→ Test
→ Metrics
→ Curves
→ Confusion Matrix
→ Error Analysis
→ Efficiency
→ Manual Test
→ Export
```

Mục tiêu không chỉ là **train được model**, mà phải tạo ra một experiment:

```text
Có thể tái lập
+
Có thể kiểm tra
+
Có thể so sánh
+
Có thể đưa trực tiếp vào báo cáo nghiên cứu
```
