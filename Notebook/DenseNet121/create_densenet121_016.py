"""Build the self-contained EXP-016 notebook from its tested inference script."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "densenet121_transfer_5seed_016.py").read_text(encoding="utf-8")
TARGET = HERE / "densenet121_transfer_5seed_016.ipynb"
CELLS = []


def add(kind, source):
    cell = {"cell_type": kind,
            "id": hashlib.sha256(f"{len(CELLS)}:{source}".encode()).hexdigest()[:8],
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    CELLS.append(cell)


add("markdown", """# EXP-DENSENET121-TRANSFER-5SEEDS-016 — Final Test V08

Đánh giá **686 ảnh Test V08** bằng đúng 5 checkpoint đã khóa theo Validation
trong EXP-014. Chỉ inference; không train lại, không đổi checkpoint và không chọn
lại seed sau khi xem Test. Seed triển khai vẫn là **123** theo
`validation_selection_lock.json` của EXP-014.

Notebook xuất Accuracy, Macro F1, Recall từng lớp, confusion matrix và dự đoán
từng ảnh cho mỗi seed. Nó kiểm tra SHA256 của ZIP V08, từng ảnh Test, config,
Validation lock và từng checkpoint trước khi chấm.

Trên Colab, mount Google Drive. Notebook dùng hai ZIP tại
`MyDrive/Cashew_Leaf_model_result/training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL.zip`
và `MyDrive/Cashew_Leaf_model_result/Dataset/CashewData_Split (8).zip`, hoặc tự ghép
các phần trong `MyDrive/Cashew_Leaf_model_result/EXP-016_INPUT_PARTS` đã tải lên Drive.
Mỗi phần và ZIP sau khi ghép đều được kiểm tra SHA256. Nếu dữ liệu nằm chỗ khác,
điền đường dẫn vào `SOURCE_ZIP_OVERRIDE`, `DATASET_ZIP_OVERRIDE` hoặc
`PARTS_DIR_OVERRIDE` ở cell kế tiếp. Kết quả được ghi vào thư mục
`EXP-DENSENET121-TRANSFER-5SEEDS-016`.
""")

source = SOURCE.split('if __name__ == "__main__":', 1)[0]
parts = [
    source.split("def set_global_seed(seed):", 1)[0],
    "def set_global_seed(seed):" + source.split("def set_global_seed(seed):", 1)[1].split("def verify_source(archive):", 1)[0],
    "def verify_source(archive):" + source.split("def verify_source(archive):", 1)[1].split("def write_seed_results(", 1)[0],
    "def write_seed_results(" + source.split("def write_seed_results(", 1)[1],
]
add("code", """# Mount only when running on Colab.
from pathlib import Path
if Path('/content').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')

# Nếu ZIP không ở vị trí mặc định, điền đường dẫn tuyệt đối trong Colab.
# Ví dụ: '/content/drive/MyDrive/MyFolder/EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL.zip'
SOURCE_ZIP_OVERRIDE = None
DATASET_ZIP_OVERRIDE = None
PARTS_DIR_OVERRIDE = None
OUTPUT_DIR_OVERRIDE = None
""")
add("markdown", "## 1. Cấu hình và mô hình EXP-014")
add("code", parts[0] + parts[1])
add("markdown", "## 2. Kiểm tra nguồn và giải nén đúng Test V08")
add("code", parts[2])
add("markdown", "## 3. Inference đủ 5 checkpoint và xuất báo cáo")
add("code", parts[3])
add("code", """# Tự tìm ZIP theo tên trong Drive nếu không điền đường dẫn ở cell đầu.
results = run(source_zip=SOURCE_ZIP_OVERRIDE,
              dataset_zip=DATASET_ZIP_OVERRIDE,
              output_dir=OUTPUT_DIR_OVERRIDE,
              parts_dir=PARTS_DIR_OVERRIDE)
display(results)
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
if len({cell["id"] for cell in CELLS}) != len(CELLS):
    raise RuntimeError("Notebook cell IDs must be unique")
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Created", TARGET)
