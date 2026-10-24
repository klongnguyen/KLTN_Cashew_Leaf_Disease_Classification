"""Build the self-contained inference-only EXP-009 Colab notebook."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "vit_transfer_xgboost_009.py"
TARGET = HERE / "vit_transfer_xgboost_009.ipynb"


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def code(value):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": value.splitlines(True)}


source = RUNNER.read_text(encoding="utf-8")
source = source.split('\nif __name__ == "__main__":\n')[0]
cells = [
    markdown("""# EXP-VIT-TRANSFER-XGBOOST-009 — Final Test V08 và Historical Holdout

Notebook này chỉ inference. Nguồn bất biến là EXP-008 FULL và năm checkpoint DeiT của EXP-007.
Khóa Validation EXP-008 chọn **seed 123 / classifier A** cho triển khai. Notebook chấm A (ViT
softmax), B (linear probe) và C (XGBoost) trên cả năm seed để báo cáo, không chọn lại model
từ Test/Holdout. Test V08 có 686 ảnh. Historical Holdout có 49 ảnh, loại đúng 1 ảnh trùng
byte với Test, còn 48 ảnh. Cả hai tập đã từng được xem; đây là đánh giá hồi cứu, không phải
blind field test mới.

Bản `009.1-validation-replay` kiểm tra classifier B/C bằng embedding Validation đã lưu
trong EXP-008. Nếu embedding mới sai khác nhẹ theo môi trường chạy, notebook ghi chẩn đoán
vào `validation_replay/` rồi tiếp tục inference; nếu classifier không tái hiện được dữ liệu
đã lưu, notebook dừng để tránh xuất kết quả sai.

Nguồn có thể nằm trong `MyDrive/Cashew_Leaf_model_result/training_results/current_dataset`
và `MyDrive/Cashew_Leaf_model_result/Dataset`. Nếu không có file trong thư mục mount,
notebook tải theo Drive file ID đã ghi trong EXP-008. Kết quả ghi tại
`MyDrive/Cashew_Leaf_model_result/training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-009`.
Các biến override ở cell cuối cho phép dùng đường dẫn khác.
"""),
    code("""# Cell 01 — Install the versions used by EXP-008 (packages only; no model training)
import pathlib, subprocess, sys
if pathlib.Path('/content').exists():
    packages = [
        'tensorflow==2.21.0', 'keras==3.15.1', 'keras-hub==0.32.0',
        'xgboost==3.1.1', 'scikit-learn==1.7.2', 'pandas==2.2.3',
        'joblib==1.5.2', 'google-api-python-client',
    ]
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--prefer-binary', *packages])
"""),
    code("# Cell 02 — Inference-only evaluator (self-contained)\n" + source + "\n"),
    code("""# Cell 03 — Run Final Test V08 and Historical Holdout
# Set an override to an existing file/folder if your Drive layout differs.
DATASET_OVERRIDE = None       # CashewData_Split (8).zip
HOLDOUT_OVERRIDE = None       # RL_Cashew_holdout.zip
SOURCE_008_OVERRIDE = None    # EXP-VIT-TRANSFER-XGBOOST-008_FULL.zip
SOURCE_007_OVERRIDE = None    # EXP-007 folder or EXP-007_FULL.zip
RESULT_ROOT_OVERRIDE = None  # parent directory of EXP-VIT-TRANSFER-XGBOOST-009

results = run(
    dataset_override=DATASET_OVERRIDE,
    holdout_override=HOLDOUT_OVERRIDE,
    source_008_override=SOURCE_008_OVERRIDE,
    source_007_override=SOURCE_007_OVERRIDE,
    result_root_override=RESULT_ROOT_OVERRIDE,
)
display(results)
print('Locked deployment: seed 123 / A (from EXP-008 Validation lock).')
"""),
    code("""# Cell 04 — Package results and upload to the requested Drive folder
# Re-run this cell alone if a Drive upload is interrupted; inference will not repeat.
from google.colab import auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

RESULT_FOLDER_ID = '1DGKTnzsvVc60RIEn5lQgflR3s-loM24G'
result_parent = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else Path('/content/drive/MyDrive/Cashew_Leaf_model_result/training_results/current_dataset')
result_dir = result_parent / EXPERIMENT
require(read_json(result_dir / 'execution_status.json')['status'] == 'COMPLETED', 'Finish Cell 03 first')
archive_path = Path('/content') / f'{EXPERIMENT}_RESULTS.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for item in sorted(result_dir.rglob('*')):
        if item.is_file() and '_work' not in item.relative_to(result_dir).parts:
            archive.write(item, arcname=f'{EXPERIMENT}/{item.relative_to(result_dir).as_posix()}')
require(zipfile.ZipFile(archive_path).testzip() is None, 'Result ZIP CRC failed')
print('Result ZIP SHA256:', digest(archive_path))
auth.authenticate_user()
credentials, _ = google.auth.default()
service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
existing = service.files().list(
    q=f"'{RESULT_FOLDER_ID}' in parents and name = '{archive_path.name}' and trashed = false",
    fields='files(id,name,size)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()['files']
if existing:
    print('Result ZIP already exists in target folder:', existing[0]['id'])
else:
    media = MediaFileUpload(str(archive_path), mimetype='application/zip', resumable=True)
    uploaded = service.files().create(body={'name': archive_path.name, 'parents': [RESULT_FOLDER_ID]},
                                      media_body=media, fields='id,name,size,webViewLink',
                                      supportsAllDrives=True).execute()
    print('Uploaded:', uploaded)
"""),
]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
