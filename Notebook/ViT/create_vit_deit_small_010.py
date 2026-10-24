"""Build the output-free EXP-010 Colab notebook from reviewed source cells."""

import json
import textwrap
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


cells = [
    markdown("""
    # EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010 — Stage A

    Gate 0 → Gate 1 → Gate 2 → Gate 3 → Gate 4. DeiT-Small distilled pretrained,
    Frozen Backbone rồi Partial Fine-tuning (2 encoder blocks cuối) cho đúng 5 seed.

    Notebook chỉ huấn luyện bằng **Train V08** và chọn/đánh giá bằng **Validation V08**.
    Test V08 chỉ được đọc trong bước kiểm tra toàn vẹn ZIP, không giải nén cho mô hình,
    không inference. Historical Holdout không được đọc. Không chỉnh sửa EXP-007/008/009 hay WebApp.

    Trước khi chạy, bật GPU trong Colab: Runtime → Change runtime type → GPU.
    Hai tệp đầu vào: ZIP V08 tại `MyDrive/Cashew_Leaf_model_result/Dataset/`
    và nhánh Git `exp/vit-deitsmall-transfer-010`. Kết quả lưu vào Drive theo từng seed.
    Nếu Colab ngắt giữa seed, thư mục seed chưa hoàn tất được giữ lại để kiểm tra;
    seed đã hoàn tất được xác thực SHA256 và dùng lại khi chạy tiếp.
    """),
    code("""
    # Cell 01 — Frozen paths and source. Change overrides only for file location.
    DATASET_OVERRIDE = None  # existing CashewData_Split (8).zip
    RESULT_ROOT_OVERRIDE = None  # parent directory of EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010
    SOURCE_REPO_OVERRIDE = None  # local clone of the EXP-010 Git branch, if needed
    EXPERIMENT_ID = 'EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010'
    SOURCE_BRANCH = 'exp/vit-deitsmall-transfer-010'
    SOURCE_URL = 'https://github.com/klongnguyen/KLTN_Cashew_Leaf_Disease_Classification.git'
    """),
    code("""
    # Cell 02 — Mount Drive, install the pinned environment, claim the source branch.
    from google.colab import drive
    drive.mount('/content/drive')
    import os, pathlib, subprocess, sys
    from pathlib import Path
    drive_root = Path('/content/drive/MyDrive/Cashew_Leaf_model_result')
    source = Path(SOURCE_REPO_OVERRIDE) if SOURCE_REPO_OVERRIDE else Path('/content/exp010_source')
    if SOURCE_REPO_OVERRIDE is None and not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--single-branch',
                        '--branch', SOURCE_BRANCH, SOURCE_URL, str(source)], check=True)
    assert (source / 'scripts/vit_deit_small/train.py').is_file(), 'EXP-010 source branch missing'
    assert (source / 'Notebook/ViT/vit_deit_small_transfer_5seeds_010.ipynb').is_file(), 'Wrong source branch'
    source_commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    exp_source = source / 'training_results/current_dataset' / EXPERIMENT_ID
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
                    '-r', str(exp_source / 'requirements.lock.txt')], check=True)
    print('Source Git commit:', source_commit)
    """),
    code("""
    # Cell 03 — Resolve V08 and result folder; preserve frozen config/baseline files.
    import hashlib, shutil
    expected_zip_sha = '75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7'
    default_dataset = drive_root / 'Dataset' / 'CashewData_Split (8).zip'
    if DATASET_OVERRIDE:
        dataset_zip = Path(DATASET_OVERRIDE)
    elif default_dataset.is_file():
        dataset_zip = default_dataset
    else:
        matches = list(Path('/content/drive/MyDrive').rglob('CashewData_Split (8).zip'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one V08 ZIP on MyDrive; found {len(matches)}: {matches[:5]}')
        dataset_zip = matches[0]
    result_root = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else drive_root / 'training_results/current_dataset'
    output = result_root / EXPERIMENT_ID
    output.mkdir(parents=True, exist_ok=True)
    for name in ('experiment_config.json', 'requirements.lock.txt',
                 'baseline_tiny_validation_008.csv', 'baseline_tiny_008_per_seed.json'):
        src, dest = exp_source / name, output / name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
            raise ValueError(f'Frozen input differs: {dest}')
        if not dest.is_file():
            shutil.copyfile(src, dest)
    (output / 'source_git_commit.txt').write_text(source_commit + '\\n')
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest().upper()
    assert sha256(dataset_zip) == expected_zip_sha, f'Wrong V08 ZIP: {dataset_zip}'
    scripts = source / 'scripts/vit_deit_small'
    subprocess.run([sys.executable, str(scripts / 'capture_environment.py'),
                    str(output / 'environment.json')], check=True)
    print('Dataset:', dataset_zip)
    print('Output:', output)
    """),
    markdown("""
    ## Gate 0 — Audit V08

    Check ZIP SHA/CRC, 6906 ảnh và class order, SHA từng ảnh, ảnh hỏng,
    exact duplicates và pHash cross-split candidates. Test chỉ được đọc để kiểm tra
    toàn vẹn; không được chuyển vào training dataset.
    """),
    code("""
    # Cell 04 — Gate 0
    import json
    subprocess.run([sys.executable, str(scripts / 'dataset_audit.py'),
                    str(dataset_zip), str(output)], check=True)
    audit = json.loads((output / 'dataset_audit.json').read_text())
    assert audit['total_images'] == 6906 and audit['decode_errors'] == 0
    assert audit['cross_split_exact_duplicates'] == 0
    print('Gate 0 PASS:', audit['counts'])
    print('pHash candidates requiring source/group review:', audit['cross_split_phash_le_6_candidates'])
    """),
    markdown("""
    ## Gate 1 — Pretrained model và kỹ thuật

    Kiểm tra 5 logits, preprocessing một lần, gradient Stage 1/Stage 2,
    đúng hai block cuối, save/reload parity trên ảnh Train thật.
    """),
    code("""
    # Cell 05 — Gate 1 on the current Colab runtime
    import pandas as pd, zipfile
    frame = pd.read_csv(output / 'data_manifest.csv')
    smoke_root = output / '_work' / 'CashewData_Split'
    with zipfile.ZipFile(dataset_zip) as archive:
        for label in (0, 1):
            row = frame[(frame.split == 'train') & (frame.label == label)].iloc[0]
            target = smoke_root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read('CashewData_Split/' + row.relative_path)
            assert hashlib.sha256(raw).hexdigest().upper() == row.sha256
            target.write_bytes(raw)
    env = os.environ.copy()
    env['KAGGLEHUB_CACHE'] = str(output / '_preset_cache')
    subprocess.run([sys.executable, str(scripts / 'smoke_test.py'), str(smoke_root),
                    str(output / 'data_manifest.csv'), str(output), str(output / '_preset_cache')],
                   env=env, check=True)
    smoke = json.loads((output / 'gate1_smoke.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['model_parameters'] == 21668357
    print('Gate 1 PASS; GPU:', smoke['gpu'])
    """),
    markdown("""
    ## Gates 2–4 — 5 seeds, Validation và báo cáo

    Mỗi seed có hai checkpoint riêng. Chọn epoch, stage, seed và quyết định
    PROMOTED/NOT PROMOTED bằng Validation V08 theo config đã khóa. Xuất dự đoán
    từng ảnh, ma trận nhầm lẫn, learning curves, phân tích lỗi, mean ± sample SD,
    selection lock và inference package nghiên cứu. Không chạy Test/Holdout.
    """),
    code("""
    # Cell 06 — Full run. Needs GPU and may take multiple hours.
    import tensorflow as tf
    gpu = tf.config.list_physical_devices('GPU')
    if not gpu:
        raise RuntimeError('Gate 2 blocked: no TensorFlow GPU. Enable Colab GPU; protocol remains unchanged.')
    subprocess.run([sys.executable, str(scripts / 'train.py'), str(dataset_zip), str(output)],
                   env=env, check=True)
    lock = json.loads((output / 'selection/validation_selection_lock.json').read_text())
    print('Gates 2–4 complete:', lock['status'])
    print('Selected research checkpoint:', lock['selected'])
    """),
    code("""
    # Cell 07 — Review artifacts after all gates pass.
    from IPython.display import display, Markdown
    display(pd.read_csv(output / 'aggregate/validation_mean_std.csv'))
    display(pd.read_csv(output / 'aggregate/baseline_tiny_vs_small.csv'))
    display(Markdown((output / 'analysis/SCIENTIFIC_REPORT.md').read_text(encoding='utf-8')))
    print('Results:', output)
    """),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "vit_deit_small_transfer_5seeds_010.ipynb"}},
        "nbformat": 4, "nbformat_minor": 5}

target = Path(__file__).with_name("vit_deit_small_transfer_5seeds_010.ipynb")
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(target)
