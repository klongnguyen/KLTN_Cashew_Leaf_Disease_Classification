"""Build a standalone DenseNet121 + YOLO26s training notebook from EXP-033."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent
source = ROOT / "densenet121_yolo_combined_033.ipynb"
target = ROOT / "densenet121_yolo_combined_training_034.ipynb"
nb = json.loads(source.read_text(encoding="utf-8"))
cells = nb["cells"]

def get(i):
    return "".join(cells[i]["source"])

def put(i, s):
    cells[i]["source"] = s.splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["execution_count"] = None
        cells[i]["outputs"] = []

def change(s, old, new):
    assert old in s, old
    return s.replace(old, new)

put(0, """# EXP-034 — Huấn luyện DenseNet121 + YOLO26s

Notebook này huấn luyện và đánh giá **một pipeline kết hợp**. DenseNet121 được huấn luyện 5 seed theo hai giai đoạn; YOLO26s đã khóa cung cấp đặc trưng vùng bệnh. Bộ phân loại kết hợp được fit bằng clean Validation theo pHash group. Chỉ các dự đoán và vùng hiển thị của pipeline kết hợp được đánh giá trên Test và Holdout.

Kết quả gồm lịch sử huấn luyện, cấu hình khóa, các chỉ số phân loại/định vị, biểu đồ, ảnh ví dụ, pipeline triển khai, `ANALYSIS.zip` và `FULL.zip`. Chạy lần lượt tất cả cell trong Colab GPU. Cell cuối upload hai ZIP lên Google Drive. `EXP034_DRY_RUN=1` chỉ dùng để kiểm tra notebook với dữ liệu tổng hợp, không tạo kết quả thực nghiệm.
""")
put(1, """## Quy trình

1. Cấu hình và xác minh ba nguồn dữ liệu cùng YOLO26s weights.
2. Kiểm tra trùng ảnh, chạy YOLO để lấy bằng chứng vùng bệnh, chọn ngưỡng box bằng YOLO Validation.
3. Huấn luyện DenseNet121 5 seed từ đầu theo hai giai đoạn; lưu checkpoint, lịch sử và dự đoán Validation.
4. Fit logistic stacker của pipeline kết hợp trên clean Validation theo GroupKFold; khóa `C`, temperature và ngưỡng từ chối trước khi đọc Test/Holdout.
5. Đánh giá **chỉ pipeline kết hợp** trên Test, Holdout và YOLO Test localization; xuất CSV, biểu đồ, ảnh ví dụ và kiểm tra pipeline triển khai.
6. Tạo ZIP và xác minh upload lên Google Drive.

Holdout 09 từng được xem trong EXP-032, vì vậy báo cáo gọi đây là holdout mô tả, không coi là blind benchmark. Holdout không có nhãn box; đo định vị bằng YOLO Test.
""")

s = get(2)
s = change(s, '# EXP-033 configuration and imports', '# EXP-034 combined training configuration and imports')
s = change(s, 'EXPERIMENT_ID = "EXP-DENSENET121-YOLO-COMBINED-033"', 'EXPERIMENT_ID = "EXP-DENSENET121-YOLO-COMBINED-TRAINING-034"')
s = s[:s.index('REUSE_032_MODELS = False')] + s[s.index('UPLOAD_RESULT_ZIPS_TO_DRIVE = True'):]
s = change(s, 'EXP033_DRY_RUN', 'EXP034_DRY_RUN')
s = s.replace('Synthetic fixture only; never an EXP-032 result.', 'Synthetic fixture only; never a real experiment result.')
s = change(s, 'exp033_inputs', 'exp034_inputs')
s = change(s, 'print("EXP-033",', 'print("EXP-034",')
s = s.replace('LEGACY_EXPERIMENT_ID = "EXP-DENSENET121-YOLO-PIPELINE-032"\n', '')
s = s.replace('BASELINE_EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05"\n', '')
put(2, s)

s = get(6).replace('YOLO-only localization on its untouched Test', 'YOLO Validation box-threshold selection for the combined display')
s = s.replace('Frozen EXP-032 R1 display threshold; no Holdout tuning.', 'Frozen display threshold from YOLO Validation; no Holdout tuning.')
put(6, s)
s = get(7)
s = change(s, '# Three-method fusion and evaluation; F3a is an appendix control.', '# Single combined DenseNet121 + YOLO26s fusion and evaluation.')
s = change(s, 'RULES = ("F0","F3","F3a","F3b")', 'RULES = ("F3",)')
s = change(s, 'cols = {"F3":FEATURE_COLUMNS,"F3a":PROB_COLUMNS,"F3b":FEATURE_COLUMNS[5:]}[kind]', 'assert kind == "F3"\n    cols = FEATURE_COLUMNS')
s = change(s, 'for rule in ("F3","F3a","F3b"):', 'for rule in RULES:')
s = change(s, '    if rule == "F0":\n        return frame[PROB_COLUMNS].to_numpy(dtype=float)\n', '    assert rule == "F3"\n')
s = change(s, '        if rule=="F0":\n            raw_oof[rule]=clean[PROB_COLUMNS].to_numpy(dtype=float)\n            continue\n', '')
s = s[:s.index('def paired_bootstrap(')] + s[s.index('def risk_coverage('):]
put(7, s)

put(8, '''# EXP-034 evaluation helpers: one frozen combined model only.
ALL_RESULTS, ALL_TAXONOMY, ALL_CALIBRATION, ALL_ABSTENTION = [],[],[],[]
ALL_BOOTSTRAP, ALL_CURVES = [],[]
PART_PREDICTIONS = {}

def evaluate_part(part, val, test, holdout, models, config):
    frames={"validation_clean":val,"test":test,"holdout":holdout,
            "holdout_n124":holdout[~holdout.evidence_key.isin(KNOWN_DUPLICATES)].copy()}
    if not DRY_RUN:
        assert len(frames["holdout_n124"])==124
    bysplit={}
    for split,frame in frames.items():
        rows,tax,cal,abst,preds=report_split(part,split,frame,models,config)
        ALL_RESULTS.extend(rows);ALL_TAXONOMY.extend(tax);ALL_CALIBRATION.extend(cal);ALL_ABSTENTION.extend(abst)
        ALL_CURVES.extend(risk_coverage(part,split,frame,preds))
        bysplit[split]=(frame,preds)
    test_frame,test_preds=bysplit["test"];hold_frame,hold_preds=bysplit["holdout"]
    rng=np.random.default_rng(32034)
    ti=rng.integers(0,len(test_frame),(10000,len(test_frame)))
    hi=rng.integers(0,len(hold_frame),(10000,len(hold_frame)))
    ty=test_frame.true_label.map(CLASS_INDEX).to_numpy()
    hy=hold_frame.true_label.map(CLASS_INDEX).to_numpy()
    tc=(test_preds["F3"][0]==ty).astype(float)
    hc=(hold_preds["F3"][0]==hy).astype(float)
    dist=tc[ti].mean(axis=1)-hc[hi].mean(axis=1)
    ALL_BOOTSTRAP.append({"part":part,"split":"gap_test_minus_holdout","rule":"F3",
        "delta_accuracy":float(tc.mean()-hc.mean()),
        "bootstrap_low":float(np.quantile(dist,.025)),"bootstrap_high":float(np.quantile(dist,.975))})
    clear=test_frame[~test_frame.evidence_key.isin(NEAR_KEYS["test"])]
    p=apply_temperature(rule_probs(clear,"F3",models),config["temperature"]["F3"])
    y=clear.true_label.map(CLASS_INDEX).to_numpy()
    ALL_RESULTS.append({"part":part,"split":"test_no_yolo_train_near","rule":"F3",
                        "n":len(y),"correct":int((p.argmax(1)==y).sum()),
                        "accuracy":float((p.argmax(1)==y).mean())})
    PART_PREDICTIONS[part]=bysplit
    return bysplit
''')

s = get(9)
s = change(s, '# EXP-033 two-stage DenseNet recipe, five saved deployment models, Validation lock.', '# EXP-034 DenseNet121 training, five seeds, two stages, then combined-model fit.')
s = change(s, 'service=drive_client() if (UPLOAD_RESULT_ZIPS_TO_DRIVE or REUSE_032_MODELS) else None', 'service=drive_client() if UPLOAD_RESULT_ZIPS_TO_DRIVE else None')
s = s[:s.index('    if REUSE_032_MODELS:')] + s[s.index('    train_ds_frame=INVENTORY["train"]'):]
s = change(s, 'if not deployed.is_file() and service and not REUSE_032_MODELS:', 'if not deployed.is_file() and service:')
s = s.replace('            if bool(meta.get("reused_from_exp032",False)) != REUSE_032_MODELS:\n                raise RuntimeError("Cannot mix reused and retrained seed artifacts in one EXP-033 folder")\n', '')
s = s.replace('            if REUSE_032_MODELS:\n                assert meta["deployment_sha256"]==MODEL_032_SHA256[seed]\n', '')
s = change(s, '"deployment_sha256":file_sha256(deployed),"stages":stages,\n               "reused_from_exp032":False}', '"deployment_sha256":file_sha256(deployed),"stages":stages}')
s = s.replace('    if choiceB["F3"]!=3 and not DRY_RUN:\n        print("WARNING: combined C differs from EXP-032 expectation of 3; inspect CV table")\n', '')
put(9, s)

s = get(10)
s = s[:s.index('def ap_at(')] + s[s.index('BOX_FRAMES["yolo_test"] ='):]
s = s[:s.index('for subset_name,keys in (("all",test_keys)')] + s[s.index('SELECTED_DEFAULT="F3"'):]
s = s.replace('for row in LOCALIZATION:\n    row.setdefault("mode","detector_only")\n', '')
s = s.replace('("paired_bootstrap.csv",ALL_BOOTSTRAP)', '("combined_gap_bootstrap.csv",ALL_BOOTSTRAP)')
put(10, s)

s = (ROOT / 'DENSENET121_YOLO_COMBINED_ONLY_FIGURES_CELL.py').read_text(encoding='utf-8')
s = s.replace('EXP-033', 'EXP-034').replace('exp033_summary.json', 'combined_summary.json')
s = s.replace('EXP-032/033', 'EXP-034')
s = s.replace('print("McNemar cần hai phương pháp trên cùng ảnh nên không áp dụng cho cell chỉ một mô hình.")\n', '')
put(11, s)

s = get(12)
tree = ast.parse(s)
module = ast.literal_eval(tree.body[0].value)
module = module.replace('EXP-033', 'EXP-034')
module = change(module, '        if self.rule in ("F3","F3a","F3b"):\n            data=feature[None,:] if self.rule=="F3" else (feature[None,:5] if self.rule=="F3a" else feature[None,5:])\n            model=self.stackers[self.rule]\n            fitted=model.predict_proba(data)[0]\n            p=np.zeros(len(self.classes));p[model[-1].classes_]=fitted', '        assert self.rule == "F3"\n        model=self.stackers["F3"]\n        fitted=model.predict_proba(feature[None,:])[0]\n        p=np.zeros(len(self.classes));p[model[-1].classes_]=fitted')
s = 'PIPELINE_MODULE_SOURCE = ' + repr(module) + '\n' + s[s.index('# Export and hard consistency check.'):]
s = s.replace('"rule":"Combined method declared before Test and Holdout"', '"rule":"Single combined model frozen before Test and Holdout"')
put(12, s)

put(13, '## Báo cáo mô hình kết hợp\n\nCác bảng, biểu đồ và ảnh dưới đây chỉ mô tả DenseNet121 + YOLO26s.\n')
put(14, (ROOT / 'EXP-034_COMBINED_TRAINING_REPORT_CELL.py').read_text(encoding='utf-8'))
put(15, '## Xuất kết quả\n\nCell tiếp theo kiểm tra đầy đủ artifact, tạo hai ZIP và upload lên Google Drive.\n')
s = get(16).replace('EXP-033 final audit', 'EXP-034 final audit')
a=s.index('required = ['); b=s.index('analysis_files =',a)
s=s[:a]+'''required = ["combined_results.csv", "combined_training_stages.csv",
            "combined_training_history.csv", "combined_stacker_cv.csv",
            "combined_summary.json", "pipeline_consistency.json", "README.md",
            "frozen_config.json", "validation_selection_lock.json",
            "localization_test.csv", "combined_gap_bootstrap.csv"]
for name in required:
    if not (AGGREGATE / name).is_file():
        raise FileNotFoundError(AGGREGATE / name)
expected_figures = [
    *(f"combined_only/{name}" for name in (
        "01_accuracy_and_gap.png", "02_class_recall.png", "03_confusion_matrices.png",
        "04_error_types.png", "05_calibration.png", "06_risk_coverage.png",
        "07_confidence_distribution.png", "08_fused_display_localization.png")),
    "00_densenet_component_training_curves.png",
    "00_combined_stacker_grouped_cv.png", "00_combined_stacker_refit.png",
    *(f"holdout_{cls}.png" for cls in CLASSES[:4]),
    "holdout_combined_errors.png"]
missing_figures = [name for name in expected_figures
                   if not (AGGREGATE / "figures" / name).is_file()]
if missing_figures:
    raise RuntimeError(
        "Missing report PNGs: " + ", ".join(missing_figures) +
        ". Run the statistical figures cell, then the combined training report cell, "
        "and retry this export cell. The saved models do not need retraining.")
assert (OUT_DIR / "cashew_pipeline.py").is_file()
assert all((OUT_DIR / "pipeline_bundle" / f"seed_{seed}" / "deployment_model.keras").is_file()
           for seed in SEEDS)

''' + s[b:]
put(16,s)

for i,c in enumerate(cells):
    c['id'] = f'exp034-cell-{i:02d}'
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(target, len(cells), 'cells')
