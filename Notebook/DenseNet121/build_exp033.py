"""Patch the EXP-032 R1 notebook into the EXP-033 notebook by cell header."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "cashew_densenet_yolo_pipeline_032.ipynb"
TARGET = ROOT / "densenet121_yolo_combined_033.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))


def src(cell):
    return "".join(cell["source"])


def find(header):
    hits = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and src(cell).startswith(header)]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one cell headed {header!r}; found {len(hits)}")
    return hits[0]


def replace(cell, old, new):
    source = src(cell)
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one replacement of {old[:70]!r}; found {source.count(old)}")
    cell["source"] = (source.replace(old, new)).splitlines(keepends=True)


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


overview = notebook["cells"][0]
set_source(overview, """# EXP-033 — DenseNet121 kết hợp YOLO26s

Một thí nghiệm với ba cách dự đoán: DenseNet một mình, YOLO một mình và DenseNet + YOLO. Cùng bộ ảnh và năm seed đã khóa; mô hình kết hợp là phương án được đề xuất, không được chọn theo Holdout. Holdout 09 là đánh giá mô tả đã được xem trước, không phải kiểm định mù.

Chạy lần lượt các cell trong Colab GPU. `REUSE_032_MODELS=False` sẽ train lại năm seed; có thể bật chế độ tái sử dụng với ZIP và SHA256 đã khóa. Cell cuối tạo và upload `ANALYSIS.zip` cùng `FULL.zip` khi tất cả kiểm tra đạt.
""")
guide = markdown("""## Cách đọc các con số

- **Độ chính xác:** phần ảnh được gọi đúng. Ví dụ 95/100 ảnh đúng nghĩa là 95%.
- **Khoảng tin cậy 95%:** khoảng bất định do số ảnh hữu hạn; 95% trên 100 ảnh có khoảng rộng hơn 95% trên 1.000 ảnh.
- **Recall theo lớp:** phần ảnh thật của một bệnh được nhận ra. Ví dụ 17/20 ảnh bệnh thán thư được nhận đúng là recall 85%.
- **Khoảng cách Test–Holdout:** độ chính xác Test trừ Holdout, tính bằng điểm phần trăm. Ví dụ 96% trừ 90% là 6 điểm.
- **Ba loại lỗi:** bỏ sót bệnh (gọi bệnh là khỏe/không phải lá điều), nhầm loại bệnh, và báo bệnh giả trên ảnh không bệnh.
- **Khoảng bootstrap:** lấy lại mẫu các ảnh nhiều lần để xem chênh lệch giữa hai cách dự đoán có ổn định không. Khoảng chứa 0 chưa cho thấy cải thiện rõ.
- **McNemar:** so hai cách dự đoán trên chính cùng ảnh; chỉ những ảnh một cách đúng và cách kia sai mới ảnh hưởng p-value.
- **ECE/hiệu chuẩn:** khoảng cách giữa độ tin cậy dự đoán và tần suất đúng thực tế; càng gần 0 càng tốt.
- **Từ chối dự đoán:** ảnh thiếu chắc chắn được chuyển cho chuyên gia; coverage là phần ảnh máy vẫn tự kết luận.
- **Precision, recall, mAP50, IoU 0.5 của box:** precision là phần box vẽ đúng, recall là phần tổn thương thật tìm thấy. Box được xem là khớp khi phần giao chia phần hợp ít nhất 0,5; mAP50 tóm tắt chất lượng qua các mức confidence.
""")
notebook["cells"].insert(1, guide)

# The embedded EXP-029 probability study is deliberately absent in EXP-033.
notebook["cells"] = [cell for cell in notebook["cells"] if not (
    cell["cell_type"] == "code" and src(cell).startswith("# 01a. FROZEN EXP-029"))]
notebook["cells"] = [cell for cell in notebook["cells"] if not (
    cell["cell_type"] == "code" and (src(cell).startswith("# EXP-032 — render validation figures")
    or src(cell).startswith("# EXP-032 — upload an ANALYSIS snapshot")
    or src(cell).startswith("# DenseNet121 + YOLO only (F3)")))]

config = find("# EXP-032 configuration and imports")
replace(config, "# EXP-032 configuration and imports", "# EXP-033 configuration and imports")
replace(config, 'EXPERIMENT_ID = "EXP-DENSENET121-YOLO-PIPELINE-032-R1"',
        'EXPERIMENT_ID = "EXP-DENSENET121-YOLO-COMBINED-033"')
replace(config, 'RUN_PART_A, RUN_PART_B = True, True',
        'RUN_PART_B = True\nREUSE_032_MODELS = False\nMODELS_032_DRIVE_FILE_ID = ""\nMODEL_032_SHA256 = {42: "46e03914b7afbdcf981bec8d81228688482c06226707a5037d8152bc1e129916",\n    123: "e2f9ccb11f049c4dd7ed61613e989c9aefecba3848291fee44b2c5f67c0b34ab",\n    2026: "1219d832eeb41ad973e2bdf35aaed91f1f2e832ec192991f37fe9f57ce94755f",\n    3407: "abcdaf65e504d6731137954183d087a2184c9533326ab4bc56b2e81b53f1f33b",\n    7777: "f962439a44b747bee47114c86742cc80a69c59b919f714674f6d364ae785bade"}\nif REUSE_032_MODELS and not MODELS_032_DRIVE_FILE_ID.strip():\n    raise ValueError("Set MODELS_032_DRIVE_FILE_ID before reusing models")')
replace(config, 'os.environ.get("EXP032_DRY_RUN")', 'os.environ.get("EXP033_DRY_RUN")')
replace(config, 'RUN_PART_A, RUN_PART_B, UPLOAD_RESULT_ZIPS_TO_DRIVE = True, True, False',
        'RUN_PART_B, UPLOAD_RESULT_ZIPS_TO_DRIVE = True, False')
replace(config, 'STAGE_ROOT = PROJECT_ROOT / "exp032_inputs"',
        'STAGE_ROOT = PROJECT_ROOT / "exp033_inputs"')
replace(config, 'TAU = 0.60\n', '')
replace(config, 'print("EXP-032",', 'print("EXP-033",')
replace(config, '      "Part A", RUN_PART_A, "Part B", RUN_PART_B)',
        '      "DenseNet training", RUN_PART_B)')

stage = find("# Stage pinned inputs.")
stage_source = src(stage)
start = stage_source.index('    # EXP-031 embedded 15-table bundle.')
end = stage_source.index('    split_archive =', start)
set_source(stage, stage_source[:start] + stage_source[end:])
replace(stage, 'synthetic_exp032', 'synthetic_exp033')
replace(stage, 'rng = np.random.default_rng(32)', 'rng = np.random.default_rng(33)')
replace(stage, '("train", CLASSES, 2), ("val", CLASSES, 2),',
        '("train", CLASSES, 2), ("val", CLASSES, 6),')

leakage = find("# Frozen EXP-029 probabilities and pixel/pHash leakage audit.")
leakage_source = src(leakage)
set_source(leakage, '# Pixel/pHash leakage audit.\nPROB_COLUMNS = [f"prob_{name}" for name in CLASSES]\n\n' +
           leakage_source[leakage_source.index('def pixel_sha256(path):'):])

detector = find("# One detector pass per classifier split;")
replace(detector, '    if label:\n        subset = subset[subset.cls == label]\n', '')
replace(detector, '        if len(kept) >= 20:\n            break\n', '')
replace(detector, 'def display_boxes(frame, box_conf, label=None, abstain=False):',
        'def display_boxes(frame, box_conf, label=None, abstain=False, draw_cap=None):')
replace(detector, '            kept.append({"cls":row.cls,"conf":float(row.conf),"xyxy":xy})\n',
        '            kept.append({"cls":row.cls,"conf":float(row.conf),"xyxy":xy})\n            if draw_cap is not None and len(kept) >= draw_cap:\n                break\n')

local_val = find("# YOLO-only localization on its untouched Test")
replace(local_val,
        'BOX_CONF=max(np.arange(.05,.951,.05),key=lambda t:(macro_f1_at(t),-t))',
        'BOX_CONF=0.25  # Frozen EXP-032 R1 display threshold; no Holdout tuning.')
replace(local_val, 'print("BOX_CONF from YOLO Val",BOX_CONF,"macro F1",macro_f1_at(BOX_CONF),',
        'print("Frozen BOX_CONF",BOX_CONF,"YOLO Val macro F1",macro_f1_at(BOX_CONF),')

fusion = find("# Fusion and evaluation, shared unchanged by Parts A and B.")
replace(fusion, '# Fusion and evaluation, shared unchanged by Parts A and B.',
        '# Three-method fusion and evaluation; F3a is an appendix control.')
replace(fusion, 'RULES = ("F0","F1","F2","F3","F3a","F3b")',
        'RULES = ("F0","F3","F3a","F3b")')
fusion_source = src(fusion)
start = fusion_source.index('def gated_probs(')
end = fusion_source.index('def fit_stackers(', start)
set_source(fusion, fusion_source[:start] + fusion_source[end:])
fusion_source = src(fusion)
start = fusion_source.index('    if rule == "F1":')
end = fusion_source.index('    model = models[rule]', start)
set_source(fusion, fusion_source[:start] + fusion_source[end:])
fusion_source = src(fusion)
start = fusion_source.index('def freeze_fusion(')
end = fusion_source.index('def wilson(', start)
new_freeze = '''def freeze_fusion(part,clean,models,choices):
    # Every reported stacker Validation probability comes from a held-out pHash group.
    global OOF_CALIBRATED
    y=clean.true_label.map(CLASS_INDEX).to_numpy()
    groups=np.array([VAL_GROUPS[k] for k in clean.evidence_key])
    folds=list(GroupKFold(5).split(clean,y,groups))
    raw_oof={}
    for rule in RULES:
        if rule=="F0":
            raw_oof[rule]=clean[PROB_COLUMNS].to_numpy(dtype=float)
            continue
        x=matrix(clean,rule)
        out=np.zeros((len(clean),len(CLASSES)),dtype=float)
        for train,val in folds:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=choices[rule],max_iter=1500))
            model.fit(x[train],y[train])
            out[np.ix_(val,model[-1].classes_)]=model.predict_proba(x[val])
        assert np.allclose(out.sum(axis=1),1,atol=1e-5)
        raw_oof[rule]=out
    temperatures={}
    OOF_CALIBRATED={}
    abstention={}
    for rule,raw in raw_oof.items():
        fit=lambda ids:minimize_scalar(lambda t:log_loss(y[ids],apply_temperature(raw[ids],t),
                       labels=np.arange(len(CLASSES))),bounds=(.5,5),method="bounded").x
        temperatures[rule]=float(fit(np.arange(len(y))))
        calibrated=np.zeros_like(raw)
        for train,val in folds:
            calibrated[val]=apply_temperature(raw[val],fit(train))
        OOF_CALIBRATED[rule]=calibrated
        abstention[rule]=abstain_threshold(y,calibrated)
    if abstention["F3"]<=0:
        print("WARNING: combined abstention threshold is zero; report coverage explicitly")
    config={"experiment":EXPERIMENT_ID,"part":part,"classes":CLASSES,
            "detector_classes":DETECTOR_CLASSES,"feature_columns":FEATURE_COLUMNS,
            "C":choices,"abstain_thresholds":abstention,"temperature":temperatures,
            "temperature_note":"Stacker and temperature Validation predictions are cross-fitted by pHash group",
            "box_conf":float(BOX_CONF),"box_nms_iou":.5,"max_draw_boxes":60,
            "seeds":SEEDS,"selection_split":"clean Validation (grouped OOF)",
            "holdout_used_for_selection":False}
    path=AGGREGATE/"frozen_config.json"
    if path.exists() and not DRY_RUN:
        assert json.loads(path.read_text(encoding="utf-8"))==config
    else:
        write_json(path,config)
    print("Frozen config SHA256",file_sha256(path))
    return config

'''
set_source(fusion, fusion_source[:start] + new_freeze + fusion_source[end:])
replace(fusion, 'for thr in np.unique(np.r_[0,confidence]):',
        'for thr in np.unique(confidence):')
replace(fusion, '        p=apply_temperature(rule_probs(frame,rule,models),config["temperature"][rule]); pred=p.argmax(axis=1)',
        '        p=(OOF_CALIBRATED[rule] if split=="validation_clean" else\n           apply_temperature(rule_probs(frame,rule,models),config["temperature"][rule])); pred=p.argmax(axis=1)')

evaluate = find("# Part A: frozen EXP-029 probabilities.")
evaluate_source = src(evaluate)
set_source(evaluate, '# EXP-033 evaluation helpers.\n' +
           evaluate_source[evaluate_source.index('ALL_RESULTS,'):evaluate_source.index('if RUN_PART_A:')])

train = find("# Part B: EXP-029 two-stage recipe")
replace(train, '# Part B: EXP-029 two-stage recipe, five saved deployment models, Validation lock.',
        '# EXP-033 two-stage DenseNet recipe, five saved deployment models, Validation lock.')
replace(train, '        legacy=False\n', '')
train_source = src(train)
start = train_source.index('        if not remote:\n            name=f"{LEGACY_EXPERIMENT_ID}_seed_{seed}.zip"')
end = train_source.index('        if not remote: return False', start)
set_source(train, train_source[:start] + train_source[end:])
replace(train, '            members=[member for member in archive.infolist()\n                     if (not legacy or member.filename in allowed)]\n            if legacy and {member.filename for member in members}!=allowed:\n                raise RuntimeError("Legacy seed archive lacks model or summary")\n',
        '            members=archive.infolist()\n')
train_source=src(train)
start=train_source.index('        if not deployed.is_file():\n            legacy=')
end=train_source.index('        if deployed.is_file() and meta_path.is_file():', start)
set_source(train,train_source[:start] +
           '        if not deployed.is_file() and service and not REUSE_032_MODELS:\n            restore_seed(service,seed)\n' +
           train_source[end:])
replace(train, '            stages.append({"stage":stage,"val_loss":loss,"epochs":len(history.history["loss"]),',
        '            stages.append({"stage":stage,"history":history.history,"val_loss":loss,"epochs":len(history.history["loss"]),')
replace(train, '"deployment_sha256":file_sha256(deployed),"stages":stages}',
        '"deployment_sha256":file_sha256(deployed),"stages":stages,\n               "reused_from_exp032":False}')
replace(train, '    service=drive_client() if UPLOAD_RESULT_ZIPS_TO_DRIVE else None',
        '    service=drive_client() if (UPLOAD_RESULT_ZIPS_TO_DRIVE or REUSE_032_MODELS) else None\n    if REUSE_032_MODELS:\n        import io\n        from googleapiclient.http import MediaIoBaseDownload\n        metadata=service.files().get(fileId=MODELS_032_DRIVE_FILE_ID,\n            fields="id,name,size,md5Checksum",supportsAllDrives=True).execute()\n        archive_path=OUT_DIR.parent/(EXPERIMENT_ID+"_source_models_032.zip")\n        if not archive_path.is_file():\n            request=service.files().get_media(fileId=MODELS_032_DRIVE_FILE_ID,supportsAllDrives=True)\n            with archive_path.open("wb") as handle:\n                downloader=MediaIoBaseDownload(handle,request,chunksize=8<<20)\n                done=False\n                while not done: _,done=downloader.next_chunk(num_retries=3)\n        assert archive_path.stat().st_size==int(metadata["size"])\n        if metadata.get("md5Checksum"):\n            assert file_md5(archive_path)==metadata["md5Checksum"]\n        with zipfile.ZipFile(archive_path) as source_zip:\n            assert source_zip.testzip() is None\n            for seed in SEEDS:\n                suffix=f"/seed_{seed}/deployment_model.keras"\n                matches=[name for name in source_zip.namelist() if name.endswith(suffix)]\n                assert len(matches)==1, (seed,matches)\n                data=source_zip.read(matches[0])\n                assert hashlib.sha256(data).hexdigest()==MODEL_032_SHA256[seed]\n                seed_dir=OUT_DIR/f"seed_{seed}"\n                seed_dir.mkdir(exist_ok=True)\n                model_path=seed_dir/"deployment_model.keras"\n                if model_path.exists():\n                    assert file_sha256(model_path)==MODEL_032_SHA256[seed]\n                else:\n                    model_path.write_bytes(data)\n                summary={"seed":seed,"deployment_sha256":MODEL_032_SHA256[seed],\n                    "reused_from_exp032":True,"source_drive_id":MODELS_032_DRIVE_FILE_ID,\n                    "stages":[],"selection_primary_metric":"val_loss"}\n                write_json(seed_dir/"seed_summary.json",summary)\n        print("Verified all five reused EXP-032 model SHA256 values")')
replace(train, '    valB=assemble(ensemble_B("validation"),"validation")',
        '    valB=assemble(ensemble_B("validation"),"validation")')
replace(train, '    modelsB,choiceB=fit_stackers(cleanB,"B")',
        '    modelsB,choiceB=fit_stackers(cleanB,"B")\n    print("Grouped-CV C values",choiceB)\n    if choiceB["F3"]!=3 and not DRY_RUN:\n        print("WARNING: combined C differs from EXP-032 expectation of 3; inspect CV table")')
replace(train, 'suffix=f"/seed_{seed}/deployment_model.keras"',
        'suffix=f"seed_{seed}/deployment_model.keras"')
replace(train, '            meta=json.loads(meta_path.read_text(encoding="utf-8"))\n            assert file_sha256(deployed)==meta["deployment_sha256"]',
        '            meta=json.loads(meta_path.read_text(encoding="utf-8"))\n            if bool(meta.get("reused_from_exp032",False)) != REUSE_032_MODELS:\n                raise RuntimeError("Cannot mix reused and retrained seed artifacts in one EXP-033 folder")\n            assert file_sha256(deployed)==meta["deployment_sha256"]\n            if REUSE_032_MODELS:\n                assert meta["deployment_sha256"]==MODEL_032_SHA256[seed]')
train_source = src(train)
if train_source.count('if service:\n') != 2:
    raise RuntimeError('Unexpected seed-upload branch count')
set_source(train, train_source.replace('if service:\n',
    'if service and UPLOAD_RESULT_ZIPS_TO_DRIVE:\n'))
replace(train, '    bysplitB=evaluate_part("B",cleanB,testB,holdB,modelsB,configB)',
        '    bysplitB=evaluate_part("B",cleanB,testB,holdB,modelsB,configB)\n    for split_name,(frame,preds) in bysplitB.items():\n        for rule,(labels,probs) in preds.items():\n            output=frame[["evidence_key","true_label"]].copy()\n            output["predicted_label"]=[CLASSES[i] for i in labels]\n            for i,cls in enumerate(CLASSES): output[f"prob_{cls}"]=probs[:,i]\n            output.to_csv(AGGREGATE/f"predictions_{split_name}_{rule}.csv",index=False)')
train_source=src(train)
start=train_source.index('    for split,ref in (("validation",.9737)')
end=train_source.index('    pd.DataFrame([r for r in ALL_RESULTS',start)
set_source(train,train_source[:start]+train_source[end:])

loc = find('GT_TEST=read_yolo_gt("test")')
loc_source = src(loc)
start = loc_source.index('# Predeclared default-rule decision')
end = loc_source.index('    # Measure the boxes actually displayed', start)
set_source(loc,loc_source[:start] + 'SELECTED_DEFAULT="F3"  # Proposed before Test/Holdout; never selected by Holdout.\nif RUN_PART_B:\n' + loc_source[end:])
replace(loc, 'display_boxes(source,BOX_CONF,row.fused_label,row.abstain)',
        'display_boxes(source,BOX_CONF,row.fused_label,row.abstain)')
combined_figures = (ROOT / 'DENSENET121_YOLO_COMBINED_ONLY_FIGURES_CELL.py').read_text(encoding='utf-8')
notebook['cells'].insert(notebook['cells'].index(loc) + 1, code(combined_figures))

# Keep the EXP-032 pipeline body, then patch its inference and consistency checks.
pipeline = find('PIPELINE_MODULE_SOURCE = ')
pipeline_source = src(pipeline)
module = ast.literal_eval(ast.parse(pipeline_source).body[0].value)
module = module.replace('Standalone EXP-032 Cashew + YOLO pipeline.',
                        'Standalone EXP-033 Cashew + YOLO pipeline.')
module = module.replace('        if self.rule in ("F1","F2"):\n            threshold=.60 if self.rule=="F1" else .05\n            rank=np.argsort(-p)\n            if rank[0]==self.classes.index("anthracnose") and evidence["max_conf"]["anthracnose"]<threshold:\n                p[rank[0]],p[rank[1]]=p[rank[1]],p[rank[0]]\n        elif self.rule in ("F3","F3a","F3b"):',
                        '        if self.rule in ("F3","F3a","F3b"):')
module = module.replace('        shown=[]\n        if not abstain and label in self.detector_classes:',
                        '        shown=[]\n        if not abstain and label in self.detector_classes:')
module = module.replace('b for b in boxes if b["cls"]==label and b["conf"]>=self.config["box_conf"]',
                        'b for b in boxes if b["conf"]>=self.config["box_conf"]')
module = module.replace('if all(self._iou(box["xyxy"],old["xyxy"])<=.5 for old in shown):',
                        'if all(box["cls"]!=old["cls"] or self._iou(box["xyxy"],old["xyxy"])<=.5 for old in shown):')
module = module.replace('if len(shown)>=20: break', 'if len(shown)>=60: break')
module = module.replace('"version":"EXP-032"', '"version":"EXP-033"')
module = module.replace('"action":"expert review" if abstain else "predict"',
                        '"action":"cần chuyên gia kiểm tra" if abstain else ("không phát hiện bệnh" if label in ("healthy","not_cashew_leaf") else "predict")')
assert 'self.rule in ("F1","F2")' not in module
pipeline_body=pipeline_source[pipeline_source.index('# Export and hard consistency check.') :]
pipeline_body=pipeline_body.replace('"Predeclared Part B decision conditions; not selected by maximum Holdout score"',
                                    '"Combined method declared before Test and Holdout"')
pipeline_body=pipeline_body.replace('display_boxes(stored,BOX_CONF,actual["label"],actual["abstain"])',
                                    'display_boxes(stored,BOX_CONF,actual["label"],actual["abstain"],draw_cap=60)')
pipeline_body=pipeline_body.replace('    print("Pipeline consistency PASS",len(fixed_keys),"Holdout images + no-detection + grayscale")',
                                    '    write_json(AGGREGATE/"pipeline_consistency.json",{"status":"PASS",\n        "holdout_images":len(fixed_keys),"probability_atol":1e-4,\n        "same_boxes":True,"no_detection_and_grayscale":True})\n    print("Pipeline consistency PASS",len(fixed_keys),"Holdout images + no-detection + grayscale")')
pipeline_body=pipeline_body[:pipeline_body.index('    for cls in CLASSES[:4]:')]
set_source(pipeline, 'PIPELINE_MODULE_SOURCE = ' + repr(module) + '\n' + pipeline_body)

# A separate report cell contains the charts, qualitative panels and plain summary.
report_source=(ROOT/'EXP-033_REPORT_CELL.py').read_text(encoding='utf-8')
export_source=(ROOT/'EXP-033_EXPORT_CELL.py').read_text(encoding='utf-8')
old_export=find('# Report, optional blind hook, integrity manifest, immutable ZIP upload.')
position=notebook['cells'].index(old_export)
notebook['cells'][position:position+1]=[
    markdown('## Báo cáo kết quả\n\nCác bảng dưới đây được tạo từ dự đoán từng ảnh. Hình được lưu ở `aggregate/figures/` và hiện ngay dưới cell.\n'),
    code(report_source),
    markdown('## Kết luận ngắn\n\nCell tiếp theo hiển thị kết luận tự động từ các số đã tính; bản đầy đủ được lưu trong `aggregate/README.md`.\n'),
    code(export_source)]

for index, cell in enumerate(notebook['cells']):
    cell.setdefault('id', f'exp033-{index:02d}')
    if cell['cell_type']=='code':
        cell['execution_count']=None
        cell['outputs']=[]
        ast.parse(src(cell))
    cell['metadata']={}
notebook['nbformat']=4
notebook['nbformat_minor']=5
assert len({cell['id'] for cell in notebook['cells']}) == len(notebook['cells'])
TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(TARGET, len(notebook['cells']), 'cells')
