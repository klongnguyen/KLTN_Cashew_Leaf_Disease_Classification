"""Read-only P0 audit of EXP-024/025 predictions and seed-42 Grad-CAM.

Run with --help. All outputs are written under a new analysis directory.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-024-256px-f9fd5d5f"
EVAL_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-025-256px"
ANALYSIS_ID = "ANALYSIS-DENSENET121-EXP025-P0-GRADCAM-001"
SEEDS = (42, 123, 2026, 3407, 7777)
CLASSES = ("anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf")
HASHES = {
    "split11": "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49",
    "holdout05": "8d5c3758b5869c3ae19238fa8d78e53ad15bb40544f4c70f632602896d80045f",
    "seed42": "fddcde3988af72b4446468a813a0d516f45663459fd7c761bec4aed58ff662c5",
}
P0 = [
    ("holdout", f"healthy/real_healthy_{n:03}.jpg", "healthy_wrong") for n in (27,29,33,34,36)
] + [
    ("holdout", "anthracnose/real_anthracnose_078.jpg", "disease_to_healthy"),
    ("holdout", "leaf_miner/real_leaf_miner_131.jpg", "disease_to_healthy"),
    ("holdout", "anthracnose/real_anthracnose_074.jpg", "disease_to_healthy"),
    ("holdout", "anthracnose/real_anthracnose_052.jpg", "disease_to_healthy"),
] + [
    ("holdout", f"healthy/real_healthy_{n:03}.jpg", "healthy_control") for n in range(22,27)
] + [
    ("validation", p, "validation_priority") for p in (
        "val/healthy/real_healthy_035.jpg", "val/healthy/healthy_1025.jpg",
        "val/healthy/healthy_1132.jpg", "val/healthy/healthy_1135.jpg",
        "val/anthracnose/anthracnose_1485.jpg", "val/anthracnose/anthracnose_276.jpg",
        "val/anthracnose/anthracnose_1256.jpg")
] + [
    ("validation", f"val/healthy/real_healthy_{n:03}.jpg", "validation_control")
    for n in (2,10,11,12,13,37)
]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def locate(root, suffix):
    if not root: return None
    root = Path(root)
    if root.is_dir():
        p = root / suffix
        return p if p.is_file() else None
    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as z:
            matches = [n for n in z.namelist() if n.endswith("/" + suffix)]
            if len(matches) == 1:
                return (root, matches[0])
    return None

def load_rows(source):
    if isinstance(source, tuple):
        with zipfile.ZipFile(source[0]) as z:
            text = z.read(source[1]).decode("utf-8-sig").splitlines()
        return list(csv.DictReader(text))
    return csv_rows(source)

def read_json(source):
    if isinstance(source, tuple):
        with zipfile.ZipFile(source[0]) as z: return json.loads(z.read(source[1]))
    return json.loads(source.read_text(encoding="utf-8"))

def norm_path(p, split):
    p = str(p).replace("\\", "/").lstrip("/")
    if split == "validation" and not p.startswith("val/"): p = "val/" + p
    return p

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("exp024", "exp025", "split11_zip", "holdout05_zip", "checkpoint"):
        ap.add_argument("--" + arg.replace("_", "-"), type=Path)
    ap.add_argument("--output", type=Path, default=Path("analysis_results") / ANALYSIS_ID)
    ap.add_argument("--holdout-inventory", type=Path, help="Previously audited 025 inventory, when the original ANALYSIS ZIP is unavailable")
    args = ap.parse_args()
    out = args.output.resolve()
    sources = [p.resolve() for p in (args.exp024,args.exp025,args.split11_zip,args.holdout05_zip,args.checkpoint) if p]
    if any(out == p or out in p.parents or p in out.parents for p in sources):
        ap.error("Output must be separate from every source")
    if out.exists() and any(out.iterdir()): ap.error("Output already contains files; use a new directory")
    tables = out / "tables"; figures = out / "figures"; out.mkdir(parents=True,exist_ok=True)
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    missing=[]; fingerprints={}; issues=[]; records=[]
    config={"analysis_id":ANALYSIS_ID,"exp024":str(args.exp024) if args.exp024 else None,
            "exp025":str(args.exp025) if args.exp025 else None,"split11_zip":str(args.split11_zip) if args.split11_zip else None,
            "holdout05_zip":str(args.holdout05_zip) if args.holdout05_zip else None,
            "checkpoint":str(args.checkpoint) if args.checkpoint else None,"seed":42,"img_size":256,
            "class_names":CLASSES,"run_training":False,"run_test":False,"run_holdout_reevaluation":False}
    (out/"analysis_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    for key,p in (("split11",args.split11_zip),("holdout05",args.holdout05_zip),("seed42",args.checkpoint)):
        if not p or not p.is_file(): missing.append({"asset":key,"reason":"MISSING"}); continue
        actual=digest(p); fingerprints[key]={"path":str(p.resolve()),"sha256":actual,"expected":HASHES[key],"verified":actual==HASHES[key]}
        if actual != HASHES[key]: issues.append(f"{key} SHA256 mismatch")
    if args.exp024 and args.exp024.is_file():
        fingerprints["exp024_full"]={"path":str(args.exp024.resolve()),"sha256":digest(args.exp024)}
    lock_source=locate(args.exp024,"validation_selection_lock.json")
    cfg_source=locate(args.exp024,"experiment_config.json")
    if not lock_source: missing.append({"asset":"EXP024 validation_selection_lock.json","reason":"MISSING"})
    if not cfg_source: missing.append({"asset":"EXP024 experiment_config.json","reason":"MISSING"})
    if lock_source and cfg_source:
        lock=read_json(lock_source); cfg=read_json(cfg_source)
        if lock.get("deployment_seed") != 42: issues.append("EXP024 deployment seed is not 42")
        if cfg.get("experiment_id") != TRAIN_ID: issues.append("EXP024 experiment ID mismatch")
        if cfg.get("dataset",{}).get("class_names") != list(CLASSES): issues.append("Class order mismatch")
        if cfg.get("dataset",{}).get("manifest_sha256") != "91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75": issues.append("Split11 manifest mismatch")
        if args.checkpoint and args.checkpoint.is_file() and args.checkpoint.name.endswith("weights.h5"):
            if lock.get("selected_checkpoint_sha256",{}).get("42") != digest(args.checkpoint): issues.append("Seed42 lock/checkpoint mismatch")
        if not isinstance(cfg_source,tuple) and digest(cfg_source)!=lock.get("config_sha256"):
            issues.append("EXP024 config SHA256 mismatch")
    for split,root,pattern,total in (("validation",args.exp024,"seed_{seed}/validation/predictions_standard.csv",1447),
                                     ("holdout",args.exp025,"real_holdout/seed_{seed}/predictions.csv",120)):
        for seed in SEEDS:
            source=locate(root,pattern.format(seed=seed))
            if not source:
                missing.append({"asset":f"{split} predictions seed {seed}","reason":"MISSING"}); continue
            data=load_rows(source)
            if len(data)!=total: issues.append(f"{split} seed {seed}: {len(data)} rows, expected {total}")
            seen=set()
            for row in data:
                path=norm_path(row.get("relative_path",""),split)
                if path in seen: issues.append(f"duplicate {split} seed {seed}: {path}")
                seen.add(path)
                true=row.get("class" if split=="holdout" else "true_label","")
                pred=row.get("predicted_class" if split=="holdout" else "predicted_label","")
                if str(true).isdigit(): true=CLASSES[int(true)]
                if str(pred).isdigit(): pred=CLASSES[int(pred)]
                records.append({"split":split,"relative_path":path,"seed":seed,"true_label":true,
                                "predicted_label":pred,"confidence":row.get("confidence",""),
                                "p_healthy":row.get("probability_healthy" if split=="holdout" else "prob_healthy",""),
                                "correct":int(true==pred),"sha256":row.get("sha256","")})
    if args.holdout_inventory and not any(r["split"]=="holdout" for r in records):
        cached=csv_rows(args.holdout_inventory)
        if len(cached)!=120: issues.append(f"Cached Holdout inventory has {len(cached)} images, expected 120")
        for row in cached:
            for seed in SEEDS:
                pred=row.get(f"seed{seed}_prediction","")
                true=row.get("true_label","")
                if not pred: continue
                records.append({"split":"holdout","relative_path":row["relative_path"],"seed":seed,
                                "true_label":true,"predicted_label":pred,
                                "confidence":row.get(f"seed{seed}_confidence",""),"p_healthy":"",
                                "correct":int(true==pred),"sha256":row.get("sha256","")})
        fingerprints["holdout_cached_inventory"]={"path":str(args.holdout_inventory.resolve()),
                                                    "sha256":digest(args.holdout_inventory),
                                                    "note":"Derived from EXP-025 ANALYSIS ZIP in earlier run; original ZIP unavailable now"}
        missing=[m for m in missing if not m["asset"].startswith("holdout predictions seed")]
    grouped=defaultdict(list)
    for r in records: grouped[(r["split"],r["relative_path"])].append(r)
    inventory=[]
    for (split,path),rs in sorted(grouped.items()):
        byseed={r["seed"]:r for r in rs}; first=byseed.get(42,rs[0]); counts=Counter(r["predicted_label"] for r in rs)
        rec={"split":split,"relative_path":path,"image_id":f"{split}:{path}","sha256":first["sha256"],
             "true_label":first["true_label"],"wrong_n_seeds":sum(1-r["correct"] for r in rs),
             "predicted_healthy_n_seeds":sum(r["predicted_label"]=="healthy" for r in rs),
             "prediction_distribution":json.dumps(counts,sort_keys=True),"seed42_correct":byseed.get(42,{}).get("correct",""),
             "seed42_predicted_label":byseed.get(42,{}).get("predicted_label",""),
             "seed42_confidence":byseed.get(42,{}).get("confidence","")}
        for seed in SEEDS:
            rec[f"seed{seed}_prediction"]=byseed.get(seed,{}).get("predicted_label","")
            rec[f"seed{seed}_confidence"]=byseed.get(seed,{}).get("confidence","")
        inventory.append(rec)
    inv_fields=list(inventory[0]) if inventory else ["split","relative_path","image_id","sha256","true_label","wrong_n_seeds","predicted_healthy_n_seeds","prediction_distribution","seed42_correct","seed42_predicted_label","seed42_confidence"] + [f"seed{s}_{x}" for s in SEEDS for x in ("prediction","confidence")]
    write_csv(tables/"error_inventory_all_seeds.csv",inventory,inv_fields)
    lookup={(r["split"],r["relative_path"]):r for r in inventory}
    priority=[]
    for split,path,group in P0:
        r=dict(lookup.get((split,path),{})); r.update({"split":split,"relative_path":path,"priority_group":group,"prediction_status":"FOUND" if r.get("image_id") else "MISSING_PREDICTIONS"}); priority.append(r)
    write_csv(tables/"p0_priority_cases.csv",priority,["priority_group","prediction_status"]+inv_fields)
    review=[]
    for i,(split,path,group) in enumerate(random.Random(42).sample(P0,len(P0)),1):
        review.append({"blind_id":f"P0-{i:03}","split":split,"relative_path":path,"current_label":path.split("/")[-2],
                       "reviewer_id":"","review_date":"","leaf_age":"unknown","leaf_color":"unknown",
                       "visible_lesion":"unknown","lesion_type_notes":"","shadow":"unknown","lighting":"unknown",
                       "blur":"unknown","background":"unknown","leaf_visibility":"unknown",
                       "source_visit":"UNKNOWN","source_tree_id":"UNKNOWN","source_leaf_id":"UNKNOWN",
                       "camera_id":"UNKNOWN","label_review":"","suggested_label":"","reviewer_notes":""})
    write_csv(tables/"review_blinded_template.csv",review,list(review[0]))
    for name,fields in (("image_quality_features.csv",["split","relative_path","image_sha256","v_mean","v_median","v_p10","v_p90","s_mean","s_median","gray_std","laplacian_variance","status"]),
                        ("group_comparison_summary.csv",["group","n","metric","mean","median","status"]),
                        ("cam_attribution_summary.csv",["split","relative_path","image_sha256","seed","true_label","predicted_label","confidence","wrong_n_seeds","predicted_healthy_n_seeds","model_sha256","source_archive_sha256","cam_target","cam_layer_or_source","cam_filename","cam_status","leaf_roi_available","saliency_on_leaf_ratio","visual_pattern","reviewer_comment","interpretation_confidence"])):
        write_csv(tables/name,[],fields)
    write_csv(tables/"missing_assets.csv",missing,["asset","reason"])
    (out/"source_fingerprints.json").write_text(json.dumps(fingerprints,indent=2),encoding="utf-8")
    status={"prediction_rows":len(records),"unique_images":len(inventory),"priority_cases":len(priority),
            "images":"BLOCKED_MISSING_IMAGES" if not args.split11_zip or not args.holdout05_zip else "NOT_RUN",
            "gradcam":"BLOCKED_MISSING_CHECKPOINT" if not args.checkpoint else ("BLOCKED_MISSING_IMAGES" if not args.split11_zip or not args.holdout05_zip else "NOT_IMPLEMENTED"),
            "issues":issues,"missing_assets":len(missing)}
    holdout=[r for r in inventory if r["split"]=="holdout"]
    healthy_wrong=[r for r in holdout if r["true_label"]=="healthy" and r["seed42_correct"]==0]
    disease_to_healthy=[r for r in holdout if r["true_label"] not in ("healthy","not_cashew_leaf") and r["seed42_predicted_label"]=="healthy"]
    status.update({"holdout_seed42_healthy_wrong":len(healthy_wrong),
                   "holdout_seed42_disease_to_healthy":len(disease_to_healthy),
                   "holdout_all_five_wrong":sum(r["wrong_n_seeds"]==5 for r in holdout)})
    (out/"execution_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    (out/"README.md").write_text("# P0 Healthy error audit\n\nRead-only source analysis. Empty feature/CAM CSV means no verified image/model processing. Review template is blinded to predictions and contains no review decisions.\n\nRun `uv run --python 3.11 --no-project python analysis/scripts/exp025_p0_audit.py --help`. Supply original archives and checkpoint. Output must be a fresh directory.\n",encoding="utf-8")
    (out/"REPORT_P0_HEALTHY_ERROR_ANALYSIS.md").write_text(
        "# P0 Error Analysis — DenseNet121 EXP-024/025\n\n## Executive summary\n"
        f"Loaded {len(records)} prediction rows for {len(inventory)} distinct images. "
        f"Seed 42: {len(healthy_wrong)} Healthy errors and {len(disease_to_healthy)} disease-to-Healthy errors on Holdout. "
        f"All-five-seed errors on Holdout: {status['holdout_all_five_wrong']}. Grad-CAM: {status['gradcam']}.\n\n"
        "## Source and checkpoint integrity\n"+json.dumps(fingerprints,indent=2)+"\n\n"
        "## Prediction error inventory\nSee `tables/error_inventory_all_seeds.csv` and `tables/p0_priority_cases.csv`. "
        "Validation predictions are unavailable when EXP-024 ANALYSIS/FULL is absent.\n\n"
        "## Image and label audit\nHuman review pending; no labels changed.\n\n"
        "## Grad-CAM and limitations\nNo attribution is claimed without verified source images, checkpoint and parity checks.\n\n"
        "## Next controlled experiment\nAwait source assets and blinded human label review. Holdout 05 has only 13 Healthy images and is historically exposed; use a new independent five-class field test for final assessment.\n",
        encoding="utf-8")
    checks=[f"{digest(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file() and p.name!="CHECKSUMS.sha256"]
    (out/"CHECKSUMS.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**status},indent=2))

if __name__=="__main__": main()
