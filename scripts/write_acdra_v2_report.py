"""Write the Vietnamese EXP-006 report from measured Validation artifacts."""

import csv
import json
import sys
import zipfile
from pathlib import Path

import yaml

from build_acdra_v2_gallery import confidence_tables, size_table
from run_acdra_v2 import EXP, FULL, REPORT, _main_class, load_class_names, sha256
from src.cashew_detection.postprocess.region_metrics import summarize


def pct(value):
    return f"{100 * float(value):.2f}%"


def number(value):
    return f"{float(value):.3f}"


def derive_test_tables():
    """Summarize the completed Test artifacts without rerunning inference or merging."""
    output = EXP / "results" / "test_locked"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    raw_path = EXP / "results" / "test_raw_detections.jsonl"
    completion = json.loads((REPORT / "test_evaluation_completed.json").read_text(encoding="utf-8"))
    if sha256(raw_path) != completion["raw_jsonl_sha256"]:
        raise RuntimeError("Pinned Test raw JSONL checksum mismatch")
    counts = [0, 0, 0]
    with raw_path.open(encoding="utf-8") as raw_stream, (
            output / "test_regions.jsonl").open(encoding="utf-8") as region_stream:
        for raw_line, region_line in zip(raw_stream, region_stream, strict=True):
            raw, display = json.loads(raw_line), json.loads(region_line)
            if raw["image_id"] != display["image_id"]:
                raise RuntimeError("Test raw/region image order mismatch")
            lookup = {str(item["id"]): int(item["class_id"])
                      for item in raw["detections"]}
            members = [str(member) for region in display["regions"]
                       for member in region["member_detection_ids"]]
            if (len(lookup) != len(raw["detections"]) or len(members) != len(lookup) or
                    set(members) != set(lookup) or
                    any(lookup[str(member)] != int(region["class_id"])
                        for region in display["regions"]
                        for member in region["member_detection_ids"])):
                raise RuntimeError("Test raw box membership or class changed")
            counts[0] += 1; counts[1] += len(lookup); counts[2] += len(display["regions"])
    if counts != [summary["images"], summary["raw_boxes"], summary["regions"]]:
        raise RuntimeError("Test raw/region count mismatch")
    with zipfile.ZipFile(FULL) as archive:
        names = load_class_names(archive)
    rows = list(csv.DictReader((output / "per_image.csv").open(encoding="utf-8")))
    for row in rows:
        row["main_class"] = _main_class(row["image_id"], names)
        for key, value in list(row.items()):
            if key in {"image_id", "main_class", "density_slice"}:
                continue
            row[key] = value == "True" if key in {
                "mixed_disease", "is_negative", "has_cross_class_overlap"} else float(value)
    class_slices = {name: summarize(row for row in rows if row["main_class"] == name)
                    for name in names.values()}
    if sum(part["images"] for part in class_slices.values()) != summary["images"]:
        raise RuntimeError("Test filename class slices do not cover every image")
    class_slices["mixed_disease"] = summarize(row for row in rows if row["mixed_disease"])
    class_slices["negative"] = summarize(row for row in rows if row["is_negative"])
    complete = {**summary, "class_slices": class_slices,
                "class_slices_source": "Saved Test per_image.csv plus filename mapping; no repeat inference or merging"}
    (REPORT / "test_summary_complete.json").write_text(
        json.dumps(complete, indent=2, ensure_ascii=False), encoding="utf-8")
    with (REPORT / "test_class_slices.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", *next(iter(class_slices.values())).keys()])
        writer.writeheader()
        for name, part in class_slices.items():
            writer.writerow({"class": name, **part})
    region_rows = list(csv.DictReader((output / "region_statistics.csv").open(encoding="utf-8")))
    for row in region_rows:
        for key, value in list(row.items()):
            if key not in {"image_id", "region_id", "class_name"}:
                row[key] = float(value)
    with (REPORT / "test_confidence_statistics.csv").open("w", newline="", encoding="utf-8") as cf, (
            REPORT / "test_region_size_statistics.csv").open("w", newline="", encoding="utf-8") as sf:
        conf = csv.DictWriter(cf, fieldnames=["operating_point", "target", "statistic",
                                               "regions_n_ge2", "roc_auc", "pr_auc"])
        size = csv.DictWriter(sf, fieldnames=["operating_point", "class", "regions",
                                               "short_side_p50", "short_side_p95", "long_side_p50",
                                               "short_side_lt32_share", "other_class_overlap_ratio_mean"])
        conf.writeheader(); size.writeheader()
        confidence_tables("test_locked", region_rows, conf)
        size_table("test_locked", region_rows, size)
    return summary, class_slices


def main():
    baseline = json.loads((EXP / "results" / "baseline_A" / "summary.json").read_text(encoding="utf-8"))
    points = yaml.safe_load((EXP / "operating_points.yaml").read_text(encoding="utf-8"))
    summaries = {"A (V005)": baseline}
    for label in points:
        summaries[label] = json.loads((EXP / "results" / label / "summary.json").read_text(encoding="utf-8"))
    confidence_path = REPORT / "confidence_statistics.csv"
    size_path = REPORT / "region_size_statistics.csv"
    confidence = list(csv.DictReader(confidence_path.open(encoding="utf-8"))) if confidence_path.exists() else []
    sizes = list(csv.DictReader(size_path.open(encoding="utf-8"))) if size_path.exists() else []
    lines = ["# Báo cáo EXP-006 — ACDRA v2 (Validation và Test khóa)", "",
             "## Phạm vi và nguồn", "",
             "Chỉ gộp bbox cùng predicted class; detector YOLO26s V005, GT, split và raw detections không đổi. Không có DenseNet/CNN verifier trong V006.",
             "Nguồn FULL V005 và dataset NEG-V03 có SHA256 trong README; checkpoint `best.pt` SHA256 `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19`.",
             "Baseline A tái hiện đúng 56.237 raw → 42.602 region, khớp cả 951 ảnh. TP raw: ghép tham lam confidence giảm dần, cùng class, IoU ≥ 0,5, mỗi GT tối đa một lần.",
             "", "## Bảng A — Baseline V005 tính lại", "",
             "| Metric | Raw | Region A |", "|---|---:|---:|"]
    for label, raw_key, region_key, fn in (
        ("GT coverage IoA ≥0,5", "raw_gt_coverage", "region_gt_coverage", pct),
        ("GT center coverage", "raw_gt_center_coverage", "region_gt_center_coverage", pct),
        ("Excess area theo GT", "raw_excess_area_ratio", "region_excess_area_ratio", pct),
        ("Display occupancy", "raw_display_occupancy", "region_display_occupancy", pct),
        ("FP/ảnh", "raw_fp_per_image", "fp_regions_per_image", number),
        ("Precision", "raw_precision", "region_precision", pct)):
        lines.append(f"| {label} | {fn(baseline[raw_key])} | {fn(baseline[region_key])} |")
    lines += ["", "## Bảng B — So sánh A và ba điểm vận hành", "",
              "| Điểm | Raw/ảnh | Region/ảnh | Giảm toàn bộ | Giảm ảnh ≥100 | Excess area | Region precision | Lost TP | p95 gộp (ms) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, s in summaries.items():
        lines.append(f"| {label} | {number(s['raw_per_image'])} | {number(s['regions_per_image'])} | "
                     f"{pct(s['reduction'])} | {pct(s['dense_ge100']['reduction'])} | "
                     f"{pct(s['region_excess_area_ratio'])} | {pct(s['region_precision'])} | "
                     f"{s['lost_tp']} | {number(s['latency_p95_ms'])} |")
    lines += ["", "Latency p50 / p95 / max (ms) và ảnh chậm nhất:", "",
              "| Điểm | p50 | p95 | max | Ảnh chậm nhất |",
              "|---|---:|---:|---:|---|"]
    for label, s in summaries.items():
        lines.append(f"| {label} | {number(s['latency_p50_ms'])} | "
                     f"{number(s['latency_p95_ms'])} | {number(s['latency_max_ms'])} | "
                     f"{s['slowest_image']} |")
    lines += ["", "## Bảng C — Slices theo mật độ và class chính", "",
              "| Điểm | Slice | Ảnh | Raw/ảnh | Region/ảnh | Giảm | Δ coverage (điểm %) | Excess area |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, s in summaries.items():
        for prefix, slices in (("density", s["slices"]), ("class", s["class_slices"])):
            for key, part in slices.items():
                if not part["images"]:
                    continue
                has_gt = not (prefix == "class" and key in
                              {"healthy", "not_cashew_leaf", "negative"})
                delta = (f"{100 * (part['region_gt_coverage'] - part['raw_gt_coverage']):+.2f}"
                         if has_gt else "N/A")
                lines.append(f"| {label} | {prefix}:{key} | {part['images']} | "
                             f"{number(part['raw_per_image'])} | {number(part['regions_per_image'])} | "
                             f"{pct(part['reduction'])} | {delta} | "
                             f"{pct(part['region_excess_area_ratio'])} |")
    lines += ["", "Ảnh âm (`healthy`, `not_cashew_leaf`) có trong Validation; bảng class và `negative` cho biết số raw/region còn lại. Coverage của slice không có GT là N/A (không có mẫu số).",
              "", "## Bảng D — Confidence của region (Validation, n≥2)", "",
              "Các giá trị max/mean/top-3 là thống kê box thành viên, **không** phải xác suất vùng. File `confidence_statistics.csv` có đủ ROC-AUC và PR-AUC cho hai target (có ≥1 TP; ≥50% thành viên TP).", "",
              "| Điểm | Target | Thống kê PR-AUC tốt nhất | ROC-AUC | PR-AUC |",
              "|---|---|---|---:|---:|"]
    for label in points:
        for target in ("has_tp", "tp_member_share_ge_half"):
            valid = [r for r in confidence if r["operating_point"] == label and
                     r["target"] == target and r["pr_auc"]]
            if valid:
                best = max(valid, key=lambda r: float(r["pr_auc"]))
                lines.append(f"| {label} | {target} | {best['statistic']} | "
                             f"{number(best['roc_auc'])} | {number(best['pr_auc'])} |")
            else:
                lines.append(f"| {label} | {target} | N/A | N/A | N/A |")
    lines += ["", "## Bảng E — Kích thước vùng và chuẩn bị crop", "",
              "| Điểm | Class | Vùng | Vùng/ảnh Validation | Cạnh ngắn p50/p95 (px) | <32 px | Region class khác chồng lấn |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in sizes:
        lines.append(f"| {row['operating_point']} | {row['class']} | {row['regions']} | "
                     f"{number(int(row['regions']) / baseline['images'])} | "
                     f"{number(row['short_side_p50'])}/{number(row['short_side_p95'])} | "
                     f"{pct(row['short_side_lt32_share'])} | "
                     f"{pct(row['other_class_overlap_ratio_mean'])} |")
    lines += ["", "GT class khác trong vùng là phép đo riêng, không đồng nhất với chồng lấn region khác class:", "",
              "| Điểm | GT khác class / diện tích region | Cặp region khác class chồng lấn | Ảnh có chồng lấn |",
              "|---|---:|---:|---:|"]
    for label, s in summaries.items():
        lines.append(f"| {label} | {pct(s['other_class_contamination'])} | "
                     f"{s['cross_class_overlap_pairs']} | {s['images_with_cross_class_overlap']} |")
    lines += ["", "Mức giảm region theo predicted class (so với raw box cùng class):", "",
              "| Điểm | Anthracnose | Leaf miner | Red rust |",
              "|---|---:|---:|---:|"]
    for label, s in summaries.items():
        by_class = s["predicted_class_reduction"]
        lines.append(f"| {label} | {pct(by_class['anthracnose']['reduction'])} | "
                     f"{pct(by_class['leaf_miner']['reduction'])} | "
                     f"{pct(by_class['red_rust']['reduction'])} |")
    lines += ["", "## Diễn giải và giới hạn", "",
              "YOLO detector không đổi; mọi thay đổi trong bảng là chất lượng và số lượng **vùng hiển thị**. Coverage có thể tăng chỉ vì bbox lớn hơn; phải xem cùng excess area và occupancy.",
              "Cạnh ngắn <32 px cho biết tỷ lệ crop sau này phải phóng to; chưa tạo crop, chưa xác minh nhãn bằng DenseNet.",
              "Mọi trial, gồm các trial C trước lần sửa chia cụm được gắn `valid_for_selection=false`, vẫn lưu trong `validation_tuning.csv`; chỉ trial hợp lệ dùng chọn ứng viên và Pareto.",
              "Ba gallery (mỗi điểm 20 ảnh ở từng nhóm sparse, moderate, dense, mixed) ở `visual_gallery/`. Chủ dự án đã duyệt balanced trước khi mở Test.",
              "", "## Test khóa", ""]
    completion = REPORT / "test_evaluation_completed.json"
    if completion.is_file():
        test, test_classes = derive_test_tables()
        locked = yaml.safe_load((EXP / "config_locked.yaml").read_text(encoding="utf-8"))
        lines += [f"Đã chạy một lần với `{locked['approved_point']}` / `{locked['trial_id']}` "
                  f"trên {test['images']} ảnh Test; 0 ảnh âm. Không chọn lại tham số từ Test.", "",
                  "Slice class chính của Test được tính từ `per_image.csv` đã lưu, "
                  "theo tiền tố tên ảnh; bước này chỉ tổng hợp báo cáo, không chạy lại YOLO "
                  "hay thuật toán gộp.", "",
                  "| Metric | Raw Test | Region Test |", "|---|---:|---:|",
                  f"| Số box/vùng | {test['raw_boxes']} | {test['regions']} |",
                  f"| Số box/vùng mỗi ảnh | {number(test['raw_per_image'])} | {number(test['regions_per_image'])} |",
                  f"| GT coverage IoA ≥0,5 | {pct(test['raw_gt_coverage'])} | {pct(test['region_gt_coverage'])} |",
                  f"| GT center coverage | {pct(test['raw_gt_center_coverage'])} | {pct(test['region_gt_center_coverage'])} |",
                  f"| Excess area theo GT | {pct(test['raw_excess_area_ratio'])} | {pct(test['region_excess_area_ratio'])} |",
                  f"| Display occupancy | {pct(test['raw_display_occupancy'])} | {pct(test['region_display_occupancy'])} |",
                  f"| FP/ảnh | {number(test['raw_fp_per_image'])} | {number(test['fp_regions_per_image'])} |",
                  f"| Precision | {pct(test['raw_precision'])} | {pct(test['region_precision'])} |", "",
                  f"Giảm toàn bộ: {pct(test['reduction'])}; ảnh ≥100 raw box: "
                  f"{pct(test['dense_ge100']['reduction'])} ({test['dense_ge100']['images']} ảnh); "
                  f"ảnh ≤10 raw box: {pct(test['sparse_le10']['reduction'])} "
                  f"({test['sparse_le10']['images']} ảnh). `lost_tp={test['lost_tp']}`.",
                  f"Đã đối chiếu độc lập raw Test JSONL với region JSONL: "
                  f"{test['raw_boxes']:,} ID raw xuất hiện đúng một lần trong "
                  "region cùng class.",
                  f"GT class khác trong region: {pct(test['other_class_contamination'])}; "
                  f"{test['cross_class_overlap_pairs']} cặp vùng khác class chồng lấn "
                  f"trên {test['images_with_cross_class_overlap']} ảnh; "
                  f"{pct(test['region_short_side_lt32_share'])} vùng có cạnh ngắn <32 px.",
                  f"Latency p50/p95/max: {number(test['latency_p50_ms'])}/"
                  f"{number(test['latency_p95_ms'])}/{number(test['latency_max_ms'])} ms; "
                  f"ảnh chậm nhất: `{test['slowest_image']}`.", "",
                  "| Slice Test | Ảnh | Raw/ảnh | Region/ảnh | Giảm | Δ coverage (điểm %) | Excess area |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for prefix, parts in (("density", test["slices"]), ("class", test_classes)):
            for key, part in parts.items():
                if not part["images"]:
                    continue
                delta = 100 * (part["region_gt_coverage"] - part["raw_gt_coverage"])
                lines.append(f"| {prefix}:{key} | {part['images']} | "
                             f"{number(part['raw_per_image'])} | {number(part['regions_per_image'])} | "
                             f"{pct(part['reduction'])} | {delta:+.2f} | "
                             f"{pct(part['region_excess_area_ratio'])} |")
        test_confidence = list(csv.DictReader((REPORT / "test_confidence_statistics.csv").open(encoding="utf-8")))
        lines += ["", "Confidence vùng trên Test (chẩn đoán sau khóa; không đổi tham số):", "",
                  "| Target | Thống kê PR-AUC cao nhất | ROC-AUC | PR-AUC |",
                  "|---|---|---:|---:|"]
        for target in ("has_tp", "tp_member_share_ge_half"):
            scored = [row for row in test_confidence if row["target"] == target and row["pr_auc"]]
            if scored:
                best = max(scored, key=lambda row: float(row["pr_auc"]))
                lines.append(f"| {target} | {best['statistic']} | "
                             f"{number(best['roc_auc'])} | {number(best['pr_auc'])} |")
            else:
                lines.append(f"| {target} | N/A | N/A | N/A |")
        lines += ["", "Phân bố kích thước vùng Test theo class nằm trong "
                  "`test_region_size_statistics.csv`; chi tiết từng vùng nằm trong "
                  "`results/test_locked/region_statistics.csv`.", "",
                  f"Trên ảnh dày, Validation giảm {pct(summaries['balanced']['dense_ge100']['reduction'])} "
                  f"và Test giảm {pct(test['dense_ge100']['reduction'])}; "
                  "đây là độ giảm của vùng hiển thị, không phải thay đổi AP của detector.",
                  f"Test giảm số vùng nhưng excess area tăng từ "
                  f"{pct(test['raw_excess_area_ratio'])} lên "
                  f"{pct(test['region_excess_area_ratio'])}; mức tăng coverage "
                  "không tự nó chứng minh crop tốt hơn. Chưa đánh giá DenseNet "
                  "hoặc chất lượng nhãn vùng cuối pipeline.", ""]
    else:
        lines += ["N/A — chưa có `config_locked.yaml` do chủ dự án duyệt. "
                  "Script Test chỉ chạy sau khóa; Test không có ảnh âm.", ""]
    (REPORT / "EXP006_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", REPORT / "EXP006_FINAL_REPORT.md")


if __name__ == "__main__":
    main()
