"""Write Vietnamese EXP-007 report from measured Validation artifacts only."""

import csv
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT
from scripts.run_acdra_v2 import CHECKPOINT_SHA256, DATA_SHA256, FULL_SHA256, sha256


def pct(value):
    return f"{100 * float(value):.2f}%"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    points = yaml.safe_load((REPORT / "operating_points_v007.yaml").read_text(encoding="utf-8"))
    trials = {row["trial_id"]: row for row in read_csv(REPORT / "validation_tuning_v007.csv")}
    summaries = {
        "Raw": json.loads((EXP / "results" / "trials" / "A_v005.json").read_text(encoding="utf-8")),
        "A": json.loads((EXP / "results" / "trials" / "A_v005.json").read_text(encoding="utf-8")),
        "V006 balanced": json.loads((EXP / "results" / "trials" / "B_e6e7d65f.json").read_text(encoding="utf-8")),
    }
    for point in points:
        summaries[point] = json.loads((EXP / "results" / point / "summary.json").read_text(encoding="utf-8"))
    crop = read_csv(REPORT / "crop_readiness.csv")
    windows = read_csv(REPORT / "crop_window_readiness.csv")
    confidence = read_csv(REPORT / "confidence_statistics_v007.csv")
    sizes = read_csv(REPORT / "image_size_distribution.csv")
    gallery = read_csv(REPORT / "visual_gallery_manifest.csv")
    labels = ("Raw", "A", "V006 balanced", *points)
    lines = ["# EXP-007 — ACDRA v3, Pareto và mức sẵn sàng cho crop", "",
             "## Nguồn và phạm vi", "",
             f"Validation gồm 951 ảnh, 56.237 raw box; checkpoint V005 SHA-256 `{CHECKPOINT_SHA256}`, "
             f"dataset ZIP SHA-256 `{DATA_SHA256}`, V005 FULL ZIP SHA-256 `{FULL_SHA256}`. "
             "Class mapping đọc từ `data_fixed.yaml`. Không huấn luyện lại detector, không tạo crop và không chạy Test.",
             "Năm trial V006 đã được chạy lại trước khi sửa mã V007 và khớp chính xác ba chỉ số tham chiếu; xem `baseline_reproduction.md`.", "",
             "## Bảng A — So sánh toàn Validation", "",
             "`region_precision_strict` là tỷ lệ region có ≥50% raw member TP. "
             "`region_precision` kiểu V006 (chỉ cần ≥1 TP) vẫn nằm trong CSV, nhưng không dùng làm chỉ số chính.", "",
             "| Điểm | Vùng/ảnh | Giảm | Excess | Strict precision | FP raw bị hấp thụ | Chồng lấn cùng class | Lost TP | p50/p95/max ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for label in labels:
        s = summaries[label]
        if label == "Raw":
            lines.append(f"| Raw | {s['raw_per_image']:.2f} | 0.00% | "
                         f"{pct(s['raw_excess_area_ratio'])} | {pct(s['raw_precision_strict'])} | "
                         f"0 | {pct(s['raw_same_class_overlap_area_ratio'])} | 0 | N/A |")
        else:
            latency = f"{s['latency_p50_ms']:.1f}/{s['latency_p95_ms']:.1f}/{s['latency_max_ms']:.1f}"
            lines.append(f"| {label} | {s['regions_per_image']:.2f} | {pct(s['reduction'])} | "
                         f"{pct(s['region_excess_area_ratio'])} | {pct(s['region_precision_strict'])} | "
                         f"{s['raw_fp_absorbed']:,} ({pct(s['raw_fp_absorbed_share'])}) | "
                         f"{pct(s['same_class_overlap_area_ratio'])} | {s['lost_tp']} | {latency} |")
    lines += ["", "### Chỉ ảnh dày (≥100 raw box)", "",
              "| Điểm | Ảnh | Vùng/ảnh | Giảm | Excess | Strict precision | Chồng lấn cùng class |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        s = summaries[label]["dense_ge100"]
        if label == "Raw":
            lines.append(f"| Raw | {s['images']} | {s['raw_per_image']:.2f} | 0.00% | "
                         f"{pct(s['raw_excess_area_ratio'])} | {pct(s['raw_precision_strict'])} | "
                         f"{pct(s['raw_same_class_overlap_area_ratio'])} |")
        else:
            lines.append(f"| {label} | {s['images']} | {s['regions_per_image']:.2f} | "
                         f"{pct(s['reduction'])} | {pct(s['region_excess_area_ratio'])} | "
                         f"{pct(s['region_precision_strict'])} | "
                         f"{pct(s['same_class_overlap_area_ratio'])} |")
    lines += ["", "Diện tích chồng lấn cùng class tính đúng một lần ở chỗ được ≥2 bbox cùng class che; "
              "mẫu số là tổng diện tích bbox (vì vậy raw và region có cùng quy tắc). "
              "Số cặp với IoU>0,1 và tỷ lệ ảnh có cặp nằm trong `validation_tuning_v007.csv`.", "",
              "## Pareto và ba ứng viên", "",
              f"Có {len(trials)} trial V007 ghi đầy đủ config hash; "
              "ba điểm dưới đây là **ứng viên chờ chủ dự án duyệt**, chưa khóa.", "",
              "| Điểm | Trial | Variant | Giảm ảnh dày | Excess | Strict precision | Chồng lấn cùng class |",
              "|---|---|---|---:|---:|---:|---:|"]
    for point, entry in points.items():
        s = summaries[point]
        lines.append(f"| {point} | `{entry['trial_id']}` | {entry['config']['variant']} | "
                     f"{pct(s['dense_ge100']['reduction'])} | {pct(s['region_excess_area_ratio'])} | "
                     f"{pct(s['region_precision_strict'])} | {pct(s['same_class_overlap_area_ratio'])} |")
    lines += ["", "Điểm giữa được chọn bằng quy tắc: trong các điểm Pareto đạt ít nhất 60% giảm "
              "trên ảnh dày, lấy điểm có excess thấp nhất. Đây là cách trình bày ba mức "
              "đánh đổi để duyệt ảnh, không phải ngưỡng loại trial.", "",
              f"Mục tiêu thăm dò đạt được ở balanced: giảm {pct(summaries['balanced']['dense_ge100']['reduction'])} "
              f"trên ảnh dày, excess {pct(summaries['balanced']['region_excess_area_ratio'])} "
              "thấp hơn mốc C2 V006 (46,47%), strict precision "
              f"{pct(summaries['balanced']['region_precision_strict'])} "
              f"cao hơn raw {pct(summaries['A']['raw_precision_strict'])}. "
              "Đây là kết quả Validation; gallery vẫn cần được duyệt trước khi khóa.", "",
              "- [Pareto giảm–excess](pareto_excess_area.png)",
              "- [Pareto giảm–strict precision](pareto_strict_precision.png)",
              "- [Pareto giảm–chồng lấn cùng class](pareto_same_class_overlap.png)",
              "- [Toàn bộ trial](validation_tuning_v007.csv) và [các điểm không bị trội](pareto_frontier_v007.csv)", "",
              "## Bảng B — Ảnh thưa (≤10 raw box)", "",
              "Cờ duyệt mềm bật khi >15% ảnh thưa có ít nhất một box được gộp; không loại tự động.", "",
              "| Điểm | Ảnh thưa bị đổi / 329 | Box được gộp | Giảm trên nhóm | Cờ duyệt |",
              "|---|---:|---:|---:|---|"]
    for label in labels[1:]:
        s = summaries[label]
        lines.append(f"| {label} | {s['sparse_images_changed']}/329 "
                     f"({pct(s['sparse_images_changed_share'])}) | {s['sparse_boxes_merged']} | "
                     f"{pct(s['sparse_le10']['reduction'])} | "
                     f"{'Cần xem ảnh' if s['sparse_review_flag'] else 'Theo dõi'} |")
    lines += ["", "## Bảng C — Mức sẵn sàng cho crop", "",
              "Vùng đủ điều kiện xác minh: `n_detections≥2` và cạnh ngắn ≥`s_min`. "
              "GT lesion được tính nếu ≥50% diện tích GT nằm trong ít nhất một vùng đủ điều kiện cùng class. "
              "Đây là phép đo hình học, chưa đánh giá DenseNet.", "",
              "| Điểm | s_min | Vùng đủ điều kiện | Raw box trong vùng | Raw TP trong vùng | GT lesion được phủ |",
              "|---|---:|---:|---:|---:|---:|"]
    for point in points:
        for threshold in (32, 48, 64):
            parts = [row for row in crop if row["point"] == point and int(row["s_min_px"]) == threshold]
            def pooled(numerator, denominator):
                a = sum(int(row[numerator]) for row in parts)
                b = sum(int(row[denominator]) for row in parts)
                return f"{a:,}/{b:,} ({pct(a / b) if b else 'N/A'})"
            lines.append(f"| {point} | {threshold} | "
                         f"{pooled('verifiable_regions', 'regions')} | "
                         f"{pooled('verifiable_raw_boxes', 'raw_boxes')} | "
                         f"{pooled('verifiable_raw_tp', 'raw_tp')} | "
                         f"{pooled('verifiable_gt', 'gt')} |")
    middle = [row for row in crop if row["point"] == "balanced" and int(row["s_min_px"]) == 32]
    middle_regions = sum(int(row["verifiable_regions"]) for row in middle)
    middle_total = sum(int(row["regions"]) for row in middle)
    middle_tp = sum(int(row["verifiable_raw_tp"]) for row in middle)
    middle_tp_total = sum(int(row["raw_tp"]) for row in middle)
    middle_gt = sum(int(row["verifiable_gt"]) for row in middle)
    middle_gt_total = sum(int(row["gt"]) for row in middle)
    lines += ["", f"**Kết luận crop:** ở điểm balanced và `s_min=32`, có "
              f"{middle_regions:,}/{middle_total:,} region ({pct(middle_regions/middle_total)}) "
              f"đủ điều kiện hình học; các vùng này chứa {middle_tp:,}/{middle_tp_total:,} "
              f"raw TP ({pct(middle_tp/middle_tp_total)}) và phủ {middle_gt:,}/{middle_gt_total:,} "
              f"GT lesion ({pct(middle_gt/middle_gt_total)}). Có vùng đủ lớn để thử crop, "
              "nhưng phần lớn region vẫn quá nhỏ hoặc chỉ có một detection; chất lượng "
              "phân loại crop chưa được kiểm chứng."]
    lines += ["", "Bảng theo từng class, tỷ lệ cạnh ngắn <32/<64 và các ngưỡng 32/48/64 px nằm trong "
              "[crop_readiness.csv](crop_readiness.csv).", "",
              "Cửa sổ giả định là hình vuông quanh tâm region, cạnh bằng `max(cạnh dài region, W_min)`; "
              "đo cắt biên trước khi kẹp vào kích thước ảnh. `other_class_region_inside` nghĩa là crop "
              "giao với region khác class, không phải GT khác class.", "",
              "| Điểm | W_min | Crop bị cắt biên | Crop gặp region khác class | Cặp crop cùng class IoU>0,5 |",
              "|---|---:|---:|---:|---:|"]
    for point in points:
        for minimum in (64, 96, 128):
            parts = [row for row in windows if row["point"] == point and int(row["W_min_px"]) == minimum]
            n = sum(int(row["regions"]) for row in parts)
            cut = sum(int(row["boundary_clipped"]) for row in parts)
            mixed = sum(int(row["other_class_region_inside"]) for row in parts)
            pairs = sum(int(row["same_class_crop_overlap_pairs_iou_gt05"]) for row in parts)
            lines.append(f"| {point} | {minimum} | {cut:,}/{n:,} ({pct(cut/n)}) | "
                         f"{mixed:,}/{n:,} ({pct(mixed/n)}) | {pairs:,} |")
    lines += ["", "[Bảng crop theo class](crop_window_readiness.csv). "
              "Ảnh Validation trong dataset ZIP có kích thước:", "",
              "| W×H | Số ảnh |", "|---|---:|"]
    for row in sizes:
        lines.append(f"| {row['width']}×{row['height']} | {row['images']} |")
    lines += ["", "Ảnh gốc độ phân giải cao hơn: **N/A** — ZIP nguồn YOLO chỉ có `images/train`, "
              "`images/val`, `images/test`; chưa xác định được cặp ảnh gốc có ánh xạ tên/GT để dùng crop. "
              "Không thể kết luận ảnh YOLO còn nguyên độ phân giải gốc chỉ từ kích thước hiện có.", "",
              "## Bảng D — Confidence region n≥2", "",
              "PR-AUC cần đọc cùng base rate của target; score mean/top3 là thống kê thành viên, không phải xác suất vùng.", "",
              "| Điểm | Target | Score | Base rate | ROC-AUC | PR-AUC |",
              "|---|---|---|---:|---:|---:|"]
    for point in points:
        for target in ("has_tp", "tp_member_share_ge_half"):
            subset = [row for row in confidence if row["point"] == point and row["target"] == target]
            for score in ("mean", "top3_mean"):
                row = next(row for row in subset if row["statistic"] == score)
                lines.append(f"| {point} | {target} | {score} | {pct(row['base_rate'])} | "
                             f"{float(row['roc_auc']):.3f} | {float(row['pr_auc']):.3f} |")
    lines += ["", "[Toàn bộ score và target](confidence_statistics_v007.csv).", "",
              "## Bảng E — Slice theo mật độ và class chính", "",
              "| Điểm | Slice | Ảnh | Giảm | Excess | Strict precision |",
              "|---|---|---:|---:|---:|---:|"]
    for point in points:
        summary = summaries[point]
        for slice_name, part in summary["slices"].items():
            lines.append(f"| {point} | density:{slice_name} | {part['images']} | "
                         f"{pct(part['reduction'])} | {pct(part['region_excess_area_ratio'])} | "
                         f"{pct(part['region_precision_strict'])} |")
        for class_name in ("anthracnose", "leaf_miner", "red_rust", "mixed_disease"):
            part = summary["class_slices"][class_name]
            lines.append(f"| {point} | class:{class_name} | {part['images']} | "
                         f"{pct(part['reduction'])} | {pct(part['region_excess_area_ratio'])} | "
                         f"{pct(part['region_precision_strict'])} |")
    lines += ["", "## Ảnh trước/sau và giới hạn suy luận", "",
              f"Gallery có {len(gallery)} ảnh. Mỗi ứng viên có 20 ảnh cho từng nhóm sparse, moderate, dense, mixed; "
              "thêm ảnh sparse bị thay đổi và 10 ảnh có chồng lấn cùng class nặng nhất.", "",
              "- [Danh sách ảnh](visual_gallery_manifest.csv)",
              "- [Cấu hình ba ứng viên](operating_points_v007.yaml)", "",
              "Test V006 đã được dùng một lần cho `B_e6e7d65f`. Vì vậy EXP-007 chưa chạy Test; "
              "nếu chủ dự án khóa một ứng viên và đánh giá tiếp, kết quả đó phải ghi là "
              "**đánh giá nội bộ, không còn là Test nguyên vẹn**. Không chọn lại tham số từ Test. "
              "Một Real Holdout độc lập mới sẽ phù hợp hơn cho báo cáo công bố.", "",
              f"Môi trường: Python {platform.python_version()}; seed gallery 42; "
              "checkpoint/dataset không đổi. Hash CSV trial: "
              f"`{sha256(REPORT / 'validation_tuning_v007.csv')}`.", ""]
    (REPORT / "EXP007_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
