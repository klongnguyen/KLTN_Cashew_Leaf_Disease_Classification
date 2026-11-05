"""
Shared result builder for both inference engines.

Rules (P0):
- The diagnosis label and its confidence ALWAYS come from the classifier
  (EXP-034 stacker or legacy DenseNet). Boxes never override them.
- A box's confidence is the detector's own score; heuristic regions carry no
  score (conf = None) and are shown as "suspected region".
- A second disease is reported only with strong detector evidence.
"""
import base64
import io
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from config import (
    DISEASE_INFO,
    MULTI_DISEASE_PROTOCOLS,
    SECONDARY_CROP_MIN_CONF,
    SECONDARY_DISEASE_MIN_BOXES,
    SECONDARY_DISEASE_MIN_CONF,
    SECONDARY_DISEASE_SINGLE_BOX_CONF,
)

DISEASE_CLASSES = ("anthracnose", "leaf_miner", "red_rust")


# ----------------------------------------------------------------------------
# Decision helpers
# ----------------------------------------------------------------------------
def abstain_decision(
    probs_sorted: List[Tuple[str, float]], threshold: float, min_margin: float
) -> Tuple[bool, Optional[str]]:
    top1 = probs_sorted[0][1]
    top2 = probs_sorted[1][1] if len(probs_sorted) > 1 else 0.0
    if top1 < threshold:
        return True, f"độ tin cậy {top1 * 100:.1f}% thấp hơn ngưỡng {threshold * 100:.1f}%"
    if top1 - top2 < min_margin:
        return True, (
            f"hai khả năng cao nhất quá sát nhau ({top1 * 100:.1f}% và {top2 * 100:.1f}%)"
        )
    return False, None


def select_display_boxes(
    candidates: List[Dict[str, Any]], primary: str, max_boxes: int
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """Keeps primary-class boxes plus boxes of confirmed secondary diseases.

    Returns (shown, confirmed_secondaries, n_suppressed).
    """
    confirmed: List[str] = []
    for cls in DISEASE_CLASSES:
        if cls == primary:
            continue
        confs = [b["conf"] for b in candidates if b["cls"] == cls and b.get("conf") is not None]
        if not confs:
            continue
        strong = [c for c in confs if c >= SECONDARY_DISEASE_MIN_CONF]
        if len(strong) >= SECONDARY_DISEASE_MIN_BOXES or max(confs) >= SECONDARY_DISEASE_SINGLE_BOX_CONF:
            confirmed.append(cls)
    allowed = {primary, *confirmed}
    shown = [b for b in candidates if b["cls"] in allowed]
    # Secondary classes: keep only their reasonably confident boxes.
    shown = [
        b for b in shown
        if b["cls"] == primary or (b.get("conf") or 0.0) >= SECONDARY_DISEASE_MIN_CONF
    ]
    shown.sort(key=lambda b: -(b.get("conf") or 0.0))
    shown = shown[:max_boxes]
    return shown, confirmed, len(candidates) - len(shown)


def select_verified_boxes(
    verified: List[Dict[str, Any]], primary: str, max_boxes: int
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """Display selection after DenseNet crop verification.

    Each input box carries cls = DenseNet crop label (or None when rejected or
    unverified) and verify = {status, yolo_cls, yolo_conf, dense_conf, crop_id}.
    Shown: boxes whose DenseNet label is the primary disease, plus boxes of a
    secondary disease confirmed by >= 1 confident crop and >= N boxes.
    """
    stats = {"proposed": len(verified), "agree": 0, "differ": 0,
             "rejected_healthy": 0, "unverified": 0, "hidden_other_disease": 0}
    for b in verified:
        st = b["verify"]["status"]
        key = {"agree": "agree", "differ": "differ",
               "rejected_healthy": "rejected_healthy"}.get(st, "unverified")
        stats[key] += 1

    labelled = [b for b in verified if b["cls"] is not None]
    confirmed: List[str] = []
    for cls in DISEASE_CLASSES:
        if cls == primary:
            continue
        # A second disease needs BOTH models to agree on it (YOLO label ==
        # DenseNet crop label) on enough boxes; DenseNet-only votes on leaf
        # fragments are too noisy to claim an extra disease.
        agreeing = [b for b in labelled if b["cls"] == cls and b["verify"]["status"] == "agree"]
        strong = any(b["verify"]["dense_conf"] >= SECONDARY_CROP_MIN_CONF for b in agreeing)
        if strong and len(agreeing) >= SECONDARY_DISEASE_MIN_BOXES:
            confirmed.append(cls)
    shown = [
        b for b in labelled
        if b["cls"] == primary or (b["cls"] in confirmed and b["verify"]["status"] == "agree")
    ]
    stats["hidden_other_disease"] = len(labelled) - len(shown)
    shown.sort(key=lambda b: -(b["verify"]["yolo_conf"]))
    return shown[:max_boxes], confirmed, stats


# ----------------------------------------------------------------------------
# Box formatting
# ----------------------------------------------------------------------------
def _crop_base64(image: Image.Image, box: List[int]) -> str:
    w, h = image.size
    x1, y1, x2, y2 = box
    px, py = max(2, int((x2 - x1) * 0.06)), max(2, int((y2 - y1) * 0.06))
    crop = image.crop((max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)))
    crop.thumbnail((120, 120), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def format_boxes(image: Image.Image, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    w, h = image.size
    out: List[Dict[str, Any]] = []
    for idx, b in enumerate(raw, 1):
        cls = b["cls"]
        info = DISEASE_INFO.get(cls, {})
        x1 = max(0, min(w - 2, int(round(b["xyxy"][0]))))
        y1 = max(0, min(h - 2, int(round(b["xyxy"][1]))))
        x2 = max(x1 + 2, min(w, int(round(b["xyxy"][2]))))
        y2 = max(y1 + 2, min(h, int(round(b["xyxy"][3]))))
        area = (x2 - x1) * (y2 - y1)
        conf = b.get("conf")
        heuristic = b.get("source") == "heuristic"
        verify = b.get("verify")
        if verify is not None:
            yinfo = DISEASE_INFO.get(verify["yolo_cls"], {})
            verify = {
                **verify,
                "yolo_label_vn": yinfo.get("short_name", verify["yolo_cls"]),
                "yolo_conf_percent": round(verify["yolo_conf"] * 100, 1),
                "dense_conf_percent": round(verify["dense_conf"] * 100, 1),
            }
        out.append(
            {
                "id": idx,
                "cls": cls,
                "label_vn": ("Vùng nghi ngờ" if heuristic else info.get("short_name", cls)),
                "full_vn": info.get("vn_name", cls),
                "source": b.get("source", "yolo"),
                "verify": verify,
                "conf": None if conf is None else round(float(conf), 3),
                "conf_percent": None if conf is None else round(float(conf) * 100, 1),
                "color_hex": info.get("color_hex", "#ef4444"),
                "badge_color": info.get("badge_color", "danger"),
                "xyxy": [x1, y1, x2, y2],
                "xyxy_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                "width_px": x2 - x1,
                "height_px": y2 - y1,
                "area_px": area,
                "area_percent": round(area / max(1, w * h) * 100, 2),
                "crop_base64": _crop_base64(image, [x1, y1, x2, y2]),
            }
        )
    return out


# ----------------------------------------------------------------------------
# Conclusion + treatment plan
# ----------------------------------------------------------------------------
def _disease_entry(cls: str) -> Dict[str, Any]:
    info = DISEASE_INFO.get(cls, {})
    return {
        "class_key": cls,
        "label_vn": info.get("vn_name", cls),
        "short_name": info.get("short_name", cls),
        "scientific_name": info.get("scientific_name", ""),
        "color_hex": info.get("color_hex", "#ef4444"),
        "badge_color": info.get("badge_color", "danger"),
        "icon": info.get("icon", "fa-biohazard"),
        "box_count": 0,
        "area_percent": 0.0,
        "max_conf": None,
        "symptoms": info.get("symptoms", ""),
        "management": info.get("management", []),
        "chemical_actives": info.get("chemical_actives", ""),
    }


def _severity(primary: str, damage: float, n_boxes: int, basis: str) -> Tuple[str, str]:
    if primary == "healthy":
        return "healthy", "An Toàn (0% Bệnh)"
    if primary == "not_cashew_leaf":
        return "none", "Không xác định (Dị vật)"
    where = "diện tích lá" if basis == "leaf" else "diện tích ảnh"
    if n_boxes == 0:
        return "low", "Chưa định vị được ổ bệnh (mức độ chưa ước lượng)"
    if damage < 5.0 and n_boxes <= 3:
        return "low", f"Nhẹ — Cấp độ 1 (ước tính {damage}% {where})"
    if damage <= 18.0:
        return "medium", f"Trung bình — Cấp độ 2 (ước tính {damage}% {where})"
    return "high", f"Nặng — Cấp độ 3 (ước tính {damage}% {where})"


PLAN_RETAKE = {
    "title": "Hướng Dẫn Chụp Lại Hình Ảnh Chuẩn Nông Nghiệp",
    "priority_note": "Hệ thống AI không nhận diện được đặc điểm hình thái của lá cây điều trong ảnh.",
    "urgent_actions": [
        "Đảm bảo đối tượng chụp là lá cây điều (Anacardium occidentale).",
        "Chụp ở cự ly gần (20–40 cm), lấy nét rõ nét hệ gân lá và các vết đốm nếu có.",
        "Tránh chụp quá tối, ngược sáng hoặc rung lắc làm mờ chi tiết viền lá.",
    ],
    "chemical_recommendations": [],
    "long_term_prevention": [],
}

PLAN_HEALTHY = {
    "title": "Phác Đồ Chăm Sóc & Duy Trì Sức Khỏe Tán Lá Điều",
    "priority_note": "Cây điều đang trong trạng thái sinh trưởng tốt, không ghi nhận vết bệnh nguy hại.",
    "urgent_actions": [
        "Tiếp tục duy trì chế độ tưới nước và thoát nước hợp lý, tránh ngập úng gốc.",
        "Kiểm tra định kỳ 7–10 ngày/lần để kịp thời bảo vệ các đợt ra đọt non, lộc hoa.",
    ],
    "chemical_recommendations": [
        {
            "target": "Phòng ngừa sinh học",
            "actives": "Không cần sử dụng thuốc bảo vệ thực vật hóa học.",
            "timing": "Có thể phun định kỳ chế phẩm nấm đối kháng Trichoderma hoặc phân bón lá hữu cơ.",
        }
    ],
    "long_term_prevention": [
        "Bón phân cân đối N-P-K theo giai đoạn sinh trưởng của cây.",
        "Bổ sung nguyên tố trung vi lượng (Bo, Kẽm, Canxi) giúp phiến lá dày cứng vững.",
    ],
}


def build_report(
    primary: str,
    confidence: float,
    abstain: bool,
    abstain_reason: Optional[str],
    boxes: List[Dict[str, Any]],
    confirmed_secondaries: List[str],
    damage: float,
    damage_basis: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (conclusion, diseases_detected, treatment_plan)."""
    info = DISEASE_INFO.get(primary, {})
    conf_pct = round(confidence * 100, 2)
    n_boxes = len(boxes)

    # Diseases list: primary first (even without boxes), then confirmed secondaries.
    diseases: List[Dict[str, Any]] = []
    if primary in DISEASE_CLASSES and not abstain:
        stats = {c: _disease_entry(c) for c in [primary, *confirmed_secondaries]}
        for b in boxes:
            d = stats.get(b["cls"])
            if d is None:
                continue
            d["box_count"] += 1
            d["area_percent"] = round(d["area_percent"] + b["area_percent"], 2)
            if b.get("conf") is not None:
                d["max_conf"] = max(d["max_conf"] or 0.0, b["conf"])
        diseases = [stats[primary]] + [stats[c] for c in confirmed_secondaries if stats[c]["box_count"] > 0]
    is_multi = len(diseases) >= 2

    sev_level, sev_vn = _severity(primary, damage, n_boxes, damage_basis)
    base = {
        "confidence_percent": conf_pct,
        "total_boxes": n_boxes,
        "total_damage_area_percent": damage,
        "damage_basis": damage_basis,
        "abstain": abstain,
        "abstain_reason": abstain_reason,
    }

    if abstain:
        conclusion = {
            **base,
            "label": primary,
            "label_vn": f"Cần Chuyên Gia Kiểm Tra (nghiêng về {info.get('short_name', primary)})",
            "is_multi_disease": False,
            "scientific_name": info.get("scientific_name", ""),
            "severity": "Thẩm Định Chuyên Gia",
            "severity_level": "abstain",
            "color_hex": "#eab308",
            "badge_color": "warning",
            "icon": "fa-triangle-exclamation",
            "symptoms": (
                f"Mô hình chưa đủ chắc chắn: {abstain_reason}. "
                "Khuyến nghị chụp lại ảnh rõ hơn hoặc chuyển cán bộ kỹ thuật nông nghiệp kiểm tra trực tiếp."
            ),
        }
        plan = {
            "title": "Khuyến Nghị Chuyển Tiếp Hồ Sơ Chuyên Gia",
            "priority_note": f"Kết quả chưa đủ tin cậy ({abstain_reason}).",
            "urgent_actions": [
                "Chụp lại lá ở cự ly gần, đủ sáng, lấy nét vào vùng vết bệnh rồi chẩn đoán lại.",
                "Thu thập thêm mẫu lá từ cây cùng tán để kiểm tra chéo.",
                "Gửi hình ảnh mẫu hoặc mang mẫu bệnh phẩm tới Chi cục Bảo vệ Thực vật địa phương.",
                "Tạm hoãn phun thuốc BVTV hóa học phổ rộng khi chưa xác định đích danh bệnh hại.",
            ],
            "chemical_recommendations": [],
            "long_term_prevention": [
                "Ghi chép diễn biến triệu chứng bệnh sau mỗi 48 giờ.",
                "Vệ sinh thông thoáng gốc cây để hạn chế mầm bệnh tiềm ẩn.",
            ],
        }
        return conclusion, [], plan

    if primary == "not_cashew_leaf":
        conclusion = {
            **base,
            "label": "not_cashew_leaf",
            "label_vn": "Cảnh Báo: Không Phải Lá Điều",
            "is_multi_disease": False,
            "scientific_name": "Outlier / Non-cashew image",
            "severity": "Không xác định",
            "severity_level": "none",
            "color_hex": "#64748b",
            "badge_color": "secondary",
            "icon": "fa-question-circle",
            "symptoms": info.get("symptoms", ""),
            "total_boxes": 0,
            "total_damage_area_percent": 0.0,
        }
        return conclusion, [], dict(PLAN_RETAKE)

    if primary == "healthy":
        conclusion = {
            **base,
            "label": "healthy",
            "label_vn": "Lá Điều Khỏe Mạnh (An Toàn)",
            "is_multi_disease": False,
            "scientific_name": "Anacardium occidentale (Healthy)",
            "severity": "An Toàn (0% Bệnh)",
            "severity_level": "healthy",
            "color_hex": "#10b981",
            "badge_color": "success",
            "icon": "fa-seedling",
            "symptoms": info.get("symptoms", ""),
            "total_boxes": 0,
            "total_damage_area_percent": 0.0,
        }
        return conclusion, [], dict(PLAN_HEALTHY)

    if is_multi:
        names = " & ".join(d["short_name"] for d in diseases)
        conclusion = {
            **base,
            "label": "multi_disease",
            "primary_label": primary,
            "label_vn": f"{info.get('vn_name', primary)} (kèm dấu hiệu: "
            + ", ".join(d["short_name"] for d in diseases[1:]) + ")",
            "is_multi_disease": True,
            "scientific_name": info.get("scientific_name", ""),
            "severity": sev_vn,
            "severity_level": sev_level,
            "color_hex": "#ef4444",
            "badge_color": "danger",
            "icon": "fa-viruses",
            "symptoms": (
                f"Bệnh chính theo DenseNet (toàn lá): {info.get('vn_name', primary)} ({conf_pct}%). "
                f"DenseNet xác minh trên các vùng YOLO đề xuất còn ghi nhận dấu hiệu rõ của: "
                + ", ".join(d["label_vn"] for d in diseases[1:])
                + f". Tổng {n_boxes} vùng được khoanh."
            ),
        }
        protocol = MULTI_DISEASE_PROTOCOLS.get(tuple(sorted(d["class_key"] for d in diseases))) or {
            "title": f"Phác Đồ Điều Trị Phức Hợp: {names}",
            "priority": "Ưu tiên xử lý bệnh chính trước, đồng thời theo dõi các dấu hiệu bệnh kèm theo.",
            "actions": [
                "Tiến hành cắt tỉa các lá có mảng hoại tử lớn đem tiêu hủy ngay.",
                "Phun thuốc đặc hiệu theo từng nhóm bệnh cách nhau 5–7 ngày để tránh tương tác thuốc.",
            ],
        }
        plan = {
            "title": protocol["title"],
            "priority_note": protocol["priority"],
            "urgent_actions": protocol["actions"],
            "chemical_recommendations": [
                {
                    "target": f"Xử lý {d['label_vn']}",
                    "actives": d["chemical_actives"],
                    "timing": "Bệnh chính ưu tiên trước; bệnh kèm theo xử lý sau 5–7 ngày nếu được xác nhận.",
                }
                for d in diseases
            ],
            "long_term_prevention": [
                "Bón phân cân đối N-P-K, hạn chế thừa phân đạm trong mùa mưa ẩm.",
                "Thường xuyên thăm vườn vào sáng sớm để phát hiện ổ dịch mới.",
            ],
        }
        return conclusion, diseases, plan

    conclusion = {
        **base,
        "label": primary,
        "label_vn": info.get("vn_name", primary),
        "is_multi_disease": False,
        "scientific_name": info.get("scientific_name", ""),
        "severity": sev_vn,
        "severity_level": sev_level,
        "color_hex": info.get("color_hex", "#ef4444"),
        "badge_color": info.get("badge_color", "danger"),
        "icon": info.get("icon", "fa-biohazard"),
        "symptoms": info.get("symptoms", ""),
    }
    located = (
        f"Khoanh được {n_boxes} vùng bệnh, ước tính {damage}% "
        f"{'diện tích lá' if damage_basis == 'leaf' else 'diện tích ảnh'}."
        if n_boxes
        else "Chưa khoanh được vùng bệnh cụ thể; kết luận dựa trên đặc trưng toàn lá."
    )
    plan = {
        "title": f"Phác Đồ Xử Lý & Điều Trị: {info.get('vn_name', primary)}",
        "priority_note": located,
        "urgent_actions": info.get("management", []),
        "chemical_recommendations": [
            {
                "target": f"Thuốc đặc trị {info.get('short_name', primary)}",
                "actives": info.get("chemical_actives", ""),
                "timing": "Phun vào lúc trời râm mát (sáng sớm hoặc chiều tà), phun ướt đều cả 2 mặt lá.",
            }
        ],
        "long_term_prevention": [
            "Cắt tỉa cành khô và cành mọc sát đất để tăng cường thông gió.",
            "Bón vôi bột khử khuẩn đất quanh gốc điều vào đầu mùa mưa.",
        ],
    }
    return conclusion, diseases, plan


def probabilities_list(classes: List[str], probs) -> List[Dict[str, Any]]:
    out = []
    for i, cls in enumerate(classes):
        info = DISEASE_INFO.get(cls, {})
        c = float(probs[i])
        out.append(
            {
                "class_id": i,
                "class_key": cls,
                "label_vn": info.get("vn_name", cls),
                "short_name": info.get("short_name", cls),
                "scientific_name": info.get("scientific_name", ""),
                "severity": info.get("severity", ""),
                "badge_color": info.get("badge_color", "primary"),
                "color_hex": info.get("color_hex", "#2563eb"),
                "icon": info.get("icon", "fa-leaf"),
                "confidence": c,
                "confidence_percent": round(c * 100, 2),
                "symptoms": info.get("symptoms", ""),
            }
        )
    out.sort(key=lambda x: -x["confidence"])
    return out
