"""
Configuration and Disease Reference Data for Cashew Leaf Disease Diagnosis.
Model: DenseNet121 Transfer Version 014 (Best Seed: 123)
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014"
    / "model_package"
    / "deployment_model.keras"
)

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014"
    / "model_package"
    / "label_map.json"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL"
    / "EXP-DENSENET121-TRANSFER-5SEEDS-014"
    / "model_package"
    / "deployment_metadata.json"
)

# Fallback checkpoint included in repository when deployment_model.keras is absent
FALLBACK_MODEL_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "archive_legacy_dataset"
    / "EXP-DENSENET121-SCRATCH-001"
    / "checkpoints"
    / "EXP-DENSENET121-SCRATCH-001_best.keras"
)

FALLBACK_LABEL_MAP = {
    "0": "anthracnose",
    "1": "healthy",
    "2": "leaf_miner",
    "3": "not_cashew_leaf",
    "4": "red_rust",
}

STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"

# Legacy engine model specs (DenseNet121 Transfer EXP-014, seed 123).
# These describe the legacy model only; EXP-034 reports its own metrics.
IMG_SIZE = 224
MODEL_NAME = "DenseNet121 Transfer EXP-014 (seed 123)"
BEST_SEED = "123"
VAL_ACCURACY = 0.9727
VAL_LOSS = 0.0925
VAL_MACRO_F1 = 0.9714

# The repository fallback checkpoint (EXP-DENSENET121-SCRATCH-001) was trained
# from scratch on the obsolete CashewData_v01 snapshot. It must never be served
# silently in place of the real model; enable only for local debugging.
ALLOW_SCRATCH_FALLBACK = False
SCRATCH_FALLBACK_NAME = "DenseNet121 Scratch EXP-001 (dataset v01 — CHỈ ĐỂ DEBUG)"

# Legacy engine decision rule (no validated stacker/threshold exists for it).
LEGACY_ABSTAIN_THRESHOLD = 0.60

# ================= EXP-034: DenseNet121 x5 + YOLO26s + Stacker F3 =================
# Engine selection: "exp034" (default) or "legacy" (DenseNet v014 + Cách 1).
# If "exp034" fails to load (missing bundle, SHA mismatch, missing packages)
# the server falls back to "legacy" and reports it in /api/model-info and in
# every prediction (model_meta.engine_warning).
MODEL_ENGINE = "exp034"

# ================= Decision architecture (thesis Fig. 3.15) =================
#   1. DenseNet x5 on the whole image: is it a cashew leaf? (gate below)
#   2. YOLO26s detects lesion boxes.
#   3. Each box is cropped (own window, small context) and classified by
#      DenseNet x5, which VERIFIES the YOLO label (see LABEL_SWITCH_MIN_CONF).
#   4. Result = the diseases present on the leaf, each with its boxes.
#      No box left -> whole-leaf DenseNet: healthy -> "healthy leaf",
#      otherwise "suspected disease, no lesion localised" (expert check).
# The whole-leaf DenseNet label and the frozen Stacker F3 are still computed
# and returned as references only (decision.densenet_label / stacker_label).
FINAL_DECISION = "boxes"
DENSENET_ABSTAIN_THRESHOLD = 0.60
# A disease is reported when at least this many verified boxes carry it.
# Image-level exact match of the disease set (YOLO NEG-V03, 300 imgs each):
#   1 box: Val 71.8% (Test 77.3%) | 2 boxes: 57.1% (58.7%) | 3 boxes: 49.5% (47.7%)
# Val healthy/negative images predicted "no disease": 18/18 for every value.
DISEASE_MIN_BOXES = 1

# ================= Out-of-distribution gate (step 0, HARD rule) =================
# DenseNet x5 runs on the whole image FIRST. The image is rejected as
# "not a cashew leaf" -- and YOLO is never run -- when:
#   (a) argmax is not_cashew_leaf, or
#   (b) the kNN distance to the cashew-leaf reference bank (train images) is
#       in the HARD zone: > knn_threshold_high, or
#   (c) only when OOD_BAND_ACTION == "reject": it is in the BAND
#       (knn_threshold, knn_threshold_high] and DenseNet is not confidently
#       cashew (p_not_cashew >= OOD_BAND_MAX_P_NOT_CASHEW or top cashew
#       probability < OOD_BAND_MIN_TOP_CASHEW), or
#   (d) the Energy score exceeds its threshold (only if OOD_USE_ENERGY).
# With OOD_BAND_ACTION == "warn" (default) a BAND image is accepted with a
# warning "differs from the training photos". Reason: the band thresholds are
# Validation quantiles (same capture conditions as train), so screenshots and
# web photos of real cashew leaves often land in the band.
# kNN embedding: for each of the 5 seeds the L2-normalised mean of the image
# and its horizontal flip; distance = 1 - mean cosine over seeds, to the k-th
# nearest bank image. Both thresholds are quantiles (0.99 / 0.999) of cashew
# VALIDATION scores only (tools/build_ood_reference.py).
# Neither YOLO, the crop verification nor the stacker can override it.
OOD_GATE_ENABLED = True
# OOD_DIR / OOD_REFERENCE_PATH / OOD_CALIBRATION_PATH / OOD_KNN_HARD_THRESHOLD
# are set per model bundle (see BUNDLES at the end of this file).
OOD_KNN_K = 5
OOD_BANK_SEED = 42  # only for a legacy single-seed bank (2-D array)
OOD_BAND_ACTION = "warn"  # "warn" | "reject" (previous behaviour)
# Hard kNN rejection threshold (per bundle, BUNDLES[...]["knn_hard_threshold"]).
# None -> knn_threshold_high of the calibration file (99.9% quantile of cashew
# Validation, too strict for real photos). EXP-034, measured 2026-11-07
# through the web API:
#   t_high | real cashew photos accepted (Holdout 126 + realdata 28)
#          | not_cashew Val+Test rejected (470) | corn x4 + other plants rejected (500)
#   0.2106 |  98.1% | 100% | 99.0%
#   0.25   | 100.0% | 100% | 95.6%   <- chosen (knee of the trade-off)
#   0.30   | 100.0% | 100% | 90.4%
#   argmax only (no kNN): corn+other 76.0% (healthy corn 25%)
# The Holdout photos were used to pick this value: report them as
# descriptive for the gate, not as a blind benchmark.
OOD_BAND_MAX_P_NOT_CASHEW = 0.01
OOD_BAND_MIN_TOP_CASHEW = 0.90
# Energy is computed and shown but NOT used to reject by default: on the
# held-out cashew Test split it wrongly rejected 39% of real field photos
# (accepted 60.7% vs 100% with argmax+kNN) while adding almost no OOD
# rejection beyond kNN (diseased corn: kNN alone 99.3-100%).
OOD_USE_ENERGY = False

# Crop verification. Every box gets its OWN square window centred on it,
# side = max(VERIFY_CROP_CONTEXT x the box's long side, VERIFY_CROP_MIN_PX),
# cut from the full-resolution original (no window sharing between boxes).
# The VERIFY_MAX_CROPS most confident boxes are verified; the others keep
# YOLO's label ("chưa xác minh").
# Box-label study (tools/eval_box_labels.py + analyze_box_labels.py), YOLO
# NEG-V03 dataset, stratified 300 Val / 300 Test images, top-16 boxes/image,
# boxes matched to ground truth at IoU >= 0.5 (false positives = "background").
# Setting selected on VALIDATION by accuracy over all boxes; Test reported once.
#   DenseNet label accuracy on true boxes (Val): 1.25x/64px 94.9% > 1.5x 92.3%
#     > 2x 87.2% > 3x/128px 86.9% (previous setting); YOLO's own label 98.8%.
#   Best verifier on Val: 1.25x crop, switch at 0.95, NO healthy rejection.
#   Test: YOLO only 98.02% label acc / 83.28% all boxes; selected verifier
#     identical (McNemar p = 1.0). Previous healthy-reject 0.50 rule: 94.28% /
#     80.78% (McNemar p ~ 2e-13 vs YOLO): it removed 22 false boxes but 102
#     true boxes.
# => With the whole-leaf DenseNet, crop verification does not improve on
#    YOLO's labels; a dedicated lesion-crop classifier is the next step.
VERIFY_CROP_CONTEXT = 1.25
VERIFY_CROP_MIN_PX = 64
VERIFY_MAX_CROPS = 16
# YOLO's label is kept unless DenseNet's top disease probability on the crop
# reaches this value (then DenseNet's label replaces it).
LABEL_SWITCH_MIN_CONF = 0.95
# DenseNet must put at least this much probability on the 3 diseases for a
# crop to count as a verification (legacy rule, not used by EXP-034 now).
VERIFY_MIN_DISEASE_MASS = 0.50
# A crop rejects its box when DenseNet sees healthy leaf with at least this
# probability (and above every disease). None = never reject (Val-selected).
VERIFY_HEALTHY_REJECT = None
# A second disease needs >= 1 agreeing crop with this raw DenseNet
# probability and >= SECONDARY_DISEASE_MIN_BOXES agreeing boxes.
SECONDARY_CROP_MIN_CONF = 0.70

# ================= Inference / display policy (P0 + P1) =================
# Abstain also when the two best classes are too close, even if top-1 is
# above the frozen EXP-034 threshold (tau = 0.504).
ABSTAIN_MIN_MARGIN = 0.15

# A second disease is reported (multi-disease) only with strong detector
# evidence; otherwise boxes of non-primary classes are hidden. A single weak
# YOLO box (conf 0.25) of another class is not a diagnosis.
SECONDARY_DISEASE_MIN_CONF = 0.50
SECONDARY_DISEASE_MIN_BOXES = 2
SECONDARY_DISEASE_SINGLE_BOX_CONF = 0.70

# Per-class display thresholds for YOLO boxes. Defaults equal the frozen
# EXP-034 box_conf (0.25); tune ONLY on Validation, never on Test/Holdout.
BOX_CONF_PER_CLASS = {"anthracnose": 0.25, "leaf_miner": 0.25, "red_rust": 0.25}

# Tiled (SAHI-style) YOLO pass, used ONLY for the boxes drawn on screen.
# The stacker always receives features from the original single 640px pass,
# exactly as in training, so classification probabilities are unchanged.
YOLO_TILED_DISPLAY = True
TILE_SIZE = 640
TILE_OVERLAP = 0.20
# The photo is first shrunk (antialiased) so its long side is at most this
# value; tiles are then cut from it. 1280 -> ~2x zoom vs the full-image pass,
# a moderate scale shift for a detector trained on whole leaves at 640px.
TILE_SOURCE_MAX_SIDE = 1280
# Tiling is skipped when the photo is not meaningfully larger than one tile.
TILE_MIN_LONG_SIDE = 800
MERGE_IOU = 0.50
MERGE_IOS = 0.80

# Boxes must lie mostly on the leaf blade (coarse colour mask).
LEAF_MASK_FILTER = True
LEAF_MASK_MIN_INSIDE = 0.50

# Antialiased pre-shrink before the classifier's tf.image.resize.
# 0 disables it (bit-exact with the frozen EXP-034 pipeline).
CLASSIFIER_PREDOWNSCALE_MAX_SIDE = 768

# Input size gate only (blur/exposure are left to the user guidance on the UI).
QUALITY = {
    "min_side_reject": 64,
    "min_side_warn": 224,
}

# ================= Model bundle selection =================
# The bundle uses DenseNet121 x5 seeds + YOLO26s best.pt + Stacker F3. Each bundle has
# its own OOD reference bank / calibration (embeddings depend on DenseNet).
ACTIVE_BUNDLE = "035"
BUNDLES = {
    "035": {
        "id": "EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04",
        "version": "EXP-035",
        "model_name": "DenseNet121 ×5 + YOLO26s NEG-V04 (EXP-035)",
        "detector": "EXP-Y26S-NEG-V04-001",
        "ood_dir": BASE_DIR / "ood" / "exp035",
        # Bank rebuilt with the EXP-035 DenseNets on split13 train (3869 cashew
        # images; zip SHA e1647b72... = source_hashes.split13); band 0.1744 /
        # 99.9% quantile 0.2104 from split13 Val. Measured 2026-11-08 via the API
        # (same 1124 images as for EXP-034):
        #   t_high | real cashew (154) | not_cashew (470) | corn x4 + other (500)
        #   0.2104 |  99.4% | 100% | 99.0%
        #   0.24   | 100.0% | 100% | 96.4%   <- chosen
        #   0.25   | 100.0% | 100% | 95.6%
        #   argmax only: corn+other 76.0%
        # Holdout photos were used to pick it (descriptive, not blind).
        "knn_hard_threshold": 0.24,
        "metrics": {
            "test_accuracy": 0.9708, "test_wilson_95": [0.9553, 0.9810], "test_macro_f1": 0.9677,
            "test_top2_accuracy": 0.9985, "test_ece15": 0.0124,
            "holdout_accuracy": 0.9444, "holdout_wilson_95": [0.8898, 0.9728],
            "validation_clean_accuracy": 0.9868,
            "test_recall": {"anthracnose": 0.9333, "healthy": 0.9576, "leaf_miner": 0.9462,
                            "red_rust": 1.0, "not_cashew_leaf": 1.0},
            "box_f1_50": {"anthracnose": 0.649, "leaf_miner": 0.541, "red_rust": 0.695},
        },
    },
}
_B = BUNDLES[ACTIVE_BUNDLE]
EXP034_ID = _B["id"]
EXP_VERSION = _B["version"]
# The bundle is used in place (not copied) and must never be edited.
EXP034_BUNDLE_DIR = PROJECT_ROOT / EXP034_ID / "pipeline_bundle"
# Verify every bundle artifact against artifact_sha256.csv at startup.
EXP034_VERIFY_SHA256 = True
EXP034_MODEL_NAME = _B["model_name"]
EXP034_METRICS = _B["metrics"]
OOD_DIR = _B["ood_dir"]
OOD_REFERENCE_PATH = OOD_DIR / "ood_reference.npz"
OOD_CALIBRATION_PATH = OOD_DIR / "ood_calibration.json"
OOD_KNN_HARD_THRESHOLD = _B["knn_hard_threshold"]

# Disease Details, Agricultural Reference & Validation Benchmark Metrics
DISEASE_INFO = {
    "anthracnose": {
        "vn_name": "Bệnh Thán Thư",
        "short_name": "Thán Thư",
        "scientific_name": "Colletotrichum gloeosporioides",
        "severity": "Cao (Nguy hiểm)",
        "benchmark_precision": 91.74,
        "benchmark_recall": 92.50,
        "benchmark_f1": 92.12,
        "badge_color": "danger",
        "color_hex": "#ef4444",
        "icon": "fa-biohazard",
        "symptoms": "Vết đốm hoại tử màu nâu sẫm hoặc đen cháy lan từ chóp và mép lá vào phiến lá. Mảng hoại tử khô giòn, mép uốn lượn có thể thấy các vòng tròn đồng tâm đặc trưng.",
        "management": [
            "Cắt tỉa toàn bộ lá và cành non bị cháy hoại tử, gom lại và tiêu hủy triệt để xa vườn.",
            "Tỉa cành tạo tán thông thoáng nhằm hạ ẩm độ bên trong tán lá, triệt tiêu môi trường bào tử nấm nảy mầm.",
            "Phun các loại thuốc trừ nấm đặc hiệu có hoạt chất Azoxystrobin, Mancozeb, Difenoconazole hoặc Propineb ngay khi phát hiện vết bệnh chớm nở."
        ],
        "chemical_actives": "Azoxystrobin, Difenoconazole, Mancozeb, Hexaconazole, Propineb"
    },
    "healthy": {
        "vn_name": "Lá Khỏe Mạnh",
        "short_name": "Khỏe Mạnh",
        "scientific_name": "Anacardium occidentale (Healthy)",
        "severity": "An toàn (0% bệnh)",
        "benchmark_precision": 100.00,
        "benchmark_recall": 95.76,
        "benchmark_f1": 97.84,
        "badge_color": "success",
        "color_hex": "#10b981",
        "icon": "fa-seedling",
        "symptoms": "Lá điều bình thường, màu xanh lục bóng khỏe tự nhiên, bề mặt phiến lá nguyên vẹn, không có tổn thương do sâu bệnh hay nấm hại.",
        "management": [
            "Duy trì chế độ tưới nước hợp lý và cung cấp dinh dưỡng cân đối N-P-K theo từng chu kỳ sinh trưởng.",
            "Bón bổ sung phân hữu cơ vi sinh và phân bón lá chứa vi lượng (kẽm, bo, canxi, magie) để tăng sức chống chịu tự nhiên.",
            "Kiểm tra vườn định kỳ 7–10 ngày/lần, đặc biệt trong các đợt cây ra lộc non và chuyển mùa."
        ],
        "chemical_actives": "Không cần sử dụng hóa chất BVTV"
    },
    "leaf_miner": {
        "vn_name": "Sâu Vẽ Bùa / Sâu Đục Lá",
        "short_name": "Sâu Vẽ Bùa",
        "scientific_name": "Acrocercops syngramma",
        "severity": "Trung bình",
        "benchmark_precision": 93.94,
        "benchmark_recall": 95.38,
        "benchmark_f1": 94.65,
        "badge_color": "warning",
        "color_hex": "#f59e0b",
        "icon": "fa-bug",
        "symptoms": "Ấu trùng sâu đục ăn nhu mô dưới lớp biểu bì lá tạo thành các đường hầm ngoằn ngoèo màu trắng bạc hoặc ánh bạc. Khi mật độ cao làm lá co quắp, biến dạng và khô từng mảng.",
        "management": [
            "Bảo tồn các loài thiên địch tự nhiên trong vườn điều (ong ký sinh, bọ rùa).",
            "Thu dọn và tiêu hủy các lá bị sâu non gây hại nặng ở các đợt lộc non.",
            "Sử dụng thuốc sinh học hoạt chất Abamectin, Emamectin benzoate hoặc dầu khoáng nông nghiệp vào sáng sớm hoặc chiều mát."
        ],
        "chemical_actives": "Abamectin, Emamectin benzoate, Dầu khoáng SK Enspray, Spinetoram"
    },
    "red_rust": {
        "vn_name": "Bệnh Rỉ Đỏ / Tảo Đỏ",
        "short_name": "Rỉ Đỏ",
        "scientific_name": "Cephaleuros virescens",
        "severity": "Trung bình - Khá",
        "benchmark_precision": 99.38,
        "benchmark_recall": 100.00,
        "benchmark_f1": 99.69,
        "badge_color": "rust",
        "color_hex": "#8b5cf6",
        "icon": "fa-virus",
        "symptoms": "Các đốm tròn nhung màu cam, đỏ gạch hoặc rỉ sắt mọc nổi gồ trên mặt lá do tảo Cephaleuros virescens ký sinh. Khi vết bệnh già đi chuyển dần sang màu nâu xám, cản trở quang hợp.",
        "management": [
            "Cắt tỉa cành nhánh già cỗi, dọn dẹp cỏ dại giúp vườn cây đón ánh nắng trực tiếp.",
            "Tránh bón thừa phân đạm (N), tăng cường kali và lân để phiến lá dày cứng vững.",
            "Phun các hợp chất gốc đồng (Booc-đô 1%, Đồng Oxychloride, Kasugamycin) khi bệnh chớm xuất hiện vào mùa mưa ẩm."
        ],
        "chemical_actives": "Đồng Oxychloride (Copper Oxychloride), Bordeaux (Booc-đô), Kasugamycin, Cuprous Oxide"
    },
    "not_cashew_leaf": {
        "vn_name": "Không Phải Lá Điều / Dị Vật",
        "short_name": "Ngoại Lai",
        "scientific_name": "Outlier / Non-cashew image",
        "severity": "Cảnh báo ảnh",
        "benchmark_precision": 99.36,
        "benchmark_recall": 100.00,
        "benchmark_f1": 99.68,
        "badge_color": "secondary",
        "color_hex": "#64748b",
        "icon": "fa-question-circle",
        "symptoms": "Ảnh được tải lên không mang đặc trưng hình thái của lá cây điều hoặc thuộc loài cây/vật thể khác.",
        "management": [
            "Vui lòng chụp lại hình ảnh lá cây điều ở cự ly gần, đủ ánh sáng tự nhiên.",
            "Đảm bảo phiến lá điều nằm ở trung tâm khung hình và không bị rung nhòe."
        ],
        "chemical_actives": "Không áp dụng"
    }
}

# Multi-disease treatment protocols
MULTI_DISEASE_PROTOCOLS = {
    ("anthracnose", "red_rust"): {
        "title": "Phác Đồ Xử Lý Phức Hợp: Bệnh Thán Thư Kết Hợp Bệnh Rỉ Đỏ (Nấm + Tảo)",
        "priority": "Ưu tiên xử lý nấm Thán thư trước để ngăn ngừa cháy khô chồi và rụng lá non, sau đó xử lý tảo Rỉ đỏ.",
        "actions": [
            "Cắt tỉa khẩn cấp các lá và chồi non có ổ hoại tử thán thư lớn, gom tiêu hủy xa vườn.",
            "Cắt tỉa thông thoáng tầng tán dưới để ánh nắng chiếu vào, giảm ẩm độ - yếu tố thuận lợi cho cả nấm và tảo phát triển.",
            "Đợt 1: Phun ngay thuốc trừ nấm phổ rộng (Azoxystrobin + Difenoconazole hoặc Mancozeb) dập dịch thán thư.",
            "Đợt 2: Cách 5-7 ngày sau, phun thuốc gốc Đồng (Đồng Oxychloride hoặc Booc-đô 1%) để quét sạch tảo rỉ đỏ bám trên mặt lá.",
            "LƯU Ý QUAN TRỌNG: Không tự ý pha trộn chung thuốc gốc Đồng với các thuốc nấm nhũ dầu EC nếu chưa thử nghiệm tương thích."
        ]
    },
    ("anthracnose", "leaf_miner"): {
        "title": "Phác Đồ Xử Lý Phức Hợp: Bệnh Thán Thư & Sâu Vẽ Bùa (Nấm + Côn Trùng)",
        "priority": "Tác hại kép: Sâu đục biểu bì tạo vết thương hở là cửa ngõ trực tiếp cho nấm thán thư xâm nhiễm ồ ạt.",
        "actions": [
            "Ngắt bỏ các lá non bị sâu đục nặng kết hợp vết cháy thán thư đem chôn lấp hoặc tiêu hủy.",
            "Phun kết hợp thuốc trị sâu sinh học (Abamectin hoặc Emamectin benzoate) với thuốc trừ nấm (Mancozeb hoặc Propineb).",
            "Phun tập trung vào đợt cây ra lộc non, thời điểm ấu trùng mới nở dưới biểu bì và bào tử nấm dễ bám dính.",
            "Phun vào sáng sớm hoặc chiều mát khi sâu hoạt động mạnh và thuốc không bị phân hủy bởi nắng gắt."
        ]
    },
    ("leaf_miner", "red_rust"): {
        "title": "Phác Đồ Xử Lý Phức Hợp: Sâu Vẽ Bùa & Bệnh Rỉ Đỏ (Côn Trùng + Tảo)",
        "priority": "Kiểm soát sâu vẽ bùa ở đọt non để bảo vệ diện tích lá, kết hợp tăng ánh sáng tán để hạn chế tảo đỏ.",
        "actions": [
            "Thu dọn lá già nhiễm tảo rỉ đỏ ở tầng đáy tán cây, dọn sạch cỏ dại xung quanh gốc điều.",
            "Phun luân phiên thuốc sinh học Abamectin diệt sâu vẽ bùa ở đọt non.",
            "Sau khi xử lý sâu, phun Đồng Oxychloride vào tầng tán bị tảo đỏ ký sinh.",
            "Bón bổ sung Lân và Kali giúp phiến lá dày, tăng sức chống chịu cả sâu hại và tảo ký sinh."
        ]
    },
    ("anthracnose", "leaf_miner", "red_rust"): {
        "title": "Phác Đồ Tổng Hợp Toàn Diện: Thán Thư + Sâu Vẽ Bùa + Rỉ Đỏ (Đa Tác Nhân)",
        "priority": "Tình trạng nhiễm bệnh nặng đa tác nhân, cần can thiệp đồng bộ theo quy trình IPM 3 bước.",
        "actions": [
            "Bước 1 (Vệ sinh cơ học): Cắt tỉa cành nhiễm bệnh nặng, dọn cỏ gốc, tỉa thông thoáng tán để tăng bức xạ nhiệt.",
            "Bước 2 (Xử lý sâu và nấm khẩn cấp): Phun phối hợp Emamectin benzoate (diệt sâu vẽ bùa) + Azoxystrobin (diệt nấm thán thư).",
            "Bước 3 (Kiểm soát tảo đỏ & phục hồi): Sau 7 ngày, phun quét gốc Đồng xử lý tảo rỉ đỏ, kết hợp tưới phân hữu cơ vi sinh bồi dưỡng rễ."
        ]
    }
}
