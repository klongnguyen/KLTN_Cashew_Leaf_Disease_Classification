"""
Configuration and Disease Reference Data for Cashew Leaf Disease Diagnosis.
Model: DenseNet121 Version 009 (Best Seed: 3407)
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009"
    / "model_package"
    / "deployment_model.keras"
)

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009"
    / "model_package"
    / "label_map.json"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "current_dataset"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL"
    / "EXP-DENSENET121-SCRATCH-5SEEDS-009"
    / "model_package"
    / "deployment_metadata.json"
)

STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"

# Model specs
IMG_SIZE = 224
MODEL_NAME = "DenseNet121 (Scratch v009)"
BEST_SEED = 3407
VAL_ACCURACY = 0.9563
VAL_LOSS = 0.1410
VAL_MACRO_F1 = 0.9551

# Disease Details, Agricultural Reference & Validation Benchmark Metrics
DISEASE_INFO = {
    "anthracnose": {
        "vn_name": "Bệnh Thán Thư",
        "scientific_name": "Colletotrichum gloeosporioides",
        "severity": "Cao (Nguy hiểm)",
        "benchmark_precision": 93.05,
        "benchmark_recall": 93.98,
        "benchmark_f1": 93.51,
        "badge_color": "danger",
        "color_hex": "#dc2626",
        "icon": "fa-biohazard",
        "symptoms": "Vết đốm màu nâu tối hoặc đen cháy hoại tử, lan dần từ chóp lá hoặc mép lá vào trong phiến lá. Vết bệnh lớn tạo mảng khô giòn, mép gợn sóng có thể thấy các vòng đồng tâm.",
        "management": [
            "Cắt tỉa toàn bộ lá và cành bị bệnh đem tiêu hủy xa vườn cây.",
            "Tạo độ thông thoáng cho tán cây điều để giảm độ ẩm bề mặt lá.",
            "Phun luân phiên thuốc phòng trừ nấm như Mancozeb, Hexaconazole, Azoxystrobin hoặc thuốc gốc đồng khi phát hiện dấu hiệu ban đầu."
        ]
    },
    "healthy": {
        "vn_name": "Lá Khỏe Mạnh",
        "scientific_name": "Anacardium occidentale (Healthy)",
        "severity": "An toàn",
        "benchmark_precision": 95.22,
        "benchmark_recall": 93.43,
        "benchmark_f1": 94.31,
        "badge_color": "success",
        "color_hex": "#15803d",
        "icon": "fa-seedling",
        "symptoms": "Lá điều bình thường, màu xanh lục bóng khỏe tự nhiên, bề mặt phiến lá nguyên vẹn, không có tổn thương do sâu bệnh hay nấm hại.",
        "management": [
            "Duy trì tưới nước và cung cấp dinh dưỡng cân đối N-P-K theo từng chu kỳ sinh trưởng.",
            "Bón bổ sung phân hữu cơ vi sinh và phân bón lá chứa vi lượng (kẽm, bo, magie).",
            "Kiểm tra định kỳ để phát hiện sớm dịch hại trong các đợt ra đọt non."
        ]
    },
    "leaf_miner": {
        "vn_name": "Sâu Vẽ Bùa / Sâu Đục Lá",
        "scientific_name": "Acrocercops syngramma",
        "severity": "Trung bình",
        "benchmark_precision": 96.30,
        "benchmark_recall": 95.12,
        "benchmark_f1": 95.71,
        "badge_color": "warning",
        "color_hex": "#b45309",
        "icon": "fa-bug",
        "symptoms": "Ấu trùng đục dưới lớp biểu bì lá tạo thành các đường hầm ngoằn ngoèo màu trắng bạc hoặc ánh bạc. Khi mật độ cao làm lá co quắp, biến dạng và khô từng mảng.",
        "management": [
            "Bảo vệ các loài thiên địch tự nhiên trong vườn (ong ký sinh, bọ rùa).",
            "Thu dọn và tiêu hủy các lá bị sâu non gây hại nặng ở các đợt lộc non.",
            "Sử dụng thuốc sinh học hoạt chất Abamectin, Emamectin benzoate hoặc dầu khoáng nông nghiệp vào sáng sớm hoặc chiều mát."
        ]
    },
    "red_rust": {
        "vn_name": "Bệnh Rỉ Đỏ / Tảo Đỏ",
        "scientific_name": "Cephaleuros virescens",
        "severity": "Trung bình - Khá",
        "benchmark_precision": 96.12,
        "benchmark_recall": 99.69,
        "benchmark_f1": 97.87,
        "badge_color": "warning",
        "color_hex": "#c2410c",
        "icon": "fa-virus",
        "symptoms": "Các đốm tròn nhung màu cam, đỏ gạch hoặc rỉ sắt mọc nổi gồ trên mặt lá. Khi vết bệnh già đi, chuyển dần sang màu nâu xám, làm giảm khả năng quang hợp của lá điều.",
        "management": [
            "Cắt tỉa cành nhánh già cỗi, dọn dẹp cỏ dại giúp vườn cây đón ánh nắng trực tiếp.",
            "Tránh bón thừa đạm (N), tăng cường kali và lân để phiến lá dày cứng vững.",
            "Phun các hợp chất gốc đồng (Booc-đô 1%, Đồng Oxyclorua, Kasugamycin) khi bệnh chớm xuất hiện vào mùa mưa ẩm."
        ]
    },
    "not_cashew_leaf": {
        "vn_name": "Không Phải Lá Điều / Dị Vật",
        "scientific_name": "Outlier / Non-cashew image",
        "severity": "Cảnh báo ảnh",
        "benchmark_precision": 97.39,
        "benchmark_recall": 94.90,
        "benchmark_f1": 96.13,
        "badge_color": "secondary",
        "color_hex": "#64748b",
        "icon": "fa-question-circle",
        "symptoms": "Ảnh được tải lên không mang đặc trưng hình thái của lá cây điều hoặc thuộc loài cây/vật thể khác.",
        "management": [
            "Vui lòng chụp lại hình ảnh lá cây điều ở cự ly gần, đủ ánh sáng tự nhiên.",
            "Đảm bảo phiến lá điều nằm ở trung tâm khung hình và không bị rung nhòe."
        ]
    }
}
