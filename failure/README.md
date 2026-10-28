# Failure / Development Archive — YOLO26

![Take history](../docs/assets/yolo_history_map50.svg)

Không phải tất cả các Take là lỗi kỹ thuật: đây là **kho nghiên cứu detector chưa chọn final**. Giữ lại các phiên bản để phân tích annotation, GT, threshold và dữ liệu, không trình bày chúng như cùng benchmark khi Test thay đổi.

| Take | Kết quả ngắn | Hồ sơ |
|---|---|---|
| 01–05 | Các bước đầu thay đổi dữ liệu/annotation, có cả kết quả không đạt | [YOLO index](YOLO26/README.md) |
| 06 | P 0,7473 · R 0,6648 · mAP50 0,7298 · mAP50-95 0,3877 | [Take 006](YOLO26/EXP-Y26S-SMALL-006/README.md) |
| 07 | P 0,6945 · R 0,6598 · mAP50 0,6993 · mAP50-95 0,3517 | [Take 007](YOLO26/EXP-Y26S-SMALL-007/README.md) |

**Take 006 và 007 đổi Test/số GT**, không là controlled ablation. Không tự lựa chọn detector final dựa trên kết quả này. Giữ nguyên bảng metrics, config và failure notes để phân tích.

[Chi tiết 7 Take](YOLO26/README.md) · [Annotation guideline](../CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md)
