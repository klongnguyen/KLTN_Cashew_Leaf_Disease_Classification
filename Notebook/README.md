# Notebook — Trục huấn luyện và kiểm định

![Research pipeline](../docs/assets/research_pipeline.svg)

| Nhánh | Notebook và script chính | Vai trò |
|---|---|---|
| DenseNet121 | [EXP-024 V11](DenseNet121/densenet121_transfer_split11_resize_024.ipynb) · [EXP-023](DenseNet121/densenet121_transfer_photometric_5seeds_023.ipynb) · [Scratch 5-seed](DenseNet121/densenet121_5seed.ipynb) | Train mới, ablation, baseline và audit |
| ResNet50 | [Transfer EXP-009](ResNet50/resnet50_transfer_5seed_009.ipynb) · [CBAM 011](ResNet50/resnet50_imagenet_cbam_5seeds_011.ipynb) · [Scratch](ResNet50/resnet50_5seed.ipynb) | Phân biệt Scratch/Transfer/CBAM |
| ViT/DeiT | [DeiT-Small 011](ViT/vit_deit_small_transfer_5seeds_011.ipynb) · [DeiT-Tiny CBAM 012](ViT/vit_deit_tiny_imagenet_cbam_5seeds_012.ipynb) · [XGBoost 009](ViT/vit_transfer_xgboost_009.ipynb) | ViT, classifier-only và study |
| YOLO26 | [train-yolov26.ipynb](Yolov26/train-yolov26.ipynb) | Detection, chưa final |
| Common | [Real Holdout utilities](common/real_holdout.py) · [audit](common/audit_holdout_archive.py) | Kiểm tra split/ảnh độc lập |

Các notebook backup/smoke và code dựng notebook được giữ để truy nguyên. **Không xem notebook backup là recipe hiện hành**; luôn đối chiếu EXP ID, dataset version và trạng thái Test trong [Training Results](../training_results/current_dataset/README.md). Không dùng Test/Holdout đã xem để chọn cấu hình.

[Protocol](../docs/RESEARCH_PROTOCOL.md) · [Tất cả experiment](../docs/EXPERIMENT_INDEX.md)
