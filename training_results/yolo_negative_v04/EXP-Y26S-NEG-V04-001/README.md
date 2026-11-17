# EXP-Y26S-NEG-V04-001

Dataset: /content/exp_yolo_neg_v04_001/CASHEW_YOLO_MASTER_FINAL_NEG_V04.zip; Train/Val/Test = 3739/1076/488.
Classes: ['anthracnose', 'leaf_miner', 'red_rust']. Lá điều khỏe, ảnh không phải lá điều và lá lạ có bệnh là ảnh nền 0 box.
Model: yolo26s.pt, seed 42, cls_pw=0.25. Trọng số: {'anthracnose': 0.7780247926712036, 'leaf_miner': 1.5752495527267456, 'red_rust': 0.6467254757881165}.
Validation mAP50=0.7144, mAP50-95=0.3998.
Leaf Miner: {'class_id': 1, 'class': 'leaf_miner', 'precision': 0.7006293147742497, 'recall': 0.7429577464788732, 'ap50': 0.7766807100476347, 'ap50_95': 0.5209897176458652, 'train_boxes': 2076, 'class_weight': 1.5752495527267456}.
Preflight: metadata/negative_image_preflight.json.
So sánh baseline: Chưa có baseline cls_pw=0 trên cùng NEG_V04.

ACDRA Validation: raw boxes=51503; display regions=39574; GT covered=34128 raw / 34885 regions. Đây là metric hiển thị, không phải mAP/recall YOLO.
