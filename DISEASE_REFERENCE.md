# Disease Reference — Cashew Leaf

Tài liệu ngắn phục vụ kiểm tra trực quan khi làm sạch dữ liệu và review annotation. Đây **không thay thế ground-truth review chuyên môn**.

## Anthracnose

Dấu hiệu thường gặp trong dataset:
- đốm nâu/đen, vùng hoại tử;
- tổn thương có thể lan từ mép/chóp lá;
- vết lớn có thể tạo mảng cháy khô;
- một số vết có cấu trúc vòng đồng tâm.

| Sample 1 | Sample 2 |
|---|---|
| ![Anthracnose 1](./img_check/ath_01.png) | ![Anthracnose 2](./img_check/ath_02.png) |

| Sample 3 | Sample 4 |
|---|---|
| ![Anthracnose 3](./img_check/ath_03.png) | ![Anthracnose 4](./img_check/ath_04.png) |

## Leaf Miner

Dấu hiệu thường gặp:
- đường hầm/vệt ngoằn ngoèo dưới biểu bì;
- vùng sáng bạc hoặc khô theo đường đục;
- tổn thương có thể liên tục hoặc thành mảng khi mật độ cao.

| Sample 1 | Sample 2 |
|---|---|
| ![Leaf Miner 1](./img_check/mine01.png) | ![Leaf Miner 2](./img_check/mine02.png) |

## Red Rust

Dấu hiệu thường gặp:
- đốm tròn màu vàng cam/đỏ gạch/rỉ sắt;
- thường tạo cụm nhiều chấm;
- vết già có thể chuyển xám nâu.

| Sample 1 | Sample 2 | Sample 3 |
|---|---|---|
| ![Red Rust 1](./img_check/red01.png) | ![Red Rust 2](./img_check/red02.png) | ![Red Rust 3](./img_check/red03.png) |

## Liên quan đến annotation

Bounding-box policy hiện tại không yêu cầu khoanh mọi dấu hiệu nhỏ nhất. Chỉ annotate lesion rõ, đủ lớn và có ý nghĩa thị giác theo:

[`CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md`](./CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md)
