# Tái hiện baseline V005 trên Validation

Nguồn: ZIP FULL V005 SHA256 `990df19b094cc14d6e5706183616245dda83d21fac3b52c20f9a2a3f5488c845`; dataset NEG-V03 SHA256 `312870474cfbab8c3fc77db7b19e821c1edf34a0787cb45592f00bde841b4dbe`;
checkpoint `best.pt` SHA256 `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19`. Mapping class đọc từ `data_fixed.yaml`.

Biến thể A chạy lại trực tiếp từ `validation_raw_detections.jsonl`: **56,237 raw box → 42,602 region**, giảm **24.2456%**.
Số region khớp `validation_per_image.csv` của V005 tại **951/951 ảnh**.
Định nghĩa TP/FP mới: ghép tham lam theo confidence giảm dần, cùng class, IoU ≥ 0,5;
mỗi GT chỉ được ghép một lần. Region có TP nếu ít nhất một box thành viên là TP.

| Metric | Raw | Region A |
|---|---:|---:|
| GT coverage IoA ≥ 0,5 | 0.7392 | 0.7573 |
| GT center coverage | 0.7630 | 0.7748 |
| Excess area ratio (GT cùng class) | 0.1471 | 0.2672 |
| Display occupancy | 0.1101 | 0.1261 |
| FP/ảnh | 23.7445 | 14.1567 |
| Precision | 0.5985 | 0.6840 |
| Lost TP | N/A | 0 |

Các metric vùng là chẩn đoán hiển thị, không phải AP/mAP detector. Coverage tăng cần đọc cùng excess area và occupancy.
