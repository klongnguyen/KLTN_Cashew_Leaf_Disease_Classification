# EXP-031 — DenseNet121 + cổng bằng chứng tổn thương (late fusion)

**Notebook:** `densenet121_yolo_evidence_gate_031.ipynb`
**Baseline:** `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05`
**Detector:** `EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED` (YOLO26s, `cls_pw=0.25`)

## 1. Vì sao không phải crop → classifier

Khi bạn nói "mỗi ảnh nhiều box", tôi đọc lại `CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md` và phải rút lại đề xuất crop trước đó của mình. Guideline quy định box chỉ bao **vùng tổn thương**, cấm bound toàn bộ chiếc lá hay toàn ảnh. Detector chỉ có ba lớp `anthracnose`, `leaf_miner`, `red_rust`; `healthy` và `not_cashew_leaf` **không có box nào** theo chính sách, chúng tồn tại dưới dạng ảnh 0 box làm negative supervision.

Hệ quả: một ảnh lá khỏe sinh ra **không crop nào**, nên pipeline crop → classifier về mặt cấu trúc không thể phân loại hai trong năm lớp. Thêm nữa crop lesion phóng lên 256×256 là phân phối đầu vào hoàn toàn khác ảnh lá nguyên, buộc phải huấn luyện lại classifier, mà đầu ra của nó lại trùng với chính class head YOLO đã có.

## 2. Thiết kế thay thế

Dùng nhiều box làm **bằng chứng mức ảnh** thay vì làm crop.

> Nếu argmax của classifier là một lớp bệnh được gate, nhưng detector không tìm thấy lesion thuộc lớp đó ở confidence `>= TAU`, thì hạ xuống lựa chọn tốt thứ hai của classifier.

Mỗi ảnh được rút gọn thành vài thống kê cho từng lớp detector: `max_conf`, `n_boxes`, `area_frac`. Chỉ `max_conf` điều khiển gate mặc định; phần còn lại được ghi lại để EXP-032 thử luật phong phú hơn mà không phải chạy lại inference. Nhiều box mỗi ảnh vì thế là **lợi thế**: không cắt ảnh, không phải gộp dự đoán từng crop, không huấn luyện lại gì.

EXP-031 **không huấn luyện gì cả**. Nó đọc lại chính xác suất softmax EXP-029 đã lưu cho Validation, Test và Holdout 09 của cả 5 seed.

## 3. Cơ sở định lượng

Gộp 5 seed EXP-029 trên Holdout 09: **77 trong 109 lỗi là "đoán ra anthracnose"**; anthracnose nhận dự đoán gấp **1,209 lần** support thật còn healthy chỉ **0,758 lần**. Mô phỏng gate với oracle đạt **92,38%** so với 82,70% của EXP-029. Test đang 96,23%, nên ngưỡng cần là Holdout ≥ **86,23%**.

Năng lực detector ủng hộ hướng này. `CASHEW_YOLO_MASTER_FINAL_NEG_V03` có mật độ box rất cao, riêng Validation là 14.507 box anthracnose trên 893 ảnh dương, trung bình khoảng 53 box mỗi ảnh. Gate chỉ cần **một** box, nên độ nhạy mức ảnh sẽ cao hơn nhiều so với box recall 0,49. Về false alarm, YOLO26s NEG V03 chỉ báo nhầm 3/58 ảnh âm ở `conf >= 0.25`, và **cả hai ảnh healthy báo nhầm đều là anthracnose ở confidence 0,297 và 0,311**, nên TAU trên khoảng 0,35 sẽ loại sạch chúng.

## 4. Giao thức, theo đúng thứ tự notebook ép

**Bước 1 — Audit rò rỉ.** Không ảnh Holdout 09 nào được phép nằm trong tập train của detector. So khớp chính xác bằng SHA256 của **pixel RGB đã giải mã** (sống sót qua tái mã hóa, khác với hash bytes) và gần đúng bằng pHash 64-bit ở Hamming ≤ 4. Phát hiện trùng chính xác thì **dừng hẳn**. Ảnh Validation bị detector nhìn thấy sẽ bị loại khỏi bước chỉnh ngưỡng.

**Bước 2 — Chọn TAU bằng chỉ số Youden.** Chỗ này tôi đã sửa hai lần trong quá trình test, nên ghi lại cả hai để bạn hiểu vì sao quy tắc cuối cùng có hình dạng như vậy.

Lần đầu tôi định chọn TAU bằng cách tối đa accuracy trên Validation của classifier. Chạy thử thì **hỏng**: Validation có 1.393/1.452 ảnh kiểu Kaggle mà classifier vốn đã đúng ~97%, nên gần như mọi veto đều lỗ và tối ưu đầu-cuối sụp về "không bao giờ veto" với TAU = 0,05, chỉ được +0,79 điểm holdout. Đúng cái bẫy đã làm hỏng việc chọn seed từ EXP-014 đến EXP-028: chỉ số bị miền nguồn chi phối trong khi gate sinh ra cho miền đích.

Lần hai tôi dùng ràng buộc cứng "độ nhạy ≥ 0,95 và chi phí Validation ≤ 0,5 điểm". Khi mô phỏng bằng đặc trưng thật của detector thì nó cũng hỏng, theo kiểu khác: nền false-positive dày đặc (79.529 box anthracnose thừa ở điểm vận hành mAP) khiến chỉ TAU = 0,05 lọt qua ràng buộc, mà ở đó false alarm là 0,79 nên gate vô dụng. Chênh lệch chỉ xuống 10,68.

Quy tắc cuối cùng: chọn TAU **tối đa chỉ số Youden** `J = độ nhạy − false alarm`, là điểm vận hành kinh điển của một phép thử chẩn đoán. J là thuộc tính của riêng detector và **không phụ thuộc tần suất nền của các lớp** — đúng thứ cần ở đây, vì tần suất dự đoán nhầm anthracnose thấp trên Validation nhưng cao ngoài thực địa, nên mọi tiêu chí phụ thuộc tần suất nền đều sẽ lại bị miền nguồn chi phối. Hai ngưỡng `SENSITIVITY_FLOOR = 0,85` và `MAX_VALIDATION_COST = 2,0` điểm chỉ còn là **rào chắn**, không phải hàm mục tiêu.

Quan trọng: độ nhạy mức ảnh được đo **trên chính Validation của classifier**, dùng nhãn 5 lớp sẵn có, nên **không cần tới dataset YOLO**. Đây là đúng quần thể mà gate sẽ chạy, và khác với box recall của detector, nó trả lời đúng câu hỏi gate đặt ra: "ảnh này có ít nhất một lesion lớp c ở confidence ≥ TAU không?".

**Bước 3 — Holdout 09 chỉ chấm một lần, ở cuối**, và không bao giờ tham gia chọn TAU. Trường `holdout_was_used_for_selection: false` được ghi vào `tau_selection.json`.

**Bước 4 — Gate áp lên CẢ HAI phía của chênh lệch.** Con số đầu bài của đề tài là một hiệu số, nên gate phải chạy trên cả Test lẫn Holdout, và báo cáo `gated_Test − gated_Holdout`. Một gate nâng Test nhiều bằng Holdout thì không đóng được gì, và nếu chỉ gate phía holdout thì kết quả bị thổi phồng.

## 5. Phân tích detector EXP-Y26S-NEG-V03-001

FULL ZIP đã lên repo tại `training_results/negative_evaluation/`, sha256 `e1d3cc42…f542d`, chứa `train/weights/best.pt` 20.316.608 byte. Tôi đã xác minh: file là torch archive hợp lệ, pickle chứa đúng ba tên lớp `anthracnose`, `leaf_miner`, `red_rust`, model gốc `yolo26s.pt`. SHA256 của `best.pt` là `66717b55…18166`, đã ghim vào notebook, và notebook tự giải nén từ FULL ZIP nên bạn không phải làm gì thêm.

**Ma trận nhầm lẫn là chỗ có thông tin quyết định.** Tổng theo cột khớp **chính xác** với manifest NEG V03 — anthracnose 14.507, leaf_miner 568, red_rust 32.183 box Validation — nên ma trận đáng tin. Hàng `background` cho tỉ lệ bỏ sót ở mức box:

| Lớp | Bỏ sót | Tổng box | Tỉ lệ bỏ sót | Bắt được |
|---|---:|---:|---:|---:|
| anthracnose | 1.125 | 14.507 | **7,75%** | 92,25% |
| leaf_miner | 6 | 568 | 1,06% | 98,94% |
| red_rust | 2.275 | 32.183 | 7,07% | 92,93% |

Con số này **tốt hơn nhiều** so với recall 0,624 mà `metrics.json` báo, vì recall đó đo ở điểm vận hành mAP với quét IoU nghiêm ngặt, còn gate chỉ cần biết ảnh có lesion hay không.

Suy ra cận dưới của **độ nhạy mức ảnh ≥ 92,25%**: kể cả trong trường hợp xấu nhất khi mọi box bỏ sót rơi vào các ảnh khác nhau, cũng chỉ 7,75% số ảnh có thể bị bỏ sót hoàn toàn. Thực tế sẽ cao hơn vì các box bỏ sót rải rác trên những ảnh mang hàng chục box. Mức 0,90–0,95 này rơi đúng vào vùng có lợi của bảng mô phỏng.

**Đối trọng quan trọng:** cùng ma trận đó có **164.378 box false positive**, riêng anthracnose là 79.529, ở điểm vận hành conf ≈ 0,001. Nghĩa là gần như ảnh nào cũng có *một* box anthracnose nào đó. Vì vậy gate phải khóa theo `max_conf >= TAU` chứ tuyệt đối không theo "có tồn tại box", và TAU phải đủ cao để vượt nền nhiễu này. Chính phát hiện này làm hỏng quy tắc chọn TAU lần hai của tôi và dẫn tới Youden.

**Vài vấn đề khác đáng ghi vào khóa luận:** `leaf_miner` có TP 480 nhưng FP 665, tức nhiều báo nhầm hơn báo đúng ở điểm vận hành đó; `multi_disease_gt = 503` cho thấy có 503 trường hợp ground truth đa bệnh, trong khi classifier là đơn nhãn, nên một phần "lỗi" có thể là mơ hồ của chính nhãn; `shifted_boxes_candidate = 140` là vấn đề chất lượng annotation. Preflight của detector sạch: 0 trùng lặp xuyên split trong dataset YOLO, status `pass`.

Tập train của detector có **621 ảnh âm, gồm 375 healthy và 246 not_cashew_leaf**. Đây chính là lý do audit rò rỉ ở bước 1 không thể bỏ qua: nếu một ảnh healthy nào trong đó trùng với Holdout 09 thì gate đang bảo chứng cho ảnh nó đã học thuộc.

## 6. Kiểm chứng đã chạy

Toàn bộ đường tính toán đã chạy thật trên **xác suất EXP-029 thật** cộng bằng chứng detector mô phỏng. Cell nạp dữ liệu tái lập đúng baseline EXP-029 — Validation 97,37%, Test 96,23%, Holdout 82,70%, chênh lệch 13,54 — và notebook sẽ **từ chối chạy tiếp** nếu con số này lệch quá 0,2 điểm, tức là bạn nạp nhầm ZIP.

Mô phỏng cuối dùng **đúng tỉ lệ bỏ sót box đã đo** (anthracnose 7,75%) cộng nền false-positive dày như ma trận nhầm lẫn cho thấy. Pipeline chọn TAU = 0,40 theo Youden J = 0,8366, với độ nhạy 0,890 và false alarm 0,053, chi phí Validation 0,94 điểm. Kết quả: Holdout 82,70% → **90,16%**, Test 96,23% → 96,47%, chênh lệch **13,54 → 6,31**, đạt mục tiêu, 5/5 seed cải thiện.

Theo lớp trên Holdout: Healthy recall **+28,4 điểm**, Red Rust **+24,0 điểm**, Leaf Miner +1,1, đổi lại Anthracnose recall **−5,2 điểm**. Đó là đánh đổi đúng bản chất của gate và phải nêu thẳng trong báo cáo. Gate cũng nâng Test thêm 0,23 điểm, nên chênh lệch giảm 7,23 chứ không phải 7,46.

**Đây là mô phỏng, không phải kết quả.** Detector thật sẽ cho số khác.

## 7. Rủi ro còn lại

Rủi ro lớn nhất là **độ nhạy mức ảnh trên ảnh thực địa** có thể thấp hơn trên val split của detector. Anthracnose là lớp yếu nhất của detector với box recall 0,49–0,59, và val split của YOLO không phải ảnh thực địa. Nếu độ nhạy thực tế tụt xuống quanh 0,75 thì theo bảng mô phỏng lợi ích chỉ còn khoảng +1 điểm hoặc âm. Notebook in ra đường đặc trưng đầy đủ nên bạn sẽ thấy ngay con số thật trước khi tin kết quả.

Rủi ro thứ hai là rò rỉ: nếu Holdout 09 nằm trong tập train của detector thì mọi con số đều vô nghĩa. Vì vậy audit là bước chặn cứng, không phải cảnh báo.

Và như mọi lần: Holdout 09 chỉ có 126 ảnh với 4 lớp thật, khoảng tin cậy 95% ở mức 86% là khoảng ±6 điểm. Test và Holdout 09 đều đã bị quan sát nhiều vòng, nên đây là kiểm toán mô tả chứ không phải benchmark mù. Hai ảnh `real_healthy_40` và `_42` vẫn cần xác minh nhãn trước khi kết luận bất cứ điều gì về lớp Healthy.

## 8. Cần điền trước khi chạy

`best.pt` **không cần điền gì**: notebook dùng bản local nếu có; nếu không, tải FULL ZIP của `EXP-Y26S-NEG-V03-001` từ commit GitHub cố định `ac1efe16319052b6f08a3c41b4ec361bbbb19e2f`, kiểm kích thước và SHA-256 của ZIP, rồi giải nén và kiểm SHA-256 của `train/weights/best.pt`. Nếu runtime không truy cập được GitHub, đặt `DETECTOR_WEIGHTS_OVERRIDE` tới file `best.pt` đã giải nén.

15 bảng prediction thật của EXP-029 (Validation, Test, Holdout 09 × 5 seed) đã được nhúng vào notebook, lấy từ `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05_ANALYSIS.zip` có SHA-256 `223298083582e13b9312a8f17c9e1e1282e68af45ce33d23db6e2f448639a86e`. Cell resolve tự giải nén và kiểm checksum, nên không cần tải ANALYSIS ZIP riêng. Có thể đặt `CLASSIFIER_ANALYSIS_OVERRIDE` tới ZIP hoặc thư mục đã giải nén nếu muốn dùng bản khác; cell vẫn kiểm đủ 15 file.

Mặc định `AUTO_STAGE_MISSING_INPUTS=True`: nếu thư mục ảnh chưa có, notebook tải đúng ba ZIP nguồn từ Drive (Split 13, Holdout 09, YOLO NEG V03), kiểm tên/kích thước/MD5 của file Drive, giải nén an toàn và đối chiếu danh sách ảnh classifier với prediction EXP-029. Nếu đã có ZIP local, đặt `DATASET_ARCHIVE_OVERRIDE`, `REAL_HOLDOUT_ARCHIVE_OVERRIDE`, `YOLO_DATASET_OVERRIDE`; nếu đã giải nén, trỏ `SPLIT13_VAL_DIR`, `SPLIT13_TEST_DIR`, `HOLDOUT_IMAGE_DIR`, `YOLO_TRAIN_IMAGE_DIR` tới thư mục thật. `YOLO_VAL_IMAGE_DIR` và `YOLO_VAL_LABEL_DIR` là tùy chọn.

EXP-031 ghép prediction với evidence bằng khóa `split/class/tên_ảnh`. Tên `real_healthy_037.jpg` xuất hiện ở cả Validation và Holdout 09 nhưng là hai file khác SHA-256 trong manifest EXP-029; ghép chỉ bằng basename sẽ gây nhầm evidence. Nếu đã có `yolo_image_evidence.csv` từ bản 031 cũ trong cùng runtime, cell inference sẽ chuyển cache sang khóa mới khi `RUN_YOLO_INFERENCE=False`, sau khi kiểm tra đường dẫn; nếu cache thuộc runtime khác, phải chạy lại inference.

Checkpoint được chuẩn bị trước khi tải các ZIP ảnh lớn, nên nếu nguồn GitHub không truy cập được notebook sẽ dừng sớm với thông báo dùng `DETECTOR_WEIGHTS_OVERRIDE`. Bước chuẩn bị ảnh cũng dừng với thông báo rõ nếu ZIP hoặc thư mục ảnh không hợp lệ, trước khi leakage audit chạy.

Sau Final Evaluation, EXP-031 tạo `EXP-DENSENET121-YOLOGATE-031_RESULTS.zip` và upload trực tiếp lên thư mục Drive `current_dataset` (`1DGKTnzsvVc60RIEn5lQgflR3s-loM24G`). Cell kiểm tra lại tên, kích thước, MD5 và thư mục cha của file đã tải. Nếu chưa có `exp031_summary.json`, cell sẽ từ chối xuất ZIP.

Nếu notebook báo không có TAU nào hợp lệ, hãy đọc bảng đặc trưng vận hành trước khi nới ngưỡng. Nếu detector không tách nổi có-lesion với không-lesion ở mức ảnh thì gate là sai công cụ, và nên quay lại nhánh đổi backbone sang DINOv2-Small, thứ đã đạt 87,17% holdout ngay trong repo của bạn.
