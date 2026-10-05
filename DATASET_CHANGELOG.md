# Dataset Changelog

Tài liệu này theo dõi các phiên bản classification dataset để tránh trộn lẫn kết quả giữa những split khác nhau.

## Current development — Cashew_dataV05

**First recorded experiments:** 2026-10-03 → 2026-10-04

V05 giữ nguyên tổng số ảnh và class counts của V04:

```text
Train      4,822
Validation 1,402
Test         687
Total      6,911
```

Tuy nhiên, ResNet50 experiment metadata ghi split fingerprints khác V04 ở Train và Validation:

- Train V04: `45a124724d96084add52d3781ff28c0d93adabc6adf4d67067327aab16ea92c7`
- Train V05: `7443257702dbba7a392a28b497a49933f5cfac7e187e991043007f8e309a795f`
- Val V04: `7529e4e0c07af37e1a9c4a366f05846cf72dea530d5f2121da5acb83c64782a0`
- Val V05: `ea4bba016c9f3abb69c0bf6f7deb3d502929707edfbeafa89f36898b7747e279`

The Test fingerprint is unchanged across V04 and V05:

`ee49ef59c791a7519c172316afd48554f92fbb1b4f47d11b57c82d6ae1624dc4`

All three latest V05 archives have identical normalized split/class/filename manifests. This version is currently **Validation complete — locked Test pending**.


## Previous completed benchmark — Cashew_dataV04

**Ngày khóa phiên bản:** 2026-09-22

V04 được tạo sau khi tiếp tục:
- loại ảnh mờ/chất lượng thấp;
- loại hoặc điều chỉnh các trường hợp có nguy cơ data leakage giữa Train/Validation/Test;
- giữ nguyên nguyên tắc nhóm các ảnh cùng nguồn/cùng lá/các biến thể liên quan trong cùng split khi cần để hạn chế leakage.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

## Previous — Cashew_dataV03

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 1,096 | 313 | 156 | 1,565 |
| `healthy` | 818 | 225 | 128 | 1,171 |
| `leaf_miner` | 919 | 262 | 131 | 1,312 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,115 | 319 | 159 | 1,593 |
| **TOTAL** | **5,049** | **1,433** | **731** | **7,213** |

## V03 → V04

| Class | V03 | V04 | Change |
|---|---:|---:|---:|
| `anthracnose` | 1,565 | 1,361 | -204 |
| `healthy` | 1,171 | 1,149 | -22 |
| `leaf_miner` | 1,312 | 1,274 | -38 |
| `not_cashew_leaf` | 1,572 | 1,572 | +0 |
| `red_rust` | 1,593 | 1,555 | -38 |
| **TOTAL** | **7,213** | **6,911** | **-302** |

V04 có ít hơn **302 ảnh**. Mức giảm không được diễn giải là mất dữ liệu đơn thuần; mục tiêu là tăng chất lượng benchmark bằng việc loại dữ liệu chất lượng thấp và giảm rủi ro leakage.

## Quy tắc so sánh

- Không so sánh trực tiếp metric V03 với V04 như một controlled architecture comparison.
- So sánh model chính thức phải dùng cùng dataset version.
- V05 is the current development dataset; V04 remains the latest completed locked-Test benchmark until V05 Test evaluation is finished.
- Test Set của từng version được khóa sau khi version đó được chốt.
