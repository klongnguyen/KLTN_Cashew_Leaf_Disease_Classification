# EXP-009 — Artifact inventory & scientific provenance

| Hạng mục | Trạng thái trên GitHub | Giải thích |
|---|---|---|
| README, method/leakage audit, comparison | **Đã lưu** | Nội dung riêng cho EXP-009 |
| 5-seed Validation/Test/Holdout CSV summaries | **Đã lưu** | Mean ± sample SD, class-level và per-seed |
| 15 confusion matrices CSV | **Đã lưu** | 5 seeds × Val/Test/Historical Holdout |
| Dataset audit, model config, pipeline smoke, selection metadata | **Đã lưu snapshot/summaries** | Snapshot rõ ràng, không tuyên bố giống byte nguồn |
| Real Val, pHash suspects, pre-fit, Train/Val diagnostics | **Đã lưu** | 5 seeds, cùng filename subgroup |
| Error audit (recurrent confident wrong) | **Đã lưu bảng top/recurrent chọn lọc** | Không có full per-image logs |
| SVG Accuracy chart | **Đã lưu, SVG dựng lại từ metric** | Không thay thế PNG source |
| `resnet50_transfer_5seed_009.ipynb` | **Chưa commit** | ZIP gốc chứa notebook ~204 KB |
| 3,785,100-byte original ANALYSIS ZIP | **Chưa commit** | SHA256 `65b77db413d3b9e0c559686116f023c4d299a666e6533fe905d8d2246ec03762` |
| FULL ZIP, 5 seed checkpoint `.weights.h5`, `.keras` deployment | **Chưa commit** | Cần canonical binary storage + checksum |
| Original PNG panel, per-epoch history / probe 50-batch, full per-image predictions | **Chưa commit** | Gốc còn trong ANALYSIS ZIP |

**Integrity:** Metadata hashes tham chiếu các file nguồn; JSON/CSV biên soạn trên GitHub có thể có SHA khác. Không suy diễn rằng mô hình có thể tải và suy luận chỉ với repo snapshot; phải có binary package kiểm tra hash.

**Kết quả EXP-009:** 95.23 ± 0.35% Val Accuracy; 92.89 ± 0.51% Test Accuracy; 75.83 ± 2.80% historical Holdout Accuracy; seed 2026 chọn bằng Val Loss. Đây là **ImageNet + partial fine-tune**, khác ResNet50 Transfer-008 **head-only frozen**.

