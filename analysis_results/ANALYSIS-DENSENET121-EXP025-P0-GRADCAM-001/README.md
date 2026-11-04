# P0 Healthy error audit

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** Phân tích read-only. Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../README.md) · [Danh mục 95 README](../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


Read-only source analysis. Empty feature/CAM CSV means no verified image/model processing. Review template is blinded to predictions and contains no review decisions.

Run `uv run --python 3.11 --no-project python analysis/scripts/exp025_p0_audit.py --help`. Supply original archives and checkpoint. Output must be a fresh directory.
