"""
Model Engine Dispatcher for Cashew Leaf Disease Diagnosis.
Selects EXP-034 (DenseNet121 x5 + YOLO26s + Stacker F3) or the legacy engine.
A fallback is never silent: the reason is attached to /api/model-info and to
every prediction (model_meta.engine_warning) so the UI can show it.
"""
import logging
from typing import Any, Dict, Optional

from config import EXP034_ID, MODEL_ENGINE

logger = logging.getLogger("CashewLeafAI.Engine")


class EngineDispatcher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EngineDispatcher, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._active_service = None
        self._engine_type = None
        self._warning: Optional[str] = None
        choice = (MODEL_ENGINE or "exp034").lower().strip()

        if choice == "exp034":
            try:
                from exp034_service import EXP034Service
                self._active_service = EXP034Service()
                self._engine_type = "exp034"
                logger.info(f"Engine Dispatcher: combined engine active ({EXP034_ID}).")
                return
            except Exception as e:  # noqa: BLE001
                logger.error(f"Engine Dispatcher: EXP-034 failed to load: {e}")
                self._warning = (
                    f"Không nạp được EXP-034 ({type(e).__name__}: {e}). "
                    "Đang dùng engine legacy (chỉ DenseNet121, khoanh vùng theo màu sắc)."
                )

        from model_service import get_legacy_service
        try:
            self._active_service = get_legacy_service()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Không có engine chẩn đoán nào khả dụng. "
                + (self._warning or "")
                + f" Legacy: {type(e).__name__}: {e}"
            ) from e
        self._engine_type = "legacy"
        legacy_warning = getattr(self._active_service, "using_fallback", False)
        if legacy_warning:
            self._warning = (self._warning or "") + " Legacy đang dùng checkpoint dự phòng SCRATCH-001 (chỉ debug)."
        logger.warning(f"Engine Dispatcher: legacy engine active. {self._warning or ''}")

    @property
    def engine_type(self) -> str:
        return self._engine_type

    def predict_image(self, image_bytes: bytes, sample_name: Optional[str] = None) -> Dict[str, Any]:
        result = self._active_service.predict_image(image_bytes, sample_name=sample_name)
        if self._warning:
            meta = result.setdefault("model_meta", {})
            existing = meta.get("engine_warning")
            meta["engine_warning"] = self._warning if not existing else f"{self._warning} {existing}"
        return result

    def get_model_info(self) -> Dict[str, Any]:
        info = dict(self._active_service.get_model_info())
        info["engine_warning"] = self._warning
        return info


# Global engine instance
model_service = EngineDispatcher()
