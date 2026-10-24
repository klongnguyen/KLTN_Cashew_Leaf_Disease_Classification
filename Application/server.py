"""
FastAPI Backend Server for Cashew Leaf Disease Diagnosis Web App.
Using DenseNet121 Transfer Version 014 (Best Checkpoint: Seed 123).
"""
import io
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import STATIC_DIR, SAMPLES_DIR, DISEASE_INFO
from model_service import model_service

app = FastAPI(
    title="Cashew Leaf Disease Diagnosis AI",
    description="Web service for diagnosing cashew leaf diseases using DenseNet121 Transfer v014 (Seed 123)",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Cashew Leaf Disease Classifier"}


@app.get("/api/model-info")
async def get_model_info():
    """Return model specifications, metrics, and disease metadata."""
    return model_service.get_model_info()


@app.get("/api/samples")
async def get_samples():
    """Return available sample images for instant testing."""
    sample_items = [
        {
            "id": "ath_01.png",
            "title": "Mẫu Bệnh Thán Thư",
            "class_hint": "anthracnose",
            "url": "/static/samples/ath_01.png",
            "description": "Lá điều có vết đốm hoại tử viền sẫm đặc trưng của thán thư."
        },
        {
            "id": "mine01.png",
            "title": "Mẫu Sâu Vẽ Bùa",
            "class_hint": "leaf_miner",
            "url": "/static/samples/mine01.png",
            "description": "Lá điều có đường đục ngoằn ngoèo ánh bạc của sâu đục lá."
        },
        {
            "id": "red01.png",
            "title": "Mẫu Bệnh Rỉ Đỏ",
            "class_hint": "red_rust",
            "url": "/static/samples/red01.png",
            "description": "Lá điều có các đốm đỏ gạch/rỉ sắt gồ lên trên mặt lá."
        }
    ]
    return {"samples": sample_items}


@app.post("/api/predict-sample/{sample_name}")
async def predict_sample(sample_name: str):
    """Diagnose a pre-loaded sample image directly."""
    sample_file = SAMPLES_DIR / sample_name
    if not sample_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy mẫu ảnh: {sample_name}"
        )

    with open(sample_file, "rb") as f:
        image_bytes = f.read()

    try:
        result = model_service.predict_image(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi chuẩn đoán: {str(e)}"
        )


@app.post("/api/predict")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Diagnose cashew leaf disease from a single uploaded image.
    Enforces strict file type checking (images only) and single file upload.
    """
    # Validate MIME type or file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if file.content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp tải lên không hợp lệ. Vui lòng chỉ tải lên 1 tệp hình ảnh (.PNG, .JPG, .JPEG, .WEBP)."
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc tệp tin: {str(e)}"
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp hình ảnh rỗng. Vui lòng chọn tệp hợp lệ."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kích thước tệp quá lớn (vượt quá giới hạn 25MB)."
        )

    try:
        result = model_service.predict_image(image_bytes)
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xảy ra trong quá trình mô hình chuẩn đoán: {str(e)}"
        )


# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the single page web application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        {"message": "Cashew Leaf AI Backend is running. Static frontend index.html not found."},
        status_code=200
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
