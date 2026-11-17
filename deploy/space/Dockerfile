# Hugging Face Space (Docker) - FastAPI backend: DenseNet121 x5 + YOLO26s (EXP-035)
FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs these shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Layout must match config.py: PROJECT_ROOT/<EXP id>/pipeline_bundle and Application/ood
COPY --chown=user Application ./Application
COPY --chown=user EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04 ./EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04

WORKDIR /home/user/app/Application
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
