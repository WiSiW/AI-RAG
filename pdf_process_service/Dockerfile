FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"
COPY app.py .
EXPOSE 8020
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8020"]