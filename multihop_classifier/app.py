import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from feature_utils import extract_features, feature_vector
from model import BertWithFeatures

MODEL_DIR = Path(os.getenv("MULTIHOP_MODEL_DIR", "models/complexity-classifier"))
MODEL_VERSION = os.getenv("MULTIHOP_MODEL_VERSION", "local-v1")
DEVICE = torch.device(os.getenv("MULTIHOP_DEVICE", "cpu"))

app = FastAPI(title="Multihop Complexity Classifier")
tokenizer = None
model = None


class PredictRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    features: dict | None = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


@app.on_event("startup")
def load_model():
    global tokenizer, model
    if not (MODEL_DIR / "model.pt").exists():
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")
    model = BertWithFeatures(str(MODEL_DIR / "encoder")).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=DEVICE))
    model.eval()


@app.get("/health")
def health():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


def predict_one(request: PredictRequest) -> dict:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    features = request.features or extract_features(request.question)
    encoded = tokenizer(
        request.question,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            encoded["input_ids"].to(DEVICE),
            encoded["attention_mask"].to(DEVICE),
            torch.tensor([feature_vector(features)], dtype=torch.float32, device=DEVICE),
        )
        probability = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
    return {
        "label": "multi" if probability >= 0.5 else "single",
        "multi_hop_probability": probability,
        "model_version": MODEL_VERSION,
        "features": features,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    return predict_one(request)


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest):
    return {"items": [predict_one(item) for item in request.items]}
