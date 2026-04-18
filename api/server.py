from __future__ import annotations

import time
import json
import hashlib
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from victim.model_def import SimpleCNN
from config import DEVICE, MODEL_PATH, LOG_PATH
from PIL import Image
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Victim model not found at: {MODEL_PATH}")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

app = FastAPI(title="Victim MNIST API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    account_id: str = Field(..., description="Unique account/user identifier")
    image: List[List[float]] = Field(..., description="28x28 nested list, values in [0,1]")

class PredictResponse(BaseModel):
    pred: int
    probs: List[float]


def validate_image(image: List[List[float]]) -> np.ndarray:
    arr = np.array(image, dtype=np.float32)
    if arr.shape != (28, 28):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image shape (28, 28), got {arr.shape}"
        )
    return arr

def hash_image(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def log_query(record: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/")
def root():
    return {"service": "Victim MNIST API", "status": "ok", "device": device}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...), account_id: str = "manual_user"):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return predict(PredictRequest(account_id=account_id, image=arr.tolist()))

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    # 1. Validate input
    img = validate_image(req.image)                    # (28,28)
    x = img.reshape(1, 1, 28, 28)                     # (1,1,28,28)

    # 2. Inference
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    # 3. Log
    log_query({
        "timestamp": time.time(),
        "account_id": req.account_id,
        "input_hash": hash_image(x),
        "input_vector": img.flatten().tolist(),
        "pred": pred,
        "probs": probs.tolist()
    })

    # 4. Return
    return PredictResponse(pred=pred, probs=probs.tolist())
