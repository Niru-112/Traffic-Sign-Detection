import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title="Traffic Sign Detection API")

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

if device == "cuda":
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

# -------------------------------
# MODEL PATH
# -------------------------------
MODEL_PATH = "models/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Model not found at {MODEL_PATH}")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO(MODEL_PATH)
model.to(device)

# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def root():
    return {"status": "Traffic Sign Detection API running 🚦"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = model(img, device=device)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

    return {"detections": detections, "count": len(detections)}
