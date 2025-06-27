from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
import numpy as np
import joblib
import sys
import os
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from model_def import AutoEncoder

model_path = "models/ae_model.pt"
scaler_path = "models/scaler.joblib"

# グローバル変数（後から app.state に格納する）
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    scaler = joblib.load(scaler_path)
    model = AutoEncoder(input_dim=66, latent_dim=8)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ Model & scaler loaded")
    yield
    # 終了処理などがあればここに書く

app = FastAPI(lifespan=lifespan)

class FeatureInput(BaseModel):
    features: list[float]

@app.post("/encode")
def encode_features(data: FeatureInput):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        z = model.encoder(x_tensor).numpy()
    return {"latent": z[0].tolist()}

@app.get("/")
def root():
    return {"message": "API is running"}
