from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import sys
import os

# scripts/ をimport対象に追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from model_def import AutoEncoder  # モデルの定義を読み込み

# FastAPIアプリのインスタンスを作成
app = FastAPI()

# モデルとスケーラーの読み込み
model_path = "models/ae_model.pt"
scaler_path = "models/scaler.joblib"

scaler = joblib.load(scaler_path)

model = AutoEncoder(input_dim=66, latent_dim=8)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 入力の定義（Pydantic）
class FeatureInput(BaseModel):
    features: list[float]  # 長さ66の特徴量ベクトル

# エンドポイントの定義
@app.post("/encode")
def encode_features(data: FeatureInput):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        z = model.encoder(x_tensor).numpy()
    return {"latent": z[0].tolist()}
