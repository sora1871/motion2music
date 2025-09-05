# api/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np, torch, joblib, os, sys

# scripts/ を import path に追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from scripts.model_def import LSTMAutoEncoder

# カレントディレクトリずれ対策（リポジトリ直下を基準にする）
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ★ 実在するほうに合わせてください（今は lstm_ae に設定）
MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_ae", "lstm_ae_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "lstm_ae", "scaler.joblib")

model, scaler = None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    # スケーラ読み込み
    scaler = joblib.load(SCALER_PATH)

    # ★ 学習時と同じハイパラに必ず合わせる
    INPUT_DIM  = 99
    HIDDEN_DIM = 128
    LATENT_DIM = 8

    model = LSTMAutoEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    print("✅ Model & scaler loaded")
    yield

app = FastAPI(lifespan=lifespan)

class SequenceInput(BaseModel):
    features: list[list[float]]  # (T,F) 例: F=99

@app.post("/encode")
def encode_features(data: SequenceInput):
    try:
        x = np.asarray(data.features, dtype=np.float32)   # (T,F)
        if x.ndim != 2:
            raise HTTPException(400, "features must be 2D: [T][F]")
        T, F = x.shape
        expected_F = model.output_layer.weight.shape[0]   # = input_dim
        if F != expected_F:
            raise HTTPException(400, f"feature dim mismatch: got {F}, expected {expected_F}")

        # 学習時と同じ前処理（各フレームにスケール）
        x_scaled = scaler.transform(x)                    # (T,F)
        x_tensor = torch.from_numpy(x_scaled).unsqueeze(0)  # (1,T,F)

        with torch.no_grad():
            # encode() が実装されていなければ forward の戻り値から z を取得
            if hasattr(model, "encode"):
                z = model.encode(x_tensor)                # (1, latent_dim)
            else:
                _, z = model(x_tensor)                    # (1, latent_dim)

        return {"latent": z.squeeze(0).tolist(), "length": T}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/")
def root():
    return {"message": "API is running"}
