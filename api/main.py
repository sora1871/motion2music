# api/main.py
# FastAPI サーバ（あなたの学習コードと完全互換の LSTMAutoEncoder）
# - POST /v1/latents/from-array : JSON配列 [L, D] を受け取り、window化->標準化->z 抽出
# - POST /v1/latents/from-csv   : CSVアップロードを受け取り、同上
# - GET  /health                : ヘルスチェック
#
# 期待する環境変数（Render の「Environment」や render.yaml で設定可）
#   MODEL_PATH        (例: /opt/render/project/src/models/lstm_0814/lstm_0814_model.p)
#   SCALER_PATH       (例: /opt/render/project/src/models/lstm_0814/scaler.joblib)
#   FEATURE_COLS_PATH (任意：学習時の列順JSON。無ければ数値列をそのまま使用)
#   INPUT_DIM=99, LATENT_DIM=8, HIDDEN_DIM=128
#   API_KEY=<任意のキー> を設定すると X-API-Key ヘッダ必須に
#
# ローカル起動:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000

import os
import io
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import joblib


# =========================
# 学習コードに合わせた実装
# =========================

def create_windows(data, window_size=20, stride=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=99, hidden_dim=128, latent_dim=8):
        super().__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)         # h_n: [1, B, H]
        h_last = h_n[-1]                            # [B, H]
        z = self.to_latent(h_last)                  # [B, Z]
        # Decode
        h_dec = self.from_latent(z)                 # [B, H]
        h_dec_seq = h_dec.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, H]
        dec_out, _ = self.decoder_lstm(h_dec_seq)   # [B, T, H]
        recon = self.output_layer(dec_out)          # [B, T, D]
        return recon, z


# =========================
# 環境 & アーティファクトの読み込み
# =========================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(ROOT, "models", "lstm_0814", "lstm_0814_model.pt")  
)
SCALER_PATH = os.environ.get(
    "SCALER_PATH",
    os.path.join(ROOT, "models", "lstm_0814", "scaler.joblib")
)
FEAT_PATH = os.environ.get(
    "FEATURE_COLS_PATH",
    os.path.join(ROOT, "models", "lstm_0814", "feature_cols.json")  # 無ければ勝手に数値列抽出
)

INPUT_DIM = int(os.environ.get("INPUT_DIM", "99"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "8"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))

API_KEY_REQ = os.environ.get("API_KEY", None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found: {path}")


# モデル/スケーラの存在チェック
_ensure_exists(MODEL_PATH, "MODEL_PATH")
_ensure_exists(SCALER_PATH, "SCALER_PATH")

# スケーラ（学習時に保存した StandardScaler）
scaler = joblib.load(SCALER_PATH)

# 学習時の列順（任意）
feature_cols: Optional[List[str]] = None
if os.path.exists(FEAT_PATH):
    try:
        with open(FEAT_PATH, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    except Exception:
        feature_cols = None  # 壊れていても動かせるようにフォールバック


# モデルのロード（構造そのまま）
model = LSTMAutoEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
state = torch.load(MODEL_PATH, map_location=DEVICE)  # 学習時は torch.save(state_dict, path)
# もし torch.save({"model_state_dict": ...}) 形式なら下のキーを使う:
if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("encoder_lstm") for k in state.keys()):
    state = state["state_dict"]
elif isinstance(state, dict) and "model_state_dict" in state and not any(k.startswith("encoder_lstm") for k in state.keys()):
    state = state["model_state_dict"]

model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()


# =========================
# ユーティリティ
# =========================

def pick_numeric_in_order(df: pd.DataFrame, expected_dim: int) -> pd.DataFrame:
    """
    学習時の列順が FEATURE_COLS_PATH に保存されていればそれを強制。
    無ければ数値列だけを選択。次元が違えば 400 を返す。
    """
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise HTTPException(400, f"入力CSVに必要な列が不足: {missing[:5]} ...")
        df_num = df[feature_cols]
    else:
        df_num = df.select_dtypes(include="number")

    if df_num.shape[1] != expected_dim:
        raise HTTPException(
            400,
            f"期待次元={expected_dim} ですが、入力は {df_num.shape[1]} 列です。列順・余計な数値列を確認してください。"
        )
    return df_num


def windows_to_scaled_tensor(X: np.ndarray, window_size: int, stride: int) -> torch.Tensor:
    """
    学習時の StandardizedMotionDataset と同じ前処理：
      - create_windows: [L,D] -> [N,T,D]
      - flatten: [N*T, D]
      - scaler.transform -> reshape: [N,T,D]
      - torch.float32 tensor へ
    """
    wins = create_windows(X, window_size, stride)  # [N,T,D]
    if wins.shape[0] == 0:
        return torch.empty((0, window_size, X.shape[1]), dtype=torch.float32, device=DEVICE)
    flat = wins.reshape(-1, X.shape[1])            # [N*T, D]
    scaled = scaler.transform(flat).reshape(wins.shape).astype(np.float32)
    return torch.from_numpy(scaled).to(DEVICE)


def verify_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY_REQ and x_api_key != API_KEY_REQ:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =========================
# FastAPI
# =========================

app = FastAPI(title="LSTM-AE Latent API (notebook-compatible)", version="1.0.0")


class ArrayRequest(BaseModel):
    data: List[List[float]]    # フレーム×特徴 = [L, D]
    window_size: int = 20
    stride: int = 1
    return_meta: bool = False


class ArrayResponse(BaseModel):
    n_windows: int
    latent_dim: int
    z: List[List[float]]
    starts: Optional[List[int]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "has_feature_cols": bool(feature_cols),
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }


@app.post("/v1/latents/from-array", response_model=ArrayResponse)
def latents_from_array(payload: ArrayRequest, x_api_key: Optional[str] = Header(default=None)):
    verify_api_key(x_api_key)

    X = np.asarray(payload.data, dtype=np.float32)  # [L, D]
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise HTTPException(400, f"data の形が不正です。期待 [L,{INPUT_DIM}] ですが {list(X.shape)} を受け取りました。")

    x_t = windows_to_scaled_tensor(X, payload.window_size, payload.stride)  # [N,T,D]
    if x_t.shape[0] == 0:
        return ArrayResponse(n_windows=0, latent_dim=LATENT_DIM, z=[], starts=[] if payload.return_meta else None)

    with torch.no_grad():
        _, z = model(x_t)                   # [N, Z]
        z_np = z.detach().cpu().numpy().tolist()

    starts = list(range(0, X.shape[0] - payload.window_size + 1, payload.stride)) if payload.return_meta else None
    return ArrayResponse(n_windows=len(z_np), latent_dim=LATENT_DIM, z=z_np, starts=starts)


@app.post("/v1/latents/from-csv", response_model=ArrayResponse)
async def latents_from_csv(file: UploadFile = File(...),
                           window_size: int = 20,
                           stride: int = 1,
                           return_meta: bool = False,
                           x_api_key: Optional[str] = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "CSV の読込に失敗しました")

    # 学習時と同じ列順で数値を取り出す（なければ数値列のみ）
    df_num = pick_numeric_in_order(df, expected_dim=INPUT_DIM)
    X = df_num.to_numpy(dtype=np.float32)  # [L, D]

    x_t = windows_to_scaled_tensor(X, window_size, stride)  # [N,T,D]
    if x_t.shape[0] == 0:
        return ArrayResponse(n_windows=0, latent_dim=LATENT_DIM, z=[], starts=[] if return_meta else None)

    with torch.no_grad():
        _, z = model(x_t)
        z_np = z.detach().cpu().numpy().tolist()

    starts = list(range(0, X.shape[0] - window_size + 1, stride)) if return_meta else None
    return ArrayResponse(n_windows=len(z_np), latent_dim=LATENT_DIM, z=z_np, starts=starts)
