# autoencoder_predict.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input-csv', type=str, required=True, help='all_features.csv のパス')
parser.add_argument('--model-dir', type=str, required=True, help='モデルとスケーラーが保存されているディレクトリ')
parser.add_argument('--output-z', type=str, default='z_vectors.csv', help='Zベクトルの出力先')
args = parser.parse_args()

# ファイル読み込み
features_df = pd.read_csv(args.input_csv)
X = features_df.drop(columns=['filename'])

# スケーラー読み込みと変換
scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# encoderモデル読み込み
encoder_path = os.path.join(args.model_dir, 'encoder.h5')
encoder = load_model(encoder_path)

# 潜在ベクトル出力
z_values = encoder.predict(X_scaled)
z_df = pd.DataFrame(z_values, columns=[f'z{i+1}' for i in range(z_values.shape[1])])
z_df.insert(0, 'filename', features_df['filename'])
z_df.to_csv(args.output_z, index=False)

print(f"✅ Zベクトルを {args.output_z} に出力しました。")