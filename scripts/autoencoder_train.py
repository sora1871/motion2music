# autoencoder_train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import argparse
import os
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--input-csv', type=str, required=True, help='入力する all_features.csv')
parser.add_argument('--output-dir', type=str, required=True, help='モデルとスケーラーを保存するディレクトリ')
parser.add_argument('--encoding-dim', type=int, default=4, help='潜在空間の次元数')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# データ読み込み
features_df = pd.read_csv(args.input_csv)
X = features_df.drop(columns=['filename'])

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# AE構築
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
h1 = Dense(16, activation='relu')(input_layer)
h2 = Dense(8, activation='relu')(h1)
encoded = Dense(args.encoding_dim, activation='tanh')(h2)

d1 = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(d1)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)


# 学習
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=4, shuffle=True, verbose=1)

# 保存
autoencoder.save(os.path.join(args.output_dir, 'autoencoder.h5'))
encoder.save(os.path.join(args.output_dir, 'encoder.h5'))
joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))

print(f"✅ モデルとスケーラーを {args.output_dir} に保存しました。")
