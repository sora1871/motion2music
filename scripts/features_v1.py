# features_v1.py
import pandas as pd
import numpy as np

def extract_features_v1(df, base_filename):
    """
    Mediapipeから抽出されたキーポイントCSV (df) から、
    動画1本分の代表的な動作特徴量を計算し、辞書として返す関数。
    """
    features = {'filename': base_filename}

    try:
        # 例：鼻の高さ（y）の平均と標準偏差
        features['nose_y_mean'] = df['P0_y'].mean()
        features['nose_y_std'] = df['P0_y'].std()

        # 左右手首のx揺れ幅
        features['l_wrist_x_range'] = df['P15_x'].max() - df['P15_x'].min()
        features['r_wrist_x_range'] = df['P16_x'].max() - df['P16_x'].min()

        # 左右足首のy上下動
        features['l_ankle_y_range'] = df['P27_y'].max() - df['P27_y'].min()
        features['r_ankle_y_range'] = df['P28_y'].max() - df['P28_y'].min()

        # 左右足のx距離差の平均
        df['step_diff'] = np.abs(df['P27_x'] - df['P28_x'])
        features['step_diff_mean'] = df['step_diff'].mean()

    except KeyError as e:
        print(f"[ERROR] 特徴量抽出失敗: キーポイント {e} が見つかりません")

    return features
