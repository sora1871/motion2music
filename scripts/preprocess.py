import pandas as pd
import numpy as np

def normalize_pose_df(df):
    x_cols = [f'P{i}_x' for i in range(33)]
    y_cols = [f'P{i}_y' for i in range(33)]

    # 全関節の重心で平行移動
    center_x = df[x_cols].mean(axis=1)
    center_y = df[y_cols].mean(axis=1)

    for i in range(33):
        df[f'P{i}_x'] -= center_x
        df[f'P{i}_y'] -= center_y

    # スケーリング（肩幅基準）
    dx = df['P11_x'] - df['P12_x']
    dy = df['P11_y'] - df['P12_y']
    shoulder_width = np.sqrt(dx**2 + dy**2)
    shoulder_width[shoulder_width == 0] = 1e-6

    for i in range(33):
        df[f'P{i}_x'] /= shoulder_width
        df[f'P{i}_y'] /= shoulder_width

    return df
