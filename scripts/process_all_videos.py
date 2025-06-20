# process_all_videos.py
import os
import glob
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import importlib


RAW_DIR = 'data/raw'
KEYPOINTS_DIR = 'data/keypoints'

# 引数の定義
#コマンド実行時に外から渡せる設定を定義
parser = argparse.ArgumentParser()
# 特徴量csvの保存場所の指定
parser.add_argument('--features-dir', type=str, default='data/features/v1_basic', help='特徴量CSVの出力先')
# 特徴量定義の入った.pyファイル名
parser.add_argument('--features-module', type=str, default='features_v1', help='特徴量抽出モジュール名（.pyファイル名）')
args = parser.parse_args()
FEATURES_DIR = args.features_dir
FEATURES_MODULE = args.features_module

# 特徴量モジュールのインポート
feature_extractor = importlib.import_module(FEATURES_MODULE)

os.makedirs(KEYPOINTS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

#関節座標をmediapipeを用いて抽出
def extract_pose_from_video(video_path, csv_output_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame)
        if result.pose_landmarks:
            row = {'frame': frame_idx}
            for i, lm in enumerate(result.pose_landmarks.landmark):
                row[f'P{i}_x'] = lm.x
                row[f'P{i}_y'] = lm.y
            results.append(row)
        frame_idx += 1
    cap.release()

    if results:
        keypoints_df = pd.DataFrame(results)
        keypoints_df.to_csv(csv_output_path, index=False)
        print(f"[INFO] Keypoints saved: {csv_output_path}")
    else:
        print(f"[WARN] No pose detected in: {video_path}")

#特徴量の抽出
#鼻の高さ平均、腕の振れ幅、左右足の差分など
def calculate_motion_features(csv_input_path, feature_output_path, base_filename):
    df = pd.read_csv(csv_input_path)
    features = feature_extractor.extract_features_v1(df, base_filename)
    features_df = pd.DataFrame([features])
    features_df.to_csv(feature_output_path, index=False)
    print(f"[INFO] Features saved: {feature_output_path}")


#data/rawにある動画を全て処理する
#既にfeatures.csvが存在する動画はスキップ
if __name__ == '__main__':
    video_files = glob.glob(os.path.join(RAW_DIR, '*.mp4'))
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        keypoints_csv = os.path.join(KEYPOINTS_DIR, f"{base_name}.csv")
        features_csv = os.path.join(FEATURES_DIR, f"{base_name}_features.csv")

        if os.path.exists(features_csv):
            print(f"[SKIP] Already processed: {base_name}")
            continue

        print(f"[START] Processing: {base_name}")
        extract_pose_from_video(video_path, keypoints_csv)
        calculate_motion_features(keypoints_csv, features_csv, base_name)

    print(f"\n✅ All videos processed. (Output: {FEATURES_DIR})")