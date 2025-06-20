import cv2
import pandas as pd
import numpy as np
import os
from mediapipe.python.solutions import pose as mp_pose

def extract_pose_from_video(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    all_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_data = []
            for lm in results.pose_landmarks.landmark:
                frame_data.extend([lm.x, lm.y])
            all_landmarks.append(frame_data)

    cap.release()

    if all_landmarks:
        df = pd.DataFrame(
            all_landmarks,
            columns=[f'P{i}_{axis}' for i in range(33) for axis in ('x', 'y')]
        )
        df.to_csv(output_csv_path, index=False)
        return True
    else:
        print(f"⚠️ No landmarks extracted from {video_path}")
        return False
