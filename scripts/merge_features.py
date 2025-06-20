# merge_features.py
import os
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--features-dir', type=str, required=True, help='特徴量CSVが保存されているディレクトリ')
args = parser.parse_args()
FEATURES_DIR = args.features_dir

output_path = os.path.join(FEATURES_DIR, 'all_features.csv')

csv_files = glob.glob(os.path.join(FEATURES_DIR, '*_features.csv'))

if not csv_files:
    print(f"[WARN] No feature files found in {FEATURES_DIR}")
    exit()

all_dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        all_dfs.append(df)
    except Exception as e:
        print(f"[ERROR] Failed to read {file}: {e}")

merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged {len(csv_files)} files into {output_path}")
