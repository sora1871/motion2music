#  motion2music API 利用ガイド

このリポジトリでは、**人の動作データ（CSV）を入力して、学習済み LSTM オートエンコーダから潜在ベクトルを取得する API** を利用できます。  
以下の手順に従えば、誰でも簡単に使えます 

---

##  1. 準備

1. **Python のインストール**  
   - 推奨: Python 3.10 

2. **ライブラリのインストール**  
   ターミナル（コマンドプロンプト）で以下を実行してください：
   ```bash
   pip install requests pandas
入力CSVを用意

学習時と同じ特徴量列数（例: 99列）の数値データ

ファイル名は input.csv としてください（任意の名前でもOKですが、下記スクリプトと合わせてください）

## 2. API エンドポイント
デプロイ済みの API は以下のURLで公開されています：

https://motion2music.onrender.com
利用可能なエンドポイント：

GET /health
https://motion2music.onrender.com/health
→ サーバ稼働チェック

POST /v1/latents/from-csv
https://motion2music.onrender.com/v1/latents/from-csv
→ CSVファイルを送信し、潜在ベクトルを取得（おすすめ）

POST /v1/latents/from-array
https://motion2music.onrender.com/v1/latents/from-array
→ JSON配列を直接送信して潜在ベクトルを取得

## 3. 使い方（CSV → 潜在ベクトルCSV）
以下のスクリプトを call_api.py という名前で保存してください：

```python
import requests
import pandas as pd

BASE_URL = "https://motion2music.onrender.com"  # ← APIのURL
INPUT_CSV = "input.csv" 
OUTPUT_CSV = "latent_series.csv" #←自分の潜在変数の出力をさせたい名前にしてください


# CSVを送信
files = {"file": open(INPUT_CSV, "rb")}
params = {"window_size": 20, "stride": 1, "return_meta": True}

res = requests.post(f"{BASE_URL}/v1/latents/from-csv", # ←この/v1/latents/from-csvを指定しないと変換できない。
                    files=files, params=params) # ← "https://motion2music.onrender.com"これだけじゃだめ
res.raise_for_status()
data = res.json()

# 潜在ベクトルをDataFrame化
df = pd.DataFrame(data["z"])
if data.get("starts") is not None:
    df.insert(0, "start_frame", data["starts"])

df.to_csv(OUTPUT_CSV, index=False)
print(f" 保存しました: {OUTPUT_CSV}")
```

## 4. 実行方法
ターミナルで以下を実行します：

```
python call_api.py
```
入力: input.csv
出力: latent_series.csv

## 5. 出力例
潜在ベクトル次元が8の場合、出力CSVは以下のようになります：

```
start_frame,z0,z1,z2,z3,z4,z5,z6,z7
0,-0.123,0.456,0.018,...
1,-0.234,0.321,-0.045,...
2,-0.201,0.310,0.012,...
...
start_frame … 各ウィンドウの開始フレーム番号

z0, z1, ... … 潜在空間の数値ベクトル
```
