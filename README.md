# 姿勢改善AIアプリケーション

卒業研究。Webカメラの映像から MediaPipe で骨格を推定し、個人の基準姿勢と比べて姿勢をスコア化・通知する。

## 必要なもの

- Anaconda（または Miniconda）
- Webカメラ（30fps以上が好ましい）

## 環境構築

```
git clone https://github.com/aoi0902/siseikaizenAI-APP.git
cd siseikaizenAI-APP
conda create -n newPy3_12 python=3.12.7
conda activate newPy3_12
pip install -r requirements.txt
```

VSCode を使う場合は、右下のインタープリター選択で `newPy3_12` を選ぶ。

## 起動

```
conda activate newPy3_12
python main.py
```

## 初回の設定

- `config.py` の `CAMERA_INDEX` を自分のカメラに合わせる（`0` = 内蔵カメラ、`1` = 外付けWebカメラ）
- 姿勢データ（`data/posture.db`）は初回起動時に自動で作られる。個人データのため Git には含めない
