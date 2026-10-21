"""main.py

アプリの起動口（エントリポイント）。
実体は ui/app_window.py の AppWindow。ここは「呼ぶだけ」の薄い入口。

実行:
    python main.py
注意: conda 環境 newpy3_12 で実行すること（base 環境は _sqlite3 のDLL読み込みに失敗する & これ以外の環境はpyside6がインストールされていない）。
"""

import os
import sys

# ── ネイティブライブラリ(MediaPipe / TensorFlow Lite)のログ抑制（第1層） ──
# MediaPipe/TFLite が出す情報・警告（XNNPACKデリゲート生成、feedback manager、
# landmark_projection の NORM_RECT 警告など）は実害はないが起動時に大量に出る。
# ライブラリ読み込み前（＝この時点）に環境変数を立てて抑制を試みる。
#   GLOG_minloglevel: 0=INFO,1=WARNING,2=ERROR,3=FATAL → 2 で INFO/WARNING を抑制
#   TF_CPP_MIN_LOG_LEVEL: 3 で TensorFlow(Lite) のログを ERROR 以上だけに
# ※ 近年の MediaPipe は C++ の absl ログを使い、これらの環境変数では抑えきれない
#   ことがある。取りこぼしは core/pose.py 側で fd=2 を一時退避して確実に黙らせる（第2層）。
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# 各層（root / core / workers / ui）を import 検索パスに通す。
# 本プロジェクトは各モジュールが兄弟名（import config / from notifier import ... 等）で
# 参照し合うため、起動口でまとめてパスを通しておく。
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "core", "workers", "ui"):
    _path = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app_window import main  # ui/app_window.py


if __name__ == "__main__":
    main()
