"""ui/feedback_view.py

日々のスコア推移グラフを画面に出すだけの部品。
（設計メモ: feedback_view_py_設計メモ.md に対応）

第7回の3段の役割分担をそのまま守る:
  集計 → pandas / 作図 → matplotlib / 埋め込み → PySide6(Qt)。

責任:
  - スコアを取り出す（get_scores 呼び出し）
  - pandas で日ごとに集計（平均/中央値）
  - matplotlib で折れ線/棒グラフを作る
  - Qt のウィンドウに埋め込んで表示（別ウィンドウは開かない）

担わないこと:
  - SQL の実行 → db/database.py（get_scores は生データを返すだけ）
  - 採点・通知・撮影・推定 → 各担当
  - 取り出し間隔・スレッド → app_window

設計の核:
  - 表示と軽い集計だけ。集計方法を変えても DB を触らない（pandas に任せる）。
  - pyplot を使わず Figure を直接作る（別ウィンドウを開かないため）。
  - オンデマンド更新（毎フレームではない）。図は使い回し、ためない。
  - メインスレッドで描く（ワーカーから触らない）。

tkinter からの移行メモ:
  - 埋め込みは FigureCanvasTkAgg → FigureCanvasQTAgg に差し替えるだけ。
    どちらも「pyplot を介さず Figure をそのまま GUI に貼る」ための橋渡しで考え方は同じ。
"""

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout

# ui/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

# 日本語ラベルが豆腐(□)にならないよう、Windows標準フォントを設定。
# 環境に無くても matplotlib が代替フォントへフォールバックするだけで落ちない。
matplotlib.rcParams["font.family"] = config.FEEDBACK_FONT
matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け回避


class FeedbackView(QWidget):
    """日次スコア推移グラフを埋め込み表示するウィジェット部品。

    使い方:
        fv = FeedbackView(get_scores=lambda: db.get_scores(uid),
                          on_refresh=app.on_refresh_feedback)
        layout.addWidget(fv)
        # 更新ボタン、または app_window から refresh() で再描画
    """

    def __init__(self, get_scores, on_refresh=None, parent=None,
                 days=None, aggregate=None, chart_type=None, figure_size=None):
        super().__init__(parent)
        # 生データの取得口（app_window が database.get_scores を束ねて渡す）。
        self._get_scores = get_scores
        self._on_refresh = on_refresh
        self.days = config.FEEDBACK_DAYS if days is None else days
        self.aggregate = config.FEEDBACK_AGGREGATE if aggregate is None else aggregate
        self.chart_type = config.FEEDBACK_CHART_TYPE if chart_type is None else chart_type
        figure_size = config.FEEDBACK_FIGURE_SIZE if figure_size is None else figure_size

        layout = QVBoxLayout(self)

        # 更新ボタン（押下で on_refresh を呼ぶ＝オンデマンド更新）。
        self._button = QPushButton("グラフを更新")
        self._button.clicked.connect(self._handle_refresh_click)
        layout.addWidget(self._button, alignment=Qt.AlignCenter)

        # 図は1つだけ作って使い回す（refresh のたびに作らない＝メモリ対策）。
        self._figure = Figure(figsize=figure_size, dpi=100)
        self._ax = self._figure.add_subplot(111)

        # Figure を Qt キャンバスへ橋渡し（別窓を開かない）。参照を保持する。
        self._canvas = FigureCanvasQTAgg(self._figure)
        # 最小高さを与え、レイアウトに押し潰されてグラフが見切れるのを防ぐ。
        self._canvas.setMinimumHeight(config.FEEDBACK_MIN_HEIGHT)
        layout.addWidget(self._canvas)

    # ── 更新ボタン押下 ──
    def _handle_refresh_click(self):
        # app_window 経由で再描画させる（配電盤を通す）。直接 refresh でも動くよう保険。
        if self._on_refresh is not None:
            self._on_refresh()
        else:
            self.refresh()

    # ── 集計（pandas・純粋。Qt不要でテスト可能） ──
    def _aggregate(self, scores):
        """生データ[{timestamp, score}] → 日付ごとの代表値(Series)。

        直近 days 日に絞り、同じ日付でまとめて平均/中央値をとる。
        データが無ければ None。
        """
        if not scores:
            return None
        df = pd.DataFrame(scores)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.normalize()  # 時刻を落として「日付」に切る

        # 直近 days 日に絞る（端末の今日を基準）。
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=self.days - 1)
        df = df[df["date"] >= cutoff]
        if df.empty:
            return None

        # 集計は pandas に任せる（mean / median）。
        return df.groupby("date")["score"].agg(self.aggregate)

    # ── 再描画（取り出し→集計→作図→埋め込み） ──
    def refresh(self):
        """最新データで集計し直し、グラフを描き直す（オンデマンド）。"""
        scores = self._get_scores() if self._get_scores is not None else None
        daily = self._aggregate(scores)

        # 同じ Axes を使い回し、中身だけ描き直す（図をためない）。
        self._ax.clear()

        if daily is None or len(daily) == 0:
            # データ無し: 落ちずに案内を出す。
            self._ax.text(0.5, 0.5, "まだ記録がありません",
                          ha="center", va="center", transform=self._ax.transAxes)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
        else:
            label = "中央値" if self.aggregate == "median" else "平均"
            if self.chart_type == "bar":
                self._ax.bar(daily.index, daily.values, color="#1565c0")
            else:
                self._ax.plot(daily.index, daily.values, marker="o", color="#1565c0")
            self._ax.set_title("日次 姿勢スコア推移")
            self._ax.set_xlabel("日付")
            self._ax.set_ylabel(f"{label}スコア")
            self._ax.set_ylim(0, 100)            # スコアは0〜100に固定
            self._ax.grid(True, alpha=0.3)
            self._figure.autofmt_xdate()         # 日付ラベルを見やすく回転

        self._figure.tight_layout()
        self._canvas.draw()                      # メインスレッドで描画


# ── 単体での動作確認 ──
def _self_test():
    from datetime import datetime, timedelta
    from PySide6.QtWidgets import QApplication

    results = []

    # 直近の日付で作ったにせスコア（今日2件・昨日3件）。
    now = datetime.now()
    today = now.strftime("%Y-%m-%d %H:%M:%S")
    yest = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    fake_scores = [
        {"timestamp": today, "score": 80.0},
        {"timestamp": today, "score": 90.0},   # 今日 平均=85
        {"timestamp": yest, "score": 60.0},
        {"timestamp": yest, "score": 70.0},
        {"timestamp": yest, "score": 80.0},     # 昨日 平均=70
    ]

    app = QApplication.instance() or QApplication([])
    fv = FeedbackView(get_scores=lambda: fake_scores)

    # ① 集計（pandas・Qt不要）: 手計算と一致するか
    daily = fv._aggregate(fake_scores)
    today_norm = pd.Timestamp(now.date())
    yest_norm = pd.Timestamp((now - timedelta(days=1)).date())
    ok_agg = (abs(daily[today_norm] - 85.0) < 1e-6 and abs(daily[yest_norm] - 70.0) < 1e-6)
    results.append(("日ごとの平均が手計算と一致(今日85/昨日70)", ok_agg))

    # ② データ無しでも落ちず None
    results.append(("データ無しはNone", fv._aggregate([]) is None))

    # ③ 埋め込み描画: 別窓を開かず refresh が通る
    fv.refresh()
    results.append(("グラフ描画が通る(別窓を開かない)", True))

    # ④ 図を使い回す: 何度 refresh しても Figure は同一（ためない）
    fig_id = id(fv._figure)
    fv.refresh()
    fv.refresh()
    results.append(("更新で図がたまらない(Figure使い回し)", id(fv._figure) == fig_id))

    # ⑤ データ無しの refresh も安全
    fv2 = FeedbackView(get_scores=lambda: [])
    fv2.refresh()
    results.append(("データ無しのrefreshも安全", True))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("feedback_view 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
