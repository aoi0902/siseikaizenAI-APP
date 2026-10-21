"""ui/session_view.py

「今回の使用中」のスコア推移をリアルタイムに折れ線で見せる部品。

feedback_view（日次グラフ）との役割分担:
  - feedback_view … DB に貯めた記録を「日ごとの平均」で振り返る（長期の傾向）。
  - session_view  … 起動してから今この瞬間までのスコアのゆらぎを追う（その場の確認）。
    DB には触らず、app_window から渡される 1 フレームごとのスコアだけを溜めて描く。

責任:
  - app_window の on_tick が渡すスコアを時系列で溜める（直近 window 秒だけ保持）
  - 一定間隔（毎フレームではない）で折れ線に描き直す
  - 良好/注意ラインを補助線として重ね、今の姿勢が保ててるか一目で分かるようにする

担わないこと:
  - 撮影・推定・採点 → 各担当 / DB 記録 → database / スレッド → app_window

設計の核:
  - 表示専用・状態は「点の列」だけ。
  - 描画は QTimer で間引く（30fps で毎回 matplotlib を描くと重いため）。
  - 溜め込みは直近 window 秒に限る（deque で古い点を捨て、メモリを一定に保つ）。
"""

import time
from collections import deque

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtWidgets import QWidget, QVBoxLayout

# ui/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

# 日本語ラベルが豆腐(□)にならないよう、Windows標準フォントを設定。
matplotlib.rcParams["font.family"] = config.FEEDBACK_FONT
matplotlib.rcParams["axes.unicode_minus"] = False


class SessionChartView(QWidget):
    """今回のセッションのスコア推移を折れ線表示するウィジェット部品。

    使い方:
        sv = SessionChartView()
        layout.addWidget(sv)
        # on_tick のたびに（採点不可 None も含めて）スコアを渡す
        sv.add_score(score)
        # 終了時にタイマーを止める
        sv.stop()
    """

    def __init__(self, parent=None, window_seconds=None, redraw_ms=None, figure_size=None):
        super().__init__(parent)
        self.window_seconds = (config.SESSION_WINDOW_SECONDS
                               if window_seconds is None else window_seconds)
        redraw_ms = config.SESSION_REDRAW_MS if redraw_ms is None else redraw_ms
        figure_size = config.SESSION_FIGURE_SIZE if figure_size is None else figure_size

        # 点の列（経過秒, スコア）。採点不可は None を入れて折れ線を切る（離席が線でつながらない）。
        self._times = deque()
        self._scores = deque()
        self._start = None   # 最初のスコアが来た時刻(monotonic)。ここを 0 秒とする。

        layout = QVBoxLayout(self)

        # 図は1つだけ作って使い回す（描き直しのたびに作らない＝メモリ対策）。
        self._figure = Figure(figsize=figure_size, dpi=100)
        self._ax = self._figure.add_subplot(111)

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(config.SESSION_MIN_HEIGHT)  # 見切れ防止
        layout.addWidget(self._canvas)

        # 最初に一度「計測待ち」を描いておく。
        self._redraw()

        # 再描画は QTimer で間引く（毎フレームは重い）。add_score は溜めるだけ。
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(redraw_ms)

    # ── app_window の on_tick から呼ばれる（溜めるだけ・描画はしない） ──
    def add_score(self, score, now=None):
        """1フレームぶんのスコアを時系列に溜める。採点不可は None のまま入れる。"""
        now = time.monotonic() if now is None else now
        if self._start is None:
            self._start = now
        elapsed = now - self._start

        self._times.append(elapsed)
        # None（採点不可）は NaN として持ち、折れ線を途切れさせる（席を外した区間を線で結ばない）。
        self._scores.append(float("nan") if score is None else float(score))

        # 直近 window 秒より古い点は捨てる（横スクロールしつつメモリを一定に保つ）。
        cutoff = elapsed - self.window_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._scores.popleft()

    # ── 再描画（QTimer が一定間隔で呼ぶ） ──
    def _redraw(self):
        self._ax.clear()

        # 数値スコアが1つも無ければ（起動直後・ずっと離席）案内を出す。
        has_value = any(s == s for s in self._scores)  # NaN 以外が1つでもあるか（NaN!=NaN）
        if not self._times or not has_value:
            self._ax.text(0.5, 0.5, "計測待ち…（カメラに姿勢が映ると記録されます）",
                          ha="center", va="center", transform=self._ax.transAxes)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
        else:
            times = list(self._times)
            scores = list(self._scores)
            self._ax.plot(times, scores, color="#1565c0", linewidth=1.5)

            # 良好(緑)・注意(黄)の目安ラインを薄く重ね、今の姿勢が保ててるか分かりやすく。
            self._ax.axhline(config.SCORE_GOOD, color="#2e7d32", linestyle="--",
                             linewidth=0.8, alpha=0.6)
            self._ax.axhline(config.SCORE_WARN, color="#f9a825", linestyle="--",
                             linewidth=0.8, alpha=0.6)

            self._ax.set_title("今回のスコア推移")
            self._ax.set_xlabel("経過時間（秒）")
            self._ax.set_ylabel("スコア")
            self._ax.set_ylim(0, 100)
            # 横軸は直近 window 秒だけを映してスクロールさせる。
            latest = times[-1]
            self._ax.set_xlim(max(0.0, latest - self.window_seconds), max(latest, 1.0))
            self._ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    # ── 終了時の後片付け ──
    def stop(self):
        """再描画タイマーを止める（ウィンドウを閉じるときに app_window が呼ぶ）。"""
        if self._timer is not None:
            self._timer.stop()


# ── 単体での動作確認 ──
def _self_test():
    from PySide6.QtWidgets import QApplication

    results = []
    app = QApplication.instance() or QApplication([])
    sv = SessionChartView(window_seconds=10)

    # ① 数値スコアを溜めると点が増える
    for i in range(5):
        sv.add_score(80.0 - i, now=float(i))
    results.append(("スコアを溜めると点が増える", len(sv._scores) == 5))

    # ② 採点不可(None)は NaN として入る（折れ線を切る）
    sv.add_score(None, now=5.0)
    results.append(("採点不可はNaNで保持", sv._scores[-1] != sv._scores[-1]))  # NaN!=NaN

    # ③ window を超えた古い点は捨てられる（直近10秒だけ保持）
    sv.add_score(50.0, now=100.0)  # 一気に時間を飛ばす → 古い点(経過0〜5秒)は全部落ちる
    results.append(("古い点はwindow外で捨てる", len(sv._times) == 1 and sv._times[0] == 100.0))

    # ④ 再描画が例外なく通る（数値あり）
    sv._redraw()
    results.append(("数値ありで再描画が通る", True))

    # ⑤ データ無し（新規）でも「計測待ち」で落ちない
    sv2 = SessionChartView()
    sv2._redraw()
    results.append(("データ無しでも再描画が通る", True))

    # ⑥ stop でタイマーが止まる
    sv.stop()
    results.append(("stopでタイマー停止", not sv._timer.isActive()))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("session_view 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
