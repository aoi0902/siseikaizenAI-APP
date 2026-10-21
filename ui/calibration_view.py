"""ui/calibration_view.py

キャリブレーションの操作と案内の画面。
（設計メモ: calibration_view_py_設計メモ.md に対応）

測定そのものはしない（app_window 経由でワーカーに委ねる）。
このファイルは「押す・案内する・結果を見せる」だけ。

責任:
  - 開始ボタンと操作の案内を表示
  - 測定中の進捗（準備カウント→測定中カウント）を表示
  - 成功/失敗の結果を表示し、やり直しを促す
  - 開始を app_window に伝える（on_calibrate を呼ぶ）

担わないこと:
  - 測定の段取り → core/calibration.py / 実測 → workers
  - 基準値の保存 → database.py / スレッド・キュー → app_window
  - ワーカー・DB・core には直接触れない（すべて app_window 経由）。

設計の核:
  - 操作と案内だけ。測定はワーカー、保存は DB、橋渡しは app_window。
  - 画面を固めない: 進捗表示は QTimer（少し後にまた呼ぶ）でメインスレッド上で回す。
  - 二重起動を防ぐ: 測定中は開始ボタンを無効化する。
  - 失敗に優しく: やり直し前提の表示にする。

tkinter からの移行メモ:
  - widget.after(ms, fn) / after_cancel → 単発の QTimer を 1 本持ち、start/stop で代替する。
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QApplication


def _ui_font(point_size, bold=False):
    """既定のUIフォントを土台に、サイズ・太さだけ変えた QFont を返す。

    QFont("", n)（空ファミリ）は "MS Sans Serif" に化けて DirectWrite 警告を出すため、
    有効な既定フォント(QApplication.font())を土台にする。
    """
    font = QApplication.font()
    font.setPointSize(point_size)
    font.setBold(bold)
    return font

# ui/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# 結果の色分け（HEX。Qt のスタイルシートで使う）。
_COLOR_OK = "#2e7d32"      # 緑（成功）
_COLOR_NG = "#c62828"      # 赤（失敗）
_COLOR_INFO = "#1565c0"    # 青（案内・測定中）


class CalibrationView(QWidget):
    """キャリブレーションの開始・案内・結果表示を行うウィジェット部品。

    使い方:
        cv = CalibrationView(on_calibrate=app.on_calibrate)
        layout.addWidget(cv)
        # 結果が出たら app_window が cv.show_result(success, message) を呼ぶ
    """

    def __init__(self, on_calibrate, parent=None, countdown_seconds=None,
                 measure_seconds=None, guidance=None):
        super().__init__(parent)
        # 開始を伝える先（app_window.on_calibrate）。
        self._on_calibrate = on_calibrate
        self.countdown_seconds = (config.CALIBRATION_COUNTDOWN_SECONDS
                                  if countdown_seconds is None else countdown_seconds)
        self.measure_seconds = (config.CALIBRATION_SECONDS
                                if measure_seconds is None else measure_seconds)
        self.guidance = guidance or config.CALIBRATION_GUIDANCE_TEXT

        # 状態
        self._measuring = False
        # tkinter の after_id 相当。単発の QTimer を 1 本だけ持ち、必要に応じ stop する。
        self._timer = None

        layout = QVBoxLayout(self)

        # 部品: 開始ボタン・案内文・状態/結果表示
        self._button = QPushButton("キャリブレーション開始")
        self._button.clicked.connect(self.on_start_pressed)
        layout.addWidget(self._button, alignment=Qt.AlignCenter)

        self._guidance_label = QLabel(self.guidance)
        self._guidance_label.setAlignment(Qt.AlignCenter)
        self._guidance_label.setWordWrap(True)
        layout.addWidget(self._guidance_label)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setFont(_ui_font(14))
        layout.addWidget(self._status_label)

    # ── 開始ボタン押下 ──
    def on_start_pressed(self):
        """準備カウント → 測定要求 → 測定中カウント、の流れを始める。"""
        if self._measuring:
            return  # 二重起動防止（測定中は無視）
        self._measuring = True
        self._button.setEnabled(False)  # ボタン無効化
        self._run_countdown(self.countdown_seconds)

    # ── 準備カウントダウン（姿勢を整える時間） ──
    def _run_countdown(self, remaining):
        if remaining > 0:
            self._set_status(f"準備… {remaining}", _COLOR_INFO)
            self._schedule(1000, lambda: self._run_countdown(remaining - 1))
        else:
            self._set_status("測定開始", _COLOR_INFO)
            # ここで初めて実測をワーカーへ要求する（準備が済んでから測り始める）。
            if self._on_calibrate is not None:
                self._on_calibrate()
            # 測定中の見た目カウント（合否は app_window の show_result で確定）。
            self._run_measure_countdown(int(self.measure_seconds))

    # ── 測定中カウントダウン（見た目。実測はワーカー） ──
    def _run_measure_countdown(self, remaining):
        if not self._measuring:
            return
        if remaining > 0:
            self._set_status(f"測定中… 残り {remaining} 秒（動かずにキープ）", _COLOR_INFO)
            self._schedule(1000, lambda: self._run_measure_countdown(remaining - 1))
        else:
            self._set_status("結果を待っています…", _COLOR_INFO)

    # ── app_window が結果を知らせるための入口 ──
    def show_result(self, success, message):
        """成功/失敗の結果を表示し、ボタンを復帰させる（やり直し可能に）。"""
        self._cancel_pending()
        self._measuring = False
        self._button.setEnabled(True)
        self._set_status(message, _COLOR_OK if success else _COLOR_NG)

    def set_status(self, message):
        """汎用の状態表示（app_window がワーカー停止などを知らせるときに使う）。"""
        self._set_status(message, _COLOR_INFO)

    # ── 内部補助 ──
    def _set_status(self, message, color):
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color: {color};")

    def _schedule(self, ms, fn):
        """tkinter の after 相当。前の予約を取り消してから単発タイマーを張る。"""
        self._cancel_pending()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(fn)
        timer.start(ms)
        self._timer = timer

    def _cancel_pending(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


# ── 単体での動作確認（にせ app_window で配線を検証） ──
def _self_test():
    from PySide6.QtWidgets import QApplication

    results = []

    calls = {"on_calibrate": 0}

    def fake_on_calibrate():
        calls["on_calibrate"] += 1

    app = QApplication.instance() or QApplication([])
    # 準備カウント・測定カウントを 0 にしてタイマーを介さず即時に進める。
    view = CalibrationView(on_calibrate=fake_on_calibrate,
                           countdown_seconds=0, measure_seconds=0)

    # ① 開始でボタンが無効になり、on_calibrate が呼ばれる
    view.on_start_pressed()
    results.append(("開始でon_calibrateが呼ばれる", calls["on_calibrate"] == 1))
    results.append(("測定中はボタン無効", view._button.isEnabled() is False))

    # ② 二重押ししても on_calibrate は増えない
    view.on_start_pressed()
    results.append(("二重起動を防ぐ", calls["on_calibrate"] == 1))

    # ③ 失敗結果 → ボタン復帰・やり直し可能
    view.show_result(False, "うまく測れませんでした。もう一度お願いします。")
    results.append(("失敗でボタン復帰", view._button.isEnabled() is True))
    results.append(("失敗後は再度開始できる(measuring解除)", view._measuring is False))

    # ④ 再挑戦 → 成功結果
    view.on_start_pressed()
    view.show_result(True, "基準値を保存しました。")
    results.append(("再挑戦で成功表示", calls["on_calibrate"] == 2 and view._measuring is False))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("calibration_view 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
