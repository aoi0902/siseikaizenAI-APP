"""ui/live_view.py

映像・骨格・スコアを表示するだけの表示専用部品。
（設計メモ: live_view_py_設計メモ.md に対応）

app_window の on_tick が「最新フレーム＋33点＋スコア」を update_view で渡してくる。
live_view はそれを描くだけ。撮影・推定・採点・DB・スレッドには一切触れない。

責任:
  - 最新フレームを PySide6 に映す
  - 33点を画素に直して骨格を重ねる
  - 現在のスコアを表示する（採点不可・検出なしの状態表示も）

設計の核:
  - 表示専用。
  - 表示先(Qt)に合わせて画像を用意する（OpenCV の BGR を QImage に渡す）。
  - 画像オブジェクト(QPixmap)の参照を保持する（GC で消える落とし穴を避ける）。
  - 毎フレーム呼ばれるので軽く保つ。

なぜ Qt では PhotoImage を使わないのか:
  - tkinter は PhotoImage（PNGバイト列）でしか画像を貼れなかったため imencode/Pillow が要った。
  - Qt は QImage が numpy のメモリ(BGR888)を直接受け取れるので、PNG 化も色変換も不要で軽い。
"""

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication


def _ui_font(point_size, bold=False):
    """既定のUIフォントを土台に、サイズ・太さだけ変えた QFont を返す。

    QFont("", n) のようにファミリ名を空で作ると、Qt が "MS Sans Serif" に
    フォールバックし DirectWrite が読み込めず警告（qt.qpa.fonts）を出す。
    有効な既定フォント(QApplication.font())を土台にすればファミリ名が入り、警告が出ない。
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


# 描画する骨格の接続（姿勢に関わる上半身・体幹・脚）。config の名前定数で取り違え防止。
_CONNECTIONS = [
    (config.LEFT_SHOULDER, config.RIGHT_SHOULDER),              # 両肩
    (config.LEFT_SHOULDER, config.LEFT_ELBOW), (config.LEFT_ELBOW, config.LEFT_WRIST),    # 左腕
    (config.RIGHT_SHOULDER, config.RIGHT_ELBOW), (config.RIGHT_ELBOW, config.RIGHT_WRIST), # 右腕
    (config.LEFT_SHOULDER, config.LEFT_HIP), (config.RIGHT_SHOULDER, config.RIGHT_HIP),    # 胴体
    (config.LEFT_HIP, config.RIGHT_HIP),                        # 腰
    (config.LEFT_HIP, config.LEFT_KNEE), (config.LEFT_KNEE, config.LEFT_ANKLE),            # 左脚
    (config.RIGHT_HIP, config.RIGHT_KNEE), (config.RIGHT_KNEE, config.RIGHT_ANKLE),        # 右脚
]

# 色（OpenCV は BGR）。線=緑、点=赤。
_LINE_COLOR = (0, 255, 0)
_POINT_COLOR = (0, 0, 255)

# スコアの色分け（CSS の色名/HEX。Qt のスタイルシートで使う）。
_COLOR_GOOD = "#2e7d32"   # 緑
_COLOR_WARN = "#f9a825"   # 黄
_COLOR_BAD = "#c62828"    # 赤
_COLOR_IDLE = "#757575"   # 灰（採点不可・検出なし）


def _draw_pose(frame_bgr, landmarks, min_visibility):
    """正規化33点を画素に直して、骨格をフレームに重ねる（その場で描画）。

    低 visibility の点・その点を含む線は描かない（チラつき防止）。
    """
    h, w = frame_bgr.shape[:2]

    def visible(i):
        return landmarks[i].visibility >= min_visibility

    def pixel(i):
        # 0〜1 の割合 ×（幅・高さ）で画素位置にする（第2・3回の描画用変換）。
        return (int(landmarks[i].x * w), int(landmarks[i].y * h))

    # 線（両端が信用できるときだけ）
    for a, b in _CONNECTIONS:
        if visible(a) and visible(b):
            cv2.line(frame_bgr, pixel(a), pixel(b), _LINE_COLOR, 2)

    # 点（信用できる点だけ）
    for i in range(len(landmarks)):
        if visible(i):
            cv2.circle(frame_bgr, pixel(i), 4, _POINT_COLOR, -1)


class LiveView(QWidget):
    """映像＋骨格＋スコアを表示するウィジェット部品。

    使い方:
        lv = LiveView()
        layout.addWidget(lv)
        lv.update_view(frame_bgr, landmarks, score, advice)   # app_window の on_tick から
    """

    def __init__(self, parent=None, display_size=None, draw_skeleton=None,
                 min_visibility=None):
        super().__init__(parent)
        self.display_size = display_size or config.LIVE_DISPLAY_SIZE  # (幅, 高さ)
        self.draw_skeleton = config.DRAW_SKELETON if draw_skeleton is None else draw_skeleton
        self.min_visibility = config.MIN_VISIBILITY if min_visibility is None else min_visibility

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 映像を貼るラベルと、スコアを出すラベル。
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        # 表示サイズぶんの領域を最初から確保する。こうしないと、フレーム到着前は
        # ラベルの高さが 0 でレイアウトに押し潰され、映像が縦に見切れてしまう。
        self._image_label.setMinimumSize(self.display_size[0], self.display_size[1])
        layout.addWidget(self._image_label, alignment=Qt.AlignCenter)

        self._score_label = QLabel("スコア: ---")
        self._score_label.setAlignment(Qt.AlignCenter)
        self._score_label.setFont(_ui_font(18, bold=True))
        layout.addWidget(self._score_label)

        # 改善提案の文言（悪姿勢時のみ表示。目を引くオレンジ）。
        self._advice_label = QLabel("")
        self._advice_label.setAlignment(Qt.AlignCenter)
        self._advice_label.setWordWrap(True)
        self._advice_label.setFont(_ui_font(14))
        self._advice_label.setStyleSheet("color: #e65100;")
        layout.addWidget(self._advice_label)

        # 現在の画像オブジェクトの参照を保持する（消える落とし穴を避ける）。
        self._pixmap = None

    # ── 表示更新（app_window の on_tick から呼ばれる） ──
    def update_view(self, frame_bgr, landmarks, score, advice=None):
        """最新フレーム＋33点＋スコア（＋改善提案）で表示を差し替える。"""
        # フレームが無ければ描画をスキップ（古い表示のまま）。
        if frame_bgr is None:
            self._update_score(score, detected=landmarks is not None)
            self._update_advice(advice)
            return

        # 軽量化のため表示サイズへ縮小（dsize=(幅,高さ)）。
        img = cv2.resize(frame_bgr, self.display_size)

        # 検出ありなら骨格を重ねる。検出なしは素のフレームのまま。
        if self.draw_skeleton and landmarks is not None and len(landmarks) > 0:
            _draw_pose(img, landmarks, self.min_visibility)

        # Qt 用の画像に変換し、参照を保持して差し替える。
        pixmap = self._to_pixmap(img)
        self._image_label.setPixmap(pixmap)
        self._pixmap = pixmap  # ← 必ず保持

        self._update_score(score, detected=landmarks is not None and len(landmarks) > 0)
        self._update_advice(advice)

    # ── 画像変換（表示先=Qt に合わせる） ──
    def _to_pixmap(self, img_bgr):
        """OpenCV の BGR フレームを Qt の QPixmap にする。

        QImage は numpy のメモリを「参照」するだけ（コピーしない）ので、その場で
        QPixmap.fromImage によりピクセルを焼き込んでしまえば、numpy 配列が解放されても安全。
        OpenCV は BGR 並びなので Format_BGR888 を使えば色変換(cvtColor)も不要。
        """
        # cv2.resize の出力は連続メモリだが、念のため連続性を保証してから渡す。
        img_bgr = np.ascontiguousarray(img_bgr)
        h, w = img_bgr.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        # copy() で Qt 側にピクセルを持たせ、numpy のバッファ寿命から切り離す。
        return QPixmap.fromImage(qimg.copy())

    # ── スコア表示（色分け＋状態表示） ──
    def _update_score(self, score, detected):
        if score is None:
            # 採点不可。検出の有無で文言を分け、古いスコアを残さない。
            if not detected:
                text, color = "検出なし（席を外していませんか）", _COLOR_IDLE
            else:
                text, color = "採点不可（キャリブレーションが必要です）", _COLOR_IDLE
        else:
            text = f"スコア: {score:.0f} / 100"
            if score >= config.SCORE_GOOD:
                color = _COLOR_GOOD
            elif score >= config.SCORE_WARN:
                color = _COLOR_WARN
            else:
                color = _COLOR_BAD
        self._score_label.setText(text)
        self._score_label.setStyleSheet(f"color: {color};")

    # ── 改善提案の表示（悪姿勢時のみ。無ければ消す） ──
    def _update_advice(self, advice):
        self._advice_label.setText(advice if advice else "")


# ── 単体での動作確認 ──
def _self_test():
    # 手作りの33点（まっすぐ座った想定）。
    LM = __import__("collections").namedtuple("LM", ["x", "y", "z", "visibility"])

    def upright():
        pts = [LM(0.5, 0.5, 0.0, 1.0) for _ in range(config.NUM_LANDMARKS)]
        pts[config.NOSE] = LM(0.5, 0.30, 0.0, 1.0)
        pts[config.LEFT_SHOULDER] = LM(0.40, 0.50, 0.0, 1.0)
        pts[config.RIGHT_SHOULDER] = LM(0.60, 0.50, 0.0, 1.0)
        pts[config.LEFT_HIP] = LM(0.43, 0.80, 0.0, 1.0)
        pts[config.RIGHT_HIP] = LM(0.57, 0.80, 0.0, 1.0)
        return pts

    results = []

    # ① 純粋描画: 黒画像に骨格を重ねると画素が変化する
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    before = frame.copy()
    _draw_pose(frame, upright(), config.MIN_VISIBILITY)
    results.append(("骨格描画でフレームが変化する", bool((frame != before).any())))

    # ② 低 visibility は描かない
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    invisible = [LM(0.5, 0.5, 0.0, 0.1) for _ in range(config.NUM_LANDMARKS)]
    _draw_pose(frame2, invisible, config.MIN_VISIBILITY)
    results.append(("低visibilityは描かれない", not (frame2 != 0).any()))

    # ③ BGR フレームを QImage 化できる（色変換不要・連続メモリ）
    qimg = QImage(np.ascontiguousarray(frame).data, 640, 480, 3 * 640,
                  QImage.Format_BGR888)
    results.append(("BGRフレームをQImage化できる", not qimg.isNull()))

    # ④ Qt 部分（オフスクリーンでも可）: 画像参照(QPixmap)が保持される
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    lv = LiveView()
    lv.update_view(frame, upright(), 95.0)
    held = lv._pixmap is not None
    lv.update_view(frame, None, None)  # 検出なし
    results.append(("Qt更新で画像参照が保持される", held))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("live_view 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
