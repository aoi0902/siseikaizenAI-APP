"""ui/calibration_review.py

キャリブレーションの測定結果を見せて、保存するかを本人に選んでもらう確認画面。

なぜ必要か: 採点は基準値との差しか見ないので、悪い姿勢で測るとその姿勢が 100 点になる。
保存する前に「どんな姿勢を基準にしようとしているか」を本人が確かめられるようにする。

責任:
  - 基準値に最も近いフレームに骨格と参考線を重ねて見せる
  - 基準値の角度と、目安を外れたときの警告を見せる
  - 保存するか、やり直すかを呼び出し元（app_window）に返す

担わないこと:
  - 画像の選択・警告の判定 → core/calibration.py
  - 基準値の保存 → app_window / database.py
"""

import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

# ui/ から実行されても、パッケージ import でも config / core を読めるように。
try:
    import config
    from features import _select_side
except ModuleNotFoundError:
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root)                       # プロジェクトルート（config 用）
    sys.path.append(os.path.join(_root, "core"))  # features 用
    import config
    from features import _select_side

from live_view import _draw_pose, bgr_to_pixmap


_GUIDE_COLOR = (0, 255, 255)    # 垂直の参考線（黄。OpenCV は BGR）
_SEGMENT_COLOR = (255, 200, 0)  # 角度を測っている 耳-肩-腰 の線（水色）
_COLOR_OK = "#2e7d32"
_COLOR_WARN = "#e65100"


def draw_review_image(image, points):
    """骨格と参考線を重ねた新しい画像を返す（元の image は書き換えない）。

    文字は描かない（cv2.putText は日本語を描けない）。角度は Qt のラベルで出す。
    """
    img = image.copy()
    _draw_pose(img, points, config.MIN_VISIBILITY)

    if config.POSTURE_MODE == "side":
        h, w = img.shape[:2]
        side = _select_side(points)

        def pixel(i):
            return int(points[i].x * w), int(points[i].y * h)

        shoulder = pixel(side.shoulder)
        # 肩を通る垂直線。耳と腰がこの線からどれだけ離れているかで姿勢を目で確かめられる。
        cv2.line(img, (shoulder[0], 0), (shoulder[0], h - 1), _GUIDE_COLOR, 1)
        cv2.line(img, pixel(side.ear), shoulder, _SEGMENT_COLOR, 3)
        cv2.line(img, shoulder, pixel(side.hip), _SEGMENT_COLOR, 3)
    return img


def baseline_text(baseline):
    """基準値の角度を 1 行の文字列にする。"""
    return " / ".join(f"{config.FEATURE_LABELS[name]} {baseline[name]:.1f}°"
                      for name in config.USE_FEATURES)


def warning_text(warnings):
    """(表示する文, 色) を返す。範囲内でも良い姿勢とは限らないことを必ず伝える。"""
    if warnings:
        return ("\n".join(warnings) + "\n目安は参考値です。画像を見て判断してください。",
                _COLOR_WARN)
    return "目安の範囲内です。ただし目安は参考値なので、画像でも確かめてください。", _COLOR_OK


def review_calibration(parent, review_frame, baseline, warnings):
    """確認画面を開く。「この姿勢で保存」なら True、「やり直す」・閉じるなら False を返す。"""
    dialog = QDialog(parent)
    dialog.setWindowTitle("キャリブレーションの確認")
    layout = QVBoxLayout(dialog)

    layout.addWidget(QLabel("この姿勢を 100 点の基準にします。良い姿勢になっていますか？"),
                     alignment=Qt.AlignCenter)

    image = QLabel()
    image.setPixmap(bgr_to_pixmap(draw_review_image(review_frame["image"], review_frame["points"])))
    layout.addWidget(image, alignment=Qt.AlignCenter)

    layout.addWidget(QLabel(baseline_text(baseline)), alignment=Qt.AlignCenter)

    text, color = warning_text(warnings)
    message = QLabel(text)
    message.setWordWrap(True)
    message.setAlignment(Qt.AlignCenter)
    message.setStyleSheet(f"color: {color};")
    layout.addWidget(message)

    buttons = QHBoxLayout()
    retry = QPushButton("やり直す")
    retry.clicked.connect(dialog.reject)
    buttons.addWidget(retry)
    save = QPushButton("この姿勢で保存")
    save.clicked.connect(dialog.accept)
    buttons.addWidget(save)
    layout.addLayout(buttons)

    return dialog.exec() == QDialog.Accepted


# ── 単体での動作確認（カメラ不要・横向きモード） ──
def _self_test():
    from collections import namedtuple

    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    LM = namedtuple("LM", ["x", "y", "z", "visibility"])
    points = [LM(0.5, 0.5, 0.0, 0.05)] * config.NUM_LANDMARKS
    points[config.RIGHT_EAR] = LM(0.55, 0.20, 0.0, 1.0)
    points[config.RIGHT_SHOULDER] = LM(0.50, 0.40, 0.0, 1.0)
    points[config.RIGHT_HIP] = LM(0.48, 0.80, 0.0, 1.0)

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    drawn = draw_review_image(image, points)
    assert not image.any(), "元の画像を書き換えている"
    # 肩(x=320)を通る垂直線が、骨格の無い画像の上端まで届いている。
    assert tuple(int(c) for c in drawn[0, 320]) == _GUIDE_COLOR
    print("PASS 参考線を描く（元の画像は書き換えない）")

    baseline = {"neck_angle": 70.0, "trunk_angle": 5.0, "kyphosis_angle": 160.0}
    assert "首の前傾 70.0°" in baseline_text(baseline)
    assert warning_text([])[1] == _COLOR_OK
    text, color = warning_text(["首の前傾が目安の外です"])
    assert color == _COLOR_WARN and "首の前傾が目安の外です" in text
    print("PASS 角度と警告の文言")

    # 実際にダイアログを組み立て、ボタンの結果が戻り値になることを確かめる。
    app = QApplication.instance() or QApplication([])
    frame = {"image": image, "points": points}

    def run(button_text):
        def press():
            dialog = app.activeModalWidget()
            next(b for b in dialog.findChildren(QPushButton) if b.text() == button_text).click()
        QTimer.singleShot(0, press)
        return review_calibration(None, frame, baseline, [])

    assert run("この姿勢で保存") is True
    assert run("やり直す") is False
    print("PASS 保存なら True、やり直しなら False")


if __name__ == "__main__":
    _self_test()
