"""core/capture.py

カメラの入出力「だけ」を担当する窓口。
（設計メモ: capture_py_設計メモ.md に対応）

責任:
  - カメラを開く / 開けたか確認する / 1フレーム取得する / 解放する
  - 必要なら左右反転（鏡表示）までの軽い整え
  - 実 fps・実解像度の確認

担わないこと:
  - ループ（→ workers/pose_worker.py）
  - 表示 imshow（→ ui/）
  - 色変換 BGR→RGB（→ 推定直前に core/pose.py）
  - 骨格推定・採点・保存・通知（→ 各担当）

約束: このファイルが外へ返す映像は「常に BGR」。RGB が必要な側が変換する。
"""

import cv2

# core/ の中から実行されても、パッケージとして import されても
# 同じく config を読めるようにする。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


class Camera:
    """1台のカメラを「開く〜解放」まで面倒を見るまとまり。

    with 文に対応しているので、確実に解放したいときは
        with Camera() as cam:
            ok, frame = cam.read_frame()
    のように使う。
    """

    def __init__(
        self,
        camera_index=None,
        frame_width=None,
        frame_height=None,
        target_fps=None,
        mirror=None,
    ):
        # 引数が省略されたら config の既定値を使う（直書きしない）
        self.camera_index = config.CAMERA_INDEX if camera_index is None else camera_index
        self.frame_width = config.FRAME_WIDTH if frame_width is None else frame_width
        self.frame_height = config.FRAME_HEIGHT if frame_height is None else frame_height
        self.target_fps = config.TARGET_FPS if target_fps is None else target_fps
        self.mirror = config.MIRROR if mirror is None else mirror

        self._cap = None
        self.open()

    # ── 開く ──
    def open(self):
        """指定番号のカメラに接続し、解像度・fps を設定する。

        開けない場合（番号違い・他アプリ使用中・接続不良）は
        曖昧に進めず、明確なエラーで停止する。
        """
        # CAP_DSHOW は Windows 向けバックエンド指定。起動が速く安定しやすい。
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # 解像度・fps は「希望」を伝えるだけ。実際に通るかは機材次第。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not cap.isOpened():
            # 後段が空の画像を掴まないよう、ここで止める。
            raise RuntimeError(
                f"カメラ(index={self.camera_index})を開けませんでした。"
                "番号違い・他アプリが使用中・接続不良の可能性があります。"
            )

        self._cap = cap
        return self

    # ── 開けたか確認 ──
    def is_opened(self):
        """接続できているかの真偽を返す。"""
        return self._cap is not None and self._cap.isOpened()

    # ── 1フレーム取得 ──
    def read_frame(self):
        """最新の1コマを取り出す。

        戻り値: (成功フラグ, 画像)
          - 成功: (True, BGR画像)
          - 失敗: (False, None)  ← 呼び出し側はこれを見て安全にループを抜ける

        色変換はここでは行わない（BGR のまま返す）。
        mirror が有効なら鏡表示用に水平反転だけ施す。
        """
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if not ok:
            # 一時的な読み込み失敗。空画像を流さず、失敗を素直に伝える。
            return False, None

        if self.mirror:
            frame = self.flip(frame)

        return True, frame

    # ── 左右反転（必要時） ──
    @staticmethod
    def flip(frame):
        """鏡のように見せるための水平反転（軽い整えのみ）。"""
        # 第2引数 1 = 左右（水平）反転
        return cv2.flip(frame, 1)

    # ── 実際の値の確認 ──
    @property
    def actual_fps(self):
        """設定が反映された実 fps。"""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def actual_resolution(self):
        """設定が反映された実解像度 (幅, 高さ)。"""
        if self._cap is None:
            return (0, 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    # ── 解放 ──
    def release(self):
        """カメラを手放す。二重に呼ばれても壊れないようにする。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── with 対応（確実な解放） ──
    def __enter__(self):
        # 既に open 済みなのでそのまま自分を返す。
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # ブロックを抜けたら例外の有無にかかわらず必ず解放する。
        self.release()
        return False  # 例外は握りつぶさず、そのまま外へ伝える。


# ── 単体での動作確認（第1段の自己テスト。本番の ui/ とは別物） ──
def _self_test():
    """カメラを開く → 表示 → q で終了 → 解放 の最小ループ。"""
    print("カメラ動作確認: q キーで終了します。")
    with Camera() as cam:
        print(f"実解像度: {cam.actual_resolution} / 実fps: {cam.actual_fps}")
        while cam.is_opened():
            ok, frame = cam.read_frame()
            if not ok:
                print("フレーム取得に失敗しました。終了します。")
                break
            cv2.imshow("capture self-test (BGR)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print("解放しました。")


if __name__ == "__main__":
    _self_test()
