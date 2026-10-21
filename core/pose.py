"""core/pose.py

MediaPipe による骨格推定「だけ」を担当する窓口。
（設計メモ: pose_py_設計メモ.md に対応）

責任:
  - MediaPipe モデルの読み込み・保持・解放
  - BGR → RGB への色変換（推定の直前）
  - 1フレームから 33 点（正規化座標 0〜1）を推定
  - 「検出できたか」を呼び出し側へ伝える

担わないこと:
  - 映像の取得（→ core/capture.py）
  - ループ（→ workers/pose_worker.py）
  - 画素への変換・描画（→ ui/live_view.py）
  - 角度・比率の計算（→ core/features.py）

入出力の約束:
  入力 = BGR フレーム + 単調増加するタイムスタンプ[ms]
  出力 = PoseResult(detected, landmarks)。landmarks は 0〜1 の正規化座標のまま。
"""

import os
import sys
import contextlib
from collections import namedtuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@contextlib.contextmanager
def _suppress_native_stderr():
    """MediaPipe/TFLite の C++ 層が標準エラー(fd=2)へ直接書くログを一時的に捨てる。

    なぜ必要か:
      これらのログ（"XNNPACK delegate"、"inference_feedback_manager"、
      "landmark_projection_calculator" など）は Python ではなく C++ 側(absl/TFLite)が
      出すため、Python の logging や GLOG_minloglevel 等の環境変数だけでは
      抑えきれないことがある。そこでネイティブ層が書き込む出力先そのもの
      （ファイル記述子 2）を、この区間だけ os.devnull に付け替えて黙らせる。

    安全性:
      付け替えるのは「ネイティブが直接書く fd」だけ。Python の例外は fd を通らず
      通常どおり送出されるので、握りつぶす心配はない（区間も最小限に限定する）。
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        # 標準エラーが差し替えられていて fileno を持たない環境では何もしない。
        yield
        return

    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)  # fd=2 を devnull へ
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)    # 元に戻す
        os.close(devnull_fd)
        os.close(saved_fd)

# core/ の中から実行されても、パッケージとして import されても config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# ── チーム内のデータの取り決め（設計メモ 2-A） ──
# MediaPipe の生の結果をそのまま回さず、自分たちの素朴な形に詰め替えて渡す。
Landmark = namedtuple("Landmark", ["x", "y", "z", "visibility"])
PoseResult = namedtuple("PoseResult", ["detected", "landmarks"])

# 検出なしのときに返す共通の「空の結果」。
NO_POSE = PoseResult(detected=False, landmarks=[])

# config の文字列 → MediaPipe の実行モードへの対応。
# config 側に mediapipe を import させない（疎結合に保つ）ためここで変換する。
_RUNNING_MODES = {
    "VIDEO": vision.RunningMode.VIDEO,
    "IMAGE": vision.RunningMode.IMAGE,
    "LIVE_STREAM": vision.RunningMode.LIVE_STREAM,
}


class PoseEstimator:
    """MediaPipe 推定器を「読み込み〜解放」まで面倒を見るまとまり。

    with 文に対応:
        with PoseEstimator() as pose:
            result = pose.estimate(bgr_frame, timestamp_ms)
    """

    def __init__(
        self,
        model_path=None,
        running_mode=None,
        num_poses=None,
        min_detection_confidence=None,
        min_tracking_confidence=None,
        min_presence_confidence=None,
    ):
        # 省略時は config の既定値を使う（直書きしない）。
        model_path = config.MODEL_PATH if model_path is None else model_path
        running_mode = config.RUNNING_MODE if running_mode is None else running_mode
        num_poses = config.NUM_POSES if num_poses is None else num_poses
        if min_detection_confidence is None:
            min_detection_confidence = config.MIN_DETECTION_CONFIDENCE
        if min_tracking_confidence is None:
            min_tracking_confidence = config.MIN_TRACKING_CONFIDENCE
        if min_presence_confidence is None:
            min_presence_confidence = config.MIN_PRESENCE_CONFIDENCE

        # モデルが無い・壊れている場合は曖昧に進めず、明確なエラーで止める。
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_path}\n"
                "models/ 以下に pose_landmarker(_lite).task を置いたか確認してください。"
            )

        if running_mode not in _RUNNING_MODES:
            raise ValueError(
                f"未知の running_mode です: {running_mode}（VIDEO/IMAGE/LIVE_STREAM）"
            )

        # ── 推定器の組み立て（読み込みは初期化で一度だけ） ──
        # model_asset_path だと MediaPipe の C++ 層がファイルを開く。この層は
        # Windows で非ASCII（日本語）パスを開けず errno=-1 で落ちる。
        # そこで Python 側でバイト列として読み込み、buffer で渡す（Python は
        # Unicode パスを正しく扱えるため、日本語パスでも安全）。
        with open(model_path, "rb") as f:
            model_buffer = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=_RUNNING_MODES[running_mode],
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # モデル読み込み時に XNNPACK / feedback_manager のログが出るので抑制する。
        with _suppress_native_stderr():
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    # ── 推定 ──
    def estimate(self, frame_bgr, timestamp_ms):
        """1フレーム（BGR）から 33 点を求める。

        引数:
          frame_bgr     … capture が返した BGR 画像
          timestamp_ms  … 単調増加するタイムスタンプ[ms]（ループ側が管理）

        戻り値: PoseResult(detected, landmarks)
          - 検出あり: (True, [Landmark×33])  ← 各値は 0〜1 の正規化座標
          - 検出なし: (False, [])            ← 呼び出し側は detected を見てから触る
        """
        if self._landmarker is None:
            # 解放後・初期化前に呼ばれた場合は使えない状態だと知らせる。
            raise RuntimeError("PoseEstimator は既に閉じられています。")

        # 推定の直前に BGR → RGB へ変換（MediaPipe は RGB 前提）。
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 動画モードは前後関係を伝えるため単調増加のタイムスタンプが必須。
        # 推論時に landmark_projection_calculator の警告が出るので抑制する。
        with _suppress_native_stderr():
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        # 人が写っていない等で空のことがある。空なら「検出なし」を返す。
        if not result.pose_landmarks:
            return NO_POSE

        # num_poses=1 想定。先頭の人の 33 点を自分たちの形に詰め替える。
        raw_landmarks = result.pose_landmarks[0]
        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in raw_landmarks
        ]
        return PoseResult(detected=True, landmarks=landmarks)

    # ── 解放 ──
    def close(self):
        """推定器を閉じる。二重に呼ばれても壊れないようにする。"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # ── with 対応（確実な解放） ──
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False  # 例外は握りつぶさず外へ伝える。


# ── 単体での動作確認（第2段の自己テスト。確認専用） ──
def _self_test():
    """capture と組み合わせ、検出した点を映像に重ねて表示する。q で終了。"""
    from capture import Camera

    print("骨格推定の動作確認: q キーで終了します。")
    timestamp_ms = 0
    with Camera() as cam, PoseEstimator() as pose:
        while cam.is_opened():
            ok, frame = cam.read_frame()
            if not ok:
                print("フレーム取得に失敗しました。終了します。")
                break

            # ループ側がタイムスタンプを管理し、単調増加させる。
            timestamp_ms += 33  # ≒30fps 相当
            result = pose.estimate(frame, timestamp_ms)

            if result.detected:
                # 確認用の最小描画。正規化座標(0〜1)を画素に直して点を打つ。
                h, w = frame.shape[:2]
                for lm in result.landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"detected: {len(result.landmarks)} pts",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    frame, "no pose", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.imshow("pose self-test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print("解放しました。")


if __name__ == "__main__":
    _self_test()
