"""workers/pose_worker.py

重い処理のループ本体（ワーカースレッド）。
（設計メモ: worker_thread_integration_設計メモ.md に対応）

役割分担（第5・6回の並行処理）:
  - ワーカー（このファイル）= 「考える」だけ:
      撮影(capture)→推定(pose)→特徴量(features)→採点(scoring) をループで回し、
      結果をキューに置く。カメラと推定器はワーカーが開いて・使って・閉じる。
  - メイン（ui/app_window 側・別ファイル）= 「見せる・知らせる・残す」だけ:
      after でキューをドレインし、描画(ui)・通知(notifier)・記録(database) へ渡す。
      画面・DB に触るのはメインだけ。基準値はメモリでワーカーに渡す。

両者の間は「キュー1本」でつなぐ。

なぜ threading で十分か（メモ §9）:
  目的は CPU 並列ではなく「重い待ち(カメラ待ち・MediaPipeのネイティブ処理)で
  画面を止めないこと(応答性)」。MediaPipe はネイティブ処理中 GIL を手放すので、
  メインの mainloop が固まらない。
"""

import time
import queue
import threading
from collections import namedtuple

# workers/ から実行されても、パッケージ import でも config / core を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root)                       # プロジェクトルート（config 用）
    sys.path.append(os.path.join(_root, "core"))  # core の各モジュール用
    import config


# ── キューで渡すもの（受け渡しの形・メモ §4） ──
# frame      … 表示用の素のフレーム（BGR）。描画は ui の担当なので landmarks も一緒に渡す。
# landmarks  … 検出した33点（描画用）。検出なしなら None。
# score      … 0〜100 の点数 ／ 採点不可なら None（scoring.UNSCORABLE）。
# advice     … 改善提案の文言（悪姿勢時のみ） ／ 提案不要なら None。
# timestamp  … そのフレームの時刻[ms]（記録・整合の参考）。
WorkerResult = namedtuple("WorkerResult", ["frame", "landmarks", "score", "advice", "timestamp"])

# キャリブレーション測定の結果（ワーカー→メイン）。保存(DB)はメインが行う。
CalibrationOutcome = namedtuple(
    "CalibrationOutcome", ["success", "user_id", "representative", "num_samples", "message"]
)

# 連続でフレーム取得に失敗したらカメラ切断とみなして安全に抜ける。
_MAX_READ_FAILURES = 100


# ── 既定のカメラ／推定器の作り方（ワーカーが所有して開く） ──
def _default_camera_factory():
    from capture import Camera
    return Camera()


def _default_estimator_factory():
    from pose import PoseEstimator
    return PoseEstimator()


class PoseWorker(threading.Thread):
    """撮影〜採点のループを回すワーカースレッド。

    使い方（メイン側の概略）:
        q = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        worker = PoseWorker(result_queue=q, baseline=baseline_from_db)
        worker.start()
        ...（メインは after で q をドレイン）...
        worker.stop(); worker.join()
    """

    def __init__(
        self,
        result_queue=None,
        stop_event=None,
        baseline=None,
        camera_factory=None,
        estimator_factory=None,
        calib_result_queue=None,
    ):
        super().__init__(daemon=True)

        # 結果キュー（映像のたまり防止に小さめ。あふれたら古いものを捨てる）。
        self.result_queue = result_queue or queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        # 停止フラグ（メインが立て、ワーカーがループ先頭で確認）。
        self.stop_event = stop_event or threading.Event()
        # キャリブレーション結果の戻し口。
        self.calib_result_queue = calib_result_queue or queue.Queue()

        # カメラ・推定器はワーカーが所有する。生成方法だけ受け取る（テストで差し替え可）。
        self._camera_factory = camera_factory or _default_camera_factory
        self._estimator_factory = estimator_factory or _default_estimator_factory
        self._camera = None
        self._estimator = None

        # ── スレッドをまたぐ状態は必ずロックで守る ──
        # 基準値（メモリ上）。再キャリブで安全に差し替えられるように。
        self._baseline = baseline
        self._baseline_lock = threading.Lock()
        # キャリブレーション要求（user_id）。
        self._calib_request = None
        self._calib_lock = threading.Lock()

        self._read_failures = 0
        # タイムスタンプは採点・キャリブで共有する単調増加カウンタ（推定器が1つのため）。
        self._timestamp_ms = 0

    # ── 基準値の受け渡し（メイン↔ワーカー、ロックで安全に） ──
    def set_baseline(self, baseline):
        """メインが新しい基準値に差し替える（再キャリブ時）。"""
        with self._baseline_lock:
            self._baseline = baseline

    def get_baseline(self):
        with self._baseline_lock:
            return self._baseline

    # ── キャリブレーション要求（カメラの持ち主は1人＝ワーカー、メモ §7） ──
    def request_calibration(self, user_id):
        """メインがキャリブレーションを要求する。ワーカーが採点を中断して測定に入る。"""
        with self._calib_lock:
            self._calib_request = user_id

    def _take_calib_request(self):
        """要求を取り出して消す（同じ要求で二重に測定しない）。"""
        with self._calib_lock:
            uid = self._calib_request
            self._calib_request = None
            return uid

    # ── 停止 ──
    def stop(self):
        """メインが停止フラグを立てる。ワーカーは次のループ先頭で抜ける。"""
        self.stop_event.set()

    def _next_timestamp(self):
        """次の単調増加タイムスタンプ[ms]を返す（採点・キャリブで共有）。"""
        self._timestamp_ms += 33  # ≒30fps相当
        return self._timestamp_ms

    # ── ループ本体 ──
    def run(self):
        # core の各処理（薄い指揮者として順番に呼ぶ）。
        from features import extract_features
        from smoothing import FeatureSmoother
        from scoring import score as score_frame
        from advice import Advisor
        import calibration

        # 改善提案はヒステリシス付きで点滅を抑える（状態を持つのでループ内で1つ保持）。
        advisor = Advisor()
        # 体幹の傾き・上体屈曲角のブレを短時間移動平均でならす（状態を持つのでループ外で1つ）。
        smoother = FeatureSmoother()

        # カメラ・推定器を開く（重い初期化はループ外で1回）。
        self._camera = self._camera_factory()
        self._estimator = self._estimator_factory()

        try:
            while not self.stop_event.is_set():
                # 1. キャリブレーション要求があれば、採点を中断して測定モードへ。
                uid = self._take_calib_request()
                if uid is not None:
                    self._run_calibration(uid, calibration)
                    continue

                # 2. 1フレーム読む。
                ok, frame = self._camera.read_frame()
                if not ok:
                    # 取得失敗。連続が続けばカメラ切断とみなして安全に抜ける。
                    self._read_failures += 1
                    if self._read_failures >= _MAX_READ_FAILURES:
                        break
                    time.sleep(0.01)  # 失敗時のCPU空回りを防ぐ
                    continue
                self._read_failures = 0

                # 3. 推定（動画モード用に単調増加タイムスタンプを渡す。採点・キャリブ共有）。
                timestamp_ms = self._next_timestamp()
                result = self._estimator.estimate(frame, timestamp_ms)

                # 4〜6. 検出なしは採点しない。検出ありなら特徴量→平滑化→採点→改善提案。
                if result.detected:
                    feats = extract_features(result.landmarks)
                    # 体幹の傾き等を直近数フレームの平均でならしてから採点する
                    # （1フレームの震えでスコアや提案がチラつくのを防ぐ）。
                    feats = smoother.smooth(feats)
                    # 基準値はメモリ上のものを使う（毎フレームDBを読まない）。
                    baseline = self.get_baseline()
                    sc = score_frame(feats, baseline)
                    # 悪姿勢なら改善提案（ヒステリシスで点滅を抑制）。
                    advice = advisor.update(feats, baseline, sc)
                    landmarks = result.landmarks
                else:
                    sc = None  # 採点不可（UNSCORABLE）
                    advice = None
                    advisor.reset()   # 離席したら判定をやり直す
                    smoother.reset()  # 離席したら移動平均もリセット（古い値を引きずらない）
                    landmarks = None

                # 7. 結果をキューに置く（映像は最新優先＝あふれたら古いものを捨てる）。
                self._put_latest(WorkerResult(frame, landmarks, sc, advice, timestamp_ms))
        finally:
            # 停止時・例外時いずれもカメラと推定器を必ず解放する（メモ §6 終了の順番）。
            if self._estimator is not None:
                self._estimator.close()
            if self._camera is not None:
                self._camera.release()

    # ── キャリブレーション測定（ワーカーのカメラを使う・保存はメイン） ──
    def _run_calibration(self, user_id, calibration):
        """採点を中断して数秒測定し、安定化した代表値をメインへ返す。

        DB保存・メモリ基準値の更新はメインが行う（DBに触るのはメインだけ）。
        """
        # タイムスタンプは採点ループと同じ単調増加カウンタを使う（推定器を共有するため、
        # 0 にリセットすると VIDEO モードが「時刻が逆戻り」でエラーになる）。
        good_samples, total = calibration.collect_samples(
            self._camera, self._estimator, timestamp_fn=self._next_timestamp)
        if len(good_samples) < config.MIN_GOOD_SAMPLES:
            self.calib_result_queue.put(CalibrationOutcome(
                success=False, user_id=user_id, representative=None,
                num_samples=len(good_samples),
                message=(f"良いフレームが不足しています"
                         f"（{len(good_samples)}/{config.MIN_GOOD_SAMPLES}、総{total}）。"
                         "正しい姿勢で動かずにもう一度お試しください。"),
            ))
            return

        representative = calibration.stabilize(good_samples)
        self.calib_result_queue.put(CalibrationOutcome(
            success=True, user_id=user_id, representative=representative,
            num_samples=len(good_samples), message="キャリブレーション完了。",
        ))

    # ── 結果をキューへ（映像は最新だけ・あふれたら古いものを捨てる、メモ §4/§14） ──
    def _put_latest(self, item):
        try:
            self.result_queue.put_nowait(item)
        except queue.Full:
            # 満杯＝メインの取り出しが追いついていない。古い1件を捨てて最新を入れる。
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(item)
            except queue.Full:
                pass


# ── 単体での動作確認（にせカメラ・にせ推定器。本物のカメラ/モデル不要） ──
def _self_test():
    from types import SimpleNamespace
    from features import extract_features

    # pose.py が返す形（x,y,z,visibility）の偽ランドマーク（まっすぐ座った想定）。
    LM = namedtuple("LM", ["x", "y", "z", "visibility"])

    def upright():
        pts = [LM(0.5, 0.5, 0.0, 1.0) for _ in range(config.NUM_LANDMARKS)]
        pts[config.NOSE] = LM(0.5, 0.30, 0.0, 1.0)
        pts[config.LEFT_SHOULDER] = LM(0.40, 0.50, 0.0, 1.0)
        pts[config.RIGHT_SHOULDER] = LM(0.60, 0.50, 0.0, 1.0)
        return pts

    class FakeCamera:
        def __init__(self):
            self.released = False
        def read_frame(self):
            return True, "FRAME"
        def release(self):
            self.released = True

    class FakeEstimator:
        def __init__(self):
            self.closed = False
        def estimate(self, frame, ts):
            return SimpleNamespace(detected=True, landmarks=upright())
        def close(self):
            self.closed = True

    # まっすぐ姿勢から基準値を作る（同じ姿勢なので採点は満点近くになるはず）。
    feats = extract_features(upright())
    baseline = {n: feats[n].value for n in config.USE_FEATURES}

    cam, est = FakeCamera(), FakeEstimator()
    result_q = queue.Queue(maxsize=2)
    worker = PoseWorker(
        result_queue=result_q,
        baseline=baseline,
        camera_factory=lambda: cam,
        estimator_factory=lambda: est,
    )

    # キャリブ測定を短くしてテストを速くする。
    orig_secs, orig_min = config.CALIBRATION_SECONDS, config.MIN_GOOD_SAMPLES
    config.CALIBRATION_SECONDS, config.MIN_GOOD_SAMPLES = 0.3, 5

    results = []
    try:
        worker.start()

        # メインの取り出し（ドレイン）を模す: 点数を集め、最後のフレームを保持。
        scores, last_frame = [], None
        deadline = time.time() + 1.0
        while time.time() < deadline and len(scores) < 5:
            try:
                item = result_q.get(timeout=0.1)
                scores.append(item.score)
                last_frame = item.frame
            except queue.Empty:
                pass
        results.append(("ワーカーが点数をキューに流す", len(scores) > 0))
        results.append(("最新フレームが取り出せる", last_frame == "FRAME"))
        results.append(("一致姿勢で高得点(>90)", bool(scores) and all(s is not None and s > 90 for s in scores)))

        # キャリブレーションをワーカー経由で要求 → カメラを奪い合わずに完了。
        worker.request_calibration(user_id=1)
        outcome = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            # ワーカーがブロックしないよう、結果キューは引き続き空にしておく。
            try:
                result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                outcome = worker.calib_result_queue.get(timeout=0.1)
                break
            except queue.Empty:
                pass
        results.append(("キャリブがワーカー経由で完了", outcome is not None and outcome.success))

        # 停止フラグでワーカーが素早く止まり、カメラが解放される。
        t0 = time.time()
        worker.stop()
        worker.join(timeout=2.0)
        stopped_quick = (not worker.is_alive()) and (time.time() - t0 < 2.0)
        results.append(("停止フラグでワーカーが止まる", stopped_quick))
        results.append(("カメラが解放される", cam.released))
        results.append(("推定器が解放される", est.closed))
    finally:
        config.CALIBRATION_SECONDS, config.MIN_GOOD_SAMPLES = orig_secs, orig_min

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("pose_worker 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
