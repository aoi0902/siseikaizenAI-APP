"""core/calibration.py

利用者の正しい姿勢を測り、ブレない基準値を作って保存する「測定の段取り役」。
（設計メモ: calibration_py_設計メモ.md に対応）

流れ: 数秒キープ → 複数フレーム収集 → 低信頼除外 → 安定化(中央値) → 特徴量 → 保存 → 読み戻し確認

責任:
  - 数秒間の測定を回し、良いフレームを集める
  - 検出なし・低 visibility のフレームを除外する
  - 集めた特徴量を安定化して 1 つの基準値にする
  - 基準値を database.py に渡して保存させ、読み戻して確認する

担わないこと（委譲先）:
  - 映像取得 → capture.py / 骨格推定 → pose.py
  - 33点→特徴量の計算 → features.py（必ずこの共通の核を通す）
  - SQL の実行そのもの → db/database.py
  - 採点 → scoring.py

設計の核: calibration.py は「いつ・何を・どの順で呼ぶか」だけを持つ薄い指揮者。
計算は features.py、保存は database.py に委ねる。これにより採点と同じものさしで
基準値が作られ、「基準と現在を比べる」が成り立つ。
"""

import time
import statistics
from collections import namedtuple
from datetime import datetime

# core/ から直接実行されても、パッケージ import でも config / features を読めるように。
try:
    import config
    from features import extract_features
except ModuleNotFoundError:
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(_here))  # プロジェクトルート（config 用）
    sys.path.append(_here)                    # core ディレクトリ（兄弟モジュール用）
    import config
    from features import extract_features


# キャリブレーションの結果。success=False のとき baseline は None。
CalibrationResult = namedtuple("CalibrationResult", ["success", "baseline", "message"])


# ── 内部: サンプル収集 ──
def collect_samples(camera, estimator, seconds=None, use_features=None,
                    min_visibility=None, timestamp_fn=None, on_frame=None):
    """数秒間、良いフレームの特徴量だけを集める。

    良いフレーム = detected が True かつ、使う特徴量がすべて valid。
    （検出なし・低 visibility・ゼロ割り等は features 側で valid=False になり除外される）

    timestamp_fn … 次のタイムスタンプ[ms]を返す関数（省略時は内部で0から+33）。
      採点ループと推定器を共有する場合は、共通の単調増加カウンタを渡すこと。
      渡さないと推定器のタイムスタンプが逆戻りし VIDEO モードがエラーになる。
    on_frame … 毎フレーム呼ぶ関数 on_frame(frame, landmarks, sample, timestamp_ms)。
      測定中の映像表示と確認画面用の画像に使う。検出なしは landmarks=None、
      良いフレームでなければ sample=None。

    戻り値: (good_samples, total_frames, raw_frames)
      good_samples … [{特徴名: 値, ...}, ...]（採用したフレームぶん）
      total_frames … 走らせた総フレーム数（除外率の確認用）
      raw_frames   … 検出できた全フレームの生の座標（除外したフレームも含む）。
                     [{"captured_at", "points", "width", "height"}, ...]
                     除外も含めるのは、式や閾値を変えたとき「どれが良いフレームか」から再計算するため
    """
    seconds = config.CALIBRATION_SECONDS if seconds is None else seconds
    use_features = config.USE_FEATURES if use_features is None else use_features
    min_visibility = config.MIN_VISIBILITY if min_visibility is None else min_visibility

    good_samples = []
    raw_frames = []
    total_frames = 0
    timestamp_ms = 0
    start = time.perf_counter()

    while time.perf_counter() - start < seconds:
        ok, frame = camera.read_frame()
        if not ok:
            continue
        total_frames += 1

        # 動画モード用に単調増加するタイムスタンプを渡す。
        if timestamp_fn is not None:
            ts = timestamp_fn()          # 共有カウンタ（採点ループと連続）
        else:
            timestamp_ms += 33           # 単体利用時は0から（≒30fps相当）
            ts = timestamp_ms
        result = estimator.estimate(frame, ts)
        if not result.detected:
            if on_frame is not None:
                on_frame(frame, None, None, ts)
            continue  # 検出なしは捨てる

        height, width = frame.shape[:2]
        raw_frames.append({"captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "points": result.landmarks, "width": width, "height": height})

        # 採点（pose_worker）と同じく実フレームの縦横比で補正する（基準値と現在値の物差しをそろえる）。
        feats = extract_features(result.landmarks, use_features, min_visibility,
                                 aspect=frame.shape[1] / frame.shape[0])
        # 使う特徴がすべて信用できるフレームだけ採用する。
        sample = None
        if all(feats[name].valid for name in use_features):
            sample = {name: feats[name].value for name in use_features}
            good_samples.append(sample)
        if on_frame is not None:
            on_frame(frame, result.landmarks, sample, ts)

    return good_samples, total_frames, raw_frames


# ── 内部: 安定化（純粋関数・カメラ/DB不要） ──
def stabilize(samples, method=None, use_features=None):
    """集めた特徴量リストを 1 つの代表値にまとめる。

    なぜ複数フレーム: MediaPipe の出力は小刻みに揺れる。1枚で決めると基準がいびつ。
    なぜ中央値    : 平均は一瞬の外れ値に引っ張られる。中央値は外れ値に強い。

    戻り値: {特徴名: 代表値} ／ サンプルが空なら None。
    """
    method = config.STABILIZE_METHOD if method is None else method
    use_features = config.USE_FEATURES if use_features is None else use_features

    if not samples:
        return None

    representative = {}
    for name in use_features:
        values = [s[name] for s in samples]
        if method == "mean":
            representative[name] = statistics.mean(values)
        else:  # 既定は median（外れ値に強い）
            representative[name] = statistics.median(values)
    return representative


# ── 内部: 確認画面の材料（純粋関数・カメラ/DB不要） ──
def pick_review_frame(candidates, representative, tolerance=None):
    """候補 [{"sample": {特徴名: 値}, ...}, ...] から基準値に最も近い 1 件を返す（空なら None）。

    保存するのは中央値なので、それに一番近い画像なら「見ている姿勢＝保存する値」になる。
    近さは採点と同じ物差し（ずれ÷許容幅の合計）で測る。
    """
    tolerance = config.TOLERANCE if tolerance is None else tolerance

    def distance(candidate):
        return sum(abs(candidate["sample"][name] - representative[name]) / tolerance[name]
                   for name in representative)

    return min(candidates, key=distance, default=None)


def check_baseline(baseline, ranges=None):
    """基準値を目安 (下限, 上限) と比べ、外れた特徴ごとの警告文のリストを返す。

    採点は基準値との差しか見ないので、悪い姿勢で測るとその姿勢が 100 点になる。
    ここで絶対値を見て、測り直しを促す（保存は止めない。目安は個人差までは見られないため）。
    """
    ranges = config.CALIBRATION_RANGES if ranges is None else ranges

    warnings = []
    for name, (low, high) in ranges.items():
        value = baseline.get(name)
        if value is None:
            continue
        if low is not None and value < low:
            direction, limit = "low", f"{low:g}°以上"
        elif high is not None and value > high:
            direction, limit = "high", f"{high:g}°以下"
        else:
            continue
        advice = config.ADVICE_MESSAGES.get((name, direction), config.ADVICE_DEFAULT_MESSAGE)
        warnings.append(f"{config.FEATURE_LABELS[name]} {value:.1f}°（目安 {limit}）: {advice}")
    return warnings


# ── 公開: キャリブレーション実行（段取り役） ──
def run_calibration(user_id, camera=None, estimator=None):
    """測定〜保存〜読み戻し確認までを一括で行う。

    引数:
      user_id   … 誰の基準値か
      camera    … 省略時はここで開いて使い終わりに閉じる
      estimator … 省略時はここで作って使い終わりに閉じる

    戻り値: CalibrationResult(success, baseline, message)
    """
    # 採取・推定・保存はそれぞれの担当に委譲（遅延 import）。
    from capture import Camera
    from pose import PoseEstimator
    from db import database

    # 呼び出し側から渡されなければ自前で用意し、自前のものだけ後で解放する。
    own_camera = camera is None
    own_estimator = estimator is None
    if own_camera:
        camera = Camera()
    if own_estimator:
        estimator = PoseEstimator()

    try:
        # 1. 良いフレームを集める
        good_samples, total, raw_frames = collect_samples(camera, estimator)

        # 2. 最低サンプル数チェック（中途半端な基準値を作らない）
        if len(good_samples) < config.MIN_GOOD_SAMPLES:
            return CalibrationResult(
                success=False,
                baseline=None,
                message=(
                    f"良いフレームが不足しています（{len(good_samples)}/"
                    f"{config.MIN_GOOD_SAMPLES}、総{total}フレーム）。"
                    "正しい姿勢で動かずにもう一度お試しください。"
                ),
            )

        # 3. 安定化（中央値で代表値を 1 つ作る）
        representative = stabilize(good_samples)

        # 4. 保存（SQL は書かない。database.py に委ねる）
        measured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        database.save_baseline(user_id, representative, measured_at, frames=raw_frames)

        # 5. 読み戻し確認（保存しっぱなしにせず妥当性を確かめる）
        saved = database.get_latest_baseline(user_id)
        if saved is None:
            return CalibrationResult(
                success=False,
                baseline=None,
                message="保存後に基準値を読み戻せませんでした。",
            )

        return CalibrationResult(
            success=True,
            baseline=saved,
            message=f"キャリブレーション完了（採用 {len(good_samples)} フレーム）。",
        )
    finally:
        # 自分で開いたものだけ確実に解放する。
        if own_estimator:
            estimator.close()
        if own_camera:
            camera.release()


# ── 単体での動作確認（安定化のみ。カメラ/DB不要） ──
def _self_test():
    # 外れ値（一瞬の検出ミス想定）を混ぜた特徴量リスト。
    samples = [
        {"neck_angle": 82.0, "trunk_angle": 4.0},
        {"neck_angle": 81.0, "trunk_angle": 5.0},
        {"neck_angle": 83.0, "trunk_angle": 3.5},
        {"neck_angle": 80.5, "trunk_angle": 4.5},
        {"neck_angle": 40.0, "trunk_angle": 35.0},  # ← 外れ値
    ]

    # モード設定に依らずこのデモ用の特徴名で安定化を確認する（明示指定）。
    uf = ["neck_angle", "trunk_angle"]
    med = stabilize(samples, method="median", use_features=uf)
    avg = stabilize(samples, method="mean", use_features=uf)
    print(f"中央値: neck_angle={med['neck_angle']:.2f}  trunk_angle={med['trunk_angle']:.2f}")
    print(f"平均値: neck_angle={avg['neck_angle']:.2f}  trunk_angle={avg['trunk_angle']:.2f}")
    print("→ 中央値は外れ値(80,40)に引っ張られず ~10/~0.5 付近、平均は引っ張られて大きいはず")

    # サンプル空 → None
    print(f"空サンプル: {stabilize([])}  ← None のはず")

    # 確認画面の画像: 基準値に最も近い候補を選ぶ（許容幅で揃えた距離）。
    rep = {"neck_angle": 70.0, "trunk_angle": 5.0}
    tol = {"neck_angle": 10.0, "trunk_angle": 1.0}
    near, far = {"sample": {"neck_angle": 75.0, "trunk_angle": 5.0}}, \
                {"sample": {"neck_angle": 70.0, "trunk_angle": 7.0}}
    assert pick_review_frame([far, near], rep, tol) is near  # 5/10=0.5 < 2/1=2
    assert pick_review_frame([], rep, tol) is None
    print("PASS 基準値に最も近い候補を選ぶ")

    # 基準値の目安: 外れた側だけ警告する。境界ちょうどと欠けた値は警告しない。
    ranges = {"neck_angle": (55.0, None), "trunk_angle": (None, 15.0)}
    assert check_baseline({"neck_angle": 55.0, "trunk_angle": 15.0}, ranges) == []
    assert check_baseline({"neck_angle": None, "trunk_angle": 5.0}, ranges) == []
    low = check_baseline({"neck_angle": 50.0, "trunk_angle": 5.0}, ranges)
    assert len(low) == 1 and "首の前傾" in low[0] and "55°以上" in low[0], low
    high = check_baseline({"neck_angle": 70.0, "trunk_angle": 20.0}, ranges)
    assert len(high) == 1 and "体幹の傾き" in high[0] and "15°以下" in high[0], high
    print("PASS 目安を外れた特徴だけ警告する")

    # 後ろにもたれた猫背: 首と体幹は目安内でも、上体屈曲角で気づける（設定値の確認）。
    if config.POSTURE_MODE == "side":
        slumped = {"neck_angle": 70.0, "trunk_angle": 10.0, "kyphosis_angle": 148.0}
        assert len(check_baseline(slumped)) == 1
        print("PASS 上体屈曲角だけが外れる姿勢も警告する")


if __name__ == "__main__":
    _self_test()
