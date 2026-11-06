"""analysis/recompute_features.py

保存した生の座標（landmarks テーブル）から、今の3特徴量と当初の候補特徴量を同じ条件で計算し直す。
実験Cで特徴量の組み合わせを比べるための表を作る。比較そのものはデータが取れてから別に行う。

使い方:
  python analysis/recompute_features.py                 （全利用者）
  python analysis/recompute_features.py --user S1 --start "2026-11-09 10:00:00"

出力（DB には書き込まない）:
  data/analysis/frames.csv     … 1フレーム1行
  data/analysis/baselines.csv  … キャリブレーション1回につき1行（各特徴量の中央値）

候補特徴量の理論:
  耳・肩・腰の3点は、位置と大きさを除くと「首の向き・体幹の向き・長さの比 L（耳肩÷胴）」の
  自由度3しか持たない。head_forward = L·cos(neck)、head_drop = L·sin(neck) なので、
  新しい情報は L の分だけ。nose_shoulder_hip は別の点（鼻）を使うので頭のうなずきが入る。
"""

import argparse
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "core")]

import config  # noqa: E402
from db import database  # noqa: E402
from features import _EPS, _Point, _kyphosis_angle, _select_side, _visible, extract_features  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "analysis")
FEATURES = ["neck_angle", "trunk_angle", "kyphosis_angle"]
CANDIDATES = ["head_forward", "head_drop", "nose_shoulder_hip"]
# 基準値テーブルとの差の許容[度]。座標を小数6桁に丸めた影響（0.001°未満）より十分大きい値。
MAX_BASELINE_DIFF = 0.01


def candidate_features(points, min_visibility=None):
    """縦横比を補正済みの33点から候補特徴量を計算する。使えない値は NaN。

    長さは胴の長さ（肩〜腰）で割る。体格やカメラとの距離によらない値にするため。
    """
    min_visibility = config.MIN_VISIBILITY if min_visibility is None else min_visibility
    side = _select_side(points)
    values = dict.fromkeys(CANDIDATES, math.nan)

    angle = _kyphosis_angle(points, side._replace(ear=config.NOSE), min_visibility)
    if angle.valid:
        values["nose_shoulder_hip"] = angle.value

    needed = (side.ear, side.shoulder, side.hip)
    if not all(_visible(points, i, min_visibility) for i in needed):
        return values
    ear, shoulder, hip = points[side.ear], points[side.shoulder], points[side.hip]
    torso = math.hypot(shoulder.x - hip.x, shoulder.y - hip.y)
    if torso < _EPS:
        return values

    # 肩より上にある耳の高さ。頭が沈むと小さくなる。
    values["head_drop"] = (shoulder.y - ear.y) / torso
    # 前向きを正にする。左右どちらを向いているかは 耳→鼻 の向きで決める。
    if _visible(points, config.NOSE, min_visibility):
        forward = math.copysign(1.0, points[config.NOSE].x - ear.x)
        values["head_forward"] = forward * (ear.x - shoulder.x) / torso
    return values


def frame_row(frame):
    """保存フレーム1件を、全特徴量の入った1行にする。"""
    aspect = frame["width"] / frame["height"]
    raw = [_Point(*p) for p in frame["points"]]
    # 今の3特徴量はアプリと同じ extract_features を通す（同じものさしを保証する）。
    feats = extract_features(raw, FEATURES, aspect=aspect)
    corrected = [_Point(p.x * aspect, p.y, p.z, p.visibility) for p in raw]
    side = _select_side(corrected)
    row = {key: frame[key] for key in ("source", "baseline_id", "score_id", "captured_at")}
    row["landmark_id"] = frame["id"]
    row["side"] = "left" if side.ear == config.LEFT_EAR else "right"
    for name in FEATURES:
        row[name] = feats[name].value if feats[name].valid else math.nan
    row.update(candidate_features(corrected))
    return row


def summarize_baselines(frames, baseline_rows):
    """キャリブレーションごとに、良いフレームの中央値を求めて基準値テーブルと照合する。

    良いフレーム = 今の3特徴量がすべて有効（core/calibration.py と同じ条件）。
    """
    calib = frames[frames["source"] == "calibration"]
    if calib.empty:
        return pd.DataFrame()
    good = calib.dropna(subset=FEATURES)
    summary = good.groupby("baseline_id")[FEATURES + CANDIDATES].median()
    summary["n_frames"] = calib.groupby("baseline_id").size()
    summary["n_good"] = good.groupby("baseline_id").size()

    table = baseline_rows.set_index("id")
    diffs = (summary[FEATURES] - table[FEATURES].reindex(summary.index)).abs()
    summary["max_diff_from_table"] = diffs.max(axis=1)
    summary["user"] = table["user"].reindex(summary.index)
    summary["name"] = table["name"].reindex(summary.index)
    return summary.reset_index()


def recompute(user=None, start=None, end=None):
    """DB を読んで (frames, baselines) の DataFrame を返す。SELECT しか実行しない。"""
    conn = database._connect()
    try:
        users = pd.read_sql_query("SELECT id, name FROM users", conn)
        baseline_rows = pd.read_sql_query(
            "SELECT b.id, b.name, u.name AS user, " + ", ".join(f"b.{n}" for n in FEATURES)
            + " FROM baseline b JOIN users u ON b.user_id = u.id", conn)
    finally:
        conn.close()

    if user is not None:
        users = users[users["name"] == user]
        if users.empty:
            raise SystemExit(f"利用者が見つかりません: {user}")

    rows = []
    for user_id, name in users.itertuples(index=False):
        for frame in database.get_landmark_frames(user_id, start=start, end=end):
            rows.append({"user": name, **frame_row(frame)})

    columns = ["user", "landmark_id", "source", "baseline_id", "score_id", "captured_at", "side",
               *FEATURES, *CANDIDATES]
    frames = pd.DataFrame(rows, columns=columns)
    return frames, summarize_baselines(frames, baseline_rows)


# ── 単体での動作確認（一時 DB を使う。カメラ不要） ──
def _self_test():
    import shutil
    import tempfile

    def pose(ear, shoulder, hip, nose, vis=1.0, width=1280, height=720):
        # 右側だけを写す。座標は画素で受け取り、MediaPipe と同じ形に正規化する。
        def norm(p):
            return _Point(p[0] / width, p[1] / height, 0.0, vis)
        points = [_Point(0.5, 0.5, 0.0, 0.05)] * config.NUM_LANDMARKS
        points[config.RIGHT_EAR] = norm(ear)
        points[config.RIGHT_SHOULDER] = norm(shoulder)
        points[config.RIGHT_HIP] = norm(hip)
        points[config.NOSE] = norm(nose)
        return points

    def row_of(points, width=1280, height=720):
        return frame_row({"id": 1, "source": "scoring", "baseline_id": None, "score_id": 1,
                          "captured_at": "t", "width": width, "height": height,
                          "points": [list(p) for p in points]})

    # 耳は肩の前に 60px、上に 80px。胴の長さは 200px。鼻は耳よりさらに前。
    pixels = {"ear": (660, 200), "shoulder": (600, 280), "hip": (600, 480), "nose": (700, 210)}
    base = pose(**pixels)
    row = row_of(base)

    # ① 手計算と一致する。
    assert abs(row["head_forward"] - 60 / 200) < 1e-9
    assert abs(row["head_drop"] - 80 / 200) < 1e-9
    print("PASS 候補特徴量が手計算と一致")

    # ② 理論: head_forward = L·cos(neck)、head_drop = L·sin(neck)（L = 耳肩÷胴）。
    length = math.hypot(60, 80) / 200
    neck = math.radians(row["neck_angle"])
    assert abs(row["head_forward"] - length * math.cos(neck)) < 1e-9
    assert abs(row["head_drop"] - length * math.sin(neck)) < 1e-9
    print("PASS 長さの比と首の向きで表せることを確認")

    # ③ 鼻だけ動かすと nose_shoulder_hip は変わるが、kyphosis_angle は変わらない（独立な情報）。
    nodded = row_of(pose(**{**pixels, "nose": (700, 260)}))
    assert abs(nodded["kyphosis_angle"] - row["kyphosis_angle"]) < 1e-9
    assert abs(nodded["nose_shoulder_hip"] - row["nose_shoulder_hip"]) > 1.0
    print("PASS nose_shoulder_hip は kyphosis_angle と独立に動く")

    # ④ 反対を向いても前方が正のまま。
    flipped = row_of(pose(ear=(540, 200), shoulder=(600, 280), hip=(600, 480), nose=(500, 210)))
    assert abs(flipped["head_forward"] - row["head_forward"]) < 1e-9
    print("PASS 向きによらず前方が正")

    # ⑤ 同じ形のポーズなら、解像度と縦横比が違っても値が変わらない（補正が効いている）。
    half = {key: (x / 2, y / 2) for key, (x, y) in pixels.items()}
    small = row_of(pose(**half, width=800, height=600), width=800, height=600)
    assert abs(small["head_forward"] - row["head_forward"]) < 1e-9
    assert abs(small["neck_angle"] - row["neck_angle"]) < 1e-9
    print("PASS 解像度・縦横比によらず同じ値")

    # ⑥ 点が見えないと NaN になる。
    unseen = pose(**pixels, vis=0.0)
    hidden = row_of(unseen)
    assert all(math.isnan(hidden[name]) for name in CANDIDATES)
    print("PASS 見えない点の値は NaN")

    # ⑦ 一時 DB で通し確認: 中央値が基準値テーブルと一致し、DB に行を作らない。
    tmp_dir = tempfile.mkdtemp(prefix="再計算_")
    original_path, original_features = config.DB_PATH, config.USE_FEATURES
    config.DB_PATH = os.path.join(tmp_dir, "test.db")
    config.USE_FEATURES = FEATURES
    try:
        database.ensure_tables()
        uid = database.get_or_create_user("S1")
        calib_frames = [{"captured_at": "2026-11-09 10:00:00", "points": points,
                         "width": 1280, "height": 720} for points in (base, base, unseen)]
        database.save_baseline(uid, {name: row[name] for name in FEATURES}, "2026-11-09 10:00:00",
                               name="S1_A1", frames=calib_frames)
        database.save_score(uid, "2026-11-09 10:00:05", 90.0, landmarks=base, image_size=(1280, 720))

        frames, baselines = recompute()
        assert len(frames) == 4
        assert set(frames["source"]) == {"calibration", "scoring"}
        one = baselines.iloc[0]
        assert (one["n_frames"], one["n_good"]) == (3, 2)
        assert one["max_diff_from_table"] < MAX_BASELINE_DIFF
        assert abs(one["head_forward"] - 60 / 200) < 1e-5
        assert one["name"] == "S1_A1" and one["user"] == "S1"
        print("PASS 中央値が基準値テーブルと一致（見えないフレームは除外）")

        try:
            recompute(user="いない人")
        except SystemExit:
            pass
        else:
            raise AssertionError("存在しない利用者でエラーにならない")
        conn = database._connect()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 1
        print("PASS 存在しない利用者はエラーで止まり、DB に行を作らない")
    finally:
        config.DB_PATH, config.USE_FEATURES = original_path, original_features
        shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="保存した座標から特徴量を計算し直す")
    parser.add_argument("--user", help="利用者名（省略時は全員）")
    parser.add_argument("--start", help='開始時刻（例: "2026-11-09 10:00:00"）')
    parser.add_argument("--end", help="終了時刻")
    parser.add_argument("--self-test", action="store_true", help="一時 DB での動作確認")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not os.path.exists(config.DB_PATH):
        raise SystemExit(f"DB がありません: {config.DB_PATH}")

    frames, baselines = recompute(args.user, args.start, args.end)
    if frames.empty:
        raise SystemExit("保存された座標がありません。記録モードで撮影するか、キャリブレーションしてください。")

    os.makedirs(OUT_DIR, exist_ok=True)
    frames.to_csv(os.path.join(OUT_DIR, "frames.csv"), index=False, encoding="utf-8")
    baselines.to_csv(os.path.join(OUT_DIR, "baselines.csv"), index=False, encoding="utf-8")
    print(f"frames.csv: {len(frames)} 行 / baselines.csv: {len(baselines)} 行 → {OUT_DIR}")
    if not baselines.empty:
        worst = baselines["max_diff_from_table"].max()
        status = "OK" if worst < MAX_BASELINE_DIFF else "要確認（アプリと計算が食い違っています）"
        print(f"基準値テーブルとの差の最大: {worst:.6f}° … {status}")


if __name__ == "__main__":
    main()
