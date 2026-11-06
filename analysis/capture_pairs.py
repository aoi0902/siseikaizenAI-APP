"""analysis/capture_pairs.py

実験B用: 「その瞬間の画像」と「その瞬間の特徴量」を1組にして保存する小さなツール。
画像を Kinovea で手動計測した CVA と、システムの neck_angle を比べるために使う。

使い方（アプリは閉じておく。カメラを取り合うため）:
  python analysis/capture_pairs.py --subject S1
  キー 1=良い姿勢 / 2=猫背 / 3=前傾 / 4=首だけ前突 で保存、q で終了

設計の要点:
  - 計算はアプリと同じ Camera / PoseEstimator / extract_features を通す（同じものさし）
  - 値は平滑化しない。画像は1フレームなので、比べる値も同じ1フレームの値にする
  - 骨格は表示用のコピーにだけ描く。保存画像に線があると計測者が引きずられる
  - 計測者には images/ だけを渡す（pairs.csv の値が見えると、それに寄せて測ってしまう）
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "core")]

import config  # noqa: E402
from db.database import _points_json  # noqa: E402
from features import extract_features  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "experiment_b")
POSTURES = {"1": "good", "2": "slouch", "3": "lean", "4": "neck"}
FEATURES = ["neck_angle", "trunk_angle", "kyphosis_angle"]
COLUMNS = ["image", "subject_id", "posture", "take", "captured_at", "width", "height",
           *FEATURES, "points"]
EXPECTED_SIZE = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
PREVIEW_SIZE = config.LIVE_DISPLAY_SIZE
WINDOW = "capture_pairs"  # imshow のタイトルは日本語が化けるので ASCII


def _next_take(images_dir, subject, posture):
    """この被験者・姿勢の次の撮影番号（撮り直しで前のファイルを消さないため）。"""
    pattern = re.compile(rf"^{re.escape(subject)}_{posture}_(\d+)\.png$")
    takes = [int(m.group(1)) for m in map(pattern.match, os.listdir(images_dir)) if m]
    return max(takes, default=0) + 1


def save_pair(out_dir, subject, posture, frame, landmarks, feats):
    """画像と特徴量を1組保存し、保存した画像のファイル名を返す。"""
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    take = _next_take(images_dir, subject, posture)
    name = f"{subject}_{posture}_{take}.png"

    # cv2.imwrite は Windows の日本語パスに保存できず、例外も出さず失敗する。
    # メモリ上で PNG に変換してから Python 側で書く。
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"画像を PNG に変換できませんでした: {name}")
    with open(os.path.join(images_dir, name), "wb") as f:
        f.write(encoded.tobytes())

    # 画像が書けてから行を足す（行だけあって画像が無い状態を作らない）。
    csv_path = os.path.join(out_dir, "pairs.csv")
    is_new = not os.path.exists(csv_path)
    height, width = frame.shape[:2]
    row = {"image": name, "subject_id": subject, "posture": posture, "take": take,
           "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "width": width, "height": height, "points": _points_json(landmarks)}
    for feature in FEATURES:
        row[feature] = feats[feature].value if feats[feature].valid else ""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return name


def compute_feats(frame, result):
    """アプリと同じ式で特徴量を計算する（平滑化はしない）。検出なしは None。"""
    if not result.detected:
        return None
    return extract_features(result.landmarks, FEATURES,
                            aspect=frame.shape[1] / frame.shape[0])


def handle_key(key, frame, result, feats, subject, out_dir):
    """キーが保存キーなら保存を試み、画面に出す文言を返す（それ以外は None）。"""
    posture = POSTURES.get(key)
    if posture is None:
        return None
    if feats is None:
        return f"NOT SAVED ({posture}): no person detected"
    if not feats["neck_angle"].valid:
        return f"NOT SAVED ({posture}): ear/shoulder not visible"
    return "saved " + save_pair(out_dir, subject, posture, frame, result.landmarks, feats)


def _draw_preview(frame, result, feats, subject, message):
    """表示用のコピーを作って描く（frame 自体には描かない）。"""
    from ui.live_view import _draw_pose

    img = cv2.resize(frame, PREVIEW_SIZE)  # 新しい配列を返すので元の frame は変わらない
    if result.detected:
        _draw_pose(img, result.landmarks, config.MIN_VISIBILITY)
    neck = feats["neck_angle"] if feats else None
    lines = [f"{subject}  1:good 2:slouch 3:lean 4:neck  q:quit",
             f"neck_angle: {neck.value:.1f}" if neck and neck.valid else "neck_angle: --"]
    if (frame.shape[1], frame.shape[0]) != EXPECTED_SIZE:
        lines.append(f"WARNING: frame {frame.shape[1]}x{frame.shape[0]}, "
                     f"expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    lines.append(message)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def run(subject, out_dir):
    from capture import Camera
    from pose import PoseEstimator

    camera = Camera()
    estimator = PoseEstimator()
    timestamp_ms = 0
    message = ""
    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok:
                continue

            timestamp_ms += 33  # VIDEO モードは単調増加が必要
            result = estimator.estimate(frame, timestamp_ms)
            feats = compute_feats(frame, result)

            cv2.imshow(WINDOW, _draw_preview(frame, result, feats, subject, message))
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "q":
                break
            # 今表示している frame をそのまま保存するので、画像と値は必ず同じ瞬間になる。
            message = handle_key(key, frame, result, feats, subject, out_dir) or message
    finally:
        estimator.close()
        camera.release()
        cv2.destroyAllWindows()


# ── 単体での動作確認（カメラ不要） ──
def _self_test():
    import json
    import shutil
    import tempfile
    from collections import namedtuple

    import numpy as np

    LM = namedtuple("LM", ["x", "y", "z", "visibility"])
    Result = namedtuple("Result", ["detected", "landmarks"])

    def pose(visibility=1.0):
        # 右側だけを写す（左側は低 visibility にして、自動選択で右側が選ばれるようにする）。
        points = [LM(0.5, 0.5, 0.0, 0.05)] * config.NUM_LANDMARKS
        points[config.RIGHT_EAR] = LM(0.55, 0.20, 0.0, visibility)
        points[config.RIGHT_SHOULDER] = LM(0.50, 0.40, 0.0, visibility)
        points[config.RIGHT_HIP] = LM(0.48, 0.80, 0.0, visibility)
        return points

    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    result = Result(True, pose())
    feats = compute_feats(frame, result)

    # 日本語パスで確かめる（cv2.imwrite だと失敗する条件）。
    out_dir = tempfile.mkdtemp(prefix="実験B_")
    try:
        # ① 保存 → 読み戻しで1画素も違わない（骨格を描いたプレビューの影響も受けない）。
        _draw_preview(frame, result, feats, "S1", "")
        message = handle_key("2", frame, result, feats, "S1", out_dir)
        assert message == "saved S1_slouch_1.png", message
        path = os.path.join(out_dir, "images", "S1_slouch_1.png")
        with open(path, "rb") as f:
            loaded = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(loaded, original)
        print("PASS 画像を無劣化で保存（骨格なし・日本語パス）")

        # ② CSV の座標から計算し直すと neck_angle が一致する。
        with open(os.path.join(out_dir, "pairs.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        points = [LM(*p) for p in json.loads(rows[0]["points"])]
        again = extract_features(points, FEATURES, aspect=1280 / 720)
        assert abs(again["neck_angle"].value - float(rows[0]["neck_angle"])) < 1e-3
        assert (rows[0]["width"], rows[0]["height"]) == ("1280", "720")
        print("PASS CSV の座標から再計算した neck_angle が一致")

        # ③ 保存しない場合: 検出なし／耳・肩が見えない／保存キー以外。
        no_person = Result(False, [])
        assert "NOT SAVED" in handle_key("1", frame, no_person, None, "S1", out_dir)
        hidden = Result(True, pose(visibility=0.0))
        assert "NOT SAVED" in handle_key("1", frame, hidden, compute_feats(frame, hidden),
                                         "S1", out_dir)
        assert handle_key("x", frame, result, feats, "S1", out_dir) is None
        assert sorted(os.listdir(os.path.join(out_dir, "images"))) == ["S1_slouch_1.png"]
        print("PASS 検出なし・耳肩が見えない・他のキーでは保存しない")

        # ④ 撮り直しは番号が増え、前のファイルと CSV の行が残る。
        assert handle_key("2", frame, result, feats, "S1", out_dir) == "saved S1_slouch_2.png"
        assert os.path.exists(path)
        with open(os.path.join(out_dir, "pairs.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["take"] for r in rows] == ["1", "2"]
        print("PASS 撮り直しで番号が増え、前の分も残る")
    finally:
        shutil.rmtree(out_dir)


def _subject_id(value):
    # 被験者ID はファイル名になるので、パスとして危険な文字を入口で弾く。
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("被験者IDは半角英数字・_・- で指定してください（例: S1）")
    return value


def main():
    parser = argparse.ArgumentParser(description="実験B用: 画像と特徴量を1組にして保存する")
    parser.add_argument("--subject", type=_subject_id, help="被験者ID（例: S1）")
    parser.add_argument("--self-test", action="store_true", help="カメラなしの動作確認")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.subject is None:
        parser.error("--subject を指定してください（例: --subject S1）")
    if getattr(config, "POSTURE_MODE", "side") != "side":
        parser.error("実験Bは横向きモード（POSTURE_MODE = 'side'）で実行してください")
    run(args.subject, OUT_DIR)


if __name__ == "__main__":
    main()
