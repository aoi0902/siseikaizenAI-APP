"""core/features.py

33点の座標を「姿勢を表す数字（特徴量）」に変換する純粋関数。
**横向き（側面）撮影 と 正面撮影 の両対応**（config.POSTURE_MODE で切り替え）。
（設計メモ: features_py_設計メモ.md を両モード仕様に拡張）

このファイルは ★ キャリブレーションと採点の両方から呼ばれる共用の核。
保存時(calibration)も採点時(scoring)も必ず同じ extract_features を通すので、
両者の計算が一致し「基準値と現在を比べる」が成り立つ。

横向き（POSTURE_MODE="side"）の特徴量:
  - neck_angle  … 選択側の「耳→肩」線が水平となす角度（CVA 相当）。
                  耳が肩の真上なら ~90°（良好）、頭が前に出るほど小さい（前傾頭位）。
  - trunk_angle … 選択側の「肩→腰」線が垂直となす角度。
                  0°なら体幹まっすぐ（良好）、大きいほど前後に傾く（猫背・前傾）。

正面（POSTURE_MODE="front"）の特徴量:
  - shoulder_tilt … 「左肩→右肩」線が水平となす角度。
                    0°なら両肩が水平（良好）、大きいほど左右で高さがずれている。
  - head_tilt     … 「左目→右目」線が水平となす角度。
                    0°なら頭がまっすぐ（良好）、大きいほど首を左右にかしげている。

共通の工夫:
  - 水平/垂直成分は絶対値で扱うので、左右どちらに傾いても符号がぶれない。
  - 角度は平行移動・拡大縮小に不変。カメラとの距離や写る大きさに左右されない。
  - 横向きでは、左右どちらの側面が写っているかを visibility から自動判定（config.FACING_SIDE）。

責任:
  - 33点（正規化）→ 特徴量（角度）への変換（モードに応じて使う点・式が変わる）
  - 必要な点が信用できるか（有効/無効）の判定

担わないこと:
  - 映像取得 → capture.py / 骨格推定 → pose.py
  - 基準値の保存 → calibration.py / db
  - 基準値との比較・採点 → scoring.py

設計の核: 純粋関数。入力(33点)だけで出力が決まり、状態を持たない。
"""

import math
from collections import namedtuple

# core/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# 1つの特徴量＝「値」と「その値が信用できるか」のペア。
Feature = namedtuple("Feature", ["value", "valid"])
INVALID = Feature(value=None, valid=False)

# 割る量がこれ未満なら（点が重なる等）向きが定まらないとみなして無効。
_EPS = 1e-6

# 側面（カメラに写る側）の点の組。
Side = namedtuple("Side", ["ear", "shoulder", "hip"])
_LEFT_SIDE = Side(config.LEFT_EAR, config.LEFT_SHOULDER, config.LEFT_HIP)
_RIGHT_SIDE = Side(config.RIGHT_EAR, config.RIGHT_SHOULDER, config.RIGHT_HIP)


# ── 補助 ──
def _convert_angle(angle_deg):
    """度で計算した角度を config の単位に変換して返す。"""
    if config.ANGLE_UNIT == "radian":
        return math.radians(angle_deg)
    return angle_deg  # 既定は度


def _visible(landmarks, index, min_visibility):
    return landmarks[index].visibility >= min_visibility


def _select_side(landmarks):
    """カメラに写っている側面（左/右）を選ぶ。

    config.FACING_SIDE が "left"/"right" なら固定。"auto" なら、左右それぞれの
    代表点（耳・肩・腰）の visibility 合計を比べ、高い側を採用する。
    """
    facing = getattr(config, "FACING_SIDE", "auto")
    if facing == "left":
        return _LEFT_SIDE
    if facing == "right":
        return _RIGHT_SIDE

    def total_vis(side):
        return (landmarks[side.ear].visibility
                + landmarks[side.shoulder].visibility
                + landmarks[side.hip].visibility)

    return _LEFT_SIDE if total_vis(_LEFT_SIDE) >= total_vis(_RIGHT_SIDE) else _RIGHT_SIDE


# ── 特徴量① 首の前傾（CVA 相当） ──
def _neck_angle(landmarks, side, min_visibility):
    """選択側の「耳→肩」線が水平となす角度。

    耳が肩の真上 → 90°（頭がまっすぐ載っている＝良好）。
    頭が前に出る → 角度が小さくなる（ストレートネック/前傾頭位）。
    """
    ear, shoulder = side.ear, side.shoulder
    if not (_visible(landmarks, ear, min_visibility)
            and _visible(landmarks, shoulder, min_visibility)):
        return INVALID

    # y は下向き。耳が肩より上なら up > 0。横方向のずれは符号を問わず絶対値で。
    up = landmarks[shoulder].y - landmarks[ear].y
    horizontal = abs(landmarks[ear].x - landmarks[shoulder].x)
    if math.hypot(up, horizontal) < _EPS:
        return INVALID

    # 水平からの角度。horizontal=0（真上）で 90°、前に出るほど小さい。
    angle_deg = math.degrees(math.atan2(up, horizontal))
    return Feature(value=_convert_angle(angle_deg), valid=True)


# ── 特徴量② 体幹の傾き ──
def _trunk_angle(landmarks, side, min_visibility):
    """選択側の「肩→腰」線が垂直となす角度。

    肩が腰の真上 → 0°（体幹がまっすぐ＝良好）。
    前後に傾く  → 角度が大きくなる（猫背・前傾）。
    """
    shoulder, hip = side.shoulder, side.hip
    if not (_visible(landmarks, shoulder, min_visibility)
            and _visible(landmarks, hip, min_visibility)):
        return INVALID

    vertical = abs(landmarks[shoulder].y - landmarks[hip].y)
    horizontal = abs(landmarks[shoulder].x - landmarks[hip].x)
    if math.hypot(vertical, horizontal) < _EPS:
        return INVALID

    # 垂直からの角度。horizontal=0（真上下）で 0°、傾くほど大きい。
    angle_deg = math.degrees(math.atan2(horizontal, vertical))
    return Feature(value=_convert_angle(angle_deg), valid=True)


# ── 特徴量③ 上体屈曲角（胸椎後弯の代理）／横向き ──
def _kyphosis_angle(landmarks, side, min_visibility):
    """選択側の「耳-肩-腰」がなす角度（肩を頂点とする）。上体屈曲角／胸椎後弯の代理。

    まっすぐ座ると 耳・肩・腰 がほぼ一直線に並ぶ → ~180°（背中がまっすぐ＝良好）。
    背中が丸まる（猫背）と、頭・肩が前に出て 耳-肩-腰 が折れ、角度が小さくなる。
    体幹の傾き(trunk_angle)と合わせて猫背を判断する。
    """
    ear, shoulder, hip = side.ear, side.shoulder, side.hip
    if not (_visible(landmarks, ear, min_visibility)
            and _visible(landmarks, shoulder, min_visibility)
            and _visible(landmarks, hip, min_visibility)):
        return INVALID

    # 肩を頂点に、肩→耳 と 肩→腰 の2ベクトルがなす角を求める。
    v1x = landmarks[ear].x - landmarks[shoulder].x
    v1y = landmarks[ear].y - landmarks[shoulder].y
    v2x = landmarks[hip].x - landmarks[shoulder].x
    v2y = landmarks[hip].y - landmarks[shoulder].y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < _EPS or n2 < _EPS:
        return INVALID  # 点が重なって向きが定まらない

    # cosθ = (v1·v2)/(|v1||v2|)。丸め誤差で±1を超えると acos が落ちるので挟む。
    cos_theta = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = math.degrees(math.acos(cos_theta))
    return Feature(value=_convert_angle(angle_deg), valid=True)


# ── 正面撮影用の特徴量 ──
def _line_tilt_from_horizontal(landmarks, i_left, i_right, min_visibility):
    """左右2点を結ぶ線が「水平」となす角度[度]を返す。

    0°=水平（左右の高さがそろっている＝良好）、傾くほど大きい。
    正面の shoulder_tilt / head_tilt はこの共通の式で計算する（見る点が違うだけ）。
    """
    if not (_visible(landmarks, i_left, min_visibility)
            and _visible(landmarks, i_right, min_visibility)):
        return INVALID

    # dx=横の広がり、dy=縦の高さ差。左右対称に扱うため両方とも絶対値。
    dx = abs(landmarks[i_right].x - landmarks[i_left].x)
    dy = abs(landmarks[i_right].y - landmarks[i_left].y)
    if math.hypot(dx, dy) < _EPS:
        return INVALID

    # 水平からの角度。dy=0（水平）で 0°、高さがずれるほど大きい。
    angle_deg = math.degrees(math.atan2(dy, dx))
    return Feature(value=_convert_angle(angle_deg), valid=True)


def _shoulder_tilt(landmarks, min_visibility):
    """両肩（左肩→右肩）の水平からの傾き。0°=両肩が水平（良好）。"""
    return _line_tilt_from_horizontal(
        landmarks, config.LEFT_SHOULDER, config.RIGHT_SHOULDER, min_visibility)


def _head_tilt(landmarks, min_visibility):
    """両目（左目→右目）の水平からの傾き。0°=頭がまっすぐ（良好）。"""
    return _line_tilt_from_horizontal(
        landmarks, config.LEFT_EYE, config.RIGHT_EYE, min_visibility)


# 特徴名 → 計算関数。
# 横向きは側面(side)を渡す (landmarks, side, min_visibility)、
# 正面は左右両方を使うので side 不要 (landmarks, min_visibility)。extract_features が振り分ける。
_SIDE_FEATURE_FUNCS = {
    "neck_angle": _neck_angle,
    "trunk_angle": _trunk_angle,
    "kyphosis_angle": _kyphosis_angle,
}
_FRONT_FEATURE_FUNCS = {
    "shoulder_tilt": _shoulder_tilt,
    "head_tilt": _head_tilt,
}


# ── 公開インターフェース ──
def extract_features(landmarks, use_features=None, min_visibility=None):
    """33点（正規化）から、名前つきの特徴量＋有効フラグを返す（モードで式が変わる）。

    引数:
      landmarks      … pose.py が返した 33点（各点 x,y,z,visibility）
      use_features   … 使う特徴名のリスト（省略時 config.USE_FEATURES）
      min_visibility … 信用する最低 visibility（省略時 config.MIN_VISIBILITY）

    戻り値: {特徴名: Feature(value, valid), ...}
      必要な点が低 visibility 等のときは valid=False で返す。

    注意: detected=False の入力は呼ばない約束（呼び出し側が detected を見てから渡す）。
    """
    if use_features is None:
        use_features = config.USE_FEATURES
    if min_visibility is None:
        min_visibility = config.MIN_VISIBILITY

    # 点数が足りない入力は、全特徴を無効にして安全に返す。
    if landmarks is None or len(landmarks) < config.NUM_LANDMARKS:
        return {name: INVALID for name in use_features}

    # モードに応じて特徴量を計算する。
    if getattr(config, "POSTURE_MODE", "side") == "front":
        # 正面: 左右両方の点を使う（側面選択は不要）。
        return {name: _FRONT_FEATURE_FUNCS[name](landmarks, min_visibility)
                for name in use_features}

    # 横向き: 写っている側面を1回だけ選び、両特徴で同じ側を使う（一貫性のため）。
    side = _select_side(landmarks)
    return {name: _SIDE_FEATURE_FUNCS[name](landmarks, side, min_visibility)
            for name in use_features}


# ── 単体での動作確認（カメラ不要。手作りのポーズでテスト。両モードを検証） ──
class _LM:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _test_side():
    """横向きモードの特徴量を検証する。"""
    def make(side, ear, shoulder, hip, seen_vis=1.0, hidden_vis=0.05):
        """指定側の耳・肩・腰を置き、反対側は低 visibility にした33点を作る。"""
        pts = [_LM(0.5, 0.5, visibility=seen_vis) for _ in range(config.NUM_LANDMARKS)]
        seen = _LEFT_SIDE if side == "left" else _RIGHT_SIDE
        hidden = _RIGHT_SIDE if side == "left" else _LEFT_SIDE
        pts[seen.ear] = _LM(*ear, visibility=seen_vis)
        pts[seen.shoulder] = _LM(*shoulder, visibility=seen_vis)
        pts[seen.hip] = _LM(*hip, visibility=seen_vis)
        for i in (hidden.ear, hidden.shoulder, hidden.hip):
            pts[i] = _LM(0.5, 0.5, visibility=hidden_vis)
        return pts

    uf = ["neck_angle", "trunk_angle", "kyphosis_angle"]
    results = []

    # ① まっすぐ（左向き）: 耳が肩の真上・肩が腰の真上 → neck~90, trunk~0, kyphosis~180
    f = extract_features(make("left", ear=(0.5, 0.25), shoulder=(0.5, 0.45), hip=(0.5, 0.75)), uf)
    results.append(("まっすぐ: neck~90", abs(f["neck_angle"].value - 90.0) < 1.0))
    results.append(("まっすぐ: trunk~0", abs(f["trunk_angle"].value - 0.0) < 1.0))
    results.append(("まっすぐ: kyphosis~180", abs(f["kyphosis_angle"].value - 180.0) < 2.0))

    # ② 前傾頭位: 耳が前へ → neck が小さくなる／耳-肩-腰が折れて kyphosis も小さくなる（猫背）
    f = extract_features(make("left", ear=(0.64, 0.27), shoulder=(0.5, 0.45), hip=(0.5, 0.75)), uf)
    results.append(("前傾頭位でneckが下がる", f["neck_angle"].value < 75.0))
    results.append(("背中が丸まるとkyphosisが下がる", f["kyphosis_angle"].value < 165.0))

    # ③ 猫背/前傾: 肩が腰より前へ → trunk が大きくなる
    f = extract_features(make("left", ear=(0.5, 0.25), shoulder=(0.6, 0.45), hip=(0.5, 0.75)), uf)
    results.append(("前傾でtrunkが増える", f["trunk_angle"].value > 10.0))

    # ④ 右向きを自動判定（右側だけ可視）: まっすぐなら neck~90, trunk~0
    f = extract_features(make("right", ear=(0.4, 0.25), shoulder=(0.4, 0.45), hip=(0.4, 0.75)), uf)
    results.append(("右向きを自動判定して算出", abs(f["neck_angle"].value - 90.0) < 1.0
                    and abs(f["trunk_angle"].value - 0.0) < 1.0))

    # ⑤ 位置・大きさを変えても角度は不変（まっすぐを別位置・別サイズで）
    f = extract_features(make("left", ear=(0.72, 0.40), shoulder=(0.72, 0.52), hip=(0.72, 0.70)), uf)
    results.append(("位置・大きさに不変", abs(f["neck_angle"].value - 90.0) < 1.0
                    and abs(f["trunk_angle"].value - 0.0) < 1.0))

    # ⑥ 低 visibility → 無効
    f = extract_features(make("left", ear=(0.5, 0.25), shoulder=(0.5, 0.45), hip=(0.5, 0.75),
                              seen_vis=0.1, hidden_vis=0.05), uf)
    results.append(("低信頼は無効", f["neck_angle"].valid is False and f["trunk_angle"].valid is False))
    return results


def _test_front():
    """正面モードの特徴量を検証する。"""
    def make(l_sh, r_sh, l_eye, r_eye, vis=1.0):
        """両肩・両目を置いた33点を作る（正面想定）。"""
        pts = [_LM(0.5, 0.5, visibility=vis) for _ in range(config.NUM_LANDMARKS)]
        pts[config.LEFT_SHOULDER] = _LM(*l_sh, visibility=vis)
        pts[config.RIGHT_SHOULDER] = _LM(*r_sh, visibility=vis)
        pts[config.LEFT_EYE] = _LM(*l_eye, visibility=vis)
        pts[config.RIGHT_EYE] = _LM(*r_eye, visibility=vis)
        return pts

    uf = ["shoulder_tilt", "head_tilt"]
    results = []

    # ① まっすぐ（両肩水平・両目水平）→ shoulder_tilt~0, head_tilt~0
    f = extract_features(make((0.40, 0.50), (0.60, 0.50), (0.45, 0.30), (0.55, 0.30)), uf)
    results.append(("正面まっすぐ: shoulder~0", abs(f["shoulder_tilt"].value - 0.0) < 1.0))
    results.append(("正面まっすぐ: head~0", abs(f["head_tilt"].value - 0.0) < 1.0))

    # ② 肩が左右で傾く（右肩を下げる）→ shoulder_tilt が増える
    f = extract_features(make((0.40, 0.50), (0.60, 0.56), (0.45, 0.30), (0.55, 0.30)), uf)
    results.append(("肩が傾くとshoulder_tiltが増える", f["shoulder_tilt"].value > 10.0))

    # ③ 頭を左右にかしげる（右目を下げる）→ head_tilt が増える
    f = extract_features(make((0.40, 0.50), (0.60, 0.50), (0.45, 0.30), (0.55, 0.36)), uf)
    results.append(("頭を傾けるとhead_tiltが増える", f["head_tilt"].value > 10.0))

    # ④ 位置・大きさを変えても水平なら 0°（平行移動・拡大縮小に不変）
    f = extract_features(make((0.20, 0.65), (0.50, 0.65), (0.28, 0.50), (0.42, 0.50)), uf)
    results.append(("正面: 位置・大きさに不変", abs(f["shoulder_tilt"].value) < 1.0
                    and abs(f["head_tilt"].value) < 1.0))

    # ⑤ 低 visibility → 無効
    f = extract_features(make((0.40, 0.50), (0.60, 0.50), (0.45, 0.30), (0.55, 0.30), vis=0.1), uf)
    results.append(("正面: 低信頼は無効",
                    f["shoulder_tilt"].valid is False and f["head_tilt"].valid is False))
    return results


def _self_test():
    # config.POSTURE_MODE を一時的に切り替えて、両モードを検証する。
    original_mode = getattr(config, "POSTURE_MODE", "side")
    all_ok = True
    try:
        for mode, runner, label in (("side", _test_side, "横向き"),
                                    ("front", _test_front, "正面")):
            config.POSTURE_MODE = mode
            print(f"--- {label}モード ---")
            for name, ok in runner():
                print(f"[{'PASS' if ok else 'FAIL'}] {name}")
                all_ok = all_ok and ok
    finally:
        config.POSTURE_MODE = original_mode
    print("features(両モード) 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
