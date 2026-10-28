"""core/scoring.py

現在の姿勢が基準値からどれだけずれているかを測り、0〜100 点に変換する純粋関数。
（設計メモ: scoring_py_設計メモ.md に対応）

採点の理論（第5回）: ずれを揃える → 重み付け → 0〜100点。
features.py が出した「今の特徴量」と、calibration が保存した「基準値(満点=100)」を
比べて、1 フレーム分の点数を返す。

責任:
  - 現在の特徴量と基準値のずれを計算
  - ずれを許容幅で公平に揃え、重み付けして 1 つにまとめる
  - まとめたずれを 0〜100 点に変換
  - 無効な特徴を採点から外す

担わないこと:
  - 特徴量の計算 → features.py
  - 基準値の保存・取得 → calibration.py / database.py
  - 時間の平均・通知判断 → notifier.py（震えをならすのは notifier の仕事）
  - スコアの記録 → database.py

設計の核（2つ）:
  1. 純粋関数（入力＝現在の特徴量・基準値・設定 だけで点数が決まる）。
  2. 1 フレーム分の点数を出すだけ。時間平均・しきい値・通知は混ぜない。
"""

# core/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# 採点不可（基準値なし／有効な特徴がゼロ）を表す。notifier 側は平均から除外する。
UNSCORABLE = None

# ゼロ割り回避用の極小値。
_EPS = 1e-9


def score(
    current_features,
    baseline,
    use_features=None,
    tolerance=None,
    weight=None,
    deadband=None,
    decay=None,
):
    """1 フレーム分の点数(0〜100)を返す。採点できなければ UNSCORABLE(None)。

    引数:
      current_features … features.extract_features の出力 {名前: Feature(value, valid)}
      baseline         … メモリ上の基準値 dict {名前: 値, ...}（get_latest_baseline 由来）
                         未キャリブレーションなら None。
      tolerance/weight … 特徴ごとの許容幅・重み（省略時 config）
      deadband/decay   … 変換曲線の設定（省略時 config）

    基準値はメモリ上のものを受け取る（毎フレーム DB を読まない＝database.py 推奨案）。
    """
    use_features = config.USE_FEATURES if use_features is None else use_features
    tolerance = config.TOLERANCE if tolerance is None else tolerance
    weight = config.WEIGHT if weight is None else weight
    deadband = config.DEADBAND if deadband is None else deadband
    decay = config.DECAY if decay is None else decay

    # 基準値が無ければ採点は成り立たない。
    if baseline is None:
        return UNSCORABLE

    # ── 特徴ごとのずれを「許容幅の何倍ずれたか」に揃えて重み付け合算 ──
    weighted_sum = 0.0   # Σ wᵢ・dᵢ（dᵢ = 揃えたずれ）
    weight_total = 0.0   # Σ wᵢ（有効な特徴ぶんだけ）

    for name in use_features:
        feat = current_features.get(name)
        # 無効な特徴は採点から外す（巨大なずれと誤解して大減点しないため）。
        if feat is None or not feat.valid or feat.value is None:
            continue

        base_val = baseline.get(name)
        if base_val is None:
            continue  # 基準値にその特徴が無い

        tol = tolerance.get(name)
        if tol is None or tol <= _EPS:
            continue  # 許容幅ゼロ/極小（設定ミス）はゼロ割り回避で外す

        w = weight.get(name, 1.0)

        # ずれを許容幅で割り、単位の違う特徴を同じ土俵に乗せる（第5回の核）。
        deviation = abs(feat.value - base_val) / tol
        weighted_sum += w * deviation
        weight_total += w

    # 有効な特徴がゼロ（体がほぼ映っていない等）→ 採点不可。
    if weight_total <= _EPS:
        return UNSCORABLE

    # 重みを「有効な特徴ぶん」で配分し直す（無効特徴を外しても公平）。
    total_deviation = weighted_sum / weight_total

    # ── 変換曲線: 許容範囲(満点) → 減点ゾーン → 0/100 で頭打ち ──
    return _deviation_to_score(total_deviation, deadband, decay)


def deviations(current_features, baseline, use_features=None, tolerance=None):
    """特徴ごとの「符号付き」ずれ（許容幅の何倍か）を返す。傾向分析の記録用。

    score() と同じ物差し (現在値 − 基準値) / 許容幅 だが、絶対値を取らず符号を残す。
    符号があれば、あとから「基準より小さい(low)/大きい(high)」＝崩れの向きを復元できる。

    戻り値: {特徴名: ずれ or None(無効)} ／ 基準値が無ければ None。
    """
    use_features = config.USE_FEATURES if use_features is None else use_features
    tolerance = config.TOLERANCE if tolerance is None else tolerance
    if baseline is None:
        return None

    result = {}
    for name in use_features:
        feat = current_features.get(name)
        base_val = baseline.get(name)
        tol = tolerance.get(name)
        if (feat is None or not feat.valid or feat.value is None
                or base_val is None or tol is None or tol <= _EPS):
            result[name] = None
            continue
        result[name] = (feat.value - base_val) / tol
    return result


def _deviation_to_score(total_deviation, deadband, decay):
    """総合ずれ → 0〜100 点。

    - deadband 以内: 100 点（呼吸・微動・座標ブレの“遊び”。これが無いと
      「ちゃんと座っているのに点が低い」になる）。
    - deadband 超過: 超えたぶんに比例して減点（少し崩れたら少し、大きく崩れたら大きく）。
    - 0 を下回らず 100 を超えない（下げ止め・頭打ち）。
    """
    # 許容範囲を超えたぶんだけが減点対象。
    effective = max(0.0, total_deviation - deadband)

    if decay <= _EPS:
        # 減点幅ゼロ（設定ミス）：許容超えで即 0、以内なら 100。
        raw = 100.0 if effective <= _EPS else 0.0
    else:
        # effective が decay に達すると 0 点になる素直な直線。
        raw = 100.0 * (1.0 - effective / decay)

    # 必ず 0〜100 の枠に収める。
    return max(0.0, min(100.0, raw))


# ── 単体での動作確認（カメラ・DB 不要。にせの値で確かめる） ──
def _self_test():
    from collections import namedtuple
    Feature = namedtuple("Feature", ["value", "valid"])

    # 横向きの基準値（首の前傾 CVA~80°、体幹の傾き~5°）。
    baseline = {"neck_angle": 80.0, "trunk_angle": 5.0}

    def feats(neck, trunk, neck_valid=True, trunk_valid=True):
        return {
            "neck_angle": Feature(neck, neck_valid),
            "trunk_angle": Feature(trunk, trunk_valid),
        }

    # このデモは neck/trunk の 2 特徴で確認する。config.POSTURE_MODE に依らず同じ検証に
    # なるよう、使う特徴・許容幅・重みを明示的に渡す（本番設定は読まない）。
    uf = ["neck_angle", "trunk_angle"]
    tol = {"neck_angle": 12.0, "trunk_angle": 12.0}
    wt = {"neck_angle": 0.6, "trunk_angle": 0.4}

    def sc(current, base=baseline):
        return score(current, base, use_features=uf, tolerance=tol, weight=wt)

    # ① 基準と一致 → ほぼ満点
    print(f"一致         : {sc(feats(80.0, 5.0)):.1f}  ← 100 のはず")

    # ② 小さなずれ（許容範囲内）→ まだ満点
    #   neck 3度/許容12 = 0.25、trunk 3度/許容12 = 0.25 → 加重平均 0.25 < deadband0.5
    print(f"小さなずれ   : {sc(feats(77.0, 8.0)):.1f}  ← 100 近いはず")

    # ③ 大きくずれ（前傾頭位＋猫背）→ 低い点
    print(f"大きなずれ   : {sc(feats(55.0, 25.0)):.1f}  ← 低いはず")

    # ④ 極端 → 0 で下げ止まり
    print(f"極端なずれ   : {sc(feats(10.0, 60.0)):.1f}  ← 0 のはず")

    # ⑤ 一部の特徴を無効に → 残り(neck)だけで採点
    print(f"片方無効     : {sc(feats(55.0, 999.0, trunk_valid=False)):.1f}"
          f"  ← trunk を無視し neck だけで採点")

    # ⑥ 基準値なし → 採点不可
    print(f"基準値なし   : {sc(feats(80.0, 5.0), None)}  ← None のはず")

    # ⑦ 有効特徴ゼロ → 採点不可
    both_invalid = feats(80.0, 5.0, neck_valid=False, trunk_valid=False)
    print(f"有効特徴ゼロ : {sc(both_invalid)}  ← None のはず")

    # ⑧ 符号付きずれ（傾向分析の記録用）: 首が基準より小さい→負、無効→None
    devs = deviations(feats(55.0, 5.0, trunk_valid=False), baseline,
                      use_features=["neck_angle", "trunk_angle"],
                      tolerance={"neck_angle": 12.0, "trunk_angle": 12.0})
    assert devs["neck_angle"] < 0 and devs["trunk_angle"] is None, devs
    assert deviations(feats(80.0, 5.0), None) is None
    print(f"符号付きずれ : {devs}  ← neck は負・trunk は None")


if __name__ == "__main__":
    _self_test()
