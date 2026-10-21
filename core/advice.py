"""core/advice.py

悪姿勢を検知したとき、どう直せばよいかの「改善提案の文言」を選ぶ純粋関数。
（卒研メモの「改善提案の文言表示（『顎を引いてください』など）」に対応）

責任:
  - スコアが閾値未満のとき、基準値から最もずれている特徴を見つける
  - その特徴とずれの向きに応じた助言文を返す

担わないこと:
  - 採点そのもの → scoring.py（点数はこのファイルに渡してもらう）
  - 文言の表示 → ui/live_view.py（ここは「どの文言か」を決めるだけ）
  - 通知の発火 → notifier.py

設計の核:
  - 純粋関数。入力（現在の特徴量・基準値・スコア・設定）だけで結果が決まる。
  - scoring と同じ「現在 vs 基準値」の比較を使うので、採点と助言の判断がぶれない。
"""

# core/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

_EPS = 1e-9


def suggest(current_features, baseline, score,
            threshold=None, use_features=None, tolerance=None, messages=None):
    """改善提案の文言を1つ返す。提案不要なら None。

    引数:
      current_features … features.extract_features の出力 {名前: Feature(value, valid)}
      baseline         … メモリ上の基準値 dict（未キャリブレーションなら None）
      score            … scoring.score の出力（0〜100 / 採点不可 None）
      threshold        … この点未満で提案を出す（省略時 config.ADVICE_SCORE_THRESHOLD）

    判定: スコアが閾値未満のとき、基準値から「許容幅の何倍ずれたか」が最大の特徴を選び、
          ずれの向き（基準より小さい=low / 大きい=high）に応じた文言を返す。
    """
    threshold = config.ADVICE_SCORE_THRESHOLD if threshold is None else threshold
    use_features = config.USE_FEATURES if use_features is None else use_features
    tolerance = config.TOLERANCE if tolerance is None else tolerance
    messages = config.ADVICE_MESSAGES if messages is None else messages

    # 採点不可・基準値なし・スコアが十分高い → 提案しない。
    if score is None or baseline is None:
        return None
    if score >= threshold:
        return None

    # 最も基準からずれている特徴（と向き）を探す。
    worst_name = None
    worst_deviation = 0.0
    worst_direction = None
    for name in use_features:
        feat = current_features.get(name)
        if feat is None or not feat.valid or feat.value is None:
            continue
        base_val = baseline.get(name)
        if base_val is None:
            continue
        tol = tolerance.get(name)
        if tol is None or tol <= _EPS:
            continue

        deviation = abs(feat.value - base_val) / tol  # 許容幅の何倍ずれたか
        if deviation > worst_deviation:
            worst_deviation = deviation
            worst_name = name
            # 基準より小さいか大きいか（文言の向きに使う）。
            worst_direction = "low" if feat.value < base_val else "high"

    # 有効な特徴が一つも無い（体が映っていない等）→ 提案しない。
    if worst_name is None:
        return None

    # 該当する文言。無ければ汎用文言。
    return messages.get((worst_name, worst_direction), config.ADVICE_DEFAULT_MESSAGE)


class Advisor:
    """改善提案にヒステリシス（遊び）を持たせ、文言の点滅を抑える状態付きラッパ。

    純粋な suggest() を、出す線(show)と消す線(hide)で挟む。
      - 非表示中: スコアが show_threshold 未満になったら文言を出す。
      - 表示中  : スコアが hide_threshold 以上に回復するまで出し続ける。
    境界（60点）付近で点いたり消えたりするフラッピングを防ぐ（notifier と同じ発想）。

    状態を持つので、ワーカー1つに1インスタンス（採点ループと同じスレッドで使う）。
    """

    def __init__(self, show_threshold=None, hide_threshold=None):
        self.show_threshold = (config.ADVICE_SCORE_THRESHOLD
                               if show_threshold is None else show_threshold)
        self.hide_threshold = (config.ADVICE_HIDE_THRESHOLD
                               if hide_threshold is None else hide_threshold)
        # 消す線は出す線より高くする（間に遊びを作る）。逆だと設計ミス。
        if self.hide_threshold < self.show_threshold:
            raise ValueError(
                f"hide_threshold({self.hide_threshold}) は "
                f"show_threshold({self.show_threshold}) 以上にしてください（遊び）。"
            )
        self._showing = False
        self._message = None

    def update(self, current_features, baseline, score):
        """1フレーム分を取り込み、今表示すべき文言（or None）を返す。"""
        # 判定材料が無い（採点不可・基準値なし）ときは今の表示を維持（点滅防止）。
        if score is None or baseline is None:
            return self._message

        if self._showing:
            # 表示中: 消す線まで回復したら消す。未回復なら最新の最悪特徴に文言を更新。
            if score >= self.hide_threshold:
                self._showing = False
                self._message = None
            else:
                msg = suggest(current_features, baseline, score,
                              threshold=self.hide_threshold)
                if msg is not None:
                    self._message = msg
        else:
            # 非表示: 出す線を割ったら出す。
            if score < self.show_threshold:
                msg = suggest(current_features, baseline, score,
                              threshold=self.show_threshold)
                if msg is not None:
                    self._showing = True
                    self._message = msg
        return self._message

    def reset(self):
        """状態を初期化（離席などで判定をやり直すとき）。"""
        self._showing = False
        self._message = None


# ── 単体での動作確認（カメラ・DB不要。にせの値で確かめる） ──
def _self_test():
    from collections import namedtuple
    Feature = namedtuple("Feature", ["value", "valid"])

    baseline = {"neck_angle": 80.0, "trunk_angle": 5.0}

    def feats(neck, trunk, neck_valid=True, trunk_valid=True):
        return {
            "neck_angle": Feature(neck, neck_valid),
            "trunk_angle": Feature(trunk, trunk_valid),
        }

    # Advisor は内部で config を読む。POSTURE_MODE に依らず横向きの文言で検証できるよう、
    # テスト中だけ config を横向き設定に固定する（本番設定は最後に戻す）。
    _saved = (config.USE_FEATURES, config.TOLERANCE, config.ADVICE_MESSAGES,
              config.ADVICE_DEFAULT_MESSAGE)
    config.USE_FEATURES = config._SIDE_USE_FEATURES
    config.TOLERANCE = config._SIDE_TOLERANCE
    config.ADVICE_MESSAGES = config._SIDE_ADVICE_MESSAGES
    config.ADVICE_DEFAULT_MESSAGE = "姿勢が崩れています。背筋を伸ばしましょう。"
    try:
        _run_advice_checks(feats, baseline)
    finally:
        (config.USE_FEATURES, config.TOLERANCE, config.ADVICE_MESSAGES,
         config.ADVICE_DEFAULT_MESSAGE) = _saved


def _run_advice_checks(feats, baseline):
    results = []

    # ① スコアが高い → 提案なし
    results.append(("高得点では提案しない", suggest(feats(80, 5), baseline, 95.0) is None))

    # ② 前傾頭位が支配的（neck が基準より小さい）＋低得点 → 顎を引く
    msg = suggest(feats(55, 6), baseline, 40.0)
    results.append(("前傾頭位→顎を引く", msg == config.ADVICE_MESSAGES[("neck_angle", "low")]))

    # ③ 体幹の傾きが支配的（trunk が基準より大きい）＋低得点 → 背筋を伸ばす
    msg = suggest(feats(78, 30), baseline, 40.0)
    results.append(("猫背・前傾→背筋を伸ばす", msg == config.ADVICE_MESSAGES[("trunk_angle", "high")]))

    # ④ 基準値なし → 提案なし
    results.append(("基準値なしは提案なし", suggest(feats(55, 6), None, 40.0) is None))

    # ⑤ 採点不可 → 提案なし
    results.append(("採点不可は提案なし", suggest(feats(55, 6), baseline, None) is None))

    # ⑥ 有効特徴ゼロ → 提案なし
    none_valid = feats(55, 30, neck_valid=False, trunk_valid=False)
    results.append(("有効特徴ゼロは提案なし", suggest(none_valid, baseline, 40.0) is None))

    # ── Advisor（ヒステリシス）の遊びを確認 ──
    adv = Advisor(show_threshold=60.0, hide_threshold=70.0)
    # 高得点: 出さない
    r1 = adv.update(feats(80, 5), baseline, 90.0)
    # 60未満: 出す（顎を引く）
    r2 = adv.update(feats(55, 6), baseline, 55.0)
    # 65（出す線と消す線の間）: まだ出し続ける（点滅しない）
    r3 = adv.update(feats(58, 6), baseline, 65.0)
    # 68: まだ出し続ける
    r4 = adv.update(feats(59, 6), baseline, 68.0)
    # 75（消す線以上）: 消える
    r5 = adv.update(feats(72, 5), baseline, 75.0)
    results.append(("遊び: 高得点では出さない", r1 is None))
    results.append(("遊び: 60未満で出す", r2 == config.ADVICE_MESSAGES[("neck_angle", "low")]))
    results.append(("遊び: 60〜70の間は出し続ける(点滅しない)", r3 is not None and r4 is not None))
    results.append(("遊び: 70以上で消える", r5 is None))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("advice 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
