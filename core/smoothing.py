"""core/smoothing.py

特徴量の短時間移動平均でブレを減らす、状態を持つ平滑化器。

なぜ features.py と分けるか:
  features.extract_features は「1フレームの座標だけで値が決まる純粋関数」（状態なし）。
  一方こちらは「直近フレームの平均」を持つので状態を持つ。設計の性質が違うので分離する。

なぜ必要か:
  MediaPipe の座標は小刻みに揺れるため、体幹の傾きや上体屈曲角のような角度は
  1フレームだと震え、スコアや改善提案がチラつく。直近 N フレームの平均でならして安定させる。

どこで使うか:
  採点ループ（workers/pose_worker.py）が、extract_features の直後・採点の直前に1回通す。
  キャリブレーション（中央値で安定化）には通さない（基準値はそのまま生の代表値にする）。
"""

from collections import deque

# core/ から実行されても、パッケージ import でも config / features を読めるように。
try:
    import config
    from features import Feature
except ModuleNotFoundError:
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(_here))  # プロジェクトルート（config 用）
    sys.path.append(_here)                    # core（features 用）
    import config
    from features import Feature


class FeatureSmoother:
    """指定した特徴量だけを、直近 window フレームの移動平均に置き換える平滑化器。

    使い方（採点ループ側）:
        smoother = FeatureSmoother()               # config の設定で作る
        feats = extract_features(landmarks)
        feats = smoother.smooth(feats)             # 対象特徴を移動平均に差し替え
        score = score_frame(feats, baseline)
        ...
        smoother.reset()                           # 離席（検出なし）でためた平均を捨てる
    """

    def __init__(self, features=None, window=None):
        # 平滑化する特徴名と窓幅（省略時は config）。
        self.features = list(config.SMOOTH_FEATURES if features is None else features)
        self.window = config.SMOOTH_WINDOW if window is None else window
        # 特徴名ごとに直近の有効値をためる固定長リング（古い値は自動で押し出される）。
        self._buffers = {name: deque(maxlen=self.window) for name in self.features}

    def smooth(self, feats):
        """特徴量 dict を受け取り、対象特徴を移動平均に置き換えた新しい dict を返す。

        - 有効値（valid かつ value あり）だけを窓にため、その平均を返す。
        - このフレームで無効な特徴は、窓に足さず・触らず、無効のまま通す
          （1フレームの取りこぼしで平均を壊さない。長く続く離席は reset で捨てる）。
        - 対象外の特徴はそのまま素通しする。
        """
        result = dict(feats)  # 対象外はそのまま
        for name in self.features:
            feat = feats.get(name)
            if feat is None:
                continue  # このモードには無い特徴（例: 正面モードの trunk_angle）
            if feat.valid and feat.value is not None:
                buf = self._buffers[name]
                buf.append(feat.value)
                result[name] = Feature(value=sum(buf) / len(buf), valid=True)
            # 無効フレームは buf を変えず、無効のまま result[name] に残す。
        return result

    def reset(self):
        """ためた値をすべて捨てる（離席明けに過去の平均を引きずらないため）。"""
        for buf in self._buffers.values():
            buf.clear()


# ── 単体での動作確認（カメラ不要。にせの特徴量で確かめる） ──
def _self_test():
    def feat(value, valid=True):
        return Feature(value=value, valid=valid)

    results = []

    # 窓3で trunk_angle を平滑化する。
    sm = FeatureSmoother(features=["trunk_angle"], window=3)

    # ① 1件目はそのまま（平均＝自分自身）
    out = sm.smooth({"trunk_angle": feat(10.0)})
    results.append(("1件目は値そのまま", abs(out["trunk_angle"].value - 10.0) < 1e-9))

    # ② 3件たまると移動平均（10,20,30 → 20）
    sm.smooth({"trunk_angle": feat(20.0)})
    out = sm.smooth({"trunk_angle": feat(30.0)})
    results.append(("移動平均になる(10,20,30→20)", abs(out["trunk_angle"].value - 20.0) < 1e-9))

    # ③ 窓を超えると古い値が抜ける（20,30,40 → 30）
    out = sm.smooth({"trunk_angle": feat(40.0)})
    results.append(("窓を超えると古い値が抜ける(→30)", abs(out["trunk_angle"].value - 30.0) < 1e-9))

    # ④ 一瞬のスパイクがならされる（急な外れ値が単独では効きにくい）
    sm2 = FeatureSmoother(features=["trunk_angle"], window=5)
    for v in (10, 10, 10, 10):
        last = sm2.smooth({"trunk_angle": feat(v)})
    spiked = sm2.smooth({"trunk_angle": feat(60.0)})  # 急なスパイク
    results.append(("スパイクがならされる(生60→平均20)", spiked["trunk_angle"].value < 30.0))

    # ⑤ 無効フレームは平均を壊さない・無効のまま通す
    sm3 = FeatureSmoother(features=["trunk_angle"], window=3)
    sm3.smooth({"trunk_angle": feat(10.0)})
    inv = sm3.smooth({"trunk_angle": feat(None, valid=False)})
    results.append(("無効フレームは無効のまま通す", inv["trunk_angle"].valid is False))
    back = sm3.smooth({"trunk_angle": feat(20.0)})  # 直前の10と今の20 → 15（壊れていない）
    results.append(("無効を挟んでも平均は保たれる(10,20→15)", abs(back["trunk_angle"].value - 15.0) < 1e-9))

    # ⑥ reset でためた値が消える
    sm3.reset()
    after = sm3.smooth({"trunk_angle": feat(50.0)})
    results.append(("resetで平均がやり直しになる(→50)", abs(after["trunk_angle"].value - 50.0) < 1e-9))

    # ⑦ 対象外の特徴はそのまま素通し
    out = sm.smooth({"neck_angle": feat(88.0)})
    results.append(("対象外はそのまま", out["neck_angle"].value == 88.0))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("smoothing 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
