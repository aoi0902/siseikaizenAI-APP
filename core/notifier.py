"""core/notifier.py

「いつ通知を出すか」を判断する状態を持つ部品。
（設計メモ: notifier_py_設計メモ.md に対応）

第6回の通知タイミング: 一定時間の平均 → しきい値 → エッジ通知・クールダウン・ヒステリシス。
メインが渡す 1 フレームごとの点数をためて平均し、姿勢が崩れ続けているときだけ
plyer に通知を出させる。

責任:
  - 点数の移動平均を保つ
  - しきい値を割ったか判断する
  - 連発を防ぐ（エッジ・クールダウン・ヒステリシス）
  - plyer に通知を出させる（出口は1か所）

担わないこと:
  - 1フレームの採点 → scoring.py / 撮影〜採点 → workers
  - 画面表示 → ui / スコアの記録 → database.py

設計の核:
  - plyer は呼び鈴にすぎず、判断はここが持つ。
  - features/scoring が純粋関数だったのに対し、notifier は「状態を持つ」
    （平均の窓・現在の状態・前回鳴らした時刻を覚える）。
  - メインスレッドで動かす（ワーカーから直接呼ばない）。
"""

import time
from collections import deque

# core/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# ── plyer 通知の出口（差し替え可能にしておく） ──
def _plyer_notify(title, message):
    """実際にデスクトップ通知を出す既定の出口。

    plyer はここでだけ import する。これにより plyer 未導入の環境でも
    （テストで別の出口を渡せば）notifier 自体は動かせる。
    """
    from plyer import notification
    notification.notify(title=title, message=message, timeout=5)


class Notifier:
    """通知のタイミングを判断する状態機械。

    使い方: メインの after 取り出し処理が、点数1件ごとに update(score) を呼ぶ。
    """

    GOOD = "good"   # 良好（監視中）
    BAD = "bad"     # 悪化（通知済み・回復待ち）

    def __init__(
        self,
        window_seconds=None,
        min_samples=None,
        fire_threshold=None,
        recover_threshold=None,
        cooldown_seconds=None,
        notify_func=None,
        title=None,
        message=None,
    ):
        self.window_seconds = config.NOTIFY_WINDOW_SECONDS if window_seconds is None else window_seconds
        self.min_samples = config.NOTIFY_MIN_SAMPLES if min_samples is None else min_samples
        self.fire_threshold = config.NOTIFY_FIRE_THRESHOLD if fire_threshold is None else fire_threshold
        self.recover_threshold = config.NOTIFY_RECOVER_THRESHOLD if recover_threshold is None else recover_threshold
        self.cooldown_seconds = config.NOTIFY_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        self.title = config.NOTIFY_TITLE if title is None else title
        self.message = config.NOTIFY_MESSAGE if message is None else message

        # 鳴らす線は戻す線より低くする（ヒステリシス）。逆だと設計ミス。
        if self.recover_threshold < self.fire_threshold:
            raise ValueError(
                f"recover_threshold({self.recover_threshold}) は "
                f"fire_threshold({self.fire_threshold}) 以上にしてください（ヒステリシス）。"
            )

        # 通知の出口（テストでは偽の関数を渡して鳴動を記録できる）。
        self._notify_func = notify_func or _plyer_notify

        # ── 状態（notifier は状態を持つ） ──
        self._samples = deque()   # (時刻, 点数) の窓
        self.state = self.GOOD
        self.last_fire_time = None

    def update(self, score, now=None):
        """1フレームの点数を取り込み、条件を満たせば通知を 1 回出す。

        引数:
          score … 0〜100 の点数。採点不可なら None（scoring.UNSCORABLE）。
          now   … 現在時刻[秒]。省略時は実時間。テストでは注入して決定的にする。

        戻り値: このフレームで通知を出したら True、出さなければ False。
        """
        if now is None:
            now = time.monotonic()

        # 1. 採点不可は平均に入れない（誰もいない椅子に通知しないため）。
        #    None と 0.0 を区別するため is not None で判定する。
        if score is not None:
            self._samples.append((now, score))

        # 2. 窓から古いサンプル（window_seconds より前）を捨てる。
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        # 3. 十分たまっていなければ判断しない（立ち上がり直後・離席明けなど）。
        if len(self._samples) < self.min_samples:
            return False

        # 4. 移動平均で判断する（生の点数は震えるため）。
        avg = sum(s for _, s in self._samples) / len(self._samples)

        notified = False
        if self.state == self.GOOD:
            # 良好 → 平均が鳴らす線を下回ったら悪化へ。
            if avg < self.fire_threshold:
                self.state = self.BAD
                # エッジ（切り替わった瞬間）に1回だけ。ただしクールダウン中は鳴らさない。
                if self.last_fire_time is None or (now - self.last_fire_time) >= self.cooldown_seconds:
                    self._fire()
                    self.last_fire_time = now
                    notified = True
        else:  # BAD
            # 悪化中 → 平均が戻す線まで回復したら良好へ（ヒステリシス）。
            if avg >= self.recover_threshold:
                self.state = self.GOOD

        return notified

    def _fire(self):
        """通知を 1 回出す（出口は1か所）。"""
        self._notify_func(self.title, self.message)

    def reset(self):
        """平均・状態・前回時刻を初期化する。"""
        self._samples.clear()
        self.state = self.GOOD
        self.last_fire_time = None


# ── 単体での動作確認（点数列を流すだけ。plyer 不要・時刻は注入で決定的に） ──
def _self_test():
    UNSCORABLE = None  # scoring.UNSCORABLE と同じ

    def make():
        """テスト用の Notifier と、鳴動を記録するリストを返す。"""
        fired = []
        n = Notifier(
            window_seconds=10, min_samples=3,
            fire_threshold=70.0, recover_threshold=80.0, cooldown_seconds=100,
            notify_func=lambda title, msg: fired.append((title, msg)),
        )
        return n, fired

    def feed(n, score, count, t_start, dt=1):
        """score を count 回、時刻 t_start から dt 刻みで流す。最後の時刻を返す。"""
        t = t_start
        for _ in range(count):
            t += dt
            n.update(score, now=t)
        return t

    results = []

    # ① 高得点に一瞬の低得点 → 平均は下がりきらず鳴らない
    n, fired = make()
    feed(n, 90, 5, 0)
    n.update(20, now=6)  # 一瞬の崩れ
    results.append(("一瞬の崩れでは鳴らない", len(fired) == 0))

    # ② 低得点が続く → 1回だけ鳴る
    n, fired = make()
    feed(n, 90, 3, 0)
    feed(n, 30, 12, 3)
    results.append(("崩れが続くと1回だけ鳴る", len(fired) == 1))

    # ③ ヒステリシス: 鳴らす線(70)と戻す線(80)の間(75)でふらつく → 暴発しない
    n, fired = make()
    feed(n, 90, 3, 0)
    feed(n, 65, 4, 3)        # 一旦割って鳴る（1回）
    feed(n, 75, 4, 7)        # 70は超えるが80未満 → BADのまま（回復しない）
    feed(n, 65, 4, 11)       # また割っても再点火しない
    results.append(("境界のふらつきで暴発しない", len(fired) == 1))

    # ④ クールダウン: 回復→すぐ再悪化（クールダウン中）は鳴らない／明け後は鳴る
    n, fired = make()
    feed(n, 30, 5, 0)        # 悪化 → 1回鳴る（last_fire~3）
    t = feed(n, 95, 12, 5)   # 回復 → GOOD
    feed(n, 30, 5, t)        # 再悪化だが now<103 でクールダウン(100)中 → 鳴らない
    after = feed(n, 95, 12, 200)   # 時間を飛ばして回復
    feed(n, 30, 5, after)    # クールダウン明け＆GOODから再悪化 → 鳴る
    results.append(("クールダウン中は鳴らない・明けで再点火", len(fired) == 2))

    # ⑤ 採点不可は平均から外す（離席中は鳴らさない）
    n, fired = make()
    feed(n, 90, 3, 0)
    for k in range(10):      # ずっと採点不可 → 窓から実点数が消え、判断材料がなくなる
        n.update(UNSCORABLE, now=20 + k)
    results.append(("採点不可を平均から外す", len(fired) == 0))

    # 結果表示
    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("notifier 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
