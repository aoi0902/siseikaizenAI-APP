"""ui/feedback_view.py

過去の計測（直近 count 回）のスコア推移を、1本のグラフにつなげて画面に出す部品。
（設計メモ: feedback_view_py_設計メモ.md に対応）

第7回の3段の役割分担をそのまま守る:
  整形 → pandas / 作図 → matplotlib / 埋め込み → PySide6(Qt)。

この部品は2つ:
  - FeedbackView     … 直近の計測のスコアを、つなげて1本の線で出す（大きめ）
  - AverageChartView … 回ごとの平均スコアを、点と線で出す（背は低め）

session_view（今回のスコア推移）との役割分担:
  - session_view  … 今この瞬間のスコアを追う（撮影中の画面）。
  - feedback_view … DB に貯めた過去の計測を、同じ見た目でつなげて振り返る。

責任:
  - スコアと計測の回を取り出す（get_scores / get_sessions 呼び出し）
  - pandas で回ごとに整える（古い順に並べ、直近 count 回に絞る）
  - matplotlib で折れ線を描く（回のあいだは詰めて、区切りに薄い縦線を入れる）
  - Qt のウィンドウに埋め込んで表示（別ウィンドウは開かない）

担わないこと:
  - SQL の実行 → db/database.py（生データを返すだけ）
  - 採点・通知・撮影・推定 → 各担当
  - 取り出し間隔・スレッド → app_window

設計の核:
  - 表示と軽い整形だけ。DB を触らない。
  - pyplot を使わず Figure を直接作る（別ウィンドウを開かないため）。
  - オンデマンド更新（毎フレームではない）。図は使い回し、ためない。
  - メインスレッドで描く（ワーカーから触らない）。

1回の数え方と見せ方:
  撮影を始めてから止めるまでが1回（scores の session_id で区別する）。
  session_id が無い古い記録は出さない。スコアが1件も残らなかった回も出さない。
  横軸の目盛りは、左端・右端・回の区切りの下に日時を出す。
    左端 = 一番古い回の開始 / 右端 = 一番新しい回の終了
    回の区切り = 前の回の終了（上）と、次の回の開始（下）
  回と回のあいだの空き時間は詰める（横軸は実際の時刻ではなく、計測した時間をつないだもの）。
  短い回は、区切りの日時の文字が重ならないよう、最低限の幅まで広げて描く。
  終了が記録されていない回（アプリが途中で落ちた等）は、最後のスコアの時刻を終了とする。
"""

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout

# ui/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

# 日本語ラベルが豆腐(□)にならないよう、Windows標準フォントを設定。
# 環境に無くても matplotlib が代替フォントへフォールバックするだけで落ちない。
matplotlib.rcParams["font.family"] = config.FEEDBACK_FONT
matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け回避

# 何回分を出すか（config に FEEDBACK_SESSIONS があればそれを使う。無ければ10回）。
_DEFAULT_COUNT = 10
# 回と回のあいだに空ける幅（全体の長さに対する割合）。
_GAP_RATIO = 0.03
# 短い回でも、目盛りの文字が重ならないよう、全体の長さに対するこの割合まで幅を広げる。
_MIN_SPAN_RATIO = 0.12


# ── 整形（pandas・純粋。Qt不要でテスト可能） ──
def prepare_sessions(scores, sessions=None, count=_DEFAULT_COUNT):
    """生データ → 回ごとの点の並び（古い順・直近 count 回）。

    scores   … [{timestamp, score, session_id}]
    sessions … [{id, started_at, ended_at}]（任意。無ければスコアの時刻で代用）
    戻り値   … [{"started", "ended", "x0", "x1", "xs", "ys"}, ...]。出せるデータが無ければ None。
      x は「計測した時間を、回のあいだを詰めてつないだ」秒数。
    """
    if not scores:
        return None
    df = pd.DataFrame(scores)
    # 古い記録など、回の番号が無いデータは使わない。
    if "session_id" not in df.columns:
        return None
    df = df[df["session_id"].notna()].copy()
    if df.empty:
        return None
    df["session_id"] = df["session_id"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 計測の回の開始・終了（sessions 表）を id で引けるようにする。
    meta_by_id = {int(s["id"]): s for s in (sessions or [])}

    # 古い順に並べ、直近 count 回に絞る。
    ids = sorted(df["session_id"].unique())[-count:]

    # 1周目: 各回の開始・終了・長さを決める。
    parts = []
    for sid in ids:
        rows = df[df["session_id"] == sid].sort_values("timestamp")
        first_ts = rows["timestamp"].iloc[0]
        last_ts = rows["timestamp"].iloc[-1]
        meta = meta_by_id.get(int(sid)) or {}
        started = pd.to_datetime(meta["started_at"]) if meta.get("started_at") else first_ts
        ended = pd.to_datetime(meta["ended_at"]) if meta.get("ended_at") else last_ts
        started = min(started, first_ts)   # スコアが開始より前になることは無いが念のため
        ended = max(ended, last_ts)        # 終了が空・古い場合は最後のスコアまで広げる
        span = max((ended - started).total_seconds(), 1.0)   # 長さ0でも幅を持たせる
        parts.append({"rows": rows, "started": started, "ended": ended, "span": span})

    # 2周目: 回のあいだを少しだけ空けて、左から順に並べる。
    # 短すぎる回は最低限の幅まで広げる（中の点も同じ割合で広げて、位置関係は保つ）。
    total = sum(p["span"] for p in parts)
    gap = max(1.0, total * _GAP_RATIO)
    min_width = total * _MIN_SPAN_RATIO
    result = []
    cursor = 0.0
    for i, p in enumerate(parts):
        # 両端の回は、端の日時の文字と区切りの文字が重ならないよう、さらに広めにする。
        is_end = i == 0 or i == len(parts) - 1
        width = max(p["span"], min_width * (1.6 if is_end else 1.0))
        scale = width / p["span"]
        offsets = (p["rows"]["timestamp"] - p["started"]).dt.total_seconds()
        result.append({
            "started": p["started"],
            "ended": p["ended"],
            "x0": cursor,
            "x1": cursor + width,
            "xs": [cursor + float(o) * scale for o in offsets],
            "ys": [float(v) for v in p["rows"]["score"]],
        })
        cursor += width + gap
    return result


class FeedbackView(QWidget):
    """過去の計測のスコア推移を、つなげて埋め込み表示するウィジェット部品。

    使い方:
        fv = FeedbackView(get_scores=lambda: db.get_scores(uid),
                          get_sessions=lambda: db.get_sessions(uid),
                          on_refresh=app.on_refresh_feedback)
        layout.addWidget(fv)
        # 更新ボタン、または app_window から refresh() で再描画
    """

    def __init__(self, get_scores, on_refresh=None, parent=None,
                 count=None, figure_size=None, get_sessions=None, show_button=True):
        super().__init__(parent)
        # 生データの取得口（app_window が database の関数を束ねて渡す）。
        self._get_scores = get_scores
        self._get_sessions = get_sessions   # 無ければスコアの時刻から開始・終了を決める
        self._on_refresh = on_refresh
        # 直近何回分を出すか。
        self.count = getattr(config, "FEEDBACK_SESSIONS", _DEFAULT_COUNT) if count is None else count
        figure_size = config.FEEDBACK_FIGURE_SIZE if figure_size is None else figure_size

        layout = QVBoxLayout(self)

        # 更新ボタン（押下で on_refresh を呼ぶ＝オンデマンド更新）。
        # 画面側が自前のボタンを置くときは show_button=False にして隠す。
        self._button = None
        if show_button:
            self._button = QPushButton("グラフを更新")
            self._button.clicked.connect(self._handle_refresh_click)
            layout.addWidget(self._button, alignment=Qt.AlignCenter)

        # 図は1つだけ作って使い回す（refresh のたびに作らない＝メモリ対策）。
        self._figure = Figure(figsize=figure_size, dpi=100)
        self._ax = self._figure.add_subplot(111)

        # Figure を Qt キャンバスへ橋渡し（別窓を開かない）。参照を保持する。
        self._canvas = FigureCanvasQTAgg(self._figure)
        # 最小高さを与え、レイアウトに押し潰されてグラフが見切れるのを防ぐ。
        self._canvas.setMinimumHeight(config.FEEDBACK_MIN_HEIGHT)
        layout.addWidget(self._canvas)

    # ── 更新ボタン押下 ──
    def _handle_refresh_click(self):
        # app_window 経由で再描画させる（配電盤を通す）。直接 refresh でも動くよう保険。
        if self._on_refresh is not None:
            self._on_refresh()
        else:
            self.refresh()

    # ── 整形は prepare_sessions に任せる（テストからも呼べるよう残してある） ──
    def _prepare(self, scores, sessions=None):
        return prepare_sessions(scores, sessions, self.count)

    # ── 再描画（取り出し→整形→作図→埋め込み） ──
    def refresh(self):
        """最新データで整形し直し、グラフを描き直す（オンデマンド）。"""
        scores = self._get_scores() if self._get_scores is not None else None
        sessions = self._get_sessions() if self._get_sessions is not None else None
        parts = self._prepare(scores, sessions)

        # 同じ Axes を使い回し、中身だけ描き直す（図をためない）。
        self._ax.clear()

        if parts is None or len(parts) == 0:
            # データ無し: 落ちずに案内を出す。
            self._ax.text(0.5, 0.5, "まだ記録がありません",
                          ha="center", va="center", transform=self._ax.transAxes)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
        else:
            for i, part in enumerate(parts):
                # 回ごとに別々の線で描く（回のあいだは線でつながない）。
                only_one = len(part["xs"]) == 1
                self._ax.plot(part["xs"], part["ys"], color="#1565c0", linewidth=1.5,
                              marker="o" if only_one else None, markersize=3)
                # 回の区切りに薄い縦線を入れる。
                if i > 0:
                    boundary = (parts[i - 1]["x1"] + part["x0"]) / 2
                    self._ax.axvline(boundary, color="#9e9e9e", linewidth=0.8, alpha=0.6)

            # 良好(緑)・注意(黄)の目安ライン（撮影中のグラフと同じ）。
            self._ax.axhline(config.SCORE_GOOD, color="#2e7d32", linestyle="--",
                             linewidth=0.8, alpha=0.6)
            self._ax.axhline(config.SCORE_WARN, color="#f9a825", linestyle="--",
                             linewidth=0.8, alpha=0.6)

            # 横軸の目盛り: 左端=最初の開始 / 右端=最後の終了 /
            # 回の区切り=前の回の終了（上）と次の回の開始（下）。
            x_min, x_max = parts[0]["x0"], parts[-1]["x1"]
            ticks = [x_min]
            tick_labels = [f"{parts[0]['started']:%m/%d %H:%M}"]
            for i in range(1, len(parts)):
                ticks.append((parts[i - 1]["x1"] + parts[i]["x0"]) / 2)
                tick_labels.append(f"{parts[i - 1]['ended']:%m/%d %H:%M}\n"
                                   f"{parts[i]['started']:%m/%d %H:%M}")
            ticks.append(x_max)
            tick_labels.append(f"{parts[-1]['ended']:%m/%d %H:%M}")
            self._ax.set_xticks(ticks)
            self._ax.set_xticklabels(tick_labels, fontsize=8)   # 文字を小さめにして重なりにくくする
            labels = self._ax.get_xticklabels()
            labels[0].set_ha("left")     # 左端の文字は右へ、右端の文字は左へ寄せて見切れを防ぐ
            labels[-1].set_ha("right")
            self._ax.set_xlim(x_min, x_max)

            self._ax.set_title(f"姿勢スコア推移（直近{len(parts)}回）")
            self._ax.set_ylabel("スコア")
            self._ax.set_ylim(0, 100)            # スコアは0〜100に固定
            self._ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()                      # メインスレッドで描画


class AverageChartView(QWidget):
    """回ごとの平均スコアを、点と線で出す背の低いグラフ（部品）。

    フィードバック画面と、傾向フィードバック画面の両方で使う。
    使い方:
        av = AverageChartView(get_scores=lambda: db.get_scores(uid),
                              get_sessions=lambda: db.get_sessions(uid))
        layout.addWidget(av)
        av.refresh()   # 最新データで描き直す
    """

    def __init__(self, get_scores, get_sessions=None, parent=None,
                 count=None, figure_size=None):
        super().__init__(parent)
        self._get_scores = get_scores
        self._get_sessions = get_sessions
        self.count = getattr(config, "FEEDBACK_SESSIONS", _DEFAULT_COUNT) if count is None else count
        figure_size = (getattr(config, "FEEDBACK_AVG_FIGURE_SIZE", (5.0, 1.8))
                       if figure_size is None else figure_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 図は1つだけ作って使い回す（描き直しのたびに作らない＝メモリ対策）。
        self._figure = Figure(figsize=figure_size, dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        # 背を低めにしつつ、見切れない最小の高さを持たせる。
        self._canvas.setMinimumHeight(getattr(config, "FEEDBACK_AVG_MIN_HEIGHT", 150))
        layout.addWidget(self._canvas)

    def _averages(self, parts):
        """回ごとの平均スコアのリスト（古い順）。"""
        return [sum(p["ys"]) / len(p["ys"]) for p in parts]

    def refresh(self):
        """最新データで整形し直し、グラフを描き直す（オンデマンド）。"""
        scores = self._get_scores() if self._get_scores is not None else None
        sessions = self._get_sessions() if self._get_sessions is not None else None
        parts = prepare_sessions(scores, sessions, self.count)

        self._ax.clear()

        if parts is None or len(parts) == 0:
            self._ax.text(0.5, 0.5, "まだ記録がありません",
                          ha="center", va="center", transform=self._ax.transAxes)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
        else:
            means = self._averages(parts)
            xs = list(range(len(parts)))
            self._ax.plot(xs, means, color="#1565c0", linewidth=1.8, marker="o", markersize=5)

            # 点の近くに平均の数字を添える（上端に近いときは点の下に出す）。
            for x, v in zip(xs, means):
                offset = -13 if v > 85 else 6
                self._ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                                  xytext=(0, offset), ha="center", fontsize=8)

            # 良好(緑)・注意(黄)の目安ライン（ほかのグラフと同じ）。
            self._ax.axhline(config.SCORE_GOOD, color="#2e7d32", linestyle="--",
                             linewidth=0.8, alpha=0.6)
            self._ax.axhline(config.SCORE_WARN, color="#f9a825", linestyle="--",
                             linewidth=0.8, alpha=0.6)

            # 横軸は、各回の開始の日付と時刻。
            self._ax.set_xticks(xs)
            self._ax.set_xticklabels([f"{p['started']:%m/%d}\n{p['started']:%H:%M}" for p in parts],
                                     fontsize=8)
            self._ax.set_xlim(-0.5, len(parts) - 0.5)
            self._ax.set_title(f"回ごとの平均スコア（直近{len(parts)}回）", fontsize=10)
            self._ax.set_ylabel("平均")
            self._ax.set_ylim(0, 100)
            self._ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()                      # メインスレッドで描画


# ── 単体での動作確認 ──
def _self_test():
    from datetime import datetime, timedelta
    from PySide6.QtWidgets import QApplication

    results = []

    def stamp(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # にせデータ: 1回目（10/29 10:00〜10:10・3件）、2回目（10/30 09:00〜09:02・2件）、
    #            回の番号が無い古い記録（使わない）。
    t1 = datetime(2026, 10, 29, 10, 0, 0)
    t2 = datetime(2026, 10, 30, 9, 0, 0)
    fake_scores = [
        {"timestamp": stamp(t1 + timedelta(seconds=5)), "score": 60.0, "session_id": 1},
        {"timestamp": stamp(t1 + timedelta(seconds=300)), "score": 70.0, "session_id": 1},
        {"timestamp": stamp(t1 + timedelta(seconds=595)), "score": 80.0, "session_id": 1},
        {"timestamp": stamp(t2 + timedelta(seconds=10)), "score": 85.0, "session_id": 2},
        {"timestamp": stamp(t2 + timedelta(seconds=60)), "score": 90.0, "session_id": 2},
        {"timestamp": stamp(t2), "score": 10.0, "session_id": None},   # 古い記録
    ]
    fake_sessions = [
        {"id": 1, "started_at": stamp(t1), "ended_at": stamp(t1 + timedelta(seconds=600))},
        {"id": 2, "started_at": stamp(t2), "ended_at": stamp(t2 + timedelta(seconds=120))},
    ]

    app = QApplication.instance() or QApplication([])
    fv = FeedbackView(get_scores=lambda: fake_scores, get_sessions=lambda: fake_sessions)

    # ① 回ごとに分かれ、古い記録は除かれ、スコアが元のまま並ぶ
    parts = fv._prepare(fake_scores, fake_sessions)
    ok_split = (len(parts) == 2
                and parts[0]["ys"] == [60.0, 70.0, 80.0]
                and parts[1]["ys"] == [85.0, 90.0])
    results.append(("回ごとに分かれ、スコアがそのまま並ぶ(古い記録は除外)", ok_split))

    # ② 回の中は開始からの秒数、回のあいだは詰めて少しだけ空く
    ok_x = (parts[0]["xs"][0] == 5.0 and parts[0]["x1"] == 600.0
            and parts[1]["x0"] > parts[0]["x1"]
            and parts[1]["x0"] < parts[0]["x1"] + 60          # 1日空いても数十秒しか空かない
            # 2回目は短い（120秒）ので、最低限の幅まで広がる。中の点も同じ割合で広がる。
            and abs(parts[1]["xs"][0] - (parts[1]["x0"] + 10.0 * (parts[1]["x1"] - parts[1]["x0"]) / 120.0)) < 1e-6)
    results.append(("回の中は実時間・回のあいだは詰める", ok_x))

    # ③ 左端は最初の開始、右端は最後の終了
    ok_ends = (parts[0]["started"] == pd.Timestamp(t1)
               and parts[-1]["ended"] == pd.Timestamp(t2 + timedelta(seconds=120)))
    results.append(("左端=最初の開始 / 右端=最後の終了", ok_ends))

    # ③' 目盛り: 左端=開始 / 区切り=前の終了と次の開始(2行) / 右端=終了
    fv.refresh()
    texts = [t.get_text() for t in fv._ax.get_xticklabels()]
    results.append(("区切りに前の終了と次の開始が出る",
                    texts == ["10/29 10:00", "10/29 10:10\n10/30 09:00", "10/30 09:02"]))

    # ④ 終了が空の回（落ちた回）は、最後のスコアの時刻が終了になる
    open_sessions = [{"id": 1, "started_at": stamp(t1), "ended_at": None},
                     {"id": 2, "started_at": stamp(t2), "ended_at": stamp(t2 + timedelta(seconds=120))}]
    parts_open = fv._prepare(fake_scores, open_sessions)
    results.append(("終了が空なら最後のスコアの時刻を終了にする",
                    parts_open[0]["ended"] == pd.Timestamp(t1 + timedelta(seconds=595))))

    # ⑤ sessions が無くても動く（スコアの時刻から決める）
    parts_nosess = fv._prepare(fake_scores, None)
    results.append(("sessionsが無くても整形できる",
                    len(parts_nosess) == 2 and parts_nosess[0]["started"] == pd.Timestamp(t1 + timedelta(seconds=5))))

    # ⑥ 回の番号が無い記録だけ / 列が無い / 空 → None
    today = stamp(t2)
    old_only = [{"timestamp": today, "score": 50.0, "session_id": None}]
    no_column = [{"timestamp": today, "score": 50.0}]
    results.append(("回の番号が無い・空ならNone",
                    fv._prepare(old_only) is None and fv._prepare(no_column) is None
                    and fv._prepare([]) is None))

    # ⑦ 直近 count 回に絞る: 12回あっても10回だけ（古い2回が落ちる）
    many = [{"timestamp": stamp(t1 + timedelta(hours=i)), "score": 50.0 + i, "session_id": i}
            for i in range(1, 13)]
    last10 = fv._prepare(many)
    results.append(("直近10回に絞る",
                    len(last10) == 10 and last10[0]["started"] == pd.Timestamp(t1 + timedelta(hours=3))))

    # ⑦' 短い回は最低限の幅まで広げ、中の点は同じ割合で広げる
    long_short = [
        {"timestamp": stamp(t1 + timedelta(seconds=100)), "score": 70.0, "session_id": 1},
        {"timestamp": stamp(t1 + timedelta(seconds=9000)), "score": 80.0, "session_id": 1},
        {"timestamp": stamp(t2 + timedelta(seconds=30)), "score": 60.0, "session_id": 2},
    ]
    long_short_sessions = [
        {"id": 1, "started_at": stamp(t1), "ended_at": stamp(t1 + timedelta(seconds=10000))},
        {"id": 2, "started_at": stamp(t2), "ended_at": stamp(t2 + timedelta(seconds=60))},
    ]
    ls = fv._prepare(long_short, long_short_sessions)
    total_real = 10000.0 + 60.0
    ok_min = ((ls[1]["x1"] - ls[1]["x0"]) >= total_real * _MIN_SPAN_RATIO - 1e-6
              and ls[1]["x0"] <= ls[1]["xs"][0] <= ls[1]["x1"]
              and abs((ls[1]["xs"][0] - ls[1]["x0"]) / (ls[1]["x1"] - ls[1]["x0"]) - 0.5) < 1e-6)
    results.append(("短い回は最低限の幅まで広げる(点の位置関係は保つ)", ok_min))

    # ⑧ 描画が通る（別窓を開かない）／1回・1件だけでも描ける
    fv.refresh()
    fv_one = FeedbackView(get_scores=lambda: [{"timestamp": today, "score": 75.0, "session_id": 1}])
    fv_one.refresh()
    results.append(("グラフ描画が通る(1件だけでも可)", True))

    # ⑨ 図を使い回す: 何度 refresh しても Figure は同一（ためない）
    fig_id = id(fv._figure)
    fv.refresh()
    fv.refresh()
    results.append(("更新で図がたまらない(Figure使い回し)", id(fv._figure) == fig_id))

    # ⑩ データ無しの refresh も安全
    fv2 = FeedbackView(get_scores=lambda: [])
    fv2.refresh()
    results.append(("データ無しのrefreshも安全", True))

    # ⑪ 回ごとの平均グラフ: 平均が手計算と一致し、描画も通る
    av = AverageChartView(get_scores=lambda: fake_scores, get_sessions=lambda: fake_sessions)
    means = av._averages(prepare_sessions(fake_scores, fake_sessions))
    results.append(("回ごとの平均が手計算と一致(1回目70/2回目87.5)", means == [70.0, 87.5]))
    av.refresh()
    results.append(("平均グラフの描画が通る", True))

    # ⑫ 平均グラフ: データ無し・1回だけでも落ちない
    AverageChartView(get_scores=lambda: []).refresh()
    AverageChartView(get_scores=lambda: [{"timestamp": today, "score": 75.0, "session_id": 1}]).refresh()
    results.append(("平均グラフはデータ無し・1回だけでも安全", True))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("feedback_view 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
