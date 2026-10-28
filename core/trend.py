"""core/trend.py

記録されたスコアから「その人の姿勢の傾向」を読み取り、フィードバック文を作る。
（ホーム画面の「傾向フィードバック」から好きなときに呼ばれる）

責任:
  - 直近 N 日の記録を集計し、傾向の「事実」を dict で返す（analyze）
  - 事実を config のテンプレートに当てはめて文章にする（render）

担わないこと:
  - SQL → db/database.py（get_scores の生データを受け取るだけ）
  - 表示 → ui/app_window.py
  - リアルタイムの助言 → advice.py（こちらは期間全体の傾向）

設計の核:
  - 純粋関数。入力（記録の一覧・今の時刻・設定）だけで結果が決まる＝同じデータなら同じ文章。
  - 事実(analyze)と文章(render)を分ける。文言を変えても分析は触らない。
  - 崩れ方の内訳は「悪姿勢のときだけ」を数える（条件付き分布 P(原因 | 悪姿勢)）。
    平均をとると前に出た(−)と反った(＋)が打ち消し合って傾向が消えるため、
    1件ずつ「一番ずれた特徴と向き」を決めて、その頻度を数える。
  - 内訳列が NULL の過去データは、点数だけで出せる指標（平均・悪姿勢率・時間帯）にだけ使う。
"""

from datetime import datetime

import numpy as np
import pandas as pd

# core/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


def analyze(rows, now=None):
    """記録 → 傾向の事実 dict。

    rows … database.get_scores の出力 [{"timestamp", "score", "dev_<特徴名>"...}, ...]
    now  … 基準時刻（省略時は現在）。テストで固定できるよう引数にしている。

    戻り値（データ不足なら enough=False と件数だけ）:
      mean_score / prev_mean_score(比較不可なら None) / bad_ratio /
      cause(内訳不足なら None) / worst_hour(比較不可なら None)
    """
    now = pd.Timestamp(now or datetime.now())
    span = pd.Timedelta(days=config.TREND_DAYS)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"enough": False, "num_samples": 0}
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 今の期間 (now-N日, now] と、その前の期間 (now-2N日, now-N日]。
    cur = df[(df["timestamp"] > now - span) & (df["timestamp"] <= now)]
    prev = df[(df["timestamp"] > now - 2 * span) & (df["timestamp"] <= now - span)]

    if len(cur) < config.TREND_MIN_SAMPLES:
        return {"enough": False, "num_samples": len(cur)}

    is_bad = cur["score"] < config.SCORE_WARN
    return {
        "enough": True,
        "num_samples": len(cur),
        "mean_score": float(cur["score"].mean()),
        "prev_mean_score": (float(prev["score"].mean())
                            if len(prev) >= config.TREND_MIN_SAMPLES else None),
        "bad_ratio": float(is_bad.mean()),
        "cause": _main_cause(cur[is_bad]),
        "worst_hour": _worst_hour(cur),
    }


def _main_cause(bad):
    """悪姿勢の記録だけを見て、一番多い「崩れ方」(特徴, 向き) とその割合を返す。"""
    cols = [f"dev_{n}" for n in config.USE_FEATURES if f"dev_{n}" in bad.columns]
    if not cols:
        return None
    # 内訳が全部 NULL の行（内訳を記録する前の過去データ・体が一部しか映っていない）は除外。
    devs = bad[cols].astype(float).dropna(how="all")
    if len(devs) < config.TREND_MIN_CAUSE_SAMPLES:
        return None

    # 各行で |ずれ| が最大の特徴を選び、その符号で向きを決める（advice.suggest と同じ判定）。
    arr = devs.to_numpy()
    idx = np.nanargmax(np.abs(arr), axis=1)
    worst = arr[np.arange(len(arr)), idx]
    causes = pd.DataFrame({
        "feature": [cols[i][len("dev_"):] for i in idx],
        "direction": np.where(worst < 0, "low", "high"),
    })
    shares = causes.value_counts(normalize=True)  # 出現頻度 → 割合
    feature, direction = shares.idxmax()
    return {"feature": feature, "direction": direction,
            "share": float(shares.max()), "num_samples": len(devs)}


def _worst_hour(cur):
    """時刻(時)ごとの平均スコアが一番低い時間帯。比べる相手が無ければ None。"""
    g = cur.groupby(cur["timestamp"].dt.hour)["score"].agg(["mean", "count"])
    g = g[g["count"] >= config.TREND_MIN_HOUR_SAMPLES]  # 数件だけの時間帯で決めつけない
    if len(g) < 2:
        return None
    hour = g["mean"].idxmin()
    return {"hour": int(hour), "mean_score": float(g.loc[hour, "mean"])}


def render(facts):
    """事実 dict → フィードバック文（テンプレートに当てはめるだけ）。"""
    days = config.TREND_DAYS
    if not facts["enough"]:
        return (f"まだデータが足りません（直近{days}日で {facts['num_samples']}"
                f" / {config.TREND_MIN_SAMPLES} 件）。もう少し計測してから生成してください。")

    mean = facts["mean_score"]
    lines = [f"直近{days}日間の平均スコアは {mean:.0f} 点です。"]

    prev = facts["prev_mean_score"]
    if prev is None:
        lines.append(f"（その前の{days}日間の記録が少ないため、比較はまだできません）")
    elif mean - prev >= 1:
        lines.append(f"その前の{days}日間より {mean - prev:.0f} 点よくなっています。")
    elif prev - mean >= 1:
        lines.append(f"その前の{days}日間より {prev - mean:.0f} 点下がっています。")
    else:
        lines.append(f"その前の{days}日間とほぼ同じです。")

    lines.append(f"記録した時間のうち {facts['bad_ratio']:.0%} が"
                 f"崩れた姿勢（{config.SCORE_WARN:.0f}点未満）でした。")

    cause = facts["cause"]
    if cause is not None:
        label, tip = config.TREND_CAUSES.get(
            (cause["feature"], cause["direction"]),
            (cause["feature"], config.ADVICE_DEFAULT_MESSAGE))
        lines.append(f"崩れたときに一番多いのは「{label}」です（崩れた時間の {cause['share']:.0%}）。")
        lines.append(f"→ {tip}")
    elif facts["bad_ratio"] > 0:
        lines.append("崩れ方の内訳は、記録がもう少したまると表示されます。")

    hour = facts["worst_hour"]
    if hour is not None:
        lines.append(f"時間帯では {hour['hour']}時台が一番低めです（平均 {hour['mean_score']:.0f} 点）。"
                     "この時間の前に休憩やストレッチを入れてみましょう。")

    return "\n".join(lines)


def generate(rows, now=None):
    """記録 → フィードバック文（ホーム画面から呼ぶ入口）。"""
    return render(analyze(rows, now))


# ── 単体での動作確認（合成データ。DB・カメラ不要） ──
def _self_test():
    # モードに依存せず同じ検証ができるよう、横向きの設定に固定する（最後に戻す）。
    saved = (config.USE_FEATURES, config.TREND_CAUSES)
    config.USE_FEATURES = ["neck_angle", "trunk_angle", "kyphosis_angle"]
    config.TREND_CAUSES = config._SIDE_TREND_CAUSES
    try:
        now = datetime(2026, 10, 21, 18, 0, 0)

        def row(day, hour, i, score, neck=None, kyph=None):
            return {"timestamp": f"2026-10-{day:02d} {hour:02d}:{i // 60:02d}:{i % 60:02d}",
                    "score": score, "dev_neck_angle": neck,
                    "dev_trunk_angle": None, "dev_kyphosis_angle": kyph}

        rows = []
        # 今の期間: 10時台は良好 200件、15時台は悪姿勢 200件（首前 150 / 猫背 50）。
        rows += [row(20, 10, i, 90.0, neck=0.1, kyph=0.1) for i in range(200)]
        rows += [row(20, 15, i, 40.0, neck=-2.0, kyph=-0.5) for i in range(150)]
        rows += [row(20, 15, 150 + i, 40.0, neck=0.2, kyph=-1.8) for i in range(50)]
        # 内訳なしの過去形式の悪姿勢 100件 → 悪姿勢率には入るが内訳からは除外される。
        rows += [row(19, 15, i, 40.0) for i in range(100)]
        # 前の期間: 平均 50 点。
        rows += [row(10, 10, i, 50.0) for i in range(300)]

        facts = analyze(rows, now)
        assert facts["enough"] and facts["num_samples"] == 500, facts
        assert abs(facts["mean_score"] - 60.0) < 1e-9, facts      # (200*90+300*40)/500
        assert abs(facts["prev_mean_score"] - 50.0) < 1e-9, facts
        assert abs(facts["bad_ratio"] - 0.6) < 1e-9, facts         # 300/500
        cause = facts["cause"]
        assert (cause["feature"], cause["direction"]) == ("neck_angle", "low"), cause
        assert abs(cause["share"] - 0.75) < 1e-9 and cause["num_samples"] == 200, cause
        assert facts["worst_hour"]["hour"] == 15, facts

        text = render(facts)
        assert "頭が前に出る" in text and "10 点よくなっています" in text, text
        print(text)

        # データ不足 → 案内文。空でも落ちない。
        assert not analyze(rows[:10], now)["enough"]
        assert "足りません" in generate([], now)
        print("trend 自己テスト: すべて OK")
    finally:
        config.USE_FEATURES, config.TREND_CAUSES = saved


if __name__ == "__main__":
    _self_test()
