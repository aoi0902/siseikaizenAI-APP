"""analysis/summarize_experiment_a.py

実験A（キャリブレーションの再現性）の集計。baseline テーブルから `S1_A1` 形式の名前の行を拾い、
被験者ごと・特徴量ごとに 平均・標準偏差・範囲 を出す。揃っている被験者の分だけ集計するので、
途中（2人分など）で実行してもよい。

使い方:
  python analysis/summarize_experiment_a.py
  python analysis/summarize_experiment_a.py --self-test

出力（DB には書き込まない。DB は読み取り専用で開く）:
  画面の表 と data/analysis/experiment_a.csv

判定の理論:
  標準偏差 SD を「満点を保つ幅」= DEADBAND × 許容幅 と比べる。測り直すたびに基準値がこの幅を
  超えて動くと、同じ姿勢でもキャリブレーションのたびに点数が変わるため。
  SD は標本標準偏差（n−1 で割る）。5回という少ない標本から母集団のばらつきを推定するため。
"""

import os
import re
import sqlite3
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402

FEATURES = ["neck_angle", "trunk_angle", "kyphosis_angle"]
NAME_PATTERN = r"(S\d+)_A(\d+)"
OUT_PATH = os.path.join(ROOT, "data", "analysis", "experiment_a.csv")
TRUNK_PRIOR_SD = 10.2  # 先行研究の trunk_angle の個人内ばらつき[度]（評価実験計画案 3節）


def load_trials(db_path=config.DB_PATH):
    """`S?_A?` 形式の baseline を読む。同じ名前が複数あれば最後に保存した行を使う。"""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = pd.read_sql_query(
            f"SELECT id, name, measured_at, {', '.join(FEATURES)} FROM baseline ORDER BY id", conn)
    rows = rows[rows["name"].fillna("").str.fullmatch(NAME_PATTERN)].copy()
    rows[["subject", "trial"]] = rows["name"].str.extract(NAME_PATTERN)
    rows["trial"] = rows["trial"].astype(int)
    return rows.drop_duplicates("name", keep="last").sort_values(["subject", "trial"])


def summarize(trials):
    """被験者 × 特徴量 で 回数・平均・SD・範囲 と、満点を保つ幅との比較を返す。"""
    long = trials.melt(id_vars=["subject"], value_vars=FEATURES, var_name="feature").dropna()
    stats = long.groupby(["subject", "feature"], sort=False)["value"].agg(
        n="count", mean="mean", sd="std", min="min", max="max").reset_index()
    stats["range"] = stats["max"] - stats["min"]
    stats["limit"] = stats["feature"].map(lambda f: config.DEADBAND * config.TOLERANCE[f])
    stats["sd_ok"] = stats["sd"] < stats["limit"]
    return stats.drop(columns=["min", "max"])


def main():
    trials = load_trials()
    if trials.empty:
        print("S?_A? 形式のキャリブレーションがありません。")
        return
    print("■ 試行一覧")
    print(trials[["subject", "trial", "measured_at", *FEATURES]].round(1).to_string(index=False))
    counts = trials.groupby("subject")["trial"].count()
    short = counts[counts < 5]
    if not short.empty:
        print(f"\n※ 5回に満たない被験者: {dict(short)}")

    stats = summarize(trials)
    print("\n■ 被験者 × 特徴量（sd_ok = SD < 満点を保つ幅 limit）")
    print(stats.round(2).to_string(index=False))
    trunk = stats[stats["feature"] == "trunk_angle"]
    print(f"\n参考: trunk_angle の SD と先行研究の個人内ばらつき {TRUNK_PRIOR_SD}°")
    print(trunk[["subject", "sd"]].round(2).to_string(index=False))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    stats.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")  # Excel で文字化けしないよう BOM 付き
    print(f"\n保存: {OUT_PATH}")


def _self_test():
    trials = pd.DataFrame({
        "subject": ["S1"] * 3 + ["S2"] * 2,
        "neck_angle": [70.0, 72.0, 74.0, 60.0, 80.0],
        "trunk_angle": [10.0, 10.0, 10.0, 5.0, None],
        "kyphosis_angle": [150.0] * 5,
    })
    s = summarize(trials).set_index(["subject", "feature"])
    assert s.loc[("S1", "neck_angle"), "mean"] == 72.0
    assert s.loc[("S1", "neck_angle"), "sd"] == 2.0           # n−1 で割る
    assert s.loc[("S1", "neck_angle"), "range"] == 4.0
    assert bool(s.loc[("S1", "neck_angle"), "sd_ok"])          # 2.0 < 9.75
    assert not bool(s.loc[("S2", "neck_angle"), "sd_ok"])      # 14.1 > 9.75
    assert s.loc[("S2", "trunk_angle"), "n"] == 1               # 欠損は数えない
    assert re.fullmatch(NAME_PATTERN, "S1_A3") and not re.fullmatch(NAME_PATTERN, "姿勢_1118_0953")
    print("self-test OK")


if __name__ == "__main__":
    _self_test() if "--self-test" in sys.argv else main()
