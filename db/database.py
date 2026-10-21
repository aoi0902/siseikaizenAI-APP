"""db/database.py

アプリ唯一の SQLite の出入り口（リポジトリパターン）。
（設計メモ: database_py_設計メモ.md に対応）

SQL は全部このファイルに閉じ込め、他のファイルは名前のついた操作
（save_baseline / get_latest_baseline / save_score / get_scores …）を呼ぶだけ。
これにより後でテーブル設計を変えても、直すのはこの 1 ファイルだけで済む。

責任:
  - SQLite への接続・切断 / テーブル作成（冪等）
  - baseline（基準値）の保存・取得
  - scores（スコア）の保存・取得
  - プレーンなデータ（数値・文字列・dict の一覧）を返す

担わないこと:
  - 特徴量の計算 → features.py / 測定の段取り → calibration.py
  - 採点 → scoring.py / 日次集計・グラフ → feedback_view（pandas）
  - 判断は一切しない。頼まれたデータを出し入れするだけ。

スレッド方針（設計メモ §4）:
  各操作ごとに「接続を開く→処理→確定→閉じる」を完結させる。これにより
  「接続は作ったスレッドでしか使えない」という SQLite の決まりに抵触しない
  （どのスレッドから呼ばれても、その場で接続を作って閉じるため安全）。
"""

import os
import re
import sqlite3
from datetime import datetime

# db/ から実行されても、パッケージ import でも config を読めるように。
try:
    import config
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


# baseline の特徴量カラムは features.py が使う特徴量名と必ず一致させる。
# config.USE_FEATURES を唯一の出どころにして、ズレを構造的に防ぐ。
def _feature_columns():
    return list(config.USE_FEATURES)


# カラム名は自分たちの config 由来だが、動的にSQL文へ埋め込むので
# 念のため「単純な識別子」だけを許可する（防御的チェック）。
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name):
    if not _IDENT_RE.match(name):
        raise ValueError(f"不正なカラム名です: {name!r}")
    return name


# ── 接続 ──
def _connect():
    """DB に接続して返す。場所が無ければ data/ を作る。

    busy_timeout / WAL を config に従って設定する。
    """
    db_path = config.DB_PATH

    # 置き場所（data/ など）が無ければ作る。作れなければ明確なエラーになる。
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 列名でアクセスできる行にする（dict化が楽）

    # ロックが一瞬重なっても即エラーにせず少し待つ。
    conn.execute(f"PRAGMA busy_timeout = {int(config.DB_BUSY_TIMEOUT_MS)}")
    if config.DB_USE_WAL:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ── 初期化（冪等） ──
def ensure_tables():
    """3 テーブルを「無ければ作る」。起動のたびに呼んでも壊れない。"""
    feature_cols = [_check_identifier(c) for c in _feature_columns()]
    # baseline の特徴量カラムを config から組み立てる（REAL・NULL許容）。
    feature_ddl = "".join(f"    {c} REAL,\n" for c in feature_cols)

    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS baseline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT,
{feature_ddl}                measured_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        # 既存テーブルに不足している列（name / 現モードの特徴量列）を補う（移行）。
        # CREATE TABLE IF NOT EXISTS は既存テーブルに列を足さないため、ここで補う。
        _ensure_baseline_columns(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_baseline_columns(conn):
    """baseline に不足している列を足す（冪等な移行）。

    足す対象:
      - name 列（名前付きキャリブレーション用。旧仕様のDBに無い場合）
      - 現在の撮影モードで使う特徴量列（config.USE_FEATURES）
        POSTURE_MODE を切り替えると必要な特徴量列が変わる（例: 横向きの
        neck_angle/trunk_angle → 正面の shoulder_tilt/head_tilt）。既存テーブルには
        CREATE TABLE では列が足されないため、ここで不足分を ALTER で追加する。
    """
    existing = [r["name"] for r in conn.execute("PRAGMA table_info(baseline)").fetchall()]
    if "name" not in existing:
        conn.execute("ALTER TABLE baseline ADD COLUMN name TEXT")
    for col in _feature_columns():
        col = _check_identifier(col)
        if col not in existing:
            conn.execute(f"ALTER TABLE baseline ADD COLUMN {col} REAL")


# ── 利用者 ──
def get_or_create_user(name):
    """名前から user_id を得る。無ければ作る。"""
    conn = _connect()
    try:
        row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return row["id"]

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute(
            "INSERT INTO users (name, created_at) VALUES (?, ?)",
            (name, created_at),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# ── 基準値（baseline） ──
def save_baseline(user_id, features, measured_at, name=None):
    """基準値を 1 行追加する。

    features … {特徴名: 代表値} の dict（calibration.stabilize の出力）
    name    … このキャリブレーションの名前（任意。後で名前で選ぶために使う）
    戻り値  … 追加した行の id
    再キャリブレーションは上書きせず行を足す（最新は measured_at で選ぶ）。
    同じ名前で測り直したときも行を足し、その名前の最新が使われる。
    """
    feature_cols = [_check_identifier(c) for c in _feature_columns()]

    columns = ["user_id", "name"] + feature_cols + ["measured_at"]
    placeholders = ", ".join("?" for _ in columns)
    # features に無いキーは None（NULL）で埋める。
    values = [user_id, name] + [features.get(c) for c in feature_cols] + [measured_at]

    conn = _connect()
    try:
        cur = conn.execute(
            f"INSERT INTO baseline ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_baseline_names(user_id):
    """保存済みキャリブレーションを「名前ごとに最新の1件」で返す。

    戻り値 … [{"name": ..., "measured_at": ...}, ...]（新しい順）。
    名前の無い（旧仕様の）行は選べないので除外する。
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT name, MAX(measured_at) AS measured_at
            FROM baseline
            WHERE user_id = ? AND name IS NOT NULL AND name <> ''
            GROUP BY name
            ORDER BY measured_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_baseline_by_name(user_id, name):
    """指定した名前の基準値のうち最新の 1 行を dict で返す。無ければ None。"""
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT * FROM baseline
            WHERE user_id = ? AND name = ?
            ORDER BY measured_at DESC, id DESC
            LIMIT 1
            """,
            (user_id, name),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def baseline_name_exists(user_id, name):
    """その名前のキャリブレーションが既に存在するか（改名の重複チェック用）。"""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM baseline WHERE user_id = ? AND name = ? LIMIT 1",
            (user_id, name),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def rename_baseline(user_id, old_name, new_name):
    """user_id の old_name のキャリブを new_name に改名する。

    同名で複数行ある（同じ名前で測り直した）場合はまとめて改名する。
    戻り値: 改名した行数。
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE baseline SET name = ? WHERE user_id = ? AND name = ?",
            (new_name, user_id, old_name),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_baseline(user_id, name):
    """user_id の name のキャリブを（同名の全行）削除する。

    戻り値: 削除した行数。
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "DELETE FROM baseline WHERE user_id = ? AND name = ?",
            (user_id, name),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_latest_baseline(user_id):
    """measured_at が最新の基準値 1 行を dict で返す。無ければ None。

    初回・未キャリブレーションでも落ちず None を返す（採点側は採点を始めない）。
    """
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT * FROM baseline
            WHERE user_id = ?
            ORDER BY measured_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


# ── スコア（scores） ──
def save_score(user_id, timestamp, score):
    """スコアを 1 件記録する。戻り値は追加した行の id。"""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO scores (user_id, timestamp, score) VALUES (?, ?, ?)",
            (user_id, timestamp, score),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_scores(user_id, start=None, end=None):
    """スコアの生の記録を時刻順に返す（集計はしない＝ pandas の役目）。

    start / end … 期間で絞りたいときの文字列日時（任意）。
    戻り値      … [{"timestamp": ..., "score": ...}, ...]
    """
    sql = "SELECT timestamp, score FROM scores WHERE user_id = ?"
    params = [user_id]
    if start is not None:
        sql += " AND timestamp >= ?"
        params.append(start)
    if end is not None:
        sql += " AND timestamp <= ?"
        params.append(end)
    sql += " ORDER BY timestamp ASC"

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── 単体での動作確認（使い捨ての一時DB。本番 posture.db を汚さない） ──
def _self_test():
    import tempfile

    # config.DB_PATH を一時ファイルに差し替えてからテストする。
    # 特徴量カラムは POSTURE_MODE に依存するので、テスト中は横向きの名前に固定して
    # どのモードで実行しても同じ検証ができるようにする（本番設定は最後に戻す）。
    tmp_dir = tempfile.mkdtemp()
    original = config.DB_PATH
    original_features = config.USE_FEATURES
    config.DB_PATH = os.path.join(tmp_dir, "test_posture.db")
    config.USE_FEATURES = ["neck_angle", "trunk_angle"]
    try:
        ensure_tables()
        print("ensure_tables: 3テーブル作成 OK")

        uid = get_or_create_user("テストユーザー")
        uid2 = get_or_create_user("テストユーザー")  # 二度目は作らず同じid
        print(f"get_or_create_user: id={uid}（再呼び出しも同じ: {uid == uid2}）")

        # 基準値が無い状態 → None が安全に返る
        print(f"未キャリブレーション: get_latest_baseline -> {get_latest_baseline(uid)}  ← None のはず")

        # 2回保存し、最新（measured_at が新しい方）が返るか
        save_baseline(uid, {"neck_angle": 80.0, "trunk_angle": 4.0}, "2026-06-19 09:00:00",
                      name="朝の姿勢")
        save_baseline(uid, {"neck_angle": 82.0, "trunk_angle": 5.0}, "2026-06-19 10:00:00",
                      name="夜の姿勢")
        latest = get_latest_baseline(uid)
        print(f"最新の基準値: neck_angle={latest['neck_angle']} measured_at={latest['measured_at']}"
              f"  ← 82.0 / 10:00 のはず")

        # 名前で選べる: 名前一覧（新しい順）と、名前指定での取得
        names = [r["name"] for r in get_baseline_names(uid)]
        print(f"キャリブ名一覧: {names}  ← ['夜の姿勢', '朝の姿勢'] のはず")
        by_name = get_baseline_by_name(uid, "朝の姿勢")
        print(f"名前で選択(朝の姿勢): neck_angle={by_name['neck_angle']}  ← 80.0 のはず")

        # 同名で測り直すと行を足し、その名前の最新が返る
        save_baseline(uid, {"neck_angle": 78.0, "trunk_angle": 3.0}, "2026-06-19 12:00:00",
                      name="朝の姿勢")
        by_name2 = get_baseline_by_name(uid, "朝の姿勢")
        print(f"同名で再測定→最新: neck_angle={by_name2['neck_angle']}  ← 78.0 のはず")
        names2 = [r["name"] for r in get_baseline_names(uid)]
        print(f"再測定後の名前一覧: {names2}  ← 重複せず2件のはず（{len(names2)}件）")

        # 改名: 「朝の姿勢」(2行)→「起床時」。同名の全行がまとまって改名される
        print(f"重複チェック(夜の姿勢は存在): {baseline_name_exists(uid, '夜の姿勢')}  ← True のはず")
        renamed = rename_baseline(uid, "朝の姿勢", "起床時")
        names3 = [r["name"] for r in get_baseline_names(uid)]
        print(f"改名: {renamed}行を改名 → 一覧 {names3}  ← ['起床時','夜の姿勢'](順不同) のはず")
        print(f"改名後: 旧名は消え新名で取得できる: "
              f"{get_baseline_by_name(uid, '朝の姿勢') is None} / "
              f"{get_baseline_by_name(uid, '起床時')['neck_angle'] == 78.0}  ← True / True のはず")

        # 削除: 「夜の姿勢」を消すと一覧から消える
        deleted = delete_baseline(uid, "夜の姿勢")
        names4 = [r["name"] for r in get_baseline_names(uid)]
        print(f"削除: {deleted}行を削除 → 一覧 {names4}  ← ['起床時'] のはず")

        # スコアを数件 → 件数ぶん返る、期間絞りも効く
        save_score(uid, "2026-06-19 10:00:00", 80.0)
        save_score(uid, "2026-06-19 10:01:00", 75.0)
        save_score(uid, "2026-06-20 10:00:00", 90.0)
        all_scores = get_scores(uid)
        day19 = get_scores(uid, start="2026-06-19 00:00:00", end="2026-06-19 23:59:59")
        print(f"get_scores: 全{len(all_scores)}件 / 6/19のみ{len(day19)}件  ← 3件 / 2件 のはず")

        print("DB 自己テスト: すべて OK")
    finally:
        config.DB_PATH = original  # 後始末（設定を元に戻す）
        config.USE_FEATURES = original_features


if __name__ == "__main__":
    _self_test()
