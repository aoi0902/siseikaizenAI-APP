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

import json
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


# scores に付ける「特徴ごとの符号付きずれ」の列名（例: dev_neck_angle）。
# 傾向分析（core/trend.py）が「どこが・どちら向きに崩れたか」を読むために使う。
def _deviation_column(feature_name):
    return _check_identifier(f"dev_{feature_name}")


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
    """4 テーブル（users / baseline / scores / landmarks）を「無ければ作る」。起動のたびに呼んでも壊れない。"""
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
        _ensure_score_columns(conn)
        # 生の座標（研究用の原本）。1フレーム＝1行で、33点は JSON にまとめる。
        # 使い道は「1フレーム分をまとめて extract_features に通し直す」だけなので、
        # 読み出す単位（フレーム）で保存するのが素直（1点1行の縦持ちだと33倍の行になる）。
        # 縦横比の補正を後から再現できるよう、フレームの実サイズも残す。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS landmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                baseline_id INTEGER,
                score_id INTEGER,
                captured_at TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                points TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (baseline_id) REFERENCES baseline(id),
                FOREIGN KEY (score_id) REFERENCES scores(id)
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


def _ensure_score_columns(conn):
    """scores に現モードの「符号付きずれ」列を足す（冪等な移行）。

    既存の行は NULL のまま残る（内訳を記録する前のデータ）。
    傾向分析では、点数だけで出せる指標には使い、崩れ方の内訳からは除外する。
    """
    existing = [r["name"] for r in conn.execute("PRAGMA table_info(scores)").fetchall()]
    for name in _feature_columns():
        col = _deviation_column(name)
        if col not in existing:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col} REAL")


# ── 生の座標（landmarks） ──
# frame は dict: {"captured_at": "YYYY-MM-DD HH:MM:SS", "points": 33点(各 x,y,z,visibility),
#                 "width": 画像の幅, "height": 画像の高さ}
def _points_json(points):
    """33点を [[x, y, z, visibility], ...] の JSON 文字列にする（小数 LANDMARK_DECIMALS 桁に丸める）。"""
    d = config.LANDMARK_DECIMALS
    return json.dumps([[round(p.x, d), round(p.y, d), round(p.z, d), round(p.visibility, d)]
                       for p in points])


def _insert_landmarks(conn, user_id, source, frame, baseline_id=None, score_id=None):
    """1フレーム分を landmarks に足す（commit は呼び出し側がまとめて行う）。"""
    conn.execute(
        "INSERT INTO landmarks (user_id, source, baseline_id, score_id, captured_at,"
        " image_width, image_height, points) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, source, baseline_id, score_id, frame["captured_at"],
         frame["width"], frame["height"], _points_json(frame["points"])),
    )


def get_landmark_frames(user_id, source=None, baseline_id=None, start=None, end=None):
    """保存した生の座標を時刻順に返す（特徴量の再計算・比較用）。

    戻り値 … [{"id", "source", "baseline_id", "score_id", "captured_at",
               "width", "height", "points": [[x, y, z, visibility], ...]}, ...]
    """
    sql = ("SELECT id, source, baseline_id, score_id, captured_at, image_width, image_height,"
           " points FROM landmarks WHERE user_id = ?")
    params = [user_id]
    for col, op, value in (("source", "=", source), ("baseline_id", "=", baseline_id),
                           ("captured_at", ">=", start), ("captured_at", "<=", end)):
        if value is not None:
            sql += f" AND {col} {op} ?"
            params.append(value)
    sql += " ORDER BY captured_at ASC, id ASC"

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [{"id": r["id"], "source": r["source"], "baseline_id": r["baseline_id"],
                 "score_id": r["score_id"], "captured_at": r["captured_at"],
                 "width": r["image_width"], "height": r["image_height"],
                 "points": json.loads(r["points"])} for r in rows]
    finally:
        conn.close()


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
def save_baseline(user_id, features, measured_at, name=None, frames=None):
    """基準値を 1 行追加する。

    features … {特徴名: 代表値} の dict（calibration.stabilize の出力）
    name    … このキャリブレーションの名前（任意。後で名前で選ぶために使う）
    frames  … 測定中に検出した全フレームの生の座標（任意）。baseline と同じトランザクションで
              landmarks に保存する（片方だけ残る中途半端な状態を作らないため）
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
        baseline_id = cur.lastrowid
        for frame in frames or []:
            _insert_landmarks(conn, user_id, "calibration", frame, baseline_id=baseline_id)
        conn.commit()
        return baseline_id
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

    そのキャリブの生の座標（landmarks）も同じトランザクションで消す（参照先の無い行を残さない）。
    戻り値: 削除した baseline の行数。
    """
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM landmarks WHERE baseline_id IN"
            " (SELECT id FROM baseline WHERE user_id = ? AND name = ?)",
            (user_id, name),
        )
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
def save_score(user_id, timestamp, score, deviations=None, landmarks=None, image_size=None):
    """スコアを 1 件記録する。戻り値は追加した行の id。

    deviations … {特徴名: 符号付きずれ or None}（scoring.deviations の出力・任意）。
                 現モードの特徴だけを dev_<名前> 列へ入れる。
    landmarks  … その瞬間の33点（記録モードのときだけ渡す）。scores と同じトランザクションで
                 landmarks に保存し、score_id で対応づける
    image_size … (幅, 高さ)。landmarks を渡すときは必須（縦横比の補正を後から再現するため）
    """
    if landmarks is not None and image_size is None:
        raise ValueError("landmarks を保存するときは image_size (幅, 高さ) が必要です。")
    deviations = deviations or {}
    names = _feature_columns()
    cols = ", ".join(_deviation_column(n) for n in names)
    marks = ", ".join("?" for _ in names)
    conn = _connect()
    try:
        cur = conn.execute(
            f"INSERT INTO scores (user_id, timestamp, score, {cols}) VALUES (?, ?, ?, {marks})",
            (user_id, timestamp, score, *(deviations.get(n) for n in names)),
        )
        score_id = cur.lastrowid
        if landmarks is not None:
            frame = {"captured_at": timestamp, "points": landmarks,
                     "width": image_size[0], "height": image_size[1]}
            _insert_landmarks(conn, user_id, "scoring", frame, score_id=score_id)
        conn.commit()
        return score_id
    finally:
        conn.close()


def get_scores(user_id, start=None, end=None):
    """スコアの生の記録を時刻順に返す（集計はしない＝ pandas の役目）。

    start / end … 期間で絞りたいときの文字列日時（任意）。
    戻り値      … [{"timestamp": ..., "score": ..., "dev_<特徴名>": ... or None}, ...]
    """
    dev_cols = "".join(f", {_deviation_column(n)}" for n in _feature_columns())
    sql = f"SELECT timestamp, score{dev_cols} FROM scores WHERE user_id = ?"
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
        print("ensure_tables: 4テーブル作成 OK")

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

        # 内訳つきで保存 → dev_ 列で読める。内訳なしの行は NULL(None)
        save_score(uid, "2026-06-21 10:00:00", 50.0,
                   deviations={"neck_angle": -1.5, "trunk_angle": None})
        rows = get_scores(uid)
        assert rows[-1]["dev_neck_angle"] == -1.5 and rows[-1]["dev_trunk_angle"] is None
        assert rows[0]["dev_neck_angle"] is None
        # 再起動を模して ensure_tables を再実行しても壊れない（移行の冪等性）
        ensure_tables()
        print("scores の内訳列: 保存・取得・移行の冪等性 OK")

        # 生の座標: キャリブ（baseline と一緒）と採点（score と一緒）で保存 → 読み戻し
        from collections import namedtuple
        P = namedtuple("P", ["x", "y", "z", "visibility"])
        pts = [P(0.1234567891, 0.5, -0.25, 0.99)] * 33
        frame = {"captured_at": "2026-06-22 10:00:00", "points": pts, "width": 640, "height": 480}
        bid = save_baseline(uid, {"neck_angle": 70.0, "trunk_angle": 5.0}, "2026-06-22 10:00:00",
                            name="座標つき", frames=[frame, frame])
        sid = save_score(uid, "2026-06-22 10:00:05", 90.0, landmarks=pts, image_size=(640, 480))
        calib = get_landmark_frames(uid, source="calibration", baseline_id=bid)
        scored = get_landmark_frames(uid, source="scoring")
        assert len(calib) == 2 and calib[0]["width"] == 640 and len(calib[0]["points"]) == 33
        assert calib[0]["points"][0][0] == 0.123457, calib[0]["points"][0]  # 6桁に丸める
        assert len(scored) == 1 and scored[0]["score_id"] == sid
        # landmarks だけ渡して画像サイズが無いのは誤り
        try:
            save_score(uid, "2026-06-22 10:00:06", 90.0, landmarks=pts)
            raise AssertionError("image_size なしで保存できてしまった")
        except ValueError:
            pass
        # キャリブを消すと、その座標も消える
        delete_baseline(uid, "座標つき")
        assert get_landmark_frames(uid, source="calibration") == []
        assert len(get_landmark_frames(uid, source="scoring")) == 1
        print("landmarks: 保存（キャリブ・採点）・6桁丸め・読み戻し・連動削除 OK")

        print("DB 自己テスト: すべて OK")
    finally:
        config.DB_PATH = original  # 後始末（設定を元に戻す）
        config.USE_FEATURES = original_features


if __name__ == "__main__":
    _self_test()
