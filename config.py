"""アプリ全体の定数・設定をまとめる場所。

各モジュールは値を直書きせず、ここから読み込む。
（capture.py 設計メモ §4 / pose.py 設計メモ §5・§2-A・§2-B に対応）
"""

import os

# このファイル（プロジェクトルート）の絶対パス。
# モデルや DB の場所を「実行時のカレントディレクトリ」に依存させないため。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── カメラ設定（core/capture.py が参照） ──
CAMERA_INDEX = 1       # 使うカメラの番号。0=内蔵 / 1=外付け webカメラ
FRAME_WIDTH = 640      # 希望する横解像度[px]
FRAME_HEIGHT = 480     # 希望する縦解像度[px]
TARGET_FPS = 30        # 希望するフレームレート[fps]
MIRROR = True          # 左右反転して鏡のように見せるか


# ── 骨格推定設定（core/pose.py が参照） ──
# モデルファイルの場所。models/ 以下に置いた .task を絶対パスで指す。
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

RUNNING_MODE = "VIDEO"   # 推定の動かし方。VIDEO=同期的な動画モード（推奨）
NUM_POSES = 1            # 検出する人数（利用者は 1 人）

# 信頼度（0〜1）。「使ってみて調整する」値。まずは標準的な 0.5 から。
MIN_DETECTION_CONFIDENCE = 0.82   # 検出を採用する最低信頼度
MIN_TRACKING_CONFIDENCE = 0.5    # 追跡を続ける最低信頼度
MIN_PRESENCE_CONFIDENCE = 0.7   # 体が写っていると見なす最低信頼度


# ── ランドマーク名↔番号の対応表（pose / features / live_view が共有） ──
# 番号での取り違えを防ぐため、コードでは landmarks[NOSE] のように名前で参照する。
# 出典: Google「Pose landmark detection guide」(33点)
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_LANDMARKS = 33   # モデルが返す点の総数


# ══════════════════════════════════════════════════════════════════════════
# ★ 撮影モードの切り替え（横向き / 正面）★
#   ここ 1 か所を変えるだけで、使う特徴量・許容幅・重み・助言文・案内文・通知文が
#   まとめて切り替わる（GUI ではなく config で切り替える方針）。
#     "side"  … 横向き（側面）撮影に特化。首の前傾(CVA)と体幹の傾きを見る。
#     "front" … 正面撮影に特化。肩の左右の傾きと頭の左右の傾きを見る。
# 注意: モードを変えたら基準値は測り直すこと（特徴量が変わるため、以前のキャリブは
#       別モードの値になり採点不可になる）。DB のテーブル列は自動で追加される。
# ══════════════════════════════════════════════════════════════════════════
POSTURE_MODE = "side"

_IS_FRONT = (POSTURE_MODE == "front")


def _by_mode(side_value, front_value):
    """POSTURE_MODE に応じて side / front のどちらの設定を使うか選ぶ小さな補助。"""
    return front_value if _IS_FRONT else side_value


# ── 特徴量設定（core/features.py が参照） ──
# 横向き:
#   neck_angle     = 首の前傾（選択側の 耳→肩 線が水平となす角度。CVA相当。90°=良好/小さい=前傾頭位）
#   trunk_angle    = 体幹の傾き（選択側の 肩→腰 線が垂直となす角度。0°=良好/大きい=前傾）
#   kyphosis_angle = 上体屈曲角／胸椎後弯（選択側の 耳-肩-腰 がなす角度。~180°=まっすぐ良好／
#                    小さいほど背中が丸まる＝猫背）。trunk_angle と合わせて猫背を判断する。
# 正面:
#   shoulder_tilt = 肩の左右の傾き（両肩を結ぶ線が水平となす角度。0°=良好/大きい=左右で高さがずれる）
#   head_tilt     = 頭の左右の傾き（両目を結ぶ線が水平となす角度。0°=良好/大きい=首をかしげている）
_SIDE_USE_FEATURES = ["neck_angle", "trunk_angle", "kyphosis_angle"]
_FRONT_USE_FEATURES = ["shoulder_tilt", "head_tilt"]
USE_FEATURES = _by_mode(_SIDE_USE_FEATURES, _FRONT_USE_FEATURES)

# カメラに向ける体の側面（横向きモードでのみ意味を持つ）。
# "auto"=左右で visibility が高い側を自動選択 / "left" / "right"
FACING_SIDE = "auto"

ANGLE_UNIT = "degree"    # 角度の単位。"degree"（度・推奨）/ "radian"（弧度）
MIN_VISIBILITY = 0.5     # この visibility 未満の点は信用せず、その特徴を無効にする

# ── 特徴量の平滑化（短時間移動平均。core/pose_worker.py が採点の直前に使う） ──
# 体幹の傾きや上体屈曲角は 1 フレームだと小刻みに揺れ、スコアがチラつく。直近数フレームの
# 平均でブレをならす。猫背判定に使う「体幹の傾き」「上体屈曲角」を対象にする。
# 平均は採点・改善提案の直前だけに適用し、キャリブレーション（中央値で安定化）には影響しない。
SMOOTH_FEATURES = ["trunk_angle", "kyphosis_angle"]  # 平滑化する特徴量（横向きの猫背関連角）
SMOOTH_WINDOW = 10  # 平均をとる直近フレーム数（≒30fps で約 0.3 秒。大きいほど滑らかだが遅れる）


# ── キャリブレーション設定（core/calibration.py が参照） ──
CALIBRATION_SECONDS = 5       # 正しい姿勢をキープして測定する秒数
MIN_GOOD_SAMPLES = 30         # 基準値を作るのに必要な「良いフレーム」の最低数
STABILIZE_METHOD = "median"   # 安定化の方法。"median"（外れ値に強い・推奨）/ "mean"


# ── データベース設定（db/database.py が参照） ──
DB_PATH = os.path.join(BASE_DIR, "data", "posture.db")  # DBファイル本体の場所
DB_BUSY_TIMEOUT_MS = 5000   # ロック時に待つ時間[ms]（複数接続が重なった時の保険）
DB_USE_WAL = False          # WALモード。推奨案（メインだけが触る）では不要なので既定OFF


# ── 採点設定（core/scoring.py が参照） ──
# 特徴ごとの許容幅。「これくらいずれたら致命的」という幅で、ずれをこの幅で割って
# 単位の違う特徴を同じ土俵（許容幅の何倍ずれたか）に揃える。features の名前に対応。
_SIDE_TOLERANCE = {
    "neck_angle": 13.0,     # 首の前傾（CVA相当）[度]
    "trunk_angle": 12.0,    # 体幹の傾き[度]
    "kyphosis_angle": 15.0, # 上体屈曲角（耳-肩-腰）[度]。基準から15°丸まると致命的とみなす
}
_FRONT_TOLERANCE = {
    "shoulder_tilt": 8.0,  # 肩の左右の傾き[度]（正面は基準0°付近なので横向きより厳しめ）
    "head_tilt": 8.0,      # 頭の左右の傾き[度]
}
TOLERANCE = _by_mode(_SIDE_TOLERANCE, _FRONT_TOLERANCE)

# 特徴ごとの重み。姿勢への影響が大きい特徴を大きくする。
# 横向きは猫背を「体幹の傾き＋上体屈曲角」の合わせ技で見るため、両者に厚めに配分する。
_SIDE_WEIGHT = {
    "neck_angle": 0.4,      # 首の前傾
    "trunk_angle": 0.3,     # 体幹の傾き（猫背の一因）
    "kyphosis_angle": 0.3,  # 上体屈曲角（背中の丸まり＝猫背の主因）
}
_FRONT_WEIGHT = {
    "shoulder_tilt": 0.6,  # 肩の水平を重め（左右差は姿勢の崩れに直結）
    "head_tilt": 0.4,
}
WEIGHT = _by_mode(_SIDE_WEIGHT, _FRONT_WEIGHT)

# 満点を保つ許容範囲の広さ（許容幅の倍数）。呼吸・微動・座標ブレの“遊び”。
DEADBAND = 0.75
# 減点の急さ。deadband を超えてから、さらにこの倍数ぶんずれると 0 点になる。
DECAY = 2.0


# ── 通知設定（core/notifier.py が参照） ──
NOTIFY_WINDOW_SECONDS = 30      # 移動平均をとる時間の長さ[秒]
NOTIFY_MIN_SAMPLES = 10         # 判断に必要な最低サンプル数（立ち上がり直後は判断しない）
NOTIFY_FIRE_THRESHOLD = 70.0    # 鳴らす線。平均がこれ未満で通知
NOTIFY_RECOVER_THRESHOLD = 80.0 # 良好に戻す線（ヒステリシス。fire より高くする）
NOTIFY_COOLDOWN_SECONDS = 60    # 一度鳴らしたら次まで待つ時間[秒]（鳴らしすぎ防止）

# 通知文（前向き・控えめに。責めない）。本文はモードに合わせて言い換える。
NOTIFY_TITLE = "姿勢をチェック"
NOTIFY_MESSAGE = _by_mode(
    "姿勢が崩れています。背筋を伸ばしましょう。",
    "姿勢が崩れています。正面をまっすぐ向きましょう。",
)


# ── 改善提案の文言（core/advice.py が判定し、ui/live_view.py が表示） ──
ADVICE_SCORE_THRESHOLD = 60.0   # 表示し始める線。このスコア未満で改善提案を出す
ADVICE_HIDE_THRESHOLD = 70.0    # 消す線。表示中はこのスコア以上に回復するまで消さない（点滅防止の遊び）
# (特徴名, 方向) → 助言文。方向 low=基準より小さい / high=基準より大きい。
# 横向き:
#   neck_angle  low =頭が前に出ている(前傾頭位) / high=後ろに反りすぎ
#   trunk_angle high=体が前後に傾いている(猫背・前傾) / low=反りすぎ
_SIDE_ADVICE_MESSAGES = {
    ("neck_angle", "low"): "顎を引いて、頭を肩の真上に戻しましょう。",
    ("neck_angle", "high"): "頭が後ろに傾いています。自然な位置に戻しましょう。",
    ("trunk_angle", "high"): "背筋を伸ばして、体を起こしましょう。",
    ("trunk_angle", "low"): "体が反りすぎています。力を抜いて自然に座りましょう。",
    # kyphosis_angle low = 基準より角度が小さい＝背中が丸まっている（猫背）
    ("kyphosis_angle", "low"): "背中が丸まっています（猫背）。胸を開いて背すじを伸ばしましょう。",
    ("kyphosis_angle", "high"): "背中を反りすぎています。力を抜いて自然な姿勢に戻しましょう。",
}
# 正面:
#   shoulder_tilt high=基準より肩が左右に傾いている / low=（基準より水平＝通常は減点されない）
#   head_tilt     high=基準より頭が左右に傾いている / low=（同上）
_FRONT_ADVICE_MESSAGES = {
    ("shoulder_tilt", "high"): "肩の高さが左右で傾いています。両肩を水平に戻しましょう。",
    ("shoulder_tilt", "low"): "肩の力を抜いて、左右の高さをそろえましょう。",
    ("head_tilt", "high"): "頭が左右に傾いています。まっすぐ正面を向きましょう。",
    ("head_tilt", "low"): "頭をまっすぐ正面に保ちましょう。",
}
ADVICE_MESSAGES = _by_mode(_SIDE_ADVICE_MESSAGES, _FRONT_ADVICE_MESSAGES)
# 上記に該当しないときの汎用文言。
ADVICE_DEFAULT_MESSAGE = _by_mode(
    "姿勢が崩れています。背筋を伸ばしましょう。",
    "姿勢が崩れています。正面をまっすぐ向きましょう。",
)


# ── ワーカー／統合設定（workers/pose_worker.py・統合層が参照） ──
QUEUE_MAXSIZE = 2     # 結果キューの上限。映像がたまり過ぎないよう小さめ（あふれたら古いものを捨てる）
SAVE_SCORE_EVERY = 30 # 何フレームごとにスコアをDB記録するか（毎フレームは多すぎるので間引く。≒30fpsで1秒に1回）


# ── 画面・統合設定（ui/app_window.py が参照） ──
TICK_INTERVAL_MS = 30        # QTimer でキューを取り出す間隔[ms]（滑らかさと負荷の兼ね合い）
WINDOW_TITLE = "姿勢改善AI"   # ウィンドウのタイトル
# ウィンドウの初期サイズ "幅x高さ"。左右2カラム（左=映像/右=グラフ2種）が
# 手動リサイズなしで収まるよう、映像640px＋グラフ列が並ぶ幅・高さを確保する。
WINDOW_SIZE = "1280x780"
DEFAULT_USER_NAME = "default"  # 既定の利用者名（users テーブルに用意される）

# ── キャリブレーション操作画面（ui/calibration_view.py が参照） ──
CALIBRATION_COUNTDOWN_SECONDS = 3  # 測定前の準備カウント（姿勢を整える時間）
# 案内文。モードごとに、必要な部位が写る座り方を促す。
_SIDE_GUIDANCE_TEXT = (
    "正しい姿勢で数秒じっとしてください。\n"
    "体の横（側面）をカメラに向け、耳・肩・腰が写るように座りましょう（左右どちらでも可）。"
)
_FRONT_GUIDANCE_TEXT = (
    "正しい姿勢で数秒じっとしてください。\n"
    "カメラに正面を向け、顔（両目）と両肩が写るように、まっすぐ座りましょう。"
)
CALIBRATION_GUIDANCE_TEXT = _by_mode(_SIDE_GUIDANCE_TEXT, _FRONT_GUIDANCE_TEXT)


# ── ライブ表示設定（ui/live_view.py が参照） ──
LIVE_DISPLAY_SIZE = (640, 480)  # 表示する映像の大きさ (幅, 高さ)。小さくすると軽い
DRAW_SKELETON = True            # 骨格を重ねて描くか
# 画像変換の方式。"imencode"=Pillow不要(既定)。"pillow"=PIL(ImageTk)を使う（要 Pillow導入）
IMAGE_BACKEND = "imencode"
SCORE_GOOD = 80.0   # これ以上は緑（良好）
SCORE_WARN = 60.0   # これ以上は黄（注意）、未満は赤（崩れ）


# ── フィードバック（グラフ）設定（ui/feedback_view.py が参照） ──
FEEDBACK_DAYS = 7                  # 表示する期間（直近N日）
FEEDBACK_AGGREGATE = "mean"        # 日ごとの代表値。"mean"（平均）/ "median"（中央値）
FEEDBACK_CHART_TYPE = "line"       # グラフ種類。"line"（折れ線）/ "bar"（棒）
FEEDBACK_FIGURE_SIZE = (5.0, 2.6)  # 図の大きさ（インチ）
FEEDBACK_FONT = "MS Gothic"        # グラフの日本語フォント（Windows標準）
FEEDBACK_MIN_HEIGHT = 220          # キャンバスの最小高さ[px]（見切れ防止。これ未満に縮まない）


# ── 今回の使用グラフ（セッション中のスコア推移。ui/session_view.py が参照） ──
# 日次グラフ(feedback)が「日ごとの平均」を見るのに対し、こちらは「起動してから今までの
# スコアのゆらぎ」をリアルタイムに折れ線で見せる（同じ姿勢を続けられているかの確認用）。
SESSION_WINDOW_SECONDS = 120       # 横軸に映す直近の秒数（これより古い点は捨ててスクロール）
SESSION_REDRAW_MS = 1000           # 再描画の間隔[ms]（毎フレーム描くと重いので間引く）
SESSION_FIGURE_SIZE = (5.0, 2.6)   # 図の大きさ（インチ）
SESSION_MIN_HEIGHT = 220           # キャンバスの最小高さ[px]（見切れ防止）
