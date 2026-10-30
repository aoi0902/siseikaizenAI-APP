"""ui/app_window.py

メインスレッドの指揮者（アプリの中央配電盤）。
（設計メモ: app_window_py_設計メモ.md に対応）

すべての部品（core / db / workers / notifier / 各 view）を 1 つに束ねる。

役割:
  - ウィンドウを組み立て、各 view を配置する
  - Qt のイベントループを回し、QTimer でキューを取り出して各部品へ配る
  - 起動と終了の段取り（ワーカー開始・停止フラグ・join・解放）
  - DB を持ち、スコアを記録させる

担わないこと:
  - 骨格の描画 → live_view / グラフ → feedback_view
  - 採点・通知の判断 → scoring / notifier
  - キャリブ測定そのもの → workers / calibration
  - SQL の実行 → db/database.py

設計の核:
  - 薄い指揮者（計算も描画もしない。「いつ・どの部品に・何を渡すか」だけ）。
  - DB と GUI に触るのはこのメインスレッドだけ。ワーカーとはキュー1本でつなぐ。
  - イベントループを止めない（on_tick は短く。重い処理は分離）。

tkinter からの移行メモ:
  - tk.Tk().mainloop()          → QApplication.exec()
  - root.after(ms, on_tick)     → QTimer（timeout を on_tick につなぎ、一定間隔で発火）
    ※ tkinter は after を毎回張り直したが、QTimer は start(interval) で自動反復するので
      on_tick の末尾で自分を再予約する必要はない。
  - root.protocol("WM_DELETE_WINDOW", on_close) → QWidget.closeEvent を捕まえて on_close
"""

import queue
from datetime import datetime

# ui/ から実行されても、パッケージ import でも config / core / workers / db を読めるように。
try:
    import config
except ModuleNotFoundError:
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root)                          # プロジェクトルート（config / db 用）
    sys.path.append(os.path.join(_root, "core"))     # notifier 等
    sys.path.append(os.path.join(_root, "workers"))  # pose_worker
    import config

from notifier import Notifier
from pose_worker import PoseWorker


def _load_database():
    """db/database.py を読み込んで返す（注入が無いときの既定）。"""
    from db import database
    return database


def _parse_window_size(size_text, default=(900, 700)):
    """config.WINDOW_SIZE の "幅x高さ" 文字列を (幅, 高さ) に分解する。

    tkinter の geometry 形式("900x700")を Qt の resize(w, h) 用に読み替えるだけ。
    解釈できなければ既定値にフォールバックして落ちないようにする。
    """
    try:
        w_text, h_text = str(size_text).lower().split("x")
        return int(w_text), int(h_text)
    except (ValueError, AttributeError):
        return default


def _ui_font(point_size, bold=False):
    """既定のUIフォントを土台に、サイズ・太さだけ変えた QFont を返す。

    QFont("", n) のようにファミリ名を空で作ると、Qt が "MS Sans Serif" に
    フォールバックし DirectWrite が読み込めず警告（qt.qpa.fonts）を出す。
    有効な既定フォント(QApplication.font())を土台にすればファミリ名が入り警告が出ない。
    """
    from PySide6.QtWidgets import QApplication
    font = QApplication.font()
    font.setPointSize(point_size)
    font.setBold(bold)
    return font


def _make_main_window(on_close):
    """closeEvent を捕まえて on_close を呼ぶ QWidget を作って返す。

    tkinter の protocol("WM_DELETE_WINDOW", on_close) に相当。
    PySide6 import を関数内に閉じ込め、_self_test（GUI不要）では読み込まないようにする。
    """
    from PySide6.QtWidgets import QWidget

    class _MainWindow(QWidget):
        def closeEvent(self, event):
            # ×ボタンやウィンドウ破棄で必ずワーカーを止める（カメラ掴みっぱなし防止）。
            if on_close is not None:
                on_close()
            event.accept()

    return _MainWindow()


class AppWindow:
    """アプリ全体の起動・取り出し・終了を司る中央配電盤。

    テスト用に database / notifier / worker / 各 view を差し替えられる。
    """

    def __init__(
        self,
        user_name=None,
        database=None,
        notifier=None,
        save_score_every=None,
        tick_interval_ms=None,
    ):
        self.user_name = user_name or config.DEFAULT_USER_NAME
        # DB は注入が無ければ実物を遅延読込（sqlite3 を base 環境で触らせない配慮）。
        self.database = database if database is not None else _load_database()
        self.notifier = notifier if notifier is not None else Notifier()
        self.save_score_every = save_score_every or config.SAVE_SCORE_EVERY
        self.tick_interval_ms = tick_interval_ms or config.TICK_INTERVAL_MS

        # 持ち物（メインスレッドが所有）
        self.app = None           # QApplication（Qt のイベントループ本体）
        self.root = None          # メインウィンドウ(QWidget)
        self.stack = None         # 画面切り替え（ホーム/撮影/フィードバック/管理）
        self.home_page = None
        self.capture_page = None
        self.feedback_page = None
        self.trend_page = None    # 傾向フィードバック画面
        self.trend_label = None   # 傾向フィードバックの文章
        self.manage_page = None   # キャリブレーションデータの管理画面
        self.live_view = None
        self.calibration_view = None
        self.feedback_view = None
        self.session_view = None  # 今回のスコア推移グラフ
        self.score_label = None
        self.capture_button = None  # 撮影 ON/OFF トグル
        self.record_checkbox = None  # 記録モード（実験用）の ON/OFF
        self.calib_selector = None  # 使用するキャリブレーション（名前で選ぶ）
        self.manage_list = None     # 管理画面のキャリブ一覧
        self.manage_status = None   # 管理画面の操作結果表示
        self.result_queue = None
        self.worker = None
        self.user_id = None
        self.baseline = None      # 現在の基準値（メモリ。撮影ONのたびワーカーへ渡す）
        self.timer = None         # on_tick を回す QTimer（root.after の置き換え）

        # 内部状態
        self._score_counter = 0   # スコア記録の間引きカウンタ
        self.recording = False    # 記録モード: ON の間だけ生の座標も保存する
        self.capturing = False    # 撮影（カメラ・ワーカー）が動いているか
        self._closing = False

    # ── 起動 ──
    def start(self):
        """DB準備 → 基準値読込 → 画面組み立て → QTimer・イベントループ開始。

        撮影（カメラ・ワーカー）はここでは開始しない。まずホーム画面を出し、
        利用者が「撮影する」を選んだときに start_capture() で初めてカメラを開く。
        """
        # 1. DB を用意し、利用者と最新基準値（メモリ）を読む。
        self.database.ensure_tables()
        self.user_id = self.database.get_or_create_user(self.user_name)
        self.baseline = self.database.get_latest_baseline(self.user_id)

        # 2. 画面を組み立てる（ホーム画面が最初に見える）。
        self.build_ui()

        # 3. on_tick を一定間隔で回す QTimer を張る。撮影OFFの間は on_tick は何もしない。
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(self.tick_interval_ms)

        # 4. ウィンドウを表示してイベントループへ。
        self.root.show()
        self.app.exec()

    # ── 撮影（カメラ・ワーカー）の ON/OFF ──
    def start_capture(self):
        """カメラ・ワーカーを開いて撮影を始める（既に動いていれば何もしない）。"""
        if self.worker is not None:
            return
        # キューとワーカーを用意して開始（カメラ・推定器はワーカーが開く）。
        self.result_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.worker = PoseWorker(result_queue=self.result_queue, baseline=self.baseline)
        self.worker.start()
        self.capturing = True
        self._update_capture_button()

    def stop_capture(self):
        """撮影を止めてカメラを解放する（既に止まっていれば安全に何もしない）。"""
        if self.worker is not None:
            self.worker.stop()              # 停止フラグ（ワーカーがカメラを解放して抜ける）
            self.worker.join(timeout=2.0)   # 終了を待つ
            self.worker = None
        self.result_queue = None
        self.capturing = False
        # 停止中と分かるよう表示を落ち着かせる。
        if self.score_label is not None:
            self.score_label.setText("撮影停止中")
        self._update_capture_button()

    def _on_toggle_recording(self, checked):
        """記録モードのチェックボックスに合わせて、生の座標を保存するかを切り替える。"""
        self.recording = bool(checked)

    def toggle_capture(self):
        """撮影 ON/OFF を切り替える（撮影画面のトグルボタンから）。"""
        if self.capturing:
            self.stop_capture()
        else:
            self.start_capture()

    def _update_capture_button(self):
        """トグルボタンの文言を現在の状態に合わせる。"""
        if self.capture_button is None:
            return
        self.capture_button.setText("撮影を停止" if self.capturing else "撮影を再開")

    # ── 画面の組み立て ──
    def build_ui(self):
        """QApplication・ウィンドウ・各画面を作り、閉じるフックを付ける。

        画面は QStackedWidget で 3 枚を切り替える:
          ホーム（メニュー）／撮影（映像・スコア・今回グラフ）／フィードバック（日次グラフ）。
        起動直後はホームを表示し、そこから撮影・フィードバックを選ぶ。
        """
        import sys
        from PySide6.QtWidgets import QApplication, QStackedWidget

        # QApplication はプロセスに1つ。既にあれば使い回す（多重生成を避ける）。
        self.app = QApplication.instance() or QApplication(sys.argv)

        # closeEvent を捕まえる QWidget（× で確実にワーカーを止めるため）。
        self.root = _make_main_window(self.on_close)
        self.root.setWindowTitle(config.WINDOW_TITLE)
        width, height = _parse_window_size(config.WINDOW_SIZE)
        self.root.resize(width, height)

        # 画面を積んで切り替える stack を root に敷く。
        from PySide6.QtWidgets import QVBoxLayout
        root_layout = QVBoxLayout(self.root)
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack)

        self.home_page = self._build_home_page()
        self.capture_page = self._build_capture_page()
        self.feedback_page = self._build_feedback_page()
        self.trend_page = self._build_trend_page()
        self.manage_page = self._build_manage_page()
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.capture_page)
        self.stack.addWidget(self.feedback_page)
        self.stack.addWidget(self.trend_page)
        self.stack.addWidget(self.manage_page)

        self.stack.setCurrentWidget(self.home_page)  # 最初はホーム

    # ── ホーム画面（メニュー） ──
    def _build_home_page(self):
        """撮影・フィードバックを選ぶメニュー画面を作って返す。"""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)

        title = QLabel(config.WINDOW_TITLE)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(_ui_font(28, bold=True))
        layout.addWidget(title)

        subtitle = QLabel("メニューを選んでください")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(_ui_font(14))
        layout.addWidget(subtitle)
        layout.addSpacing(20)

        # 「撮影する」→ カメラを ON にして撮影画面へ。
        btn_capture = QPushButton("撮影する")
        btn_capture.setFont(_ui_font(16))
        btn_capture.setMinimumHeight(56)
        btn_capture.clicked.connect(self.go_capture)
        layout.addWidget(btn_capture)

        # 「フィードバックを見る」→ 日次グラフ画面へ。
        btn_feedback = QPushButton("フィードバックを見る")
        btn_feedback.setFont(_ui_font(16))
        btn_feedback.setMinimumHeight(56)
        btn_feedback.clicked.connect(self.go_feedback)
        layout.addWidget(btn_feedback)

        # 「傾向フィードバック」→ 記録から姿勢のクセを読み取って文章で返す画面へ。
        btn_trend = QPushButton("傾向フィードバック")
        btn_trend.setFont(_ui_font(16))
        btn_trend.setMinimumHeight(56)
        btn_trend.clicked.connect(self.go_trend)
        layout.addWidget(btn_trend)

        # 「キャリブレーションの管理」→ 名前変更・削除の画面へ。
        btn_manage = QPushButton("キャリブレーションの管理")
        btn_manage.setFont(_ui_font(16))
        btn_manage.setMinimumHeight(56)
        btn_manage.clicked.connect(self.go_manage)
        layout.addWidget(btn_manage)

        layout.addStretch(2)
        return page

    # ── 撮影画面 ──
    def _build_capture_page(self):
        """映像・スコア・キャリブ・今回グラフ＋操作（ON/OFF・戻る）を作って返す。

        レイアウトは左右2カラム:
          左 = ライブ映像＋スコア＋キャリブ操作 / 右 = 今回のスコア推移＋操作ボタン。
        縦一列だと映像(480px)＋グラフで高さが足りず見切れるため、横に振り分ける。
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                       QPushButton, QComboBox, QCheckBox)
        from live_view import LiveView
        from calibration_view import CalibrationView
        from session_view import SessionChartView

        page = QWidget()
        root_layout = QHBoxLayout(page)

        # ── 左カラム: 映像・スコア・キャリブ操作 ──
        left = QVBoxLayout()

        self.live_view = LiveView()
        left.addWidget(self.live_view)

        self.score_label = QLabel("スコア: ---")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(_ui_font(16))
        left.addWidget(self.score_label)

        # 使用するキャリブレーションの選択（名前で選ぶ）。選ぶとその基準値で採点する。
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("使用するキャリブレーション:"))
        self.calib_selector = QComboBox()
        # textActivated は「利用者が選んだとき」だけ発火する（プログラムからの再構築では鳴らない）。
        self.calib_selector.textActivated.connect(self._on_select_calibration)
        selector_row.addWidget(self.calib_selector, 1)
        left.addLayout(selector_row)

        # キャリブレーション（ボタンは calibration_view が持ち、押下で on_calibrate）
        self.calibration_view = CalibrationView(on_calibrate=self.on_calibrate)
        left.addWidget(self.calibration_view)
        left.addStretch(1)

        # 起動時点で保存済みのキャリブ名を選択肢に入れる（現在の基準値を選択状態に）。
        self._refresh_calibration_selector(
            select=self.baseline.get("name") if self.baseline else None)

        # ── 右カラム: 今回グラフ＋操作 ──
        right = QVBoxLayout()

        self.session_view = SessionChartView()
        right.addWidget(self.session_view)

        # 撮影 ON/OFF トグル（撮影中はカメラを止め、停止中は再開できる）。
        self.capture_button = QPushButton("撮影を停止")
        self.capture_button.setFont(_ui_font(14))
        self.capture_button.setMinimumHeight(44)
        self.capture_button.clicked.connect(self.toggle_capture)
        right.addWidget(self.capture_button)

        # 記録モード（実験用）: ON の間だけ、スコアと同じタイミングで生の座標も保存する。
        # 常時保存すると DB が膨らむため実験中だけ使う。画面に状態が見えるので入れ忘れに気づける。
        self.record_checkbox = QCheckBox("記録モード（実験用：座標も保存）")
        self.record_checkbox.toggled.connect(self._on_toggle_recording)
        right.addWidget(self.record_checkbox)

        # ホームへ戻る（戻るときはカメラを解放する）。
        btn_home = QPushButton("ホームへ戻る")
        btn_home.setMinimumHeight(36)
        btn_home.clicked.connect(self.go_home)
        right.addWidget(btn_home)
        right.addStretch(1)

        # 左は映像(640px)が横に切れないよう最小幅を確保し、伸縮は右列に任せる。
        left_container, right_container = QWidget(), QWidget()
        left_container.setLayout(left)
        right_container.setLayout(right)
        left_container.setMinimumWidth(config.LIVE_DISPLAY_SIZE[0] + 40)
        root_layout.addWidget(left_container, 0)
        root_layout.addWidget(right_container, 1)
        return page

    # ── フィードバック画面 ──
    def _build_feedback_page(self):
        """日次スコアグラフ＋ホームへ戻るボタンを作って返す。"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
        from feedback_view import FeedbackView

        page = QWidget()
        layout = QVBoxLayout(page)

        # フィードバック: 日次グラフ（重いので更新時だけ描く）
        self.feedback_view = FeedbackView(
            get_scores=lambda: self.database.get_scores(self.user_id),
            on_refresh=self.on_refresh_feedback,
        )
        layout.addWidget(self.feedback_view)

        btn_home = QPushButton("ホームへ戻る")
        btn_home.setMinimumHeight(36)
        btn_home.clicked.connect(self.go_home)
        layout.addWidget(btn_home)
        return page

    # ── 傾向フィードバック画面 ──
    def _build_trend_page(self):
        """生成ボタン＋フィードバック文＋ホームへ戻るボタンを作って返す。"""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("傾向フィードバック")
        title.setFont(_ui_font(18, bold=True))
        layout.addWidget(title)
        layout.addWidget(QLabel(f"直近{config.TREND_DAYS}日間の記録から、姿勢のクセを読み取ります。"))

        btn_generate = QPushButton("フィードバックを生成")
        btn_generate.setFont(_ui_font(14))
        btn_generate.setMinimumHeight(44)
        btn_generate.clicked.connect(self.on_generate_trend)
        layout.addWidget(btn_generate)

        # 結果の文章（折り返し・選択コピー可）。
        self.trend_label = QLabel("「フィードバックを生成」を押してください。")
        self.trend_label.setFont(_ui_font(13))
        self.trend_label.setWordWrap(True)
        self.trend_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.trend_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.trend_label, 1)

        btn_home = QPushButton("ホームへ戻る")
        btn_home.setMinimumHeight(36)
        btn_home.clicked.connect(self.go_home)
        layout.addWidget(btn_home)
        return page

    # ── キャリブレーション管理画面（名前変更・削除） ──
    def _build_manage_page(self):
        """保存済みキャリブレーションの一覧＋名前変更・削除・戻るボタンを作って返す。"""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                       QPushButton, QListWidget)

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("キャリブレーションデータの管理")
        title.setFont(_ui_font(18, bold=True))
        layout.addWidget(title)
        layout.addWidget(QLabel("名前を変更・削除したいデータを選んでください。"))

        # 保存済みキャリブの一覧。
        self.manage_list = QListWidget()
        layout.addWidget(self.manage_list, 1)

        # 操作結果の表示（成功=緑・失敗=赤）。
        self.manage_status = QLabel("")
        layout.addWidget(self.manage_status)

        # 操作ボタン: 名前変更・削除。
        btn_row = QHBoxLayout()
        btn_rename = QPushButton("名前を変更")
        btn_rename.setMinimumHeight(36)
        btn_rename.clicked.connect(self._on_rename_calibration)
        btn_row.addWidget(btn_rename)

        btn_delete = QPushButton("削除")
        btn_delete.setMinimumHeight(36)
        btn_delete.clicked.connect(self._on_delete_calibration)
        btn_row.addWidget(btn_delete)
        layout.addLayout(btn_row)

        btn_home = QPushButton("ホームへ戻る")
        btn_home.setMinimumHeight(36)
        btn_home.clicked.connect(self.go_home)
        layout.addWidget(btn_home)
        return page

    # ── 画面遷移 ──
    def go_capture(self):
        """撮影画面へ。カメラを ON にしてから表示する。"""
        self.start_capture()
        self.stack.setCurrentWidget(self.capture_page)

    def go_feedback(self):
        """フィードバック画面へ。最新データで日次グラフを描き直してから表示する。"""
        if self.feedback_view is not None:
            self.feedback_view.refresh()
        self.stack.setCurrentWidget(self.feedback_page)

    def go_trend(self):
        """傾向フィードバック画面へ（生成は利用者がボタンを押したときだけ）。"""
        self.stack.setCurrentWidget(self.trend_page)

    def go_manage(self):
        """管理画面へ。最新の一覧を作り直してから表示する。"""
        self._refresh_manage_list()
        if self.manage_status is not None:
            self.manage_status.setText("")
        self.stack.setCurrentWidget(self.manage_page)

    def go_home(self):
        """ホーム画面へ。撮影中なら止めてカメラを解放する。"""
        self.stop_capture()
        self.stack.setCurrentWidget(self.home_page)

    # ── 管理画面の中身（一覧・名前変更・削除） ──
    def _refresh_manage_list(self):
        """保存済みキャリブ名で一覧を作り直す（各行に名前と測定日時を表示）。"""
        if self.manage_list is None:
            return
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem
        self.manage_list.clear()
        for row in self.database.get_baseline_names(self.user_id):
            name = row["name"]
            measured_at = row.get("measured_at", "")
            item = QListWidgetItem(f"{name}    （{measured_at}）")
            # 表示文字列に日時を足すので、本当の名前は UserRole に持たせる。
            item.setData(Qt.UserRole, name)
            self.manage_list.addItem(item)

    def _selected_calibration_name(self):
        """一覧で選択中のキャリブ名を返す。未選択なら None。"""
        if self.manage_list is None:
            return None
        from PySide6.QtCore import Qt
        item = self.manage_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _set_manage_status(self, message, ok=True):
        if self.manage_status is not None:
            self.manage_status.setText(message)
            self.manage_status.setStyleSheet(f"color: {'#2e7d32' if ok else '#c62828'};")

    def _on_rename_calibration(self):
        """選択中のキャリブを別名に変更する（重複名は拒否）。"""
        name = self._selected_calibration_name()
        if name is None:
            self._set_manage_status("変更する行を選んでください。", ok=False)
            return

        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self.root, "名前の変更", "新しい名前を入力してください：", text=name)
        if not ok:
            return  # キャンセル
        new_name = (new_name or "").strip()
        if not new_name:
            self._set_manage_status("名前が空です。変更しませんでした。", ok=False)
            return
        if new_name == name:
            return  # 変化なし
        if self.database.baseline_name_exists(self.user_id, new_name):
            self._set_manage_status(f"「{new_name}」は既にあります。別の名前にしてください。", ok=False)
            return

        self.database.rename_baseline(self.user_id, name, new_name)
        # 使用中のキャリブを改名したら、メモリ側もその新名で読み直して整合を取る。
        if self._current_baseline_name() == name:
            self._apply_baseline(self.database.get_baseline_by_name(self.user_id, new_name))

        self._refresh_manage_list()
        self._refresh_calibration_selector(select=self._current_baseline_name())
        self._set_manage_status(f"「{name}」を「{new_name}」に変更しました。", ok=True)

    def _on_delete_calibration(self):
        """選択中のキャリブを削除する（確認あり）。"""
        name = self._selected_calibration_name()
        if name is None:
            self._set_manage_status("削除する行を選んでください。", ok=False)
            return

        from PySide6.QtWidgets import QMessageBox
        answer = QMessageBox.question(
            self.root, "削除の確認",
            f"「{name}」を削除しますか？（この操作は元に戻せません）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        self.database.delete_baseline(self.user_id, name)
        # 使用中のキャリブを消したら、残っている最新に切り替える（無ければ None＝採点不可）。
        if self._current_baseline_name() == name:
            self._apply_baseline(self.database.get_latest_baseline(self.user_id))

        self._refresh_manage_list()
        self._refresh_calibration_selector(select=self._current_baseline_name())
        self._set_manage_status(f"「{name}」を削除しました。", ok=True)

    def _current_baseline_name(self):
        """現在採点に使っている基準値の名前（無ければ None）。"""
        return self.baseline.get("name") if self.baseline else None

    def _apply_baseline(self, baseline):
        """メモリ基準値を差し替え、撮影中ならワーカーにも反映する。"""
        self.baseline = baseline
        if self.worker is not None:
            self.worker.set_baseline(baseline)

    # ── QTimer の取り出しループ（心臓部） ──
    def on_tick(self):
        """キューをドレインし、notifier・DB・live_view へ配る（QTimer が反復発火）。

        撮影OFF（ワーカー無し）のときは何もしない。QTimer は回り続けるが空振りするだけ。
        """
        if self.worker is None or self.result_queue is None:
            return

        items = self._drain()
        last = self._dispatch_results(items)

        # 最新の1フレームだけ描画し、スコア表示・改善提案も更新する。
        if last is not None:
            if self.live_view is not None:
                self.live_view.update_view(last.frame, last.landmarks, last.score,
                                           getattr(last, "advice", None))
            self._update_score_label(last.score)
            # 今回のスコア推移グラフへ供給（採点不可 None も渡す＝線が途切れて離席が分かる）。
            if self.session_view is not None:
                self.session_view.add_score(last.score)

        # キャリブレーション結果が届いていれば処理する。
        self._poll_calibration()

        # ワーカーが落ちていたら無言で固まらず状態を出す。
        if self.worker is not None and not self.worker.is_alive() and not self._closing:
            self._set_calib_status("カメラ/ワーカーが停止しました。")
        # QTimer は start(interval) で自動反復するので、再予約は不要。

    def _drain(self):
        """キューにある分をすべて取り出す（点数は落とさない）。"""
        items = []
        while True:
            try:
                items.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _dispatch_results(self, items):
        """各結果を notifier・DB へ配り、描画用の最後の1件を返す。

        点数は全部処理（notifier へ渡し、間引いて DB 記録）、映像は最後の1枚だけ使う。
        """
        last = None
        for item in items:
            # 通知判断（採点不可 None も渡す。notifier 側が平均から除外する）。
            self.notifier.update(item.score)

            # スコア記録は数値のときだけ、かつ間引いて行う（毎フレームは多すぎる）。
            if item.score is not None:
                self._score_counter += 1
                if self._score_counter >= self.save_score_every:
                    self._score_counter = 0
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # 記録モード（実験用）のときだけ、その瞬間の生の座標と画像サイズも残す。
                    landmarks, image_size = None, None
                    if self.recording and item.landmarks is not None:
                        landmarks = item.landmarks
                        image_size = (item.frame.shape[1], item.frame.shape[0])
                    # 内訳（特徴ごとのずれ）も一緒に残す → 傾向フィードバックで使う。
                    self.database.save_score(self.user_id, ts, item.score,
                                             deviations=item.deviations,
                                             landmarks=landmarks, image_size=image_size)

            last = item
        return last

    def _update_score_label(self, score):
        if self.score_label is None:
            return
        text = "スコア: 採点不可" if score is None else f"スコア: {score:.0f}"
        self.score_label.setText(text)

    # ── キャリブレーション配線 ──
    def on_calibrate(self):
        """ワーカーへ測定要求（採点を一時中断して測定モードへ）。

        測定中の案内・カウントダウンは calibration_view が自前で出すので、
        ここでは状態表示をせず要求だけを伝える。
        撮影OFF（ワーカー無し）のときは測れないので、その旨を返してやり直しを促す。
        """
        if self.worker is not None:
            self.worker.request_calibration(self.user_id)
        elif self.calibration_view is not None:
            # 待ちっぱなしにならないよう、失敗として返してカウントダウンを畳む。
            self.calibration_view.show_result(False, "先に「撮影を再開」でカメラをONにしてください。")

    def _poll_calibration(self):
        """ワーカーからのキャリブ結果が届いていれば取り出して処理する。"""
        if self.worker is None:
            return
        try:
            outcome = self.worker.calib_result_queue.get_nowait()
        except queue.Empty:
            return
        self._handle_calib_outcome(outcome)

    def _handle_calib_outcome(self, outcome):
        """キャリブ結果に名前を付けて保存する。保存をキャンセルしたらやり直せる。"""
        if outcome.success:
            # 名前を尋ねてから保存する（名前でキャリブを選べるようにするため）。
            # 利用者がキャンセルしたら None が返る＝この測定結果は保存しない。
            name = self._ask_calibration_name()
            if name is None:
                # 保存キャンセル: DB・メモリ基準値・ワーカー・選択肢のいずれも触らないので、
                # 直前の状態がそのまま保たれる。ボタンを戻してやり直しを促す。
                self._show_calib_result(
                    False,
                    "保存をキャンセルしました。もう一度「キャリブレーション開始」からやり直せます。")
                return outcome
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # DB に触るのはメインだけ。保存 → 名前で読み戻し → ワーカーへ差し替え。
            # 測定中の生の座標も一緒に保存する（式を直したとき、測り直さずに再計算できるように）。
            self.database.save_baseline(outcome.user_id, outcome.representative, ts, name=name,
                                        frames=outcome.frames)
            baseline = self.database.get_baseline_by_name(outcome.user_id, name)
            # メモリの基準値も更新（撮影を止めて再開したとき新しい基準値を使うため）。
            self.baseline = baseline
            if self.worker is not None:
                self.worker.set_baseline(baseline)
            # 選択肢を作り直し、今保存した名前を選択状態にする。
            self._refresh_calibration_selector(select=name)
            # 結果メッセージに保存名を添える。
            message = f"「{name}」として保存しました。"
            self._show_calib_result(True, message)
        else:
            self._show_calib_result(False, outcome.message)
        return outcome

    def _show_calib_result(self, success, message):
        """calibration_view に結果を伝える（成功/失敗でボタン復帰・色分け表示）。"""
        if self.calibration_view is not None:
            self.calibration_view.show_result(success, message)

    def _ask_calibration_name(self):
        """キャリブレーションの保存名を尋ねる。

        戻り値:
          - 入力された名前（空のまま OK を押したら日時からの既定名）
          - None … 利用者がキャンセルした（＝この結果は保存せずやり直す）
        """
        default = "姿勢_" + datetime.now().strftime("%m%d_%H%M")
        # GUI が無い（テスト等）場合は既定名で保存に進む。
        if self.root is None:
            return default
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self.root, "キャリブレーションの保存",
            "保存する名前を入力してください。\n"
            "（キャンセルすると保存せず、やり直せます）：", text=default)
        if not ok:
            return None  # キャンセル＝保存しない（呼び出し側がやり直しを促す）
        name = (name or "").strip()
        return name if name else default

    def _on_select_calibration(self, name):
        """選択されたキャリブ名の基準値を読み込み、採点に反映する。"""
        if not name:
            return
        baseline = self.database.get_baseline_by_name(self.user_id, name)
        if baseline is None:
            return
        self.baseline = baseline
        if self.worker is not None:
            self.worker.set_baseline(baseline)

    def _refresh_calibration_selector(self, select=None):
        """保存済みキャリブ名でコンボボックスを作り直す（select があればそれを選択）。"""
        if self.calib_selector is None:
            return
        names = [r["name"] for r in self.database.get_baseline_names(self.user_id)]
        # 作り直し中の currentText 変化で誤発火しないよう signal を止める。
        self.calib_selector.blockSignals(True)
        self.calib_selector.clear()
        self.calib_selector.addItems(names)
        if select is not None and select in names:
            self.calib_selector.setCurrentText(select)
        self.calib_selector.blockSignals(False)

    def _set_calib_status(self, message):
        if self.calibration_view is not None:
            self.calibration_view.set_status(message)

    # ── グラフ更新（重いので更新時だけ） ──
    def on_refresh_feedback(self):
        """フィードバックの更新要求 → feedback_view に再集計・再描画を依頼。"""
        if self.feedback_view is not None:
            self.feedback_view.refresh()

    # ── 傾向フィードバックの生成（オンデマンド） ──
    def on_generate_trend(self):
        """記録を読んで傾向を分析し、文章を表示する。

        集計は数十万行でも pandas で1秒未満なのでメインスレッドで同期実行する
        （DB に触るのはメインだけ、という方針もそのまま守れる）。
        """
        import trend
        from datetime import timedelta

        now = datetime.now()
        # 比較に使う「前の期間」まで含めて読む（それより古い記録は読まない）。
        start = (now - timedelta(days=2 * config.TREND_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        rows = self.database.get_scores(self.user_id, start=start)
        text = trend.generate(rows, now)
        if self.trend_label is not None:
            self.trend_label.setText(f"{now:%Y-%m-%d %H:%M} 生成\n\n{text}")
        return text

    # ── 終了 ──
    def on_close(self):
        """停止フラグ → QTimer停止 → ワーカー join → 解放。

        closeEvent から呼ばれる前提なので、ここで root.close() はしない（再帰防止）。
        二重呼び出しに備えて _closing で冪等にする。
        """
        if self._closing:
            return
        self._closing = True
        # QTimer を止める（破棄処理中に on_tick が走らないように）。
        if self.timer is not None:
            self.timer.stop()
        # セッショングラフの再描画タイマーも止める。
        if self.session_view is not None:
            self.session_view.stop()
        # 撮影中ならワーカーを止めてカメラを解放する（停止中なら何もしない）。
        self.stop_capture()
        # database は操作ごとに接続を開閉する設計なので、閉じるべき常設接続は無い。


def main():
    """本番の起動口。"""
    AppWindow().start()


# ── 単体での動作確認（ディスプレイ不要。統合ロジックを偽部品で検証） ──
def _self_test():
    from collections import namedtuple
    WorkerResult = namedtuple("WorkerResult",
                              ["frame", "landmarks", "score", "timestamp", "deviations"],
                              defaults=[None])
    Outcome = namedtuple("Outcome",
                         ["success", "user_id", "representative", "num_samples", "message", "frames"],
                         defaults=[None])

    class FakeDB:
        def __init__(self):
            self.saved_scores = []
            self.saved_baselines = []
        def ensure_tables(self): pass
        def get_or_create_user(self, name): return 1
        def get_latest_baseline(self, uid):
            return {"neck_angle": 10.0, "shoulder_tilt": 0.0} if self.saved_baselines else None
        def save_score(self, uid, ts, score, deviations=None, landmarks=None, image_size=None):
            self.saved_scores.append((uid, ts, score, deviations, landmarks, image_size))
        def get_scores(self, uid, start=None, end=None): return []
        def save_baseline(self, uid, rep, ts, name=None, frames=None):
            self.saved_baselines.append((uid, rep, ts, name))
            self.saved_frames = frames
        def get_baseline_names(self, uid):
            return [{"name": n, "measured_at": ts} for (_, _, ts, n) in self.saved_baselines if n]
        def get_baseline_by_name(self, uid, name):
            for (_, rep, _, n) in reversed(self.saved_baselines):
                if n == name:
                    return dict(rep, name=name)
            return None

    class FakeNotifier:
        def __init__(self): self.updates = []
        def update(self, score, now=None): self.updates.append(score)

    class FakeWorker:
        def __init__(self):
            self.calib_result_queue = queue.Queue()
            self.stopped = False
            self.joined = False
            self.baseline = None
            self._requested = None
        def request_calibration(self, uid): self._requested = uid
        def set_baseline(self, b): self.baseline = b
        def stop(self): self.stopped = True
        def join(self, timeout=None): self.joined = True
        def is_alive(self): return not self.stopped

    results = []
    db, notifier = FakeDB(), FakeNotifier()
    app = AppWindow(database=db, notifier=notifier, save_score_every=3)
    app.user_id = 1
    app.worker = FakeWorker()

    # ① ドレイン配分: 5件（うち1件は採点不可None）を流す
    items = [
        WorkerResult("f1", None, 95.0, 1),
        WorkerResult("f2", None, 80.0, 2),
        WorkerResult("f3", None, None, 3),   # 採点不可
        WorkerResult("f4", None, 70.0, 4, {"neck_angle": -1.2}),
        WorkerResult("f5", None, 60.0, 5),
    ]
    last = app._dispatch_results(items)
    results.append(("notifierへ全件(採点不可含む)渡す", notifier.updates == [95.0, 80.0, None, 70.0, 60.0]))
    # 数値スコアは4件、save_score_every=3 なので 3件目で1回だけ記録される
    results.append(("スコア記録が間引かれる(数値4件→1回)", len(db.saved_scores) == 1))
    results.append(("描画用に最後の1件を返す", last.frame == "f5"))
    results.append(("記録に内訳(ずれ)も渡す", db.saved_scores[0][3] == {"neck_angle": -1.2}))
    results.append(("記録モードOFFでは座標を保存しない", db.saved_scores[0][4] is None))

    # ①'' 記録モードON: スコアと一緒に座標と画像サイズ(幅, 高さ)を渡す
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    app._on_toggle_recording(True)
    app._score_counter = 0
    app._dispatch_results([WorkerResult(frame, ["LM"] * 33, 75.0, 6)] * 3)
    results.append(("記録モードONで座標と画像サイズも保存",
                    db.saved_scores[-1][4] == ["LM"] * 33 and db.saved_scores[-1][5] == (640, 480)))
    app._on_toggle_recording(False)

    # ①' 傾向フィードバック: 記録が無くても落ちず、不足の案内を返す
    results.append(("傾向生成(記録なし)で案内文", "足りません" in app.on_generate_trend()))

    # ② キャリブ結果の処理: 成功 → save_baseline & set_baseline
    outcome = Outcome(True, 1, {"neck_angle": 12.0, "shoulder_tilt": -1.0}, 50, "完了",
                      frames=[{"captured_at": "t", "points": [], "width": 640, "height": 480}])
    app._handle_calib_outcome(outcome)
    results.append(("キャリブ成功で基準値を保存", len(db.saved_baselines) == 1))
    results.append(("キャリブの座標も一緒に保存", db.saved_frames == outcome.frames))
    results.append(("ワーカーの基準値を差し替え", app.worker.baseline is not None))

    # ②' 保存キャンセル: 名前入力を None にすると保存されず、基準値も変わらない
    saved_before = len(db.saved_baselines)
    baseline_before = app.worker.baseline
    app._ask_calibration_name = lambda: None   # 利用者がキャンセルした状況を再現
    app._handle_calib_outcome(outcome)
    results.append(("保存キャンセルでは保存しない", len(db.saved_baselines) == saved_before))
    results.append(("保存キャンセルで基準値は変わらない", app.worker.baseline is baseline_before))

    # ③ 撮影ON/OFF: 停止でワーカーを止め、撮影フラグが下りて worker が外れる
    app.capturing = True
    worker_ref = app.worker              # stop_capture が worker を None にするので控えておく
    app.stop_capture()
    results.append(("撮影OFFで停止フラグ＋join", worker_ref.stopped and worker_ref.joined))
    results.append(("撮影OFFで撮影フラグが下りる", app.capturing is False and app.worker is None))

    # ④ 終了手順: 停止中に閉じても安全（二重停止しない）
    app.on_close()
    results.append(("停止中でも終了が通る", app._closing is True))

    all_ok = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("app_window 自己テスト:", "すべて OK" if all_ok else "失敗あり")


if __name__ == "__main__":
    _self_test()
