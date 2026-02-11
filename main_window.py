# main_window.py

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTabWidget, QFrame
)
from ui.widgets import panel, ghost_panel, pill, section_label, right_tab_placeholder
from ui.style import DARK_STYLE
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from ui.widgets import StatusDot, panel_inner
from PyQt6.QtWidgets import QSizePolicy




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Revia Controller")
        self.resize(1600, 950)

        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        left = self._build_left()
        center = self._build_center()
        right = self._build_right()

        main_layout.addWidget(left, 0)
        main_layout.addWidget(center, 1)
        main_layout.addWidget(right, 0)

        self.setStyleSheet(DARK_STYLE)

    def _build_left(self) -> QWidget:
        p = panel()
        p.setFixedWidth(260)

        inner = p.findChild(QFrame, "PanelInner")
        lay = inner.layout()
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Avatar image
        avatar = QLabel()
        avatar.setObjectName("AvatarImage")
        avatar.setFixedSize(220, 220)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pix = QPixmap("assets/avatar.png")
        if not pix.isNull():
            avatar.setPixmap(
                pix.scaled(
                    avatar.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        lay.addWidget(avatar, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Name row (status dot + name)
        name_row = QWidget()
        name_row_lay = QHBoxLayout(name_row)
        name_row_lay.setContentsMargins(0, 0, 0, 0)
        name_row_lay.setSpacing(8)

        status_dot = StatusDot("on", size=10)
        status_dot.setObjectName("StatusDot")

        name = QLabel(
            '<span style="color:#ffffff; font-weight:600;">Astra</span> '
            '<span style="color:#7fb3ff; font-weight:300;">(Assistant)</span>'
        )

        name_row_lay.addStretch(1)
        name_row_lay.addWidget(status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        name_row_lay.addWidget(name, alignment=Qt.AlignmentFlag.AlignVCenter)
        name_row_lay.addStretch(1)

        lay.addWidget(name_row)

        lay.addSpacing(6)

        stt_pill = pill("STT Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        lay.addWidget(stt_pill)
        stt_pill.toggle.toggled.connect(lambda on: print("STT toggled:", on))

        tts_pill = pill("TTS Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        lay.addWidget(tts_pill)

        vision_pill = pill("Vision Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        lay.addWidget(vision_pill)

        lay.addSpacing(14)
        lay.addWidget(section_label("Modes"))
        lay.addWidget(ghost_panel("Passive (Background)", height=36))
        lay.addWidget(ghost_panel("Interactive (Voice)", height=36))
        lay.addWidget(ghost_panel("Teaching / Explain", height=36))
        lay.addWidget(ghost_panel("Debug", height=36))

        lay.addSpacing(14)
        lay.addWidget(section_label("Profiles"))
        lay.addWidget(ghost_panel("Default", height=32))
        lay.addWidget(ghost_panel("Technical", height=32))
        lay.addWidget(ghost_panel("Custom", height=32))

        lay.addStretch(1)
        return p

    def _build_center(self) -> QWidget:
        p = panel()

        inner = p.findChild(QFrame, "PanelInner")
        lay = inner.layout()
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        # ---------------- Top Bar ----------------
        topbar = panel()
        topbar.setFixedHeight(60)

        topbar_inner = topbar.findChild(QFrame, "PanelInner")
        tl = topbar_inner.layout()
        tl.setContentsMargins(12, 8, 12, 8)

        top = QLabel(
            "Runtime: Active   |   Model: -   |   VRAM: -   |   RAM: -   |   GPU: -   |   CPU: -   |   Health: -"
        )
        top.setObjectName("TopBarText")
        tl.addWidget(top)

        lay.addWidget(topbar, 0)

        # ---------------- Assistant Status ----------------
        status = panel("Assistant Status", title_object="StatusPanelTitle")
        title = status.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")
        status_inner = status.findChild(QFrame, "PanelInner")
        sl = status_inner.layout()
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setSpacing(6)

        lines = QLabel(
            "• Listening...\n"
            "• Processing Command...\n"
            "• Vision: Idle\n"
            "• Generating Response..."
        )
        lines.setObjectName("MonoInfo")
        sl.addWidget(lines)

        lay.addWidget(status, 0)

        # ---------------- Activity ----------------
        activity = panel("Activity", title_object="ActivityPanelTitle")
        title = activity.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")
        act_inner = activity.findChild(QFrame, "PanelInner")
        al = act_inner.layout()
        al.setContentsMargins(12, 12, 12, 12)
        al.setSpacing(6)

        act = QLabel(
            'User: "Analyze this screenshot and explain the chart"\n'
            'AI: "Sure! The chart shows..."'
        )
        act.setObjectName("MonoInfo")
        act.setWordWrap(True)

        al.addWidget(act)
        lay.addWidget(activity, 1)
        # ---------------- Inference ----------------
        inf = panel("Inference",  title_object="ActivityPanelTitle")
        title = inf.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")
        inf_inner = inf.findChild(QFrame, "PanelInner")

        old = inf_inner.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)

        il = QHBoxLayout(inf_inner)
        il.setContentsMargins(12, 12, 12, 12)
        il.setSpacing(12)

        stats = QLabel("LLM: -\nLatency: -\nTokens/sec: -\nTTFT: -\nContext: -")
        stats.setObjectName("MonoInfo")
        stats.setFixedWidth(240)
        il.addWidget(stats)

        il.addWidget(ghost_panel("Preview / Whiteboard / Vision Frame", height=220), 1)

        lay.addWidget(inf, 2)

        # Explicit stretch control (prevents weird expansion)
        lay.setStretch(0, 0)  # topbar
        lay.setStretch(1, 0)  # status
        lay.setStretch(2, 1)  # activity
        lay.setStretch(3, 1)  # inference

        return p

    def _build_right(self) -> QWidget:
        p = panel()
        p.setFixedWidth(490)

        inner = p.findChild(QFrame, "PanelInner")
        lay = inner.layout()
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(15)

        tabs = QTabWidget()
        tabs.setObjectName("RightTabs")

        tabs.addTab(right_tab_placeholder("Behavior controls go here"), "Behavior")
        tabs.addTab(right_tab_placeholder("LLM provider/model settings go here"), "LLM")
        tabs.addTab(right_tab_placeholder("STT/TTS/Vision settings go here"), "Voice & Vision")
        tabs.addTab(right_tab_placeholder("None / Low / Strict + categories"), "Filters")
        tabs.addTab(right_tab_placeholder("Allowed / Filtered / Rewritten logs"), "Logs")
        tabs.addTab(right_tab_placeholder("Graphs + meters go here"), "System")

        lay.addWidget(tabs, 1)
        return p
