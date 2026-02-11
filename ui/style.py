# ui/style.py

DARK_STYLE = """
QWidget {
    background: #0f141c;
    color: #d8e1ee;
    font-size: 13px;
}

/* Base panels */
QFrame#Panel, QFrame#PanelInner {
    background: #151c27;
    border: 1px solid #263246;
    border-radius: 12px;
}

QLabel#PanelTitle {
    background: transparent;
    font-weight: 700;
    font-size: 14px;
    padding-left: 4px;
}

QLabel#SectionLabel {
    color: #9fb3cc;
    font-weight: 700;
    margin-top: 4px;
    margin-bottom: 2px;
}


/* Define accent style once */
QLabel[class="AccentTitle"] {
    background: #151c27;
    font-size: 25px;
    font-weight: 800;
    border-radius: 9px;
    padding: 6px 16px;
}

QLabel#ActivityPanelTitle {

    background: #151c27;
    font-size: 25px;
    font-weight: 800;
    color: #8fc9ff;  
    border-radius: 9px;
    padding: 6px 16px;
}

QLabel#StatusPanelTitle {

    background: #151c27;
    font-size: 25px;
    font-weight: 800;
    color: #8fc9ff;  
    border-radius: 9px;
    padding: 6px 16px;
}


QLabel#TopBarText {
    color: #c7d3e6;
    font-size: 18px;
    border: 1px solid rgba(70, 110, 170, 90); 
    border-radius: 9px;   
    padding: 6px 16px;
}

QLabel#MonoInfo {
    color: #c7d3e6;
    line-height: 1.4;
     font-size: 18px;
}

/* Sidebar title */
QLabel#SidebarTitle {
    font-weight: 800;
    font-size: 16px;
}

/* Pills */
QFrame#Pill {
    background: #101824;
    border: 1px solid #2a3b55;
    border-radius: 10px;
}
QLabel#PillTitle { font-weight: 700; }
QLabel#PillSub { color: #9fb3cc; font-size: 12px; }

/* Ghost placeholders */
QFrame#GhostPanel {
    background: #0f1621;
    border: 1px dashed #2b3b53;
    border-radius: 10px;
}
QLabel#GhostText {
    color: #8fa6c3;
}

/* Tabs */
QTabWidget#RightTabs::pane {
    border: 1px solid #263246;
    border-radius: 12px;
    background: #151c27;
    top: -1px;
}
QTabBar::tab {
    background: #101824;
    border: 1px solid #263246;
    padding: 8px 12px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    color: #a9bdd6;
}
QTabBar::tab:selected {
    background: #151c27;
    color: #d8e1ee;
    border-bottom-color: #151c27;
}


/* Status dot base */
QWidget#StatusDot {
    border-radius: 5px; 
}

/* Green (on) */
QWidget#StatusDot[status="on"] {
    background-color: #33d17a;
    border: 1px solid #2bd46f;
}

/* Yellow (warn) */
QWidget#StatusDot[status="warn"] {
    background-color: #f9c74f;
    border: 1px solid #ffd166;
}

/* Red (off) */
QWidget#StatusDot[status="off"] {
    background-color: #ef476f;
    border: 1px solid #ff5c7a;
}

/* Toggle Switch */
QCheckBox#ToggleSwitch {
    background: #1a2433;
    border-radius: 10px;
}

QCheckBox#ToggleSwitch::indicator {
    width: 20px;
    height: 20px;
}

QCheckBox#ToggleSwitch::indicator:unchecked {
    background-color: #2b3b53;
    border-radius: 10px;
}

QCheckBox#ToggleSwitch::indicator:checked {
    background-color: #33d17a;
    border-radius: 10px;
}


"""
