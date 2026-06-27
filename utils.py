import streamlit as st
import streamlit.components.v1 as components

_LIGHT_CSS = """<style>
:root {
  --bg-base:    #f0f5fa;
  --bg-panel:   #e4ecf4;
  --bg-card:    #ffffff;
  --bg-card-2:  #f7fafd;
  --bg-input:   #f8fbfe;
  --bg-glass:   rgba(255,255,255,0.70);
  --teal:       #0288a7;
  --teal-10:    rgba(2,136,167,0.07);
  --teal-15:    rgba(2,136,167,0.12);
  --teal-20:    rgba(2,136,167,0.17);
  --teal-border:rgba(2,136,167,0.28);
  --teal-glow:  rgba(2,136,167,0.12);
  --orange:     #d9540a;
  --orange-deep:#c24808;
  --orange-10:  rgba(217,84,10,0.07);
  --orange-border:rgba(217,84,10,0.26);
  --orange-glow:rgba(217,84,10,0.14);
  --blue:       #2563eb;
  --blue-10:    rgba(37,99,235,0.07);
  --blue-border:rgba(37,99,235,0.26);
  --text-hi:    #0d1b2a;
  --text-mid:   #3d5166;
  --text-lo:    rgba(61,81,102,0.50);
  --border:     rgba(0,0,0,0.08);
  --border-hi:  rgba(0,0,0,0.13);
}
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(2,136,167,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 85% 110%, rgba(217,84,10,0.03) 0%, transparent 60%),
    var(--bg-base) !important;
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container, section.main>div {
  background: transparent !important;
  color: var(--text-hi) !important;
}
[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-mid) !important; }
h1 { color: var(--text-hi) !important; text-shadow: none !important; }
h2 { color: var(--text-hi) !important; }
h2::after { background: linear-gradient(90deg, var(--teal-10), transparent) !important; }
h3 { color: var(--text-mid) !important; }
p, .stMarkdown p { color: var(--text-mid) !important; }
hr { background: linear-gradient(90deg, transparent, var(--border-hi) 30%, var(--border-hi) 70%, transparent) !important; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #ffffff, #f7fafd) !important;
  border-color: var(--border) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricValue"]>div { color: var(--text-hi) !important; }
[data-testid="stMetricLabel"]>div { color: var(--text-lo) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(135deg, #ffffff, #f7fafd) !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] {
  background: #ffffff !important;
  box-shadow: 0 12px 40px rgba(0,0,0,0.14) !important;
}
[data-baseweb="popover"] [role="option"] { color: var(--text-mid) !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
  background: var(--bg-input) !important;
  border-color: var(--teal-border) !important;
  color: var(--teal) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"]>div {
  background: var(--bg-input) !important;
}
[data-testid="stVerticalBlock"]>[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, #ffffff, #f7fafd) !important;
  border-color: var(--border) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
.stSuccess,[data-testid="stNotificationContentSuccess"] {
  background: linear-gradient(90deg,rgba(2,136,167,0.06),rgba(2,136,167,0.02)) !important;
}
.stInfo,[data-testid="stNotificationContentInfo"] {
  background: linear-gradient(90deg,rgba(37,99,235,0.06),rgba(37,99,235,0.02)) !important;
}
.stWarning,[data-testid="stNotificationContentWarning"] {
  background: linear-gradient(90deg,rgba(217,84,10,0.06),rgba(217,84,10,0.02)) !important;
}
[data-testid="stSidebarNav"] li a { color: var(--text-mid) !important; }
[data-testid="stSidebarNav"] li a:hover {
  background: var(--teal-10) !important;
  color: var(--teal) !important;
}
[data-testid="stSidebarNav"] li [aria-current="page"] {
  background: var(--teal-15) !important;
  color: var(--teal) !important;
}
</style>"""


def render_theme_toggle():
    """Ajoute un toggle Jour/Sombre dans la sidebar. Appeler après le header sidebar."""
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True

    st.sidebar.toggle("Mode sombre", key="dark_mode")

    if not st.session_state["dark_mode"]:
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)
