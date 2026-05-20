"""
inference_cnn.py — WATLOW QC SYSTEM v5
Exécuter : streamlit run inference_cnn.py
Prérequis : best_model_v2.h5 dans ./models/
"""

import streamlit as st
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# ════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WATLOW · Contrôle Qualité",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

*, html, body, [class*="css"], .stApp {
    background-color: #060d16 !important;
    font-family: 'Exo 2', sans-serif !important;
    color: #b8ccd8 !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }

/* HEADER */
.wq-header {
    background: linear-gradient(90deg, #060d16 0%, #0c1f35 50%, #060d16 100%);
    border: 1px solid #00e5c830;
    border-top: 2px solid #00e5c8;
    border-radius: 6px;
    padding: 24px 36px 20px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.wq-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #fff; letter-spacing: 0.12em;
    text-transform: uppercase; line-height: 1; margin: 0;
}
.wq-title em { color: #00e5c8; font-style: normal; }
.wq-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; color: #00e5c860;
    letter-spacing: 0.18em; margin-top: 6px;
}
.wq-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem; color: #00e5c850;
    letter-spacing: 0.12em; text-align: right; line-height: 1.8;
}

/* STATUS BAR */
.wq-status {
    display: flex; align-items: center; gap: 16px;
    background: #0c1a26; border: 1px solid #1a2e40;
    border-radius: 4px; padding: 7px 16px; margin-bottom: 22px;
}
.wq-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00e5c8; flex-shrink: 0;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.wq-st  { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#00e5c8; letter-spacing:0.1em; }
.wq-sep { color:#1a2e40; }

/* UPLOAD */
[data-testid="stFileUploader"] > div:first-child {
    background: #0c1a26 !important;
    border: 1px dashed #00e5c840 !important;
    border-radius: 6px !important; padding: 20px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: #00e5c8 !important; background: #0f2030 !important;
}
[data-testid="stFileUploader"] button {
    background: #0c1a26 !important; border: 1px solid #00e5c840 !important;
    color: #00e5c8 !important; font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important; border-radius: 3px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #00e5c810 !important; border-color: #00e5c8 !important;
}
[data-testid="uploadedFileData"] {
    background: #0c1a26 !important; border: 1px solid #1a2e40 !important;
    border-radius: 4px !important;
}

/* LABEL */
.wq-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.28em; color: #00e5c8;
    border-left: 3px solid #00e5c8; padding-left: 8px; margin: 0 0 12px 0;
}

/* VERDICT */
.wq-verdict-ok {
    background: linear-gradient(90deg, #00251a, #003825);
    border: 1px solid #00e5c8; border-left: 5px solid #00e5c8;
    border-radius: 5px; padding: 18px 28px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center;
}
.wq-verdict-nok {
    background: linear-gradient(90deg, #200008, #300010);
    border: 1px solid #ff003c; border-left: 5px solid #ff003c;
    border-radius: 5px; padding: 18px 28px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center;
}
.wq-vt-ok  { font-family:'Rajdhani',sans-serif; font-size:1.7rem; font-weight:700; color:#00e5c8; letter-spacing:0.18em; }
.wq-vt-nok { font-family:'Rajdhani',sans-serif; font-size:1.7rem; font-weight:700; color:#ff003c; letter-spacing:0.18em; }
.wq-vs     { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#60788c; letter-spacing:0.1em; margin-top:4px; }
.wq-vtime  { font-family:'Share Tech Mono',monospace; font-size:0.62rem; letter-spacing:0.1em; text-align:right; }

/* POINT CARDS */
.wq-point {
    display: flex; align-items: center; gap: 10px;
    padding: 11px 14px; border-radius: 4px; margin-bottom: 6px;
    background: #0c1a26; border-left: 3px solid #1a3a50;
}
.wq-point-nok { background: #160010; border-left-color: #ff003c; }
.wq-pname { font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:600; color:#c8d8e8; flex:1; letter-spacing:0.04em; }
.wq-pscore    { font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#405060; }
.wq-pok-txt   { font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#00e5c8; font-weight:600; }
.wq-pnok-txt  { font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#ff003c; font-weight:600; }
.wq-bar-bg    { background:#1a2e40; border-radius:2px; height:3px; margin-top:5px; }

/* METRICS */
.wq-metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:16px; }
.wq-metric  { background:#0c1a26; border:1px solid #1a2e40; border-radius:4px; padding:12px; text-align:center; }
.wq-mval    { font-family:'Rajdhani',sans-serif; font-size:1.6rem; font-weight:700; color:#00e5c8; line-height:1; }
.wq-mval-nok{ color:#ff003c; }
.wq-mlbl    { font-family:'Share Tech Mono',monospace; font-size:0.55rem; color:#405060; letter-spacing:0.12em; margin-top:4px; }

/* WAITING */
.wq-wait {
    text-align:center; padding:80px 20px;
    border:1px dashed #1a2e40; border-radius:6px;
}
.wq-wait-icon { font-size:3rem; opacity:0.15; margin-bottom:14px; }
.wq-wait-txt  { font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#00e5c840; letter-spacing:0.22em; }
.wq-wait-sub  { font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#1a2e40; margin-top:8px; letter-spacing:0.12em; }

hr { border-color:#1a2e40 !important; margin:16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════
POINT_NAMES = [
    'N°5  — Marquage fils',
    'N°6  — Manchons',
    'N°12 — Serre-câbles',
    'N°14 — Broage',
    'N°17 — Accessoires',
]
POINT_SHORT = ['Marquage','Manchons','Serre-câbles','Broage','Accessoires']
ZONES = {
    0: (0.0, 0.00, 1.0, 0.20),
    1: (0.0, 0.00, 1.0, 0.20),
    2: (0.0, 0.30, 1.0, 0.60),
    3: (0.0, 0.20, 1.0, 0.50),
    4: (0.0, 0.00, 1.0, 1.00),
}
COLORS    = ['#ff003c','#ff6b00','#ff00aa','#0088ff','#00e5c8']
THRESHOLD = 0.5
MODELS_DIR = Path('models')

# ════════════════════════════════════════════════════════
# FONCTIONS
# ════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(p):
    return keras.models.load_model(p)

def preprocess(img: Image.Image, size: int) -> np.ndarray:
    arr = np.array(img.convert('RGB').resize((size,size)), dtype=np.float32)/255.0
    return np.expand_dims(arr, axis=0)

def draw_annotated(img: Image.Image, scores, threshold) -> plt.Figure:
    w, h   = img.size
    fig_w  = 5.0
    fig_h  = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#060d16')
    ax.set_facecolor('#060d16')
    ax.imshow(img)
    ax.axis('off')
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    defauts = [i for i,s in enumerate(scores) if s > threshold]

    if not defauts:
        ax.text(w/2, h/2, '✓  CONFORME',
                fontsize=14, color='#00e5c8',
                ha='center', va='center', fontweight='bold',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#00251a',
                         alpha=0.92, edgecolor='#00e5c8', linewidth=1.5))
    else:
        for idx in defauts:
            x1p,y1p,x2p,y2p = ZONES[idx]
            x1=int(x1p*w); y1=int(y1p*h)
            bw=int((x2p-x1p)*w); bh=int((y2p-y1p)*h)
            c = COLORS[idx]
            ax.add_patch(patches.Rectangle(
                (x1,y1),bw,bh, linewidth=2, edgecolor=c,
                facecolor=c, alpha=0.18))
            ax.add_patch(patches.Rectangle(
                (x1,y1),bw,bh, linewidth=1.5, edgecolor=c,
                facecolor='none', linestyle='--', alpha=0.55))
            ax.text(x1+6, y1+bh*0.1,
                    f'⚠  {POINT_SHORT[idx]}  [{scores[idx]:.2f}]',
                    fontsize=8, color='white', fontweight='bold',
                    fontfamily='monospace',
                    bbox=dict(facecolor=c, alpha=0.88,
                             boxstyle='round,pad=0.3', edgecolor='none'))

    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    return fig

# ════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════
MODELS_DIR.mkdir(exist_ok=True)
h5_files = list(MODELS_DIR.glob('*.h5')) + list(MODELS_DIR.glob('*.keras'))

# ════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════
now        = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
model_name = h5_files[0].name.upper() if h5_files else "—"

st.markdown(f"""
<div class="wq-header">
    <div>
        <div class="wq-title">WATLOW <em>QC</em> SYSTEM</div>
        <div class="wq-subtitle">
            // CONTRÔLE QUALITÉ CÂBLAGE INDUSTRIEL · CNN MULTI-LABEL · 5 POINTS DE CONTRÔLE
        </div>
    </div>
    <div class="wq-badge">
        MODÈLE · {model_name}<br>
        {now}
    </div>
</div>
""", unsafe_allow_html=True)

if not h5_files:
    st.markdown("""
    <div style="background:#160010;border:1px solid #ff003c;border-radius:5px;
                padding:20px;font-family:'Share Tech Mono',monospace;
                font-size:0.8rem;color:#ff003c;line-height:2.2;">
    ⚠  ERREUR — Aucun modèle .h5 détecté dans ./models/<br>
    <span style="color:#405060">
    → Téléchargez best_model_v2.h5 depuis Google Drive<br>
    → Placez-le dans le dossier models/ et rechargez la page
    </span></div>
    """, unsafe_allow_html=True)
    st.stop()

model    = load_model(str(h5_files[0]))
IMG_SIZE = model.input_shape[1]

st.markdown(f"""
<div class="wq-status">
    <div class="wq-dot"></div>
    <span class="wq-st">SYSTÈME OPÉRATIONNEL</span>
    <span class="wq-sep">|</span>
    <span class="wq-st">MODÈLE CHARGÉ</span>
    <span class="wq-sep">|</span>
    <span class="wq-st">RÉSOLUTION : {IMG_SIZE}×{IMG_SIZE} px</span>
    <span class="wq-sep">|</span>
    <span class="wq-st">5 POINTS DE CONTRÔLE</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2], gap="large")

# ── COLONNE GAUCHE : Upload + image originale ──
with col_left:
    st.markdown('<div class="wq-label">// acquisition image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "IMAGE", type=['jpg','jpeg','png'],
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="wq-label">// image originale</div>', unsafe_allow_html=True)
        st.image(Image.open(uploaded), use_container_width=True)
    else:
        st.markdown("""
        <div class="wq-wait">
            <div class="wq-wait-icon">⚡</div>
            <div class="wq-wait-txt">EN ATTENTE D'IMAGE</div>
            <div class="wq-wait-sub">chargez une image pour démarrer l'analyse</div>
        </div>
        """, unsafe_allow_html=True)

# ── COLONNE DROITE : Résultats ──
with col_right:
    if not uploaded:
        st.markdown("""
        <div style="height:420px;display:flex;align-items:center;
                    justify-content:center;border:1px dashed #1a2e40;
                    border-radius:6px;
                    font-family:'Share Tech Mono',monospace;
                    font-size:0.65rem;color:#1a2e40;
                    text-align:center;letter-spacing:0.15em;">
            LES RÉSULTATS S'AFFICHERONT ICI
        </div>
        """, unsafe_allow_html=True)
    else:
        img       = Image.open(uploaded)
        img_array = preprocess(img, IMG_SIZE)

        t0   = time.time()
        pred = model.predict(img_array, verbose=0)[0]
        dt   = round((time.time()-t0)*1000, 1)

        defauts = [i for i,s in enumerate(pred) if s > THRESHOLD]
        all_ok  = len(defauts) == 0
        n_def   = len(defauts)

        # ── VERDICT ──
        if all_ok:
            st.markdown(f"""
            <div class="wq-verdict-ok">
                <div>
                    <div class="wq-vt-ok">✓  CÂBLE CONFORME</div>
                    <div class="wq-vs">AUCUN DÉFAUT · TOUS LES 5 POINTS VALIDÉS</div>
                </div>
                <div class="wq-vtime" style="color:#00e5c840;">
                    INFÉRENCE<br>{dt} ms
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pts = '  ·  '.join([POINT_SHORT[i] for i in defauts])
            st.markdown(f"""
            <div class="wq-verdict-nok">
                <div>
                    <div class="wq-vt-nok">✗  DÉFAUT DÉTECTÉ</div>
                    <div class="wq-vs">{n_def} POINT(S) NON CONFORME(S)  ·  {pts}</div>
                </div>
                <div class="wq-vtime" style="color:#ff003c40;">
                    INFÉRENCE<br>{dt} ms
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── IMAGE ANNOTÉE + POINTS DE CONTRÔLE ──
        c1, c2 = st.columns([1,1], gap="medium")

        with c1:
            st.markdown('<div class="wq-label">// zones analysées</div>', unsafe_allow_html=True)
            fig = draw_annotated(img, pred, THRESHOLD)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown('<div class="wq-label">// points de contrôle</div>', unsafe_allow_html=True)

            cards = ""
            for i, score in enumerate(pred):
                is_nok = score > THRESHOLD
                pct    = int(score*100)
                bc     = COLORS[i] if is_nok else '#00e5c8'
                st_lbl = f'<span class="wq-pnok-txt">⚠ NOK</span>' if is_nok \
                         else f'<span class="wq-pok-txt">✓ OK</span>'
                cards += f"""
                <div class="wq-point {'wq-point-nok' if is_nok else ''}">
                    <div style="flex:1">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;gap:6px;">
                            <span class="wq-pname">{POINT_NAMES[i]}</span>
                            <div style="display:flex;align-items:center;
                                        gap:8px;flex-shrink:0;">
                                <span class="wq-pscore">{score:.3f}</span>
                                {st_lbl}
                            </div>
                        </div>
                        <div class="wq-bar-bg">
                            <div style="width:{pct}%;background:{bc};
                                        height:3px;border-radius:2px;"></div>
                        </div>
                    </div>
                </div>"""
            st.markdown(cards, unsafe_allow_html=True)

            # ── MÉTRIQUES ──
            nok_cls = "wq-mval-nok" if n_def > 0 else "wq-mval"
            conformite = int((5-n_def)/5*100)
            st.markdown(f"""
            <div class="wq-metrics">
                <div class="wq-metric">
                    <div class="wq-mval {nok_cls}">{n_def} / 5</div>
                    <div class="wq-mlbl">POINTS DÉFECTUEUX</div>
                </div>
                <div class="wq-metric">
                    <div class="wq-mval">{conformite}%</div>
                    <div class="wq-mlbl">CONFORMITÉ</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;
            font-family:'Share Tech Mono',monospace;
            font-size:0.55rem;color:#1a2e40;">
    <span>WATLOW QC SYSTEM v5.0 · CNN MULTI-LABEL · 5 POINTS DE CONTRÔLE</span>
    <span>TENSORFLOW {tf.__version__} · RÉSOLUTION {IMG_SIZE}×{IMG_SIZE}px</span>
</div>
""", unsafe_allow_html=True)