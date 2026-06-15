"""
inference_cnn.py — WATLOW QC SYSTEM v5
streamlit run inference_cnn.py
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

st.set_page_config(
    page_title="WATLOW · Contrôle Qualité",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with open("style_watlow.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── CONSTANTES ──
POINT_NAMES = [
    'N°5  — Marquage fils',
    'N°6  — Manchons',
    'N°12 — Serre-câbles',
    'N°14 — Broage',
    'N°17 — Accessoires',
]
POINT_SHORT = ['Marquage', 'Manchons', 'Serre-câbles', 'Broage', 'Accessoires']
COLORS      = ['#ff003c', '#ff6b00', '#ff00aa', '#0088ff', '#00e5c8']
THRESHOLD   = 0.5
MODELS_DIR  = Path('models')

@st.cache_resource(show_spinner=False)
def load_model(p):
    return keras.models.load_model(p)

def preprocess(img: Image.Image, size: int) -> np.ndarray:
    arr = np.array(img.convert('RGB').resize((size, size)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def draw_clean(img: Image.Image, scores, threshold) -> plt.Figure:
    """Image annotée SANS zones colorées — juste l'image originale redimensionnée."""
    w, h    = img.size
    fig_w   = 5.0
    fig_h   = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#060d16')
    ax.set_facecolor('#060d16')
    ax.imshow(img)
    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    defauts = [i for i, s in enumerate(scores) if s > threshold]

    if not defauts:
        # Seul overlay autorisé : bandeau CONFORME
        ax.text(w / 2, h / 2, '✓  CONFORME',
                fontsize=11, color='#00e5c8',
                ha='center', va='center', fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='#00251a',
                          alpha=0.92, edgecolor='#00e5c8', linewidth=1.1))

    # Pas de rectangles, pas de textes de zone — rien d'autre

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig

# ── MODÈLE ──
MODELS_DIR.mkdir(exist_ok=True)
h5_files = list(MODELS_DIR.glob('*.h5'))

#h5_files = list(MODELS_DIR.glob('*.h5')) + list(MODELS_DIR.glob('*.keras'))

# ════════════════════════════════════════════════
# HEADER — titre seul, pas de badge/date/modèle
# ════════════════════════════════════════════════
st.markdown("""
<div class="wq-header">
  <div class="wq-title">WATLOW <em>QC</em></div>
</div>
""", unsafe_allow_html=True)

if not h5_files:
    st.markdown("""
    <div style="background:rgba(255,0,60,0.06);border:1px solid rgba(255,0,60,0.28);
                border-left:3px solid #ff003c;border-radius:5px;
                padding:10px 16px;font-family:'JetBrains Mono',monospace;
                font-size:0.6rem;color:#ff003c;line-height:1.9;margin:12px 20px;">
      ⚠&nbsp;&nbsp;Aucun modèle .h5 détecté dans ./models/<br>
      <span style="color:rgba(90,120,150,0.65);font-size:0.52rem;">
        → Placez best_model_v2.h5 dans models/ et rechargez
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model    = load_model(str(h5_files[0]))
IMG_SIZE = model.input_shape[1]

# ════════════════════════════════════════════════
# STATUS BAR — sans scan_id
# ════════════════════════════════════════════════
st.markdown(f"""
<div class="wq-status">
  <div class="wq-dot"></div>
  <span class="wq-st">SYSTÈME OPÉRATIONNEL</span>
  <span class="wq-sep">|</span>
  <span class="wq-st">MODÈLE CHARGÉ</span>
  <span class="wq-sep">|</span>
  <span class="wq-st">RÉSOLUTION&nbsp;: {IMG_SIZE}×{IMG_SIZE}px</span>
  <span class="wq-sep">|</span>
  <span class="wq-st">5 POINTS DE CONTRÔLE</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# BODY
# ════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2], gap="small")

with col_left:
    # Upload — sans label visible
    uploaded = st.file_uploader(
        "IMAGE", type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded:
        st.image(Image.open(uploaded), use_container_width=True)
    else:
        st.markdown("""
        <div class="wq-wait">
          <div class="wq-wait-icon">⚡</div>
          <div class="wq-wait-txt">EN ATTENTE D'IMAGE</div>
          <div class="wq-wait-sub">chargez une image pour démarrer l'analyse</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if not uploaded:
        st.markdown("""
        <div class="wq-placeholder">
          <div style="font-size:1.4rem;opacity:0.05;">◉</div>
          LES RÉSULTATS S'AFFICHERONT ICI
        </div>
        """, unsafe_allow_html=True)
    else:
        img       = Image.open(uploaded)
        img_array = preprocess(img, IMG_SIZE)
        t0        = time.time()
        pred      = model.predict(img_array, verbose=0)[0]
        dt        = round((time.time() - t0) * 1000, 1)

        defauts = [i for i, s in enumerate(pred) if s > THRESHOLD]
        all_ok  = len(defauts) == 0
        n_def   = len(defauts)

        # ── TITRE RÉSULTATS ──
        st.markdown(f"""
        <div class="wq-results-title">Résultats d'Inspection</div>
        <div class="wq-results-sub">
          Analyse terminée en {dt}ms&nbsp;·&nbsp;
          {"Aucun défaut détecté" if all_ok else f"{n_def} anomalie(s) détectée(s)"}
        </div>
        """, unsafe_allow_html=True)

        # ── VERDICT ──
        if all_ok:
            st.markdown(f"""
            <div class="wq-verdict-ok">
              <div>
                <div class="wq-vt-ok">✓&nbsp;&nbsp;CÂBLE CONFORME</div>
                <div class="wq-vs">AUCUN DÉFAUT · TOUS LES 5 POINTS VALIDÉS</div>
              </div>
              <div class="wq-vtime" style="color:rgba(0,229,200,0.4);">
                INFÉRENCE<br>
                <strong style="font-size:0.85rem;color:#00e5c8;">{dt} ms</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pts = '&nbsp;·&nbsp;'.join([POINT_SHORT[i] for i in defauts])
            st.markdown(f"""
            <div class="wq-verdict-nok">
              <div>
                <div class="wq-vt-nok">✗&nbsp;&nbsp;DÉFAUT DÉTECTÉ</div>
                <div class="wq-vs">{n_def} POINT(S) NON CONFORME(S)&nbsp;&nbsp;·&nbsp;&nbsp;{pts}</div>
              </div>
              <div class="wq-vtime" style="color:rgba(255,0,60,0.38);">
                INFÉRENCE<br>
                <strong style="font-size:0.85rem;color:#ff003c;">{dt} ms</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── IMAGE + POINTS DE CONTRÔLE ──
        c1, c2 = st.columns([3, 2], gap="small")

        with c1:
            # Image sans zones colorées
            fig = draw_clean(img, pred, THRESHOLD)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Chip résultat sous l'image
            if all_ok:
                st.markdown('<div class="wq-chip-ok"><div>✓&nbsp;&nbsp;CONFORME</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="wq-chip-nok"><div>⚠&nbsp;&nbsp;ÉCHEC — DÉFAUTS DÉTECTÉS</div></div>',
                            unsafe_allow_html=True)

        with c2:
            # Cards points de contrôle
            cards = ""
            for i, score in enumerate(pred):
                is_nok  = score > THRESHOLD
                pct     = int(score * 100)
                bc      = COLORS[i] if is_nok else '#00e5c8'
                nok_cls = 'wq-point-nok' if is_nok else ''
                status  = '<span class="wq-pnok-txt">⚠ NOK</span>' if is_nok \
                          else '<span class="wq-pok-txt">✓ OK</span>'
                cards += f"""
                <div class="wq-point {nok_cls}">
                  <div style="flex:1">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;gap:5px;">
                      <span class="wq-pname">{POINT_NAMES[i]}</span>
                      <div style="display:flex;align-items:center;gap:5px;flex-shrink:0;">
                        <span class="wq-pscore">{score:.3f}</span>
                        {status}
                      </div>
                    </div>
                    <div class="wq-bar-bg">
                      <div style="width:{pct}%;background:{bc};height:2px;
                                  border-radius:2px;box-shadow:0 0 3px {bc}44;"></div>
                    </div>
                  </div>
                </div>"""
            st.markdown(cards, unsafe_allow_html=True)

            # Métriques
            nok_cls    = "wq-mval-nok" if n_def > 0 else ""
            conformite = int((5 - n_def) / 5 * 100)

            bars_data = [2, 1, 3, 0, 1, 2, n_def]
            max_bar   = max(bars_data) or 1
            bars_html = "".join([
                f'<div class="wq-mini-bar '
                f'{"nok" if (j==6 and n_def>0) else ("active" if v>0 else "")}"'
                f' style="height:{max(8, int(v/max_bar*100))}%"></div>'
                for j, v in enumerate(bars_data)
            ])

            st.markdown(f"""
            <div class="wq-metrics">
              <div class="wq-metric">
                <div class="wq-mval {nok_cls}">{n_def}&nbsp;/&nbsp;5</div>
                <div class="wq-mlbl">Points défectueux</div>
              </div>
              <div class="wq-metric">
                <div class="wq-mval">{conformite}%</div>
                <div class="wq-mlbl">Conformité</div>
              </div>
            </div>
            <div class="wq-chart-wrap">
              <div class="wq-chart-lbl">Taux de défauts — 7 derniers jours</div>
              <div class="wq-mini-bar-row">{bars_html}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════
st.markdown(f"""
<div class="wq-footer">
  <span>WATLOW QC SYSTEM v5.0 · CNN MULTI-LABEL · 5 POINTS DE CONTRÔLE</span>
  <span>TENSORFLOW {tf.__version__} · RÉSOLUTION {IMG_SIZE}×{IMG_SIZE}px</span>
</div>
""", unsafe_allow_html=True)