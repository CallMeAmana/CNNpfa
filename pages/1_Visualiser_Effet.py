import streamlit as st
import albumentations as A
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Visualiser Effets — Câblage industriel",
    page_icon="👁",
    layout="wide"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Carte effet */
.effet-card {
    background: var(--background-color, #fafaf8);
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.15s;
}
.effet-card:hover { border-color: #1D9E75; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
    <div style="width:38px;height:38px;border-radius:10px;background:#E1F5EE;
        border:0.5px solid #9FE1CB;display:flex;align-items:center;
        justify-content:center;font-size:1.2rem;">👁</div>
    <div>
        <div style="font-size:1.3rem;font-weight:500;color:var(--text-color);">
            Visualiser les effets</div>
        <div style="font-size:0.72rem;color:#888;letter-spacing:0.1em;
            text-transform:uppercase;">
            Prévisualisation interactive — original vs augmenté
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "actifs" not in st.session_state:
    st.session_state.actifs = set()

# ── Définition des techniques ─────────────────────────────────────────────────
TECHNIQUES = {
    "Transformations géométriques": [
        {"id": "rotation",    "nom": "Rotation",          "desc": "Tourne l'image d'un angle", "icon": "↻"},
        {"id": "flipH",       "nom": "Miroir horizontal",  "desc": "Retourne gauche ↔ droite",  "icon": "↔"},
        {"id": "flipV",       "nom": "Miroir vertical",    "desc": "Retourne haut ↕ bas",        "icon": "↕"},
        {"id": "zoom",        "nom": "Zoom",               "desc": "Agrandit le centre",         "icon": "🔍"},
        {"id": "perspective", "nom": "Perspective",        "desc": "Déforme en profondeur",      "icon": "◱"},
        {"id": "shear",       "nom": "Cisaillement",       "desc": "Incline l'image",            "icon": "▱"},
    ],
    "Ajustements visuels": [
        {"id": "brightness",  "nom": "Luminosité",         "desc": "Rend plus claire ou sombre", "icon": "☀"},
        {"id": "contrast",    "nom": "Contraste",          "desc": "Accentue les différences",   "icon": "◑"},
        {"id": "saturation",  "nom": "Saturation HSV",     "desc": "Intensifie les couleurs",    "icon": "🎨"},
        {"id": "grayscale",   "nom": "Niveaux de gris",    "desc": "Convertit en noir/blanc",    "icon": "⬤"},
        {"id": "clahe",       "nom": "CLAHE",              "desc": "Égalisation adaptative",     "icon": "▦"},
        {"id": "sepia",       "nom": "Sépia",              "desc": "Ton chaud vintage",          "icon": "🟫"},
    ],
    "Effets avancés": [
        {"id": "blur",        "nom": "Flou gaussien",      "desc": "Adoucit les détails",        "icon": "💧"},
        {"id": "motion_blur", "nom": "Flou de mouvement",  "desc": "Simule le mouvement",        "icon": "➡"},
        {"id": "noise",       "nom": "Bruit gaussien",     "desc": "Ajoute du grain",            "icon": "✦"},
        {"id": "elastic",     "nom": "Elastic Transform",  "desc": "Déforme élastiquement",      "icon": "〰"},
        {"id": "grid",        "nom": "Grid Distortion",    "desc": "Distorsion en grille",       "icon": "▦"},
        {"id": "jpeg",        "nom": "Compression JPEG",   "desc": "Simule la compression",      "icon": "📷"},
    ],
}

BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
}

# ── Fonction d'application des effets ─────────────────────────────────────────
def appliquer_effets(img: np.ndarray, actifs: set, params: dict) -> np.ndarray:
    out = img.copy()

    if "rotation" in actifs:
        angle = params.get("rotation_angle", 15.0)
        border = BORDER_MODES[params.get("rotation_border", "reflect")]
        out = A.Rotate(limit=(angle, angle), border_mode=border, p=1.0)(image=out)["image"]

    if "flipH" in actifs:
        out = A.HorizontalFlip(p=1.0)(image=out)["image"]

    if "flipV" in actifs:
        out = A.VerticalFlip(p=1.0)(image=out)["image"]

    if "zoom" in actifs:
        scale = params.get("zoom_factor", 1.3) - 1.0
        out = A.RandomScale(scale_limit=(scale, scale), p=1.0)(image=out)["image"]
        out = cv2.resize(out, (img.shape[1], img.shape[0]))

    if "perspective" in actifs:
        v = params.get("perspective_scale", 0.1)
        out = A.Perspective(scale=(v, v), p=1.0)(image=out)["image"]

    if "shear" in actifs:
        angle = params.get("shear_angle", 15.0)
        out = A.Affine(shear=(-angle, angle), p=1.0)(image=out)["image"]

    if "brightness" in actifs:
        lim = params.get("brightness_val", 0.3)
        out = A.RandomBrightnessContrast(
            brightness_limit=(lim, lim), contrast_limit=0, p=1.0)(image=out)["image"]

    if "contrast" in actifs:
        lim = params.get("contrast_val", 0.3)
        out = A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(lim, lim), p=1.0)(image=out)["image"]

    if "saturation" in actifs:
        s = params.get("saturation_val", 30)
        out = A.HueSaturationValue(
            hue_shift_limit=0, sat_shift_limit=s, val_shift_limit=0, p=1.0)(image=out)["image"]

    if "grayscale" in actifs:
        out = A.ToGray(p=1.0)(image=out)["image"]
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    if "clahe" in actifs:
        clip = params.get("clahe_clip", 4.0)
        out = A.CLAHE(clip_limit=clip, p=1.0)(image=out)["image"]

    if "sepia" in actifs:
        intensity = params.get("sepia_intensity", 0.7)
        sepia_matrix = np.array([
            [0.393 + 0.607*(1-intensity), 0.769*(1-intensity), 0.189*(1-intensity)],
            [0.349*(1-intensity), 0.686 + 0.314*(1-intensity), 0.168*(1-intensity)],
            [0.272*(1-intensity), 0.534*(1-intensity), 0.131 + 0.869*(1-intensity)]
        ])
        out_f = out.astype(np.float32) / 255.0
        sepia_img = np.clip(out_f @ sepia_matrix.T, 0, 1)
        out = (sepia_img * 255).astype(np.uint8)

    if "blur" in actifs:
        k = params.get("blur_kernel", 5)
        k = k if k % 2 == 1 else k + 1
        k = max(3, k)
        out = A.GaussianBlur(blur_limit=(k, k), p=1.0)(image=out)["image"]

    if "motion_blur" in actifs:
        k = params.get("motion_kernel", 9)
        k = k if k % 2 == 1 else k + 1
        k = max(3, k)
        out = A.MotionBlur(blur_limit=(k, k), p=1.0)(image=out)["image"]

    if "noise" in actifs:
        var = params.get("noise_var", 30.0)
        out = A.GaussNoise(var_limit=(var, var), p=1.0)(image=out)["image"]

    if "elastic" in actifs:
        alpha = params.get("elastic_alpha", 60.0)
        out = A.ElasticTransform(alpha=alpha, sigma=10, p=1.0)(image=out)["image"]

    if "grid" in actifs:
        d = params.get("grid_distort", 0.3)
        out = A.GridDistortion(distort_limit=d, p=1.0)(image=out)["image"]

    if "jpeg" in actifs:
        q = params.get("jpeg_quality", 60)
        out = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)(image=out)["image"]

    return out


# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL : sidebar gauche | panneau droit
# ════════════════════════════════════════════════════════════════════════════════
col_side, col_main = st.columns([1, 2.4], gap="large")

# ── Colonne gauche : upload + sélection des techniques ───────────────────────
with col_side:

    # Upload
    uploaded = st.file_uploader(
        "Glissez votre image ici",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="vis_upload"
    )

    if uploaded:
        img_orig = np.array(Image.open(uploaded).convert("RGB"))
        img_orig = cv2.resize(img_orig, (512, 512))
        st.session_state["img_orig"] = img_orig
        st.markdown(
            f"<div style='font-size:0.72rem;color:#888;margin-bottom:0.5rem;'>"
            f"1 image importée — 512×512px</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    # Techniques avec toggles
    params = {}
    actifs_nouveau = set()

    for famille, techniques in TECHNIQUES.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;
            margin:1rem 0 0.4rem;padding:0 2px;">
            <div style="width:12px;height:1px;background:#9FE1CB;flex-shrink:0;"></div>
            <span style="font-size:0.6rem;font-weight:500;color:#1D9E75;
                text-transform:uppercase;letter-spacing:0.18em;white-space:nowrap;">
                {famille}
            </span>
            <div style="flex:1;height:1px;background:rgba(0,229,200,0.15);"></div>
        </div>
        """, unsafe_allow_html=True)

        for tech in techniques:
            tid = tech["id"]
            is_on = tid in st.session_state.actifs

            c_tog, c_icon, c_info = st.columns([0.5, 0.4, 2.2])
            with c_tog:
                tog = st.checkbox("", value=is_on, key=f"tog_{tid}", label_visibility="collapsed")
            with c_icon:
                st.markdown(
                    f"<div style='width:26px;height:26px;border-radius:6px;"
                    f"background:{'#E1F5EE' if tog else 'transparent'};"
                    f"border:0.5px solid {'#9FE1CB' if tog else 'rgba(0,0,0,0.1)'};"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-size:13px;margin-top:2px;'>{tech['icon']}</div>",
                    unsafe_allow_html=True
                )
            with c_info:
                st.markdown(
                    f"<div style='padding-top:2px;'>"
                    f"<div style='font-size:13px;font-weight:500;"
                    f"'>{tech['nom']}</div>"
                    f"<div style='font-size:11px;color:#888;margin-top:1px;'>"
                    f"{tech['desc']}</div></div>",
                    unsafe_allow_html=True
                )

            if tog:
                actifs_nouveau.add(tid)

    st.session_state.actifs = actifs_nouveau

    # Bouton reset
    if st.session_state.actifs:
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Réinitialiser tous les effets", use_container_width=True):
            st.session_state.actifs = set()
            st.rerun()


# ── Colonne droite : aperçu + paramètres ─────────────────────────────────────
with col_main:

    if "img_orig" not in st.session_state:
        st.markdown("""
        <div style="height:420px;border:1.5px dashed rgba(0,0,0,0.1);
            border-radius:12px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:12px;
            color:#aaa;font-size:0.9rem;">
            <div style="font-size:2.5rem;opacity:0.3;">🖼</div>
            <div>Chargez une image pour commencer</div>
            <div style="font-size:0.75rem;opacity:0.6;">JPG ou PNG · max 10 Mo</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        img_orig = st.session_state["img_orig"]
        actifs = st.session_state.actifs

        # ── Paramètres des effets actifs ──────────────────────────────────────
        if actifs:
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:500;color:#1D9E75;
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.6rem;">
                Paramètres des effets actifs
            </div>
            """, unsafe_allow_html=True)

            pcols = st.columns(2)
            ci = 0

            if "rotation" in actifs:
                with pcols[ci % 2]:
                    params["rotation_angle"] = st.slider("↻ Angle rotation (°)", -45, 45, 15, 1)
                    params["rotation_border"] = st.selectbox(
                        "Remplissage bords", ["reflect", "constant", "replicate"], key="rot_border")
                ci += 1

            if "zoom" in actifs:
                with pcols[ci % 2]:
                    params["zoom_factor"] = st.slider("🔍 Facteur zoom", 1.1, 2.0, 1.3, 0.05)
                ci += 1

            if "perspective" in actifs:
                with pcols[ci % 2]:
                    params["perspective_scale"] = st.slider("◱ Échelle perspective", 0.01, 0.2, 0.1, 0.01)
                ci += 1

            if "shear" in actifs:
                with pcols[ci % 2]:
                    params["shear_angle"] = st.slider("▱ Angle cisaillement (°)", 5, 30, 15, 1)
                ci += 1

            if "brightness" in actifs:
                with pcols[ci % 2]:
                    params["brightness_val"] = st.slider("☀ Luminosité", -0.5, 0.5, 0.3, 0.05)
                ci += 1

            if "contrast" in actifs:
                with pcols[ci % 2]:
                    params["contrast_val"] = st.slider("◑ Contraste", -0.5, 0.5, 0.3, 0.05)
                ci += 1

            if "saturation" in actifs:
                with pcols[ci % 2]:
                    params["saturation_val"] = st.slider("🎨 Saturation HSV", -50, 50, 30, 1)
                ci += 1

            if "clahe" in actifs:
                with pcols[ci % 2]:
                    params["clahe_clip"] = st.slider("▦ CLAHE clip limit", 1.0, 8.0, 4.0, 0.5)
                ci += 1

            if "sepia" in actifs:
                with pcols[ci % 2]:
                    params["sepia_intensity"] = st.slider("🟫 Sépia intensité", 0.0, 1.0, 0.7, 0.05)
                ci += 1

            if "blur" in actifs:
                with pcols[ci % 2]:
                    params["blur_kernel"] = st.slider("💧 Noyau flou (px)", 3, 15, 5, 2)
                ci += 1

            if "motion_blur" in actifs:
                with pcols[ci % 2]:
                    params["motion_kernel"] = st.slider("➡ Noyau mouvement (px)", 3, 21, 9, 2)
                ci += 1

            if "noise" in actifs:
                with pcols[ci % 2]:
                    params["noise_var"] = st.slider("✦ Variance bruit", 5.0, 100.0, 30.0, 5.0)
                ci += 1

            if "elastic" in actifs:
                with pcols[ci % 2]:
                    params["elastic_alpha"] = st.slider("〰 Alpha élastique", 10.0, 200.0, 60.0, 10.0)
                ci += 1

            if "grid" in actifs:
                with pcols[ci % 2]:
                    params["grid_distort"] = st.slider("▦ Distorsion grille", 0.1, 0.5, 0.3, 0.05)
                ci += 1

            if "jpeg" in actifs:
                with pcols[ci % 2]:
                    params["jpeg_quality"] = st.slider("📷 Qualité JPEG", 10, 95, 60, 5)
                ci += 1

            st.divider()

        # ── Aperçu côte à côte ────────────────────────────────────────────────
        nb_actifs = len(actifs)
        chips = " &nbsp;·&nbsp; ".join([
            next(t["nom"] for fam in TECHNIQUES.values() for t in fam if t["id"] == a)
            for a in actifs
        ]) if actifs else "<em style='opacity:0.5;'>Aucun effet actif — activez une technique à gauche</em>"

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.8rem;">
            <div style="font-size:0.72rem;font-weight:500;color:#1D9E75;
                text-transform:uppercase;letter-spacing:0.12em;">
                Aperçu
            </div>
            <div style="font-size:0.72rem;color:#888;">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

        col_orig, col_arrow, col_aug = st.columns([10, 1, 10])

        with col_orig:
            st.markdown("<div style='text-align:center;font-size:0.7rem;"
                        "color:#888;text-transform:uppercase;letter-spacing:0.1em;"
                        "margin-bottom:6px;'>Original</div>", unsafe_allow_html=True)
            st.image(img_orig, use_container_width=True)
            h, w = img_orig.shape[:2]
            st.caption(f"RGB · {w}×{h}px")

        with col_arrow:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:100%;font-size:1.4rem;color:#9FE1CB;padding-top:60px;'>→</div>",
                unsafe_allow_html=True
            )

        with col_aug:
            st.markdown("<div style='text-align:center;font-size:0.7rem;"
                        "color:#1D9E75;text-transform:uppercase;letter-spacing:0.1em;"
                        "margin-bottom:6px;font-weight:500;'>Résultat augmenté</div>",
                        unsafe_allow_html=True)
            if actifs:
                img_aug = appliquer_effets(img_orig, actifs, params)
                st.image(img_aug, use_container_width=True)
                st.caption(f"{'RGB' if len(img_aug.shape)==3 else 'Gris'} · {img_aug.shape[1]}×{img_aug.shape[0]}px · {nb_actifs} effet(s)")

                # Téléchargement
                buf = io.BytesIO()
                Image.fromarray(img_aug).save(buf, format="JPEG", quality=95)
                st.download_button(
                    "Télécharger l'image augmentée",
                    data=buf.getvalue(),
                    file_name="image_augmentee.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            else:
                st.markdown("""
                <div style="height:260px;border:1.5px dashed rgba(0,0,0,0.08);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;color:#ccc;font-size:0.85rem;">
                    ← Activez un effet
                </div>
                """, unsafe_allow_html=True)