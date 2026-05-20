import streamlit as st
import albumentations as A
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import io
import pandas as pd
from pathlib import Path
from itertools import product as itertools_product


def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


st.set_page_config(
    page_title="Data Augmentation — Câblage industriel",
    page_icon="🔬",
    layout="wide"
)

# --- Badge


st.title("Systeme de Data Augmentation")
st.markdown("""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#4a5568;
    margin-top:-0.3rem;margin-bottom:1.5rem;">
    Configurez chaque technique &mdash; valeur max &amp; nombre de pas precis.&nbsp;&nbsp;
    <span style="color:rgba(249,115,22,0.4);">|</span>&nbsp;&nbsp;
    <span style="color:rgba(249,115,22,0.75);">Augmentation combinatoire deterministe</span>
</p>
""", unsafe_allow_html=True)
st.divider()

# ════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════
def generer_valeurs(val_max, nb_pas):
    if nb_pas <= 0:
        return [val_max]
    pas = val_max / nb_pas
    return [round(pas * i, 4) for i in range(1, nb_pas + 1)]

BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE
}

# ════════════════════════════════════════════════
# SIDEBAR — FAMILLE 1
# ════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="padding:1rem 0.5rem 0.75rem;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.4rem;">
        <div style="width:28px;height:28px;background:rgba(0,229,200,0.1);
            border:1px solid rgba(0,229,200,0.3);border-radius:6px;
            display:flex;align-items:center;justify-content:center;font-size:0.85rem;">
            &#9707;
        </div>
        <span style="font-family:'Syne',sans-serif;font-size:0.78rem;font-weight:700;
            color:#ffffff;letter-spacing:0.05em;text-transform:uppercase;">
            Augmentation
        </span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
        color:rgba(0,229,200,0.5);letter-spacing:0.15em;padding-left:36px;">
        CONFIGURATION DES TECHNIQUES
    </div>
</div>
""", unsafe_allow_html=True)



st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin:1.2rem 0 0.6rem;padding:0 2px;">
    <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
        color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
        F1 &mdash; Geometrique
    </span>
    <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
</div>
""", unsafe_allow_html=True)
 
use_rotation = st.sidebar.checkbox("Rotation", value=True)
rotation_max = st.sidebar.slider("Angle max (°)", 5, 90, 25, 5, disabled=not use_rotation)
rotation_pas = st.sidebar.number_input("Nombre de pas — Rotation", 1, 20, 5, disabled=not use_rotation)
rotation_border = st.sidebar.selectbox("Mode remplissage bords", ["reflect", "constant", "replicate"], disabled=not use_rotation)
st.sidebar.markdown("---")

use_flip_h = st.sidebar.checkbox("Flip horizontal", value=True)
use_flip_v = st.sidebar.checkbox("Flip vertical", value=False)
st.sidebar.markdown("---")

use_zoom = st.sidebar.checkbox("Zoom", value=False)
zoom_max = st.sidebar.slider("Facteur zoom max", 1.1, 2.0, 1.3, 0.1, disabled=not use_zoom)
zoom_pas = st.sidebar.number_input("Nombre de pas — Zoom", 1, 20, 3, disabled=not use_zoom)
st.sidebar.markdown("---")

use_perspective = st.sidebar.checkbox("Perspective", value=False)
perspective_max = st.sidebar.slider("Échelle perspective max", 0.01, 0.2, 0.1, 0.01, disabled=not use_perspective)
perspective_pas = st.sidebar.number_input("Nombre de pas — Perspective", 1, 10, 3, disabled=not use_perspective)
st.sidebar.markdown("---")

use_shear = st.sidebar.checkbox("Cisaillement (Shear)", value=False)
shear_max = st.sidebar.slider("Angle cisaillement max (°)", 5, 30, 15, 5, disabled=not use_shear)
shear_pas = st.sidebar.number_input("Nombre de pas — Cisaillement", 1, 10, 3, disabled=not use_shear)
st.sidebar.markdown("---")

use_shift = st.sidebar.checkbox("Translation (Shift)", value=False)
shift_max = st.sidebar.slider("Déplacement max (%)", 0.05, 0.3, 0.1, 0.05, disabled=not use_shift)
shift_pas = st.sidebar.number_input("Nombre de pas — Translation", 1, 10, 3, disabled=not use_shift)

# ════════════════════════════════════════════════
# SIDEBAR — FAMILLE 2
# ════════════════════════════════════════════════
 
st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.6rem;padding:0 2px;">
    <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
        color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
        F2 &mdash; Photometrique
    </span>
    <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
</div>
""", unsafe_allow_html=True)


use_brightness = st.sidebar.checkbox("Luminosité / Contraste", value=True)
brightness_max = st.sidebar.slider("Luminosité max", 0.1, 0.5, 0.3, 0.05, disabled=not use_brightness)
brightness_pas = st.sidebar.number_input("Nombre de pas — Luminosité", 1, 20, 3, disabled=not use_brightness)
st.sidebar.markdown("---")

use_noise = st.sidebar.checkbox("Bruit gaussien", value=False)
noise_max = st.sidebar.slider("Variance bruit max", 10, 100, 50, 5, disabled=not use_noise)
noise_pas = st.sidebar.number_input("Nombre de pas — Bruit", 1, 20, 4, disabled=not use_noise)
st.sidebar.markdown("---")

use_blur = st.sidebar.checkbox("Flou gaussien", value=False)
blur_max = st.sidebar.slider("Noyau flou max (px)", 3, 11, 7, 2, disabled=not use_blur)
blur_pas = st.sidebar.number_input("Nombre de pas — Flou", 1, 5, 2, disabled=not use_blur)
st.sidebar.markdown("---")

use_motion_blur = st.sidebar.checkbox("Flou de mouvement", value=False)
motion_max = st.sidebar.slider("Noyau mouvement max (px)", 3, 21, 9, 2, disabled=not use_motion_blur)
motion_pas = st.sidebar.number_input("Nombre de pas — Flou mouvement", 1, 5, 2, disabled=not use_motion_blur)
st.sidebar.markdown("---")

use_hsv = st.sidebar.checkbox("Saturation HSV", value=False)
hsv_max = st.sidebar.slider("Saturation max", 10, 50, 30, 5, disabled=not use_hsv)
hsv_pas = st.sidebar.number_input("Nombre de pas — HSV", 1, 10, 3, disabled=not use_hsv)
st.sidebar.markdown("---")

use_clahe = st.sidebar.checkbox("Égalisation CLAHE", value=False)
clahe_max = st.sidebar.slider("Clip limit max", 1.0, 8.0, 4.0, 0.5, disabled=not use_clahe)
clahe_pas = st.sidebar.number_input("Nombre de pas — CLAHE", 1, 8, 2, disabled=not use_clahe)
st.sidebar.markdown("---")

use_jpeg = st.sidebar.checkbox("Compression JPEG", value=False)
jpeg_min = st.sidebar.slider("Qualité JPEG min", 50, 95, 70, 5, disabled=not use_jpeg)
jpeg_pas = st.sidebar.number_input("Nombre de pas — JPEG", 1, 5, 2, disabled=not use_jpeg)
st.sidebar.markdown("---")

use_togray = st.sidebar.checkbox("Niveaux de gris (ToGray)", value=False)

# ════════════════════════════════════════════════
# SIDEBAR — FAMILLE 4
# ════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.6rem;padding:0 2px;">
    <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
        color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
        F4 &mdash; Mixage
    </span>
    <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
</div>
""", unsafe_allow_html=True)



use_mixup = st.sidebar.checkbox("Mixup", value=False)
mixup_max = st.sidebar.slider("Alpha Mixup max", 0.1, 0.5, 0.3, 0.05, disabled=not use_mixup)
mixup_pas = st.sidebar.number_input("Nombre de pas — Mixup", 1, 5, 2, disabled=not use_mixup)
st.sidebar.markdown("---")

use_cutmix = st.sidebar.checkbox("CutMix", value=False)
cutmix_max = st.sidebar.slider("Alpha CutMix max", 0.1, 1.0, 0.5, 0.1, disabled=not use_cutmix)
cutmix_pas = st.sidebar.number_input("Nombre de pas — CutMix", 1, 5, 2, disabled=not use_cutmix)

# ════════════════════════════════════════════════
# SIDEBAR — FAMILLE 5
# ════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.6rem;padding:0 2px;">
    <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
        color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
        F5 &mdash; Avancee
    </span>
    <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
</div>
""", unsafe_allow_html=True)
 


use_elastic = st.sidebar.checkbox("Elastic Transform", value=False)
elastic_max = st.sidebar.slider("Alpha élastique max", 10, 200, 80, 10, disabled=not use_elastic)
elastic_pas = st.sidebar.number_input("Nombre de pas — Elastic", 1, 10, 3, disabled=not use_elastic)
st.sidebar.markdown("---")

use_grid = st.sidebar.checkbox("Grid Distortion", value=False)
grid_max = st.sidebar.slider("Distorsion grille max", 0.1, 0.5, 0.3, 0.05, disabled=not use_grid)
grid_pas = st.sidebar.number_input("Nombre de pas — Grid", 1, 5, 2, disabled=not use_grid)
st.sidebar.markdown("---")

use_optical = st.sidebar.checkbox("Optical Distortion", value=False)
optical_max = st.sidebar.slider("Distorsion optique max", 0.05, 0.5, 0.2, 0.05, disabled=not use_optical)
optical_pas = st.sidebar.number_input("Nombre de pas — Optical", 1, 5, 2, disabled=not use_optical)

# ════════════════════════════════════════════════
# SIDEBAR — RÉSOLUTION
# ════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.markdown("""
<div style="margin:1.6rem 0 0.6rem;">
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
            color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
            Resolution de sortie
        </span>
        <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
    </div>
</div>
""", unsafe_allow_html=True)
 

output_size = st.sidebar.selectbox("Résolution", [224, 512, 768, 1024], index=1)

# ════════════════════════════════════════════════
# CONSTRUCTION TECHNIQUES ACTIVES
# ════════════════════════════════════════════════
techniques_actives = []

if use_rotation:
    valeurs = generer_valeurs(rotation_max, int(rotation_pas))
    techniques_actives.append({"nom": "Rotation", "parametre": "angle (°)", "valeurs": valeurs, "nb_pas": int(rotation_pas), "max": rotation_max, "taille_pas": round(rotation_max / int(rotation_pas), 2)})

if use_flip_h:
    techniques_actives.append({"nom": "Flip horizontal", "parametre": "actif", "valeurs": [1], "nb_pas": 1, "max": 1, "taille_pas": "-"})

if use_flip_v:
    techniques_actives.append({"nom": "Flip vertical", "parametre": "actif", "valeurs": [1], "nb_pas": 1, "max": 1, "taille_pas": "-"})

if use_zoom:
    valeurs = [round(v + 1.0, 4) for v in generer_valeurs(zoom_max - 1.0, int(zoom_pas))]
    techniques_actives.append({"nom": "Zoom", "parametre": "facteur", "valeurs": valeurs, "nb_pas": int(zoom_pas), "max": zoom_max, "taille_pas": round((zoom_max - 1.0) / int(zoom_pas), 3)})

if use_perspective:
    valeurs = generer_valeurs(perspective_max, int(perspective_pas))
    techniques_actives.append({"nom": "Perspective", "parametre": "scale", "valeurs": valeurs, "nb_pas": int(perspective_pas), "max": perspective_max, "taille_pas": round(perspective_max / int(perspective_pas), 3)})

if use_shear:
    valeurs = generer_valeurs(shear_max, int(shear_pas))
    techniques_actives.append({"nom": "Cisaillement", "parametre": "angle (°)", "valeurs": valeurs, "nb_pas": int(shear_pas), "max": shear_max, "taille_pas": round(shear_max / int(shear_pas), 2)})

if use_shift:
    valeurs = generer_valeurs(shift_max, int(shift_pas))
    techniques_actives.append({"nom": "Translation", "parametre": "déplacement", "valeurs": valeurs, "nb_pas": int(shift_pas), "max": shift_max, "taille_pas": round(shift_max / int(shift_pas), 3)})

if use_brightness:
    valeurs = generer_valeurs(brightness_max, int(brightness_pas))
    techniques_actives.append({"nom": "Luminosité", "parametre": "limit", "valeurs": valeurs, "nb_pas": int(brightness_pas), "max": brightness_max, "taille_pas": round(brightness_max / int(brightness_pas), 3)})

if use_noise:
    valeurs = generer_valeurs(noise_max, int(noise_pas))
    techniques_actives.append({"nom": "Bruit gaussien", "parametre": "variance", "valeurs": valeurs, "nb_pas": int(noise_pas), "max": noise_max, "taille_pas": round(noise_max / int(noise_pas), 1)})

if use_blur:
    raw = generer_valeurs(blur_max, int(blur_pas))
    valeurs = [max(3, int(v) if int(v) % 2 == 1 else int(v) + 1) for v in raw]
    techniques_actives.append({"nom": "Flou gaussien", "parametre": "noyau (px)", "valeurs": valeurs, "nb_pas": int(blur_pas), "max": blur_max, "taille_pas": round(blur_max / int(blur_pas), 1)})

if use_motion_blur:
    raw = generer_valeurs(motion_max, int(motion_pas))
    valeurs = [max(3, int(v) if int(v) % 2 == 1 else int(v) + 1) for v in raw]
    techniques_actives.append({"nom": "Flou mouvement", "parametre": "noyau (px)", "valeurs": valeurs, "nb_pas": int(motion_pas), "max": motion_max, "taille_pas": round(motion_max / int(motion_pas), 1)})

if use_hsv:
    valeurs = generer_valeurs(hsv_max, int(hsv_pas))
    techniques_actives.append({"nom": "Saturation HSV", "parametre": "sat_shift", "valeurs": valeurs, "nb_pas": int(hsv_pas), "max": hsv_max, "taille_pas": round(hsv_max / int(hsv_pas), 1)})

if use_clahe:
    valeurs = generer_valeurs(clahe_max, int(clahe_pas))
    techniques_actives.append({"nom": "CLAHE", "parametre": "clip_limit", "valeurs": valeurs, "nb_pas": int(clahe_pas), "max": clahe_max, "taille_pas": round(clahe_max / int(clahe_pas), 2)})

if use_jpeg:
    raw = generer_valeurs(100 - jpeg_min, int(jpeg_pas))
    valeurs = [round(100 - v, 0) for v in reversed(raw)]
    techniques_actives.append({"nom": "Compression JPEG", "parametre": "qualité min", "valeurs": valeurs, "nb_pas": int(jpeg_pas), "max": jpeg_min, "taille_pas": round((100 - jpeg_min) / int(jpeg_pas), 1)})

if use_togray:
    techniques_actives.append({"nom": "Niveaux de gris", "parametre": "actif", "valeurs": [1], "nb_pas": 1, "max": 1, "taille_pas": "-"})

if use_mixup:
    valeurs = generer_valeurs(mixup_max, int(mixup_pas))
    techniques_actives.append({"nom": "Mixup", "parametre": "alpha", "valeurs": valeurs, "nb_pas": int(mixup_pas), "max": mixup_max, "taille_pas": round(mixup_max / int(mixup_pas), 3)})

if use_cutmix:
    valeurs = generer_valeurs(cutmix_max, int(cutmix_pas))
    techniques_actives.append({"nom": "CutMix", "parametre": "alpha", "valeurs": valeurs, "nb_pas": int(cutmix_pas), "max": cutmix_max, "taille_pas": round(cutmix_max / int(cutmix_pas), 3)})

if use_elastic:
    valeurs = generer_valeurs(elastic_max, int(elastic_pas))
    techniques_actives.append({"nom": "Elastic Transform", "parametre": "alpha", "valeurs": valeurs, "nb_pas": int(elastic_pas), "max": elastic_max, "taille_pas": round(elastic_max / int(elastic_pas), 1)})

if use_grid:
    valeurs = generer_valeurs(grid_max, int(grid_pas))
    techniques_actives.append({"nom": "Grid Distortion", "parametre": "distort_limit", "valeurs": valeurs, "nb_pas": int(grid_pas), "max": grid_max, "taille_pas": round(grid_max / int(grid_pas), 3)})

if use_optical:
    valeurs = generer_valeurs(optical_max, int(optical_pas))
    techniques_actives.append({"nom": "Optical Distortion", "parametre": "distort_limit", "valeurs": valeurs, "nb_pas": int(optical_pas), "max": optical_max, "taille_pas": round(optical_max / int(optical_pas), 3)})

# ════════════════════════════════════════════════
# CALCUL x
# ════════════════════════════════════════════════
x = 1
for t in techniques_actives:
    x *= t["nb_pas"]

# ════════════════════════════════════════════════
# AFFICHAGE PARAMÈTRES GLOBAUX
# ════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
    <div style="width:36px;height:36px;background:rgba(8,145,178,0.1);
        border:1px solid rgba(8,145,178,0.25);border-radius:8px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">&#9881;</div>
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#0f2a38;">
            Parametres globaux</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            color:#3b6e82;letter-spacing:0.12em;text-transform:uppercase;">
            Configuration active</div>
    </div>
</div>
""", unsafe_allow_html=True)
if not techniques_actives:
    st.warning("Aucune technique activée. Cochez au moins une technique dans le panneau gauche.")
else:
    formule = " × ".join([f"{t['nb_pas']} ({t['nom']})" for t in techniques_actives])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Techniques actives", len(techniques_actives))
    c2.metric("Facteur x", x)
    c3.metric("Images par originale", x)

st.divider()

# ════════════════════════════════════════════════
# CHARGEMENT IMAGES
# ════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
    <div style="width:36px;height:36px;background:rgba(249,115,22,0.12);
        border:1px solid rgba(249,115,22,0.3);border-radius:8px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">&#9881;</div>
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#1a1f2e;">
            Parametres globaux</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;">
            Configuration active</div>
    </div>
</div>
""", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Sélectionnez vos images (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and techniques_actives:
    images_numpy = []
    for f in uploaded_files:
        img = np.array(Image.open(f).convert("RGB"))
        img = cv2.resize(img, (output_size, output_size))
        images_numpy.append((f.name, img))

    st.success(f"{len(images_numpy)} image(s) chargée(s)")
    st.info(f"Total images qui seront générées : {len(images_numpy)} × {x} = **{len(images_numpy) * x} images**")

    n_prev = min(len(images_numpy), 4)
    cols_orig = st.columns(n_prev)
    for i in range(n_prev):
        nom, img = images_numpy[i]
        h, w = img.shape[:2]
        cols_orig[i].image(img, caption=nom, use_container_width=True)
        cols_orig[i].caption(f"RGB | {w}×{h}px")

    st.divider()

    if st.button("Appliquer la génération", type="primary", use_container_width=True):

        resultats = []
        progress = st.progress(0, text="Génération en cours...")
        listes_valeurs = [t["valeurs"] for t in techniques_actives]
        combinaisons = list(itertools_product(*listes_valeurs))
        total = len(images_numpy) * len(combinaisons)
        compteur = 0

        for nom_fichier, img in images_numpy:
            stem = Path(nom_fichier).stem

            for idx_combo, combo in enumerate(combinaisons):
                img_aug = img.copy()
                label_parts = []

                for tech, valeur in zip(techniques_actives, combo):
                    nom_tech = tech["nom"]

                    if nom_tech == "Rotation":
                        angle = float(valeur)
                        img_aug = A.Rotate(limit=(angle, angle), border_mode=BORDER_MODES[rotation_border], p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"rot{angle}deg")

                    elif nom_tech == "Flip horizontal":
                        img_aug = A.HorizontalFlip(p=1.0)(image=img_aug)["image"]
                        label_parts.append("flipH")

                    elif nom_tech == "Flip vertical":
                        img_aug = A.VerticalFlip(p=1.0)(image=img_aug)["image"]
                        label_parts.append("flipV")

                    elif nom_tech == "Zoom":
                        scale = float(valeur) - 1.0
                        img_aug = A.RandomScale(scale_limit=(scale, scale), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"zoom{valeur}")

                    elif nom_tech == "Perspective":
                        v = float(valeur)
                        img_aug = A.Perspective(scale=(v, v), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"persp{v}")

                    elif nom_tech == "Cisaillement":
                        angle = float(valeur)
                        img_aug = A.Affine(shear=(-angle, angle), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"shear{angle}deg")

                    elif nom_tech == "Translation":
                        shift = float(valeur)
                        img_aug = A.ShiftScaleRotate(shift_limit=shift, scale_limit=0, rotate_limit=0, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"shift{shift}")

                    elif nom_tech == "Luminosité":
                        lim = float(valeur)
                        img_aug = A.RandomBrightnessContrast(brightness_limit=(lim, lim), contrast_limit=(lim, lim), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"bright{lim}")

                    elif nom_tech == "Bruit gaussien":
                        var = float(valeur)
                        img_aug = A.GaussNoise(var_limit=(var, var), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"noise{var}")

                    elif nom_tech == "Flou gaussien":
                        k = int(valeur)
                        img_aug = A.GaussianBlur(blur_limit=(k, k), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"blur{k}px")

                    elif nom_tech == "Flou mouvement":
                        k = int(valeur)
                        img_aug = A.MotionBlur(blur_limit=(k, k), p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"mblur{k}px")

                    elif nom_tech == "Saturation HSV":
                        s = int(valeur)
                        img_aug = A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=s, val_shift_limit=0, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"hsv{s}")

                    elif nom_tech == "CLAHE":
                        clip = float(valeur)
                        img_aug = A.CLAHE(clip_limit=clip, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"clahe{clip}")

                    elif nom_tech == "Compression JPEG":
                        q = int(valeur)
                        img_aug = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"jpeg{q}")

                    elif nom_tech == "Niveaux de gris":
                        img_aug = A.ToGray(p=1.0)(image=img_aug)["image"]
                        if len(img_aug.shape) == 2:
                            img_aug = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2RGB)
                        label_parts.append("gray")

                    elif nom_tech == "Mixup":
                        alpha = float(valeur)
                        autre = images_numpy[np.random.randint(len(images_numpy))][1]
                        img_aug = (alpha * img_aug.astype(np.float32) + (1 - alpha) * autre.astype(np.float32)).astype(np.uint8)
                        label_parts.append(f"mixup{alpha}")

                    elif nom_tech == "CutMix":
                        alpha = float(valeur)
                        autre = images_numpy[np.random.randint(len(images_numpy))][1]
                        h2, w2 = img_aug.shape[:2]
                        lam = np.random.beta(alpha, alpha)
                        cut_w = int(w2 * np.sqrt(1 - lam))
                        cut_h = int(h2 * np.sqrt(1 - lam))
                        cx, cy = np.random.randint(w2), np.random.randint(h2)
                        x1 = max(cx - cut_w // 2, 0)
                        y1 = max(cy - cut_h // 2, 0)
                        x2 = min(cx + cut_w // 2, w2)
                        y2 = min(cy + cut_h // 2, h2)
                        img_aug[y1:y2, x1:x2] = cv2.resize(autre, (x2 - x1, y2 - y1))
                        label_parts.append(f"cutmix{alpha}")

                    elif nom_tech == "Elastic Transform":
                        a = float(valeur)
                        img_aug = A.ElasticTransform(alpha=a, sigma=10, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"elastic{a}")

                    elif nom_tech == "Grid Distortion":
                        d = float(valeur)
                        img_aug = A.GridDistortion(distort_limit=d, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"grid{d}")

                    elif nom_tech == "Optical Distortion":
                        d = float(valeur)
                        img_aug = A.OpticalDistortion(distort_limit=d, p=1.0)(image=img_aug)["image"]
                        label_parts.append(f"optical{d}")

                img_aug = cv2.resize(img_aug, (output_size, output_size))
                label_str = "__".join(label_parts)
                nom_aug = f"{stem}__{label_str}__combo{idx_combo:04d}.jpg"
                resultats.append((nom_aug, img_aug, label_str))

                compteur += 1
                progress.progress(compteur / total, text=f"Génération : {compteur}/{total}")

        st.success(f"{len(resultats)} images générées !")
        st.divider()

        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images originales", len(images_numpy))
        c2.metric("Facteur x", x)
        c3.metric("Images générées", len(resultats))
        c4.metric("Techniques", len(techniques_actives))

        st.divider()

        # Aperçu
        st.subheader("Aperçu — 12 premières images augmentées")
        cols_res = st.columns(4)
        for i, (nom, img_aug, label) in enumerate(resultats[:12]):
            col = cols_res[i % 4]
            h, w = img_aug.shape[:2]
            c_nb = img_aug.shape[2] if len(img_aug.shape) == 3 else 1
            col.image(img_aug, use_container_width=True)
            col.caption(f"{'RGB' if c_nb == 3 else 'Gris'} | {w}×{h}px\n{label[:45]}...")

        st.divider()

        

        # Tableau caractéristiques
        st.subheader("Caractéristiques détaillées — 8 premières images")
        rows_c = []
        for nom, img_aug, label in resultats[:8]:
            h, w = img_aug.shape[:2]
            c_nb = img_aug.shape[2] if len(img_aug.shape) == 3 else 1
            rows_c.append({
                "Fichier": nom[:50],
                "Type couleur": "RGB" if c_nb == 3 else "Gris",
                "Dimensions": f"{w}×{h}",
                "Canaux": c_nb,
                "Moy. R": round(float(img_aug[:,:,0].mean()), 1) if c_nb == 3 else "-",
                "Moy. G": round(float(img_aug[:,:,1].mean()), 1) if c_nb == 3 else "-",
                "Moy. B": round(float(img_aug[:,:,2].mean()), 1) if c_nb == 3 else "-",
                "Techniques appliquées": label[:60],
            })
        st.dataframe(pd.DataFrame(rows_c), use_container_width=True)

        st.divider()

        # Téléchargement
        st.subheader("Télécharger le dataset augmenté")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for nom, img_aug, _ in resultats:
                img_pil = Image.fromarray(img_aug)
                img_bytes = io.BytesIO()
                img_pil.save(img_bytes, format="JPEG", quality=95)
                zf.writestr(nom, img_bytes.getvalue())

        st.download_button(
            label=f"Télécharger {len(resultats)} images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="dataset_augmente.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.caption("Chaque image est nommée avec les techniques et paramètres exacts utilisés.")
        