import streamlit as st
from utils import render_theme_toggle
import albumentations as A
import cv2
import numpy as np
import zipfile
import io
import os
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Data Augmentation — Câblage industriel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

st.title("Systeme d'Augmentation des images annotées")
st.markdown("""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#4a5568;
    margin-top:-0.3rem;margin-bottom:1.5rem;">
    Upload images + annotations .txt Roboflow &mdash; correspondance automatique par nom.&nbsp;&nbsp;
    <span style="color:rgba(249,115,22,0.4);">|</span>&nbsp;&nbsp;
    <span style="color:rgba(249,115,22,0.75);">Crop optionnel + augmentation aléatoire — bbox préservées</span>
</p>
""", unsafe_allow_html=True)
st.divider()

BORDER_MODES = {
    "reflect":   cv2.BORDER_REFLECT,
    "constant":  cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
}

def _sidebar_family(label):
    st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.6rem;padding:0 2px;">
    <div style="flex-shrink:0;width:16px;height:1px;background:rgba(0,229,200,0.5);"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
        color:#00e5c8;text-transform:uppercase;letter-spacing:0.2em;white-space:nowrap;">
        {label}
    </span>
    <div style="flex:1;height:1px;background:rgba(0,229,200,0.1);"></div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# SIDEBAR — EN-TÊTE
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

render_theme_toggle()

# ════════════════════════════════════════════════
# SIDEBAR — F-1 CROP AUTOMATIQUE
# ════════════════════════════════════════════════
_sidebar_family("F-1 &mdash; Crop automatique")

use_crop = st.sidebar.checkbox("Activer le crop automatique", value=True,
                                help="Élimine le fond blanc inutile avant augmentation, basé sur les annotations")
crop_margin_pct = st.sidebar.slider("Marge (%)", 0.02, 0.20, 0.08, 0.01, disabled=not use_crop)
crop_min_margin_px = st.sidebar.slider("Marge minimale (px)", 50, 400, 250, 10, disabled=not use_crop,
                                       help="Couvre les pièces non annotées (ex: pattes de bracket)")
crop_threshold = st.sidebar.slider("Seuil fond blanc", 150, 240, 200, 5, disabled=not use_crop)

# ════════════════════════════════════════════════
# SIDEBAR — F-2 NETTOYAGE FOND
# ════════════════════════════════════════════════
_sidebar_family("F-2 &mdash; Nettoyage fond")

use_clean = st.sidebar.checkbox(
    "Supprimer taches de fond", value=False,
    help="Détecte les petites taches sombres sur fond clair et les efface par inpainting"
)
clean_bg_thresh  = st.sidebar.slider(
    "Seuil luminosité fond", 150, 240, 185, 5,
    disabled=not use_clean,
    help="Pixels en dessous de ce seuil sont considérés comme sombres"
)
clean_spot_area  = st.sidebar.slider(
    "Surface max tache (px²)", 50, 5000, 800, 50,
    disabled=not use_clean,
    help="Composantes sombres plus petites que cette valeur → effacées"
)
clean_inpaint_r  = st.sidebar.slider(
    "Rayon inpainting (px)", 2, 20, 7, 1,
    disabled=not use_clean,
    help="Rayon de reconstruction autour de chaque tache effacée"
)

# ════════════════════════════════════════════════
# SIDEBAR — F0 GÉNÉRAL
# ════════════════════════════════════════════════
_sidebar_family("F0 &mdash; Général")
n_copies    = st.sidebar.slider("Copies à générer par image", 1, 50, 3,
                                 help="OK -> 2 | pclipAbsent -> 3 | ticketManquant -> 6")
output_size = st.sidebar.selectbox("Dimension max de sortie (px)", [512, 768, 1024, 1536, 2048], index=2)
st.sidebar.caption("Ratio d'aspect natif préservé après crop — aucune déformation, aucun padding ajouté")

# ════════════════════════════════════════════════
# SIDEBAR — F1 GÉOMÉTRIQUE
# ════════════════════════════════════════════════
_sidebar_family("F1 &mdash; Géométrique")

use_rotation     = st.sidebar.checkbox("Rotation", value=True)
rotation_max     = st.sidebar.slider("Angle max (°)", 5, 90, 15, 5, disabled=not use_rotation)
rotation_border  = st.sidebar.selectbox("Mode remplissage bords", ["reflect", "constant", "replicate"],
                                         disabled=not use_rotation)
rotation_pas     = st.sidebar.number_input("Nombre de pas — Rotation", 1, 20, 1, disabled=not use_rotation)
st.sidebar.markdown("---")

use_flip_h = st.sidebar.checkbox("Flip horizontal", value=True)
use_flip_v = st.sidebar.checkbox("Flip vertical",   value=False)
st.sidebar.markdown("---")

use_zoom     = st.sidebar.checkbox("Zoom", value=True)
zoom_max     = st.sidebar.slider("Facteur zoom max", 1.1, 2.0, 1.2, 0.1, disabled=not use_zoom)
zoom_pas     = st.sidebar.number_input("Nombre de pas — Zoom", 1, 20, 1, disabled=not use_zoom)
st.sidebar.markdown("---")

use_perspective  = st.sidebar.checkbox("Perspective", value=False)
perspective_max  = st.sidebar.slider("Échelle perspective max", 0.01, 0.2, 0.1, 0.01,
                                      disabled=not use_perspective)
perspective_pas  = st.sidebar.number_input("Nombre de pas — Perspective", 1, 10, 1, disabled=not use_perspective)
st.sidebar.markdown("---")

use_shear    = st.sidebar.checkbox("Cisaillement (Shear)", value=False)
shear_max    = st.sidebar.slider("Angle cisaillement max (°)", 5, 30, 15, 5, disabled=not use_shear)
shear_pas    = st.sidebar.number_input("Nombre de pas — Cisaillement", 1, 10, 1, disabled=not use_shear)
st.sidebar.markdown("---")

use_shift    = st.sidebar.checkbox("Translation (Shift)", value=True)
shift_max    = st.sidebar.slider("Déplacement max (%)", 0.05, 0.3, 0.10, 0.05, disabled=not use_shift)
shift_pas    = st.sidebar.number_input("Nombre de pas — Translation", 1, 10, 1, disabled=not use_shift)

# ════════════════════════════════════════════════
# SIDEBAR — F2 PHOTOMÉTRIQUE
# ════════════════════════════════════════════════
_sidebar_family("F2 &mdash; Photométrique")

use_brightness   = st.sidebar.checkbox("Luminosité / Contraste", value=True)
brightness_max   = st.sidebar.slider("Luminosité max", 0.1, 0.5, 0.25, 0.05, disabled=not use_brightness)
brightness_pas   = st.sidebar.number_input("Nombre de pas — Luminosité", 1, 10, 1, disabled=not use_brightness)
st.sidebar.markdown("---")

use_noise    = st.sidebar.checkbox("Bruit gaussien", value=False)
noise_max    = st.sidebar.slider("Variance bruit max", 10, 100, 20, 5, disabled=not use_noise)
noise_pas    = st.sidebar.number_input("Nombre de pas — Bruit", 1, 10, 1, disabled=not use_noise)
st.sidebar.markdown("---")

use_blur     = st.sidebar.checkbox("Flou gaussien", value=False)
blur_max     = st.sidebar.slider("Noyau flou max (px)", 3, 11, 7, 2, disabled=not use_blur)
blur_pas     = st.sidebar.number_input("Nombre de pas — Flou gaussien", 1, 10, 1, disabled=not use_blur)
st.sidebar.markdown("---")

use_motion_blur  = st.sidebar.checkbox("Flou de mouvement", value=False)
motion_max       = st.sidebar.slider("Noyau mouvement max (px)", 3, 21, 9, 2, disabled=not use_motion_blur)
motion_pas       = st.sidebar.number_input("Nombre de pas — Flou mouvement", 1, 10, 1, disabled=not use_motion_blur)
st.sidebar.markdown("---")

use_hsv      = st.sidebar.checkbox("Saturation HSV", value=True)
hsv_max      = st.sidebar.slider("Saturation max", 10, 50, 20, 5, disabled=not use_hsv)
hsv_pas      = st.sidebar.number_input("Nombre de pas — HSV", 1, 10, 1, disabled=not use_hsv)
st.sidebar.markdown("---")

use_clahe    = st.sidebar.checkbox("Égalisation CLAHE", value=False)
clahe_max    = st.sidebar.slider("Clip limit max", 1.0, 8.0, 4.0, 0.5, disabled=not use_clahe)
clahe_pas    = st.sidebar.number_input("Nombre de pas — CLAHE", 1, 10, 1, disabled=not use_clahe)
st.sidebar.markdown("---")

use_jpeg     = st.sidebar.checkbox("Compression JPEG", value=False)
jpeg_min     = st.sidebar.slider("Qualité JPEG min", 50, 95, 70, 5, disabled=not use_jpeg)
jpeg_pas     = st.sidebar.number_input("Nombre de pas — JPEG", 1, 10, 1, disabled=not use_jpeg)
st.sidebar.markdown("---")

use_togray   = st.sidebar.checkbox("Niveaux de gris (ToGray)", value=False)

# ════════════════════════════════════════════════
# SIDEBAR — F4 MIXAGE (pixel-only, bbox inchangées)
# ════════════════════════════════════════════════
_sidebar_family("F4 &mdash; Mixage")

use_mixup    = st.sidebar.checkbox("Mixup", value=False)
mixup_alpha  = st.sidebar.slider("Alpha Mixup max", 0.1, 0.5, 0.3, 0.05, disabled=not use_mixup)
mixup_pas    = st.sidebar.number_input("Nombre de pas — Mixup", 1, 10, 1, disabled=not use_mixup)
st.sidebar.markdown("---")

use_cutmix   = st.sidebar.checkbox("CutMix", value=False)
cutmix_alpha = st.sidebar.slider("Alpha CutMix max", 0.1, 1.0, 0.5, 0.1, disabled=not use_cutmix)
cutmix_pas   = st.sidebar.number_input("Nombre de pas — CutMix", 1, 10, 1, disabled=not use_cutmix)

# ════════════════════════════════════════════════
# SIDEBAR — F5 AVANCÉE
# ════════════════════════════════════════════════
_sidebar_family("F5 &mdash; Avancée")

use_elastic  = st.sidebar.checkbox("Elastic Transform", value=False)
elastic_max  = st.sidebar.slider("Alpha élastique max", 10, 200, 80, 10, disabled=not use_elastic)
elastic_pas  = st.sidebar.number_input("Nombre de pas — Elastic Transform", 1, 10, 1, disabled=not use_elastic)
st.sidebar.markdown("---")

use_grid     = st.sidebar.checkbox("Grid Distortion", value=False)
grid_max     = st.sidebar.slider("Distorsion grille max", 0.1, 0.5, 0.3, 0.05, disabled=not use_grid)
grid_pas     = st.sidebar.number_input("Nombre de pas — Grid Distortion", 1, 10, 1, disabled=not use_grid)
st.sidebar.markdown("---")

use_optical  = st.sidebar.checkbox("Optical Distortion", value=False)
optical_max  = st.sidebar.slider("Distorsion optique max", 0.05, 0.5, 0.2, 0.05, disabled=not use_optical)
optical_pas  = st.sidebar.number_input("Nombre de pas — Optical Distortion", 1, 10, 1, disabled=not use_optical)

# ════════════════════════════════════════════════
# TECHNIQUES ACTIVES
# ════════════════════════════════════════════════
techniques_actives = []
if use_rotation:    techniques_actives.append({"nom": "Rotation", "nb_pas": int(rotation_pas)})
if use_flip_h:      techniques_actives.append({"nom": "Flip horizontal", "nb_pas": 1})
if use_flip_v:      techniques_actives.append({"nom": "Flip vertical", "nb_pas": 1})
if use_zoom:        techniques_actives.append({"nom": "Zoom", "nb_pas": int(zoom_pas)})
if use_perspective: techniques_actives.append({"nom": "Perspective", "nb_pas": int(perspective_pas)})
if use_shear:       techniques_actives.append({"nom": "Cisaillement", "nb_pas": int(shear_pas)})
if use_shift:       techniques_actives.append({"nom": "Translation", "nb_pas": int(shift_pas)})
if use_brightness:  techniques_actives.append({"nom": "Luminosité / Contraste", "nb_pas": int(brightness_pas)})
if use_noise:       techniques_actives.append({"nom": "Bruit gaussien", "nb_pas": int(noise_pas)})
if use_blur:        techniques_actives.append({"nom": "Flou gaussien", "nb_pas": int(blur_pas)})
if use_motion_blur: techniques_actives.append({"nom": "Flou mouvement", "nb_pas": int(motion_pas)})
if use_hsv:         techniques_actives.append({"nom": "Saturation HSV", "nb_pas": int(hsv_pas)})
if use_clahe:       techniques_actives.append({"nom": "CLAHE", "nb_pas": int(clahe_pas)})
if use_jpeg:        techniques_actives.append({"nom": "Compression JPEG", "nb_pas": int(jpeg_pas)})
if use_togray:      techniques_actives.append({"nom": "Niveaux de gris", "nb_pas": 1})
if use_mixup:       techniques_actives.append({"nom": "Mixup", "nb_pas": int(mixup_pas)})
if use_cutmix:      techniques_actives.append({"nom": "CutMix", "nb_pas": int(cutmix_pas)})
if use_elastic:     techniques_actives.append({"nom": "Elastic Transform", "nb_pas": int(elastic_pas)})
if use_grid:        techniques_actives.append({"nom": "Grid Distortion", "nb_pas": int(grid_pas)})
if use_optical:     techniques_actives.append({"nom": "Optical Distortion", "nb_pas": int(optical_pas)})

x = 1
for technique in techniques_actives:
    x *= technique["nb_pas"]

# ════════════════════════════════════════════════
# PARAMÈTRES GLOBAUX
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
    c1, c2, c3 = st.columns(3)
    c1.metric("Techniques actives", len(techniques_actives))
    c2.metric("Facteur x",           x)
    c3.metric("Total par image",    n_copies * x + 1)

st.divider()

# ════════════════════════════════════════════════
# FONCTIONS UTILITAIRES — PARSING & FORMATAGE
# ════════════════════════════════════════════════
def parse_labels(txt_content: str):
    labels = []
    for line in txt_content.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls    = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            cx, cy, w, h = coords
            labels.append((cls, cx, cy, w, h))
        elif len(coords) % 2 == 0 and len(coords) >= 6:
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w  = x_max - x_min
            h  = y_max - y_min
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            w, h   = min(w, 1), min(h, 1)
            labels.append((cls, cx, cy, w, h))
    return labels


def format_bbox_labels(class_labels, bboxes):
    lines = []
    for cls, box in zip(class_labels, bboxes):
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def draw_bboxes(image_rgb, class_labels, bboxes, color=(0, 255, 100)):
    img = image_rgb.copy()
    h, w = img.shape[:2]
    for cls, box in zip(class_labels, bboxes):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(cls), (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def decode_image_raw(raw_bytes: bytes):
    """Décode l'image SANS resize — nécessaire pour le crop (qui a besoin
    de la résolution originale pour bien calibrer la marge en pixels)."""
    arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img_bgr


# ════════════════════════════════════════════════
# RESIZE PROPORTIONNEL SANS PADDING — garde le ratio natif, zéro pixel artificiel
# ════════════════════════════════════════════════
def resize_keep_ratio_no_pad(img_bgr, max_dimension):
    """
    Redimensionne en conservant EXACTEMENT le ratio d'aspect de l'image
    après crop, sans jamais forcer un format carré et sans ajouter de
    padding. Limite seulement la plus grande dimension à max_dimension
    pour éviter des fichiers trop lourds.

    Avantage : les coordonnées YOLO normalisées (0-1) restent valides
    sans aucun recalcul, car le ratio d'aspect ne change pas.
    """
    h, w = img_bgr.shape[:2]
    scale = min(max_dimension / w, max_dimension / h, 1.0)
    if scale >= 1.0:
        return img_bgr.copy()

    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ════════════════════════════════════════════════
# CROP AUTOMATIQUE — validé sur dataset réel (méthode calibrée)
# ════════════════════════════════════════════════
def get_threshold_bbox(img_gray, threshold=200):
    """Méthode seuillage fond blanc — filet de sécurité si pas d'annotation."""
    mask = img_gray < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        h, w = img_gray.shape
        return 0, 0, w, h
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def get_annotations_bbox(bboxes, img_w, img_h):
    """Bbox englobante de toutes les annotations YOLO (en pixels absolus)."""
    if not bboxes:
        return None
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
    for box in bboxes:
        cx, cy, w, h = box
        x_mins.append((cx - w / 2) * img_w)
        x_maxs.append((cx + w / 2) * img_w)
        y_mins.append((cy - h / 2) * img_h)
        y_maxs.append((cy + h / 2) * img_h)
    return int(min(x_mins)), int(min(y_mins)), int(max(x_maxs)), int(max(y_maxs))


def union_bbox(bbox_thresh, bbox_annot, img_w, img_h, max_area_ratio=0.7):
    """
    Le seuillage fond blanc est peu fiable sur table industrielle
    (rayures, taches) -> peut couvrir ~100% de l'image.
    On priorise donc la bbox des annotations, et on ignore le seuillage
    s'il est aberrant (couvre plus de max_area_ratio de l'image).
    """
    img_area = img_w * img_h
    thresh_area = (bbox_thresh[2] - bbox_thresh[0]) * (bbox_thresh[3] - bbox_thresh[1])
    thresh_ratio = thresh_area / img_area

    if bbox_annot is None:
        return bbox_thresh
    if thresh_ratio > max_area_ratio:
        return bbox_annot

    return (
        min(bbox_thresh[0], bbox_annot[0]),
        min(bbox_thresh[1], bbox_annot[1]),
        max(bbox_thresh[2], bbox_annot[2]),
        max(bbox_thresh[3], bbox_annot[3]),
    )


def add_margin(bbox, img_w, img_h, margin_ratio=0.08, min_margin_px=250):
    """
    Marge = max(pourcentage, minimum absolu en pixels).
    Évite de couper les pièces non annotées (ex: pattes de bracket)
    quand la bbox d'annotations est étroite.
    """
    x_min, y_min, x_max, y_max = bbox
    mw = max(int((x_max - x_min) * margin_ratio), min_margin_px)
    mh = max(int((y_max - y_min) * margin_ratio), min_margin_px)
    return (
        max(0, x_min - mw),
        max(0, y_min - mh),
        min(img_w, x_max + mw),
        min(img_h, y_max + mh),
    )


def recompute_labels_after_crop(class_labels, bboxes, crop_bbox, img_w, img_h):
    """Recalcule les coordonnées YOLO normalisées après crop."""
    cx0, cy0, cx1, cy1 = crop_bbox
    crop_w, crop_h = cx1 - cx0, cy1 - cy0

    new_class_labels, new_bboxes = [], []
    for cls, box in zip(class_labels, bboxes):
        cx, cy, w, h = box
        abs_cx, abs_cy = cx * img_w, cy * img_h
        abs_w, abs_h = w * img_w, h * img_h

        new_abs_cx = abs_cx - cx0
        new_abs_cy = abs_cy - cy0

        new_cx = new_abs_cx / crop_w
        new_cy = new_abs_cy / crop_h
        new_w = abs_w / crop_w
        new_h = abs_h / crop_h

        if 0 <= new_cx <= 1 and 0 <= new_cy <= 1:
            new_class_labels.append(cls)
            new_bboxes.append([new_cx, new_cy, min(new_w, 1.0), min(new_h, 1.0)])

    return new_class_labels, new_bboxes


def clean_background_spots(img_bgr, bg_thresh=185, max_spot_area=800, inpaint_radius=7):
    """
    Supprime les petites taches sombres sur fond clair.

    Principe :
      1. Seuillage → masque de toutes les zones sombres (taches + objet principal)
      2. Composantes connexes → filtrage par aire
      3. Seules les petites composantes (< max_spot_area px²) sont considérées
         comme des taches ; les grandes (pièce réelle) sont préservées
      4. Inpainting TELEA sur le masque des taches uniquement
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Masque binaire des zones sombres
    dark_mask = np.where(gray < bg_thresh, np.uint8(255), np.uint8(0))

    # Composantes connexes 8-voisins
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    # Masque des taches seules (petites composantes)
    spots_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_spot_area:
            spots_mask[labels == i] = 255

    if not spots_mask.any():
        return img_bgr

    return cv2.inpaint(img_bgr, spots_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)


def apply_crop(img_bgr, class_labels, bboxes, margin_ratio=0.08, min_margin_px=250, threshold=200):
    """
    Pipeline de crop complet : seuillage + bbox annotations + marge calibrée.
    Opère sur l'image en résolution ORIGINALE (avant resize).
    Retourne (image_croppée_bgr, nouvelles_class_labels, nouvelles_bboxes, info_dict).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bbox_thresh = get_threshold_bbox(gray, threshold)
    bbox_annot = get_annotations_bbox(bboxes, w, h)
    final_bbox = union_bbox(bbox_thresh, bbox_annot, w, h)
    final_bbox = add_margin(final_bbox, w, h, margin_ratio, min_margin_px)

    x0, y0, x1, y1 = final_bbox
    if x1 <= x0 or y1 <= y0:
        return img_bgr, class_labels, bboxes, {"ok": False, "reduction": 0.0}

    cropped = img_bgr[y0:y1, x0:x1]
    new_class_labels, new_bboxes = recompute_labels_after_crop(
        class_labels, bboxes, final_bbox, w, h)

    reduction = 100 * (1 - (cropped.shape[0] * cropped.shape[1]) / (h * w))
    info = {"ok": True, "reduction": reduction,
            "labels_kept": len(new_bboxes), "labels_total": len(bboxes)}
    return cropped, new_class_labels, new_bboxes, info


def build_augmentor():
    transforms = []

    # F1 — Géométrique
    if use_rotation:
        transforms.append(A.Rotate(
            limit=rotation_max,
            border_mode=BORDER_MODES[rotation_border],
            p=0.8
        ))
    if use_flip_h:
        transforms.append(A.HorizontalFlip(p=0.5))
    if use_flip_v:
        transforms.append(A.VerticalFlip(p=0.2))
    if use_zoom:
        transforms.append(A.RandomScale(scale_limit=zoom_max - 1.0, p=0.6))
    if use_perspective:
        transforms.append(A.Perspective(scale=(0.01, perspective_max), p=0.5))
    if use_shear:
        transforms.append(A.Affine(shear=(-shear_max, shear_max), p=0.5))
    if use_shift:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=shift_max, scale_limit=0, rotate_limit=0, p=0.6
        ))

    # F2 — Photométrique
    if use_brightness:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=brightness_max, contrast_limit=brightness_max, p=0.8
        ))
    if use_noise:
        transforms.append(A.GaussNoise(var_limit=(5, noise_max), p=0.5))
    if use_blur:
        k = blur_max if blur_max % 2 == 1 else blur_max + 1
        transforms.append(A.GaussianBlur(blur_limit=(3, max(3, k)), p=0.3))
    if use_motion_blur:
        k = motion_max if motion_max % 2 == 1 else motion_max + 1
        transforms.append(A.MotionBlur(blur_limit=(3, max(3, k)), p=0.4))
    if use_hsv:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=hsv_max, val_shift_limit=20, p=0.5
        ))
    if use_clahe:
        transforms.append(A.CLAHE(clip_limit=clahe_max, p=0.5))
    if use_jpeg:
        transforms.append(A.ImageCompression(
            quality_lower=jpeg_min, quality_upper=95, p=0.4
        ))
    if use_togray:
        transforms.append(A.ToGray(p=0.3))

    # F5 — Avancée (compatibles bbox)
    if use_elastic:
        transforms.append(A.ElasticTransform(alpha=elastic_max, sigma=10, p=0.4))
    if use_grid:
        transforms.append(A.GridDistortion(distort_limit=grid_max, p=0.4))
    if use_optical:
        transforms.append(A.OpticalDistortion(distort_limit=optical_max, p=0.4))

    if not transforms:
        transforms.append(A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3
        )
    )


def apply_mixup(img: np.ndarray, other: np.ndarray, alpha_max: float) -> np.ndarray:
    alpha = float(np.random.uniform(0, alpha_max))
    other_r = cv2.resize(other, (img.shape[1], img.shape[0]))
    return (alpha * img.astype(np.float32) + (1 - alpha) * other_r.astype(np.float32)).astype(np.uint8)


def apply_cutmix(img: np.ndarray, other: np.ndarray, alpha_max: float) -> np.ndarray:
    h, w = img.shape[:2]
    lam  = float(np.random.beta(alpha_max, alpha_max))
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    cx    = int(np.random.randint(w))
    cy    = int(np.random.randint(h))
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    out = img.copy()
    patch = cv2.resize(other, (x2 - x1, y2 - y1)) if (x2 > x1 and y2 > y1) else out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = patch
    return out


# ════════════════════════════════════════════════
# SECTION CHARGEMENT
# ════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
    <div style="width:36px;height:36px;background:rgba(249,115,22,0.12);
        border:1px solid rgba(249,115,22,0.3);border-radius:8px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">&#8681;</div>
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#1a1f2e;">
            Chargement des fichiers</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;">
            Correspondance automatique par nom de fichier</div>
    </div>
</div>
""", unsafe_allow_html=True)

import_mode = st.radio(
    "Mode d'import",
    ["Fichiers multiples", "Archive ZIP"],
    horizontal=True,
    label_visibility="collapsed"
)

raw_pairs  = []   # avant crop, résolution originale : {"stem", "img_bgr_orig", "class_ids", "bboxes"}
sans_label = []

# ─────────────────────────────────────────────
# MODE 1 — FICHIERS MULTIPLES
# ─────────────────────────────────────────────
if import_mode == "Fichiers multiples":
    col_img, col_lbl = st.columns(2)

    with col_img:
        img_files = st.file_uploader(
            "Images (.jpg / .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="imgs_multi"
        )

    with col_lbl:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:600;margin-bottom:0.4rem;'>"
            "Source des labels</div>",
            unsafe_allow_html=True,
        )
        label_src = st.radio(
            "label_src",
            ["Dossier local", "Fichiers individuels"],
            horizontal=False,
            label_visibility="collapsed",
            key="label_source_mode",
        )

        # ── Dossier local ──
        if label_src == "Dossier local":
            folder_path_str = st.text_input(
                "Chemin du dossier de labels",
                value=r"C:\Users\HP\Downloads\vision_project.yolov8 (1)\train - Copie\labels",
                placeholder="ex : C:\\dataset\\labels",
                key="label_folder_path",
                label_visibility="collapsed",
            )
            txt_files = None  # non utilisé dans ce mode

            # Feedback en temps réel
            if folder_path_str:
                folder_path = Path(folder_path_str.strip())
                if not folder_path.exists():
                    st.error("Dossier introuvable.")
                elif not folder_path.is_dir():
                    st.error("Ce chemin n'est pas un dossier.")
                else:
                    found_txts = list(folder_path.glob("*.txt"))
                    st.success(
                        f"{len(found_txts)} fichier(s) .txt trouvé(s)"
                        + (
                            f" — {sum(1 for t in found_txts if t.stem in {Path(f.name).stem for f in (img_files or [])})}"
                            f" correspondance(s) avec les images"
                            if img_files else ""
                        )
                    )
            else:
                folder_path = None
        # ── Fichiers individuels ──
        else:
            folder_path = None
            txt_files = st.file_uploader(
                "Annotations (.txt) — même nom que les images",
                type=["txt"],
                accept_multiple_files=True,
                key="txts_multi",
                label_visibility="collapsed",
            )

    # ── Construction du txt_map et des paires ──
    if img_files:
        # Priorité : dossier local → fichiers uploadés
        use_disk = label_src == "Dossier local" and folder_path and folder_path.is_dir()

        if use_disk:
            txt_map_disk = {p.stem: p for p in folder_path.glob("*.txt")}
        else:
            txt_map_disk = {}

        txt_map_upload = {Path(f.name).stem: f for f in (txt_files or [])}

        for img_f in img_files:
            stem = Path(img_f.name).stem
            img_bgr_orig = decode_image_raw(img_f.read())
            if img_bgr_orig is None:
                continue

            if use_disk:
                if stem not in txt_map_disk:
                    sans_label.append(img_f.name)
                    continue
                labels = parse_labels(txt_map_disk[stem].read_text(encoding="utf-8"))
            else:
                if stem not in txt_map_upload:
                    sans_label.append(img_f.name)
                    continue
                txt_map_upload[stem].seek(0)
                labels = parse_labels(txt_map_upload[stem].read().decode("utf-8"))

            if not labels:
                sans_label.append(img_f.name)
                continue
            raw_pairs.append({
                "stem":         stem,
                "img_bgr_orig": img_bgr_orig,
                "class_ids":    [l[0] for l in labels],
                "bboxes":       [[l[1], l[2], l[3], l[4]] for l in labels],
            })

# ─────────────────────────────────────────────
# MODE 2 — ARCHIVE ZIP
# ─────────────────────────────────────────────
else:
    zip_file = st.file_uploader(
        "Archive ZIP contenant les images + fichiers .txt (même nom)",
        type=["zip"],
        key="zip_upload"
    )
    if zip_file:
        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
            all_names   = zf.namelist()
            img_entries = [
                n for n in all_names
                if not os.path.basename(n).startswith(".")
                and not n.startswith("__MACOSX")
                and Path(n).suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            txt_map = {
                Path(n).stem: n for n in all_names
                if not os.path.basename(n).startswith(".")
                and Path(n).suffix.lower() == ".txt"
            }
            for img_name in img_entries:
                stem = Path(img_name).stem
                img_bgr_orig = decode_image_raw(zf.read(img_name))
                if img_bgr_orig is None:
                    continue
                if stem not in txt_map:
                    sans_label.append(Path(img_name).name)
                    continue
                labels = parse_labels(zf.read(txt_map[stem]).decode("utf-8"))
                if not labels:
                    sans_label.append(Path(img_name).name)
                    continue
                raw_pairs.append({
                    "stem":         stem,
                    "img_bgr_orig": img_bgr_orig,
                    "class_ids":    [l[0] for l in labels],
                    "bboxes":       [[l[1], l[2], l[3], l[4]] for l in labels],
                })

# ════════════════════════════════════════════════
# RÉSULTATS DU CHARGEMENT
# ════════════════════════════════════════════════
if sans_label:
    st.warning(
        f"{len(sans_label)} image(s) ignorée(s) — aucun .txt correspondant : "
        f"{', '.join(sans_label[:5])}{'…' if len(sans_label) > 5 else ''}"
    )

if not raw_pairs:
    st.info(
        "Chargez des images et leurs annotations. "
        "Le fichier .txt doit avoir **exactement le même nom** que l'image (extension différente)."
    )
    st.stop()

# ════════════════════════════════════════════════
# APPLICATION DU CROP (si activé) puis LETTERBOX RESIZE
# ════════════════════════════════════════════════
pairs = []
crop_stats = []

for rp in raw_pairs:
    img_bgr = rp["img_bgr_orig"]
    class_ids = rp["class_ids"]
    bboxes = rp["bboxes"]
    crop_info = None

    if use_clean:
        img_bgr = clean_background_spots(
            img_bgr,
            bg_thresh=clean_bg_thresh,
            max_spot_area=clean_spot_area,
            inpaint_radius=clean_inpaint_r,
        )

    if use_crop:
        img_bgr, class_ids, bboxes, crop_info = apply_crop(
            img_bgr, class_ids, bboxes,
            margin_ratio=crop_margin_pct,
            min_margin_px=crop_min_margin_px,
            threshold=crop_threshold,
        )
        crop_stats.append(crop_info)

    # Resize proportionnel APRÈS le crop — garde le ratio natif, zéro padding
    # (bboxes inchangées : coordonnées YOLO normalisées indépendantes de la résolution
    #  tant que le ratio d'aspect ne change pas)
    img_bgr_resized = resize_keep_ratio_no_pad(img_bgr, output_size)
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)

    pairs.append({
        "stem":      rp["stem"],
        "img_rgb":   img_rgb,
        "class_ids": class_ids,
        "bboxes":    bboxes,
        "crop_info": crop_info,
    })

total_annotations = sum(len(p["bboxes"]) for p in pairs)
st.success(
    f"{len(pairs)} image(s) chargée(s) avec annotation · "
    f"{total_annotations} bbox(es) au total · dimension max {output_size}px (ratio natif préservé)"
)

if use_crop and crop_stats:
    reductions = [c["reduction"] for c in crop_stats if c["ok"]]
    n_failed_crop = sum(1 for c in crop_stats if not c["ok"])
    if reductions:
        st.info(
            f"Crop appliqué sur {len(reductions)} image(s) — "
            f"réduction moyenne de surface : {np.mean(reductions):.1f}% "
            f"(min {min(reductions):.1f}% / max {max(reductions):.1f}%)"
        )
    if n_failed_crop:
        st.warning(f"{n_failed_crop} image(s) n'ont pas pu être croppées — augmentation lancée sur l'original.")
    low_reduction = [c for c in crop_stats if c["ok"] and c["reduction"] < 15]
    if low_reduction:
        st.warning(
            f"⚠ {len(low_reduction)} image(s) avec réduction < 15% — "
            "vérifie si les annotations correspondent bien à l'image (ou ajuste la marge)."
        )

expected_per_image = n_copies * x
st.info(
    f"Total images à générer : {len(pairs)} images × {n_copies} copies × {x} (facteur) "
    f"= **{len(pairs) * expected_per_image} images augmentées** (+ {len(pairs)} originaux)"
)

# Avertissement Mixup/CutMix si une seule image
if (use_mixup or use_cutmix) and len(pairs) < 2:
    st.warning("Mixup / CutMix nécessitent au moins 2 images chargées pour mélanger.")

def _encode_img_to_data_uri(img_rgb):
    import base64
    is_success, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not is_success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"

# ── Aperçu grille — montre le résultat APRÈS crop + letterbox
st.divider()
st.subheader("Aperçu — images chargées avec bounding boxes" + (" (après crop)" if use_crop else ""))

prev_items = []
for p in pairs:
    vis = draw_bboxes(p["img_rgb"], p["class_ids"], p["bboxes"])
    caption = f"{p['stem']} · {len(p['bboxes'])} bbox"
    if p["crop_info"] and p["crop_info"]["ok"]:
        caption += f" · -{p['crop_info']['reduction']:.0f}%"
    uri = _encode_img_to_data_uri(vis)
    if uri:
        prev_items.append(
            f"<div style='display:flex;flex-direction:column;align-items:center;margin:6px;'>"
            f"<img src='{uri}' style='width:180px;height:auto;border-radius:6px;object-fit:cover;'/>"
            f"<div style='font-size:0.75rem;margin-top:4px;color:#334455'>{caption}</div></div>"
        )

prev_html = (
    "<div style='max-height:420px;overflow:auto;display:flex;flex-wrap:wrap;"
    "gap:8px;padding:6px;border:1px solid rgba(0,0,0,0.04);'>"
    + "".join(prev_items) + "</div>"
)
st.markdown(prev_html, unsafe_allow_html=True)

# ── Toutes les images chargées (scrollable)
st.markdown("**Toutes les images chargées**")
imgs_html = [f"<div style='display:flex;flex-direction:column;align-items:center;margin:6px;'>"
             f"<img src='{_encode_img_to_data_uri(draw_bboxes(p['img_rgb'], p['class_ids'], p['bboxes']))}' "
             f"style='width:180px;height:auto;border-radius:6px;object-fit:cover;'/><div "
             f"style='font-size:0.75rem;margin-top:4px;color:#334455'>{p['stem']} · {len(p['bboxes'])} bbox</div></div>"
             for p in pairs]

grid_html = "".join(imgs_html)
container_html = f"<div style='max-height:420px;overflow:auto;display:flex;flex-wrap:wrap;gap:8px;padding:6px;border:1px solid rgba(0,0,0,0.04);'>{grid_html}</div>"
st.markdown(container_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# GÉNÉRATION
# ════════════════════════════════════════════════
st.divider()
if st.button("Lancer l'augmentation", type="primary", use_container_width=True):

    augmentor       = build_augmentor()
    zip_buffer      = io.BytesIO()
    total_target    = len(pairs) * n_copies * x
    total_generated = 0
    total_failed    = 0
    preview_imgs    = []

    # pré-extraction des images (déjà croppées/letterboxées) pour Mixup/CutMix
    all_imgs_rgb = [p["img_rgb"] for p in pairs]

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        progress = st.progress(0, text="Génération en cours…")

        for pair in pairs:
            stem      = pair["stem"]
            img_rgb   = pair["img_rgb"]
            class_ids = pair["class_ids"]
            bboxes    = pair["bboxes"]

            # écriture de l'original (déjà croppé/letterboxé si activé)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            _, orig_enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            zf.writestr(f"{stem}_orig.jpg", orig_enc.tobytes())
            zf.writestr(f"{stem}_orig.txt", format_bbox_labels(class_ids, bboxes))

            generated    = 0
            attempts     = 0
            max_attempts = n_copies * x * 5

            while generated < n_copies * x and attempts < max_attempts:
                attempts += 1
                try:
                    result = augmentor(image=img_rgb, bboxes=bboxes, class_labels=class_ids)
                    if not result["bboxes"]:
                        total_failed += 1
                        continue

                    aug_rgb   = result["image"]
                    aug_boxes = list(result["bboxes"])
                    aug_cls   = list(result["class_labels"])

                    # ToGray : reconvertir en RGB 3 canaux si nécessaire
                    if len(aug_rgb.shape) == 2:
                        aug_rgb = cv2.cvtColor(aug_rgb, cv2.COLOR_GRAY2RGB)

                    # F4 — Mixup (pixels uniquement, bbox inchangées)
                    if use_mixup and len(all_imgs_rgb) > 1:
                        other = all_imgs_rgb[int(np.random.randint(len(all_imgs_rgb)))]
                        aug_rgb = apply_mixup(aug_rgb, other, mixup_alpha)

                    # F4 — CutMix (pixels uniquement, bbox inchangées)
                    if use_cutmix and len(all_imgs_rgb) > 1:
                        other = all_imgs_rgb[int(np.random.randint(len(all_imgs_rgb)))]
                        aug_rgb = apply_cutmix(aug_rgb, other, cutmix_alpha)

                    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                    _, enc  = cv2.imencode(".jpg", aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    aug_txt = format_bbox_labels(aug_cls, aug_boxes)

                    fname = f"{stem}_aug{generated:03d}"
                    zf.writestr(f"{fname}.jpg", enc.tobytes())
                    zf.writestr(f"{fname}.txt", aug_txt)

                    if len(preview_imgs) < 8:
                        preview_imgs.append((aug_rgb, aug_cls, aug_boxes))

                    generated       += 1
                    total_generated += 1
                    progress.progress(
                        total_generated / total_target,
                        text=f"Génération : {total_generated}/{total_target}  ({stem})"
                    )
                except Exception:
                    total_failed += 1

        progress.empty()

    st.success(
        f"{total_generated} images générées · "
        f"{len(pairs)} originaux inclus · "
        f"{total_failed} rejet(s) (bbox hors cadre)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Originaux",       len(pairs))
    c2.metric("Copies générées", total_generated)
    c3.metric("Total dataset",   total_generated + len(pairs))
    c4.metric("Techniques",      len(techniques_actives))

    if preview_imgs:
        st.divider()
        st.subheader("Aperçu — premières copies augmentées")
        cols_res = st.columns(min(4, len(preview_imgs)))
        for i, (pi, pc, pb) in enumerate(preview_imgs):
            cols_res[i % 4].image(
                draw_bboxes(pi, pc, pb),
                caption=f"aug_{i:03d}",
                use_container_width=True
            )

    # ── Toutes les images augmentées (scrollable) — extraites depuis le ZIP en mémoire
    try:
        import base64
        with zipfile.ZipFile(io.BytesIO(zip_buffer.getvalue())) as zf2:
            aug_names = [n for n in zf2.namelist() if n.lower().endswith('.jpg') and '_aug' in n]
            if aug_names:
                thumbs = []
                for name in aug_names:
                    data = zf2.read(name)
                    b64 = base64.b64encode(data).decode('ascii')
                    thumbs.append(f"<div style='display:flex;flex-direction:column;align-items:center;margin:6px;'>"
                                  f"<img src='data:image/jpeg;base64,{b64}' style='width:160px;height:auto;border-radius:6px;object-fit:cover;'/>"
                                  f"<div style='font-size:0.72rem;margin-top:4px;color:#334455'>{name}</div></div>")

                grid = ''.join(thumbs)
                sc_html = f"<div style='max-height:520px;overflow:auto;display:flex;flex-wrap:wrap;gap:8px;padding:6px;border:1px solid rgba(0,0,0,0.04);'>{grid}</div>"
                st.markdown("**Toutes les images augmentées**")
                st.markdown(sc_html, unsafe_allow_html=True)
    except Exception:
        pass

    st.divider()
    st.subheader("Télécharger le dataset augmenté")
    st.download_button(
        label=f"Télécharger {total_generated + len(pairs)} images + annotations (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="dataset_yolo_augmente.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.caption("Chaque image est accompagnée de son fichier .txt YOLO mis à jour.")