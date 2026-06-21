import streamlit as st
import albumentations as A
import cv2
import numpy as np
import zipfile
import io
import os
from PIL import Image

st.set_page_config(page_title="YOLO Augmentation", layout="wide")
st.title("Augmentation YOLO — avec préservation des annotations")
st.caption("Upload une image + son fichier .txt YOLO → les bounding boxes sont recalculées automatiquement.")

# ─────────────────────────────────────────────
# SIDEBAR — Choix des transformations
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Transformations")

    n_copies = st.slider("Nombre de copies à générer", 1, 50, 10)

    st.subheader("Géométrique")
    use_flip_h  = st.checkbox("Flip horizontal",  value=True)
    use_flip_v  = st.checkbox("Flip vertical",    value=False)
    use_rotate  = st.checkbox("Rotation",         value=True)
    rot_limit   = st.slider("Rotation max (°)", 5, 30, 15, disabled=not use_rotate)
    use_zoom    = st.checkbox("Zoom aléatoire",   value=True)
    zoom_limit  = st.slider("Zoom max (%)", 5, 30, 15, disabled=not use_zoom) / 100
    use_shift   = st.checkbox("Translation",      value=False)
    shift_limit = st.slider("Shift max (%)", 2, 15, 5, disabled=not use_shift) / 100

    st.subheader("Photométrique")
    use_brightness = st.checkbox("Luminosité / Contraste", value=True)
    bright_limit   = st.slider("Intensité luminosité", 0.05, 0.5, 0.3, disabled=not use_brightness)
    use_noise      = st.checkbox("Bruit gaussien",     value=True)
    noise_limit    = st.slider("Bruit max",  5, 80, 30, disabled=not use_noise)
    use_blur       = st.checkbox("Flou gaussien",      value=False)
    use_hsv        = st.checkbox("HSV (saturation)",   value=True)
    use_shadow     = st.checkbox("Ombres aléatoires",  value=False)

# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────
def parse_yolo_labels(txt_content: str):
    """Lit le contenu d'un .txt YOLO → list de (class_id, cx, cy, w, h)."""
    labels = []
    for line in txt_content.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((cls, cx, cy, w, h))
    return labels

def format_yolo_labels(class_labels, bboxes):
    """Reconstruit le contenu .txt YOLO depuis les résultats Albumentations."""
    lines = []
    for cls, box in zip(class_labels, bboxes):
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)

def draw_bboxes(image_rgb, class_labels, bboxes, color=(0, 255, 100)):
    """Dessine les bounding boxes sur l'image pour prévisualisation."""
    img = image_rgb.copy()
    h, w = img.shape[:2]
    for cls, box in zip(class_labels, bboxes):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(cls), (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def build_augmentor(is_nok_mode: bool):
    """Construit le pipeline Albumentations selon les options choisies."""
    transforms = []

    if use_flip_h:
        transforms.append(A.HorizontalFlip(p=0.5))
    if use_flip_v:
        transforms.append(A.VerticalFlip(p=0.2))
    if use_rotate:
        transforms.append(A.Rotate(limit=rot_limit, p=0.8))
    if use_zoom:
        transforms.append(A.RandomScale(scale_limit=zoom_limit, p=0.6))
    if use_shift:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=0, rotate_limit=0, p=0.5))
    if use_brightness:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=bright_limit, contrast_limit=bright_limit, p=0.8))
    if use_noise:
        transforms.append(A.GaussNoise(var_limit=(5, noise_limit), p=0.5))
    if use_blur:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
    if use_hsv:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5))
    if use_shadow:
        transforms.append(A.RandomShadow(p=0.3))

    if not transforms:
        transforms.append(A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
    )

# ─────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ─────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 2])

with col_upload:
    st.subheader("1. Upload")
    img_file = st.file_uploader(
        "Image (.jpg / .png)", type=["jpg", "jpeg", "png"], key="img")
    txt_file = st.file_uploader(
        "Annotation YOLO (.txt)", type=["txt"], key="lbl")

    is_nok = st.checkbox(
        "Image de défaut (NOK) — augmentation agressive",
        help="Active des transformations plus nombreuses pour les classes sous-représentées."
    )

if img_file and txt_file:
    # Chargement image
    file_bytes = np.frombuffer(img_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Chargement labels
    txt_content = txt_file.read().decode("utf-8")
    labels = parse_yolo_labels(txt_content)

    if not labels:
        st.error("Aucune annotation valide trouvée dans le fichier .txt.")
        st.stop()

    class_ids = [l[0] for l in labels]
    bboxes    = [[l[1], l[2], l[3], l[4]] for l in labels]

    with col_preview:
        st.subheader("2. Aperçu original")
        img_with_boxes = draw_bboxes(img_rgb, class_ids, bboxes)
        st.image(img_with_boxes, caption=f"Original — {len(labels)} annotation(s)", use_column_width=True)
        st.caption(f"Classes détectées : {sorted(set(class_ids))}")

    # ─────────────────────────────────────────────
    # GÉNÉRATION
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("3. Générer les copies augmentées")

    if st.button("Lancer l'augmentation", type="primary"):
        augmentor = build_augmentor(is_nok)

        zip_buffer = io.BytesIO()
        stem = os.path.splitext(img_file.name)[0]

        generated = 0
        failed    = 0
        preview_imgs = []

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

            # Inclure l'original dans le ZIP
            _, orig_enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            zf.writestr(f"{stem}_orig.jpg", orig_enc.tobytes())
            zf.writestr(f"{stem}_orig.txt", txt_content)

            progress = st.progress(0, text="Génération en cours...")

            attempts = 0
            max_attempts = n_copies * 4

            while generated < n_copies and attempts < max_attempts:
                attempts += 1
                try:
                    result = augmentor(image=img_rgb, bboxes=bboxes, class_labels=class_ids)

                    if not result['bboxes']:
                        failed += 1
                        continue

                    aug_img  = result['image']
                    aug_cls  = result['class_labels']
                    aug_bbox = list(result['bboxes'])

                    # Encoder image
                    aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    _, aug_enc = cv2.imencode(".jpg", aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Créer label .txt
                    aug_txt = format_yolo_labels(aug_cls, aug_bbox)

                    fname = f"{stem}_aug{generated:03d}"
                    zf.writestr(f"{fname}.jpg", aug_enc.tobytes())
                    zf.writestr(f"{fname}.txt", aug_txt)

                    # Garder 4 aperçus max
                    if len(preview_imgs) < 4:
                        preview_imgs.append((aug_img, aug_cls, aug_bbox))

                    generated += 1
                    progress.progress(generated / n_copies,
                                      text=f"Génération : {generated}/{n_copies}")

                except Exception as e:
                    failed += 1

            progress.empty()

        # ─── Résultats ───
        st.success(f"✅ {generated} images générées ({failed} tentatives échouées — bbox hors cadre).")

        # Aperçu des 4 premières
        st.subheader("Aperçu des premières copies")
        preview_cols = st.columns(min(4, len(preview_imgs)))
        for i, (p_img, p_cls, p_bbox) in enumerate(preview_imgs):
            with preview_cols[i]:
                p_drawn = draw_bboxes(p_img, p_cls, p_bbox)
                st.image(p_drawn, caption=f"aug_{i:03d}", use_column_width=True)

        # ─── Téléchargement ZIP ───
        st.download_button(
            label=f"⬇ Télécharger le ZIP ({generated + 1} images + annotations)",
            data=zip_buffer.getvalue(),
            file_name=f"{stem}_augmented_{generated}copies.zip",
            mime="application/zip",
        )

        st.info(
            f"Le ZIP contient **{generated + 1} images** (original inclus) "
            f"et **{generated + 1} fichiers .txt YOLO** — zéro ré-annotation nécessaire."
        )

elif img_file and not txt_file:
    st.warning("Upload le fichier .txt YOLO correspondant à l'image.")
elif txt_file and not img_file:
    st.warning("Upload l'image correspondant au fichier .txt.")
else:
    st.info("Upload une image et son fichier d'annotation .txt YOLO pour commencer.")