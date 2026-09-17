"""
occlusion_augmenter.py - Photorealistic occlusion augmentation for faces.

Uses authentic photographic templates with alpha transparency for:
  - Class 10: sunglasses (real dark Wayfarer lenses with reflections)
  - Class 11: mask (real surgical blue/green, KN95, N95, cloth masks)

Uses the underlying semantic segmentation labels (skin, nose, mouth, eyes, ears)
to derive anatomically accurate facial anchor points and boundaries.
"""

import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Directory holding high-resolution photographic templates
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "assets" / "templates"

# Cached templates in memory for high-throughput training
_CACHED_MASK_TEMPLATES = None
_CACHED_SUNGLASSES_TEMPLATE = None


def _load_templates():
    global _CACHED_MASK_TEMPLATES, _CACHED_SUNGLASSES_TEMPLATE
    if _CACHED_MASK_TEMPLATES is not None:
        return

    _CACHED_MASK_TEMPLATES = []
    if TEMPLATES_DIR.exists():
        mask_files = ["surgical_blue.png", "surgical_green.png", "surgical.png",
                      "KN95.png", "N95.png", "cloth.png"]
        for mf in mask_files:
            p = TEMPLATES_DIR / mf
            if p.exists():
                try:
                    tpl = Image.open(p).convert("RGBA")
                    _CACHED_MASK_TEMPLATES.append((mf, tpl))
                except Exception:
                    pass

        sg_path = TEMPLATES_DIR / "sunglasses.png"
        if sg_path.exists():
            try:
                _CACHED_SUNGLASSES_TEMPLATE = Image.open(sg_path).convert("RGBA")
            except Exception:
                pass


def get_region_bbox(mask: np.ndarray, class_id: int):
    """Return (ymin, xmin, ymax, xmax) or None if class not present."""
    ys, xs = np.where(mask == class_id)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def get_region_center(mask: np.ndarray, class_id: int):
    """Return (cy, cx) centroid or None."""
    ys, xs = np.where(mask == class_id)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def apply_face_mask(
    image: np.ndarray,
    mask: np.ndarray,
    style: str = "random",
    color: str = "random",
    mask_class_id: int = 11,
) -> tuple:
    """
    Overlays an authentic photographic face mask onto the face
    and sets the covered face pixels to mask_class_id in the ground-truth mask.
    Returns: (augmented_image_rgb, augmented_mask)
    """
    _load_templates()

    h, w = mask.shape[:2]
    aug_img = image.copy()
    aug_mask = mask.copy()

    # Locate facial anchors
    mouth_box = get_region_bbox(mask, 5)  # mouth
    nose_box  = get_region_bbox(mask, 4)  # nose
    skin_box  = get_region_bbox(mask, 1)  # skin

    if mouth_box is None and nose_box is None and skin_box is None:
        return aug_img, aug_mask

    # Calculate positioning bounds
    if nose_box is not None:
        ny_min, nx_min, ny_max, nx_max = nose_box
        top_y = max(0, int(ny_min + (ny_max - ny_min) * 0.15))
        nose_cx = int((nx_min + nx_max) / 2)
    elif mouth_box is not None:
        my_min, mx_min, my_max, mx_max = mouth_box
        top_y = max(0, int(my_min - (my_max - my_min) * 1.5))
        nose_cx = int((mx_min + mx_max) / 2)
    else:
        top_y = int(h * 0.40)
        nose_cx = int(w * 0.50)

    if skin_box is not None:
        sy_min, sx_min, sy_max, sx_max = skin_box
        chin_y = min(h, int(sy_max + 2))
        left_bound  = max(0, int(sx_min + (sx_max - sx_min) * 0.04))
        right_bound = min(w, int(sx_max - (sx_max - sx_min) * 0.04))
    elif mouth_box is not None:
        my_min, mx_min, my_max, mx_max = mouth_box
        chin_y = min(h, my_max + int((my_max - my_min) * 1.5))
        left_bound = max(0, nose_cx - 50)
        right_bound = min(w, nose_cx + 50)
    else:
        chin_y = int(h * 0.90)
        left_bound = int(w * 0.20)
        right_bound = int(w * 0.80)

    target_w = max(10, right_bound - left_bound)
    target_h = max(10, chin_y - top_y)

    # Use photographic templates if available
    if _CACHED_MASK_TEMPLATES and len(_CACHED_MASK_TEMPLATES) > 0:
        _, tpl = random.choice(_CACHED_MASK_TEMPLATES)
        resized_tpl = tpl.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tpl_np = np.array(resized_tpl)

        alpha = (tpl_np[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        roi = aug_img[top_y:top_y + target_h, left_bound:left_bound + target_w]

        # Blend photographic mask
        blended = (alpha * tpl_np[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)
        aug_img[top_y:top_y + target_h, left_bound:left_bound + target_w] = blended

        # Update ground truth mask on significant alpha coverage
        mask_roi = aug_mask[top_y:top_y + target_h, left_bound:left_bound + target_w]
        is_covered = (tpl_np[:, :, 3] > 120) & (mask_roi != 0)
        mask_roi[is_covered] = mask_class_id
        aug_mask[top_y:top_y + target_h, left_bound:left_bound + target_w] = mask_roi

        return aug_img, aug_mask

    # Fallback to procedural drawing if templates unavailable
    color_img = np.full((h, w, 3), (180, 215, 240), dtype=np.uint8)
    poly_pts = np.array([
        [nose_cx, top_y],
        [right_bound, int((top_y + chin_y) / 2)],
        [nose_cx, chin_y],
        [left_bound, int((top_y + chin_y) / 2)],
    ], dtype=np.int32)
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [poly_pts], 255)
    valid_face = (mask == 1) | (mask == 4) | (mask == 5)
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=valid_face.astype(np.uint8) * 255)
    alpha = (binary_mask.astype(np.float32) / 255.0)[:, :, None]
    aug_img = (alpha * color_img + (1.0 - alpha) * aug_img).astype(np.uint8)
    aug_mask[binary_mask > 128] = mask_class_id
    return aug_img, aug_mask


def apply_sunglasses(
    image: np.ndarray,
    mask: np.ndarray,
    style: str = "random",
    tint: str = "random",
    sunglasses_class_id: int = 10,
) -> tuple:
    """
    Overlays authentic photographic sunglasses onto the eye region
    and updates ground truth mask with sunglasses_class_id.
    Returns: (augmented_image_rgb, augmented_mask)
    """
    _load_templates()

    h, w = mask.shape[:2]
    aug_img = image.copy()
    aug_mask = mask.copy()

    left_box  = get_region_bbox(mask, 2)  # left eye
    right_box = get_region_bbox(mask, 3)  # right eye

    if left_box is None and right_box is None:
        return aug_img, aug_mask

    if left_box is not None and right_box is not None:
        ly1, lx1, ly2, lx2 = left_box
        ry1, rx1, ry2, rx2 = right_box
    elif left_box is not None:
        ly1, lx1, ly2, lx2 = left_box
        eye_w = lx2 - lx1
        ry1, rx1, ry2, rx2 = ly1, lx2 + int(eye_w * 0.8), ly2, lx2 + int(eye_w * 1.8)
    else:
        ry1, rx1, ry2, rx2 = right_box
        eye_w = rx2 - rx1
        ly1, lx1, ly2, lx2 = ry1, rx1 - int(eye_w * 1.8), ry2, rx1 - int(eye_w * 0.8)

    lcx, lcy = int((lx1 + lx2) / 2), int((ly1 + ly2) / 2)
    rcx, rcy = int((rx1 + rx2) / 2), int((ry1 + ry2) / 2)

    eye_dist = max(25, int(np.hypot(rcx - lcx, rcy - lcy)))
    cx, cy = int((lcx + rcx) / 2), int((lcy + rcy) / 2)

    # Use photographic sunglasses template if available
    if _CACHED_SUNGLASSES_TEMPLATE is not None:
        sg_w = int(eye_dist * 2.3)
        sg_h = int(sg_w * (_CACHED_SUNGLASSES_TEMPLATE.height / _CACHED_SUNGLASSES_TEMPLATE.width))

        resized_sg = _CACHED_SUNGLASSES_TEMPLATE.resize((sg_w, sg_h), Image.Resampling.LANCZOS)
        sg_np = np.array(resized_sg)

        x1 = max(0, cx - sg_w // 2)
        y1 = max(0, cy - sg_h // 2)
        x2 = min(w, x1 + sg_w)
        y2 = min(h, y1 + sg_h)

        crop_sg = sg_np[:(y2 - y1), :(x2 - x1)]
        alpha = (crop_sg[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

        roi = aug_img[y1:y2, x1:x2]
        aug_img[y1:y2, x1:x2] = (alpha * crop_sg[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)

        mask_roi = aug_mask[y1:y2, x1:x2]
        is_covered = (crop_sg[:, :, 3] > 120)
        mask_roi[is_covered] = sunglasses_class_id
        aug_mask[y1:y2, x1:x2] = mask_roi

        return aug_img, aug_mask

    # Fallback to procedural rectangles
    lens_w = int(eye_dist * 0.52)
    lens_h = int(eye_dist * 0.42)
    sg_layer = np.zeros((h, w), dtype=np.uint8)
    for (px, py) in [(lcx, lcy), (rcx, rcy)]:
        cv2.rectangle(sg_layer, (px - lens_w // 2, py - lens_h // 2), (px + lens_w // 2, py + lens_h // 2), 255, -1)
    bridge_y = int((lcy + rcy) / 2 - lens_h * 0.15)
    cv2.line(sg_layer, (lcx, bridge_y), (rcx, bridge_y), 255, 4)
    sg_alpha = (sg_layer.astype(np.float32) / 255.0)[:, :, None]
    aug_img = (sg_alpha * np.array([20, 22, 26]) + (1.0 - sg_alpha) * aug_img).astype(np.uint8)
    aug_mask[sg_layer > 128] = sunglasses_class_id
    return aug_img, aug_mask


def apply_random_occlusion(
    image: np.ndarray,
    mask: np.ndarray,
    p_mask: float = 0.5,
    p_sunglasses: float = 0.5,
    mask_class_id: int = 11,
    sunglasses_class_id: int = 10,
) -> tuple:
    """Randomly applies real photographic mask, sunglasses, both, or none."""
    aug_img = image.copy()
    aug_mask = mask.copy()
    meta = {"has_mask": False, "has_sunglasses": False}

    r = random.random()
    if r < p_mask * 0.45:
        aug_img, aug_mask = apply_face_mask(aug_img, aug_mask, mask_class_id=mask_class_id)
        meta["has_mask"] = True
    elif r < p_mask * 0.90:
        aug_img, aug_mask = apply_sunglasses(aug_img, aug_mask, sunglasses_class_id=sunglasses_class_id)
        meta["has_sunglasses"] = True
    elif r < (p_mask * 0.90 + p_sunglasses * 0.40):
        # Both mask & sunglasses (heavy occlusion)
        aug_img, aug_mask = apply_face_mask(aug_img, aug_mask, mask_class_id=mask_class_id)
        aug_img, aug_mask = apply_sunglasses(aug_img, aug_mask, sunglasses_class_id=sunglasses_class_id)
        meta["has_mask"] = True
        meta["has_sunglasses"] = True

    return aug_img, aug_mask, meta
