"""
occlusion_augmenter.py - Photorealistic synthetic occlusion augmentation for faces.

Generates realistic face masks (surgical, KN95, cloth) and sunglasses (aviator, wayfarer, round)
with pixel-perfect ground-truth segmentation masks for:
  - Class 10: sunglasses
  - Class 11: mask

Uses the underlying semantic segmentation labels (skin, nose, mouth, eyes, ears)
to derive anatomically accurate facial anchor points and boundaries.
"""

import random
import numpy as np
import cv2


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
    Synthesizes a realistic face mask onto the face and updates the ground-truth mask.
    Returns: (augmented_image_rgb, augmented_mask)
    """
    h, w = mask.shape[:2]
    aug_img = image.copy()
    aug_mask = mask.copy()

    # Need mouth or nose or skin
    mouth_box = get_region_bbox(mask, 5)  # mouth
    nose_box  = get_region_bbox(mask, 4)  # nose
    skin_box  = get_region_bbox(mask, 1)  # skin

    if mouth_box is None and nose_box is None:
        return aug_img, aug_mask  # Cannot reliably place mask

    # Compute key anchors
    if nose_box is not None:
        ny_min, nx_min, ny_max, nx_max = nose_box
        top_y = int(ny_min + (ny_max - ny_min) * 0.20)
        nose_cx = int((nx_min + nx_max) / 2)
    else:
        my_min, mx_min, my_max, mx_max = mouth_box
        top_y = max(0, my_min - int((my_max - my_min) * 1.5))
        nose_cx = int((mx_min + mx_max) / 2)

    if mouth_box is not None:
        my_min, mx_min, my_max, mx_max = mouth_box
        mouth_cx = int((mx_min + mx_max) / 2)
        mouth_h = my_max - my_min
    else:
        my_min = top_y + 40
        my_max = top_y + 70
        mouth_cx = nose_cx
        mouth_h = 30

    # Chin bottom anchor
    if skin_box is not None:
        sy_min, sx_min, sy_max, sx_max = skin_box
        chin_y = min(h - 1, int(sy_max - (sy_max - my_max) * 0.15))
        left_bound  = max(0, sx_min + int((sx_max - sx_min) * 0.08))
        right_bound = min(w - 1, sx_max - int((sx_max - sx_min) * 0.08))
    else:
        chin_y = min(h - 1, my_max + mouth_h * 2)
        left_bound = max(0, nose_cx - 60)
        right_bound = min(w - 1, nose_cx + 60)

    # Lateral cheek anchors
    mid_y = int((top_y + chin_y) / 2)

    # Styles: surgical, kn95, cloth
    styles = ["surgical", "kn95", "cloth"]
    chosen_style = random.choice(styles) if style == "random" else style

    # Colors (RGB)
    color_palette = {
        "blue":      (180, 215, 240),
        "cyan":      (175, 225, 220),
        "white":     (240, 242, 245),
        "black":     (35, 38, 42),
        "navy":      (30, 45, 80),
        "grey":      (140, 145, 150),
        "burgundy":  (90, 25, 35),
    }
    if color == "random":
        chosen_color = random.choice(list(color_palette.values()))
    elif color in color_palette:
        chosen_color = color_palette[color]
    else:
        chosen_color = (180, 215, 240)

    # Build mask polygon points
    p_nose_top    = [nose_cx, top_y]
    p_left_upper  = [int(left_bound * 0.65 + nose_cx * 0.35), int(top_y + (mid_y - top_y) * 0.35)]
    p_right_upper = [int(right_bound * 0.65 + nose_cx * 0.35), int(top_y + (mid_y - top_y) * 0.35)]
    p_left_mid    = [left_bound, mid_y]
    p_right_mid   = [right_bound, mid_y]
    p_left_lower  = [int(left_bound * 0.35 + nose_cx * 0.65), int(chin_y - (chin_y - mid_y) * 0.15)]
    p_right_lower = [int(right_bound * 0.35 + nose_cx * 0.65), int(chin_y - (chin_y - mid_y) * 0.15)]
    p_chin        = [int((nose_cx + mouth_cx) / 2), chin_y]

    poly_pts = np.array([
        p_nose_top,
        p_right_upper,
        p_right_mid,
        p_right_lower,
        p_chin,
        p_left_lower,
        p_left_mid,
        p_left_upper,
    ], dtype=np.int32)

    # Smooth contour
    mask_layer = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_layer, [poly_pts], 255)
    mask_layer = cv2.GaussianBlur(mask_layer, (7, 7), 2.0)
    binary_mask = (mask_layer > 100).astype(np.uint8) * 255

    # Refine mask layer to only cover face skin / mouth / nose / beard areas
    valid_face = (mask == 1) | (mask == 4) | (mask == 5)
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=valid_face.astype(np.uint8) * 255)

    if binary_mask.sum() == 0:
        return aug_img, aug_mask

    # Visual rendering
    color_img = np.full((h, w, 3), chosen_color, dtype=np.uint8)

    # Gradient shading for 3D depth
    y_grad = np.tile(np.linspace(0.85, 1.1, h)[:, None], (1, w))
    for c in range(3):
        color_img[:, :, c] = np.clip(color_img[:, :, c].astype(np.float32) * y_grad, 0, 255).astype(np.uint8)

    if chosen_style == "surgical":
        for pleat_y in range(top_y + 18, chin_y - 8, 14):
            if pleat_y < h:
                cv2.line(color_img, (left_bound + 8, pleat_y), (right_bound - 8, pleat_y),
                         tuple(max(0, c - 28) for c in chosen_color), 2)
                cv2.line(color_img, (left_bound + 8, pleat_y + 1), (right_bound - 8, pleat_y + 1),
                         tuple(min(255, c + 25) for c in chosen_color), 1)
        # Top border seam
        cv2.polylines(color_img, [poly_pts[:3]], False, (245, 245, 245), 2)

    elif chosen_style == "kn95":
        cv2.line(color_img, (nose_cx, top_y), (nose_cx, chin_y),
                 tuple(max(0, c - 35) for c in chosen_color), 2)
        cv2.line(color_img, (nose_cx + 1, top_y), (nose_cx + 1, chin_y),
                 tuple(min(255, c + 35) for c in chosen_color), 1)

    # Ear loops (thin elastic cords towards ears or sides)
    cv2.line(color_img, (left_bound, mid_y - 10), (max(0, left_bound - 25), mid_y - 5), (230, 230, 235), 2)
    cv2.line(color_img, (right_bound, mid_y - 10), (min(w - 1, right_bound + 25), mid_y - 5), (230, 230, 235), 2)

    # Blend
    alpha = cv2.GaussianBlur(binary_mask.astype(np.float32) / 255.0, (5, 5), 1.0)
    alpha = np.stack([alpha] * 3, axis=-1)

    aug_img = (alpha * color_img + (1.0 - alpha) * aug_img).astype(np.uint8)

    # Update ground truth mask: all pixels covered by mask become mask_class_id
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
    Synthesizes realistic sunglasses onto the face and updates the ground-truth mask.
    Returns: (augmented_image_rgb, augmented_mask)
    """
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
    lens_w = int(eye_dist * 0.52)
    lens_h = int(eye_dist * 0.42)

    sg_layer = np.zeros((h, w), dtype=np.uint8)

    styles = ["aviator", "wayfarer", "round"]
    chosen_style = random.choice(styles) if style == "random" else style

    for (cx, cy) in [(lcx, lcy), (rcx, rcy)]:
        if chosen_style == "round":
            r = int((lens_w + lens_h) / 4)
            cv2.circle(sg_layer, (cx, cy), r + 4, 255, -1)
        elif chosen_style == "aviator":
            pts = np.array([
                [cx - lens_w // 2, cy - lens_h // 2],
                [cx + lens_w // 2, cy - lens_h // 2],
                [cx + int(lens_w * 0.42), cy + int(lens_h * 0.55)],
                [cx - int(lens_w * 0.35), cy + int(lens_h * 0.65)],
            ], dtype=np.int32)
            cv2.fillPoly(sg_layer, [pts], 255)
        else: # wayfarer
            x1, y1 = cx - lens_w // 2, cy - lens_h // 2
            x2, y2 = cx + lens_w // 2, cy + lens_h // 2
            cv2.rectangle(sg_layer, (x1, y1), (x2, y2), 255, -1)

    # Bridge between eyes
    bridge_y = int((lcy + rcy) / 2 - lens_h * 0.15)
    cv2.line(sg_layer, (lcx + lens_w // 4, bridge_y), (rcx - lens_w // 4, bridge_y), 255, 5)

    # Dilate slightly for smooth frame coverage
    sg_layer = cv2.dilate(sg_layer, np.ones((5, 5), np.uint8), iterations=1)

    # Dark lens color & reflection
    lens_color = (20, 22, 26)  # dark charcoal
    color_img = np.full((h, w, 3), lens_color, dtype=np.uint8)

    # Specular reflections
    for offset in [-8, 6]:
        pt1 = (lcx - lens_w // 3 + offset, lcy - lens_h // 2 + 4)
        pt2 = (lcx + offset, lcy + lens_h // 2 - 4)
        cv2.line(color_img, pt1, pt2, (110, 120, 135), 2)

        pt3 = (rcx - lens_w // 3 + offset, rcy - lens_h // 2 + 4)
        pt4 = (rcx + offset, rcy + lens_h // 2 - 4)
        cv2.line(color_img, pt3, pt4, (110, 120, 135), 2)

    # Frames contour
    contours, _ = cv2.findContours(sg_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_img, contours, -1, (12, 14, 16), 3)

    # Alpha blend
    alpha = (sg_layer.astype(np.float32) / 255.0) * 0.94
    alpha = np.stack([alpha] * 3, axis=-1)

    aug_img = (alpha * color_img + (1.0 - alpha) * aug_img).astype(np.uint8)

    # Update ground truth mask
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
    """Randomly applies mask, sunglasses, both, or none."""
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
    elif r < p_mask * 0.90 + p_sunglasses * 0.40:
        # Both mask & sunglasses
        aug_img, aug_mask = apply_face_mask(aug_img, aug_mask, mask_class_id=mask_class_id)
        aug_img, aug_mask = apply_sunglasses(aug_img, aug_mask, sunglasses_class_id=sunglasses_class_id)
        meta["has_mask"] = True
        meta["has_sunglasses"] = True

    return aug_img, aug_mask, meta
