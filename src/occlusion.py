"""
occlusion.py - Phase 2: Occlusion Estimation & Adaptive Weighting Module.

Functions:
  1. estimate_occlusion(mask, num_classes=13) -> dict
     - Calculates canonical face area, occlusion area, total percentage
     - Tri-region breakdown: upper face (eyes/forehead), mid face (nose/cheeks), lower face (mouth/chin)
     - Determines recognition viability (occlusion < 50%)
     - Computes normalized adaptive region weights for downstream face recognition
  2. visualize_occlusion_report(image, mask, occlusion_info, save_path=None) -> np.ndarray
     - Generates rich multi-panel visual diagnostic card
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def estimate_occlusion(
    mask: np.ndarray,
    mask_class_id: int = 11,
    sunglasses_class_id: int = 10,
    glasses_class_id: int = 9,
    hat_class_id: int = 12,
) -> dict:
    """
    Estimates facial occlusion from a semantic segmentation mask.

    Parameters:
      mask: (H, W) uint8 or int array with class IDs
      mask_class_id: class ID for face masks (default 11)
      sunglasses_class_id: class ID for sunglasses (default 10)
      glasses_class_id: class ID for eyeglasses (default 9)
      hat_class_id: class ID for hat/headwear (default 12)

    Returns:
      dict with comprehensive occlusion analytics:
        - total_face_pixels: int
        - occluded_pixels: int
        - occlusion_percentage: float (0.0 to 100.0)
        - is_recognizable: bool (True if occlusion_percentage < 50.0)
        - viability_status: str ("ACCEPT" | "REJECT")
        - regional_occlusion: dict (upper, mid, lower percentages)
        - adaptive_weights: dict (upper, mid, lower normalized weights summing to 1.0)
        - occlusion_breakdown: dict (mask_pct, sunglasses_pct, hat_pct)
        - bounding_boxes: dict of region bounds
    """
    h, w = mask.shape[:2]

    # Core facial anatomy classes
    # 1: skin, 2: left_eye, 3: right_eye, 4: nose, 5: mouth
    core_face_mask = (
        (mask == 1) |
        (mask == 2) |
        (mask == 3) |
        (mask == 4) |
        (mask == 5)
    )

    # Occluders
    mask_occ       = (mask == mask_class_id)
    sunglasses_occ = (mask == sunglasses_class_id)
    hat_occ        = (mask == hat_class_id)
    glasses_occ    = (mask == glasses_class_id)

    # Compute facial boundary and geometric convex hull
    face_elements = core_face_mask | mask_occ | sunglasses_occ | glasses_occ
    face_coords = np.where(face_elements)
    if len(face_coords[0]) == 0:
        return {
            "total_face_pixels": 0,
            "occluded_pixels": 0,
            "occlusion_percentage": 0.0,
            "is_recognizable": False,
            "viability_status": "NO_FACE_DETECTED",
            "regional_occlusion": {"upper": 0.0, "mid": 0.0, "lower": 0.0},
            "adaptive_weights": {"upper": 0.333, "mid": 0.333, "lower": 0.334},
            "occlusion_breakdown": {"mask": 0.0, "sunglasses": 0.0, "hat": 0.0},
            "region_bounds": None,
            "occ_mask": np.zeros((h, w), dtype=bool),
        }

    ymin, xmin = int(face_coords[0].min()), int(face_coords[1].min())
    ymax, xmax = int(face_coords[0].max()), int(face_coords[1].max())

    # Compute facial convex hull envelope to detect occluder intrusions
    contours, _ = cv2.findContours(face_elements.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10] or contours
        all_pts = np.vstack(valid_contours).reshape(-1, 2)

        # Anatomical Chin Prior: Detect if the lower face/mouth is occluded
        eyes = (mask == 2) | (mask == 3)
        eye_ys = np.where(eyes)[0]
        hair_ys = np.where(mask == 6)[0]

        if len(eye_ys) > 0:
            y_eyes = float(eye_ys.mean())
            cranial_half = (y_eyes - hair_ys.min()) if len(hair_ys) > 0 else (ymax - ymin) * 0.5
            w_face = xmax - xmin
            cx = (xmin + xmax) // 2
            expected_chin_y = min(h - 4, int(y_eyes + max(cranial_half * 1.05, w_face * 0.85)))

            # If visible face stops noticeably above expected chin (indicating mask/hand occlusion)
            if ymax < expected_chin_y - 18:
                chin_contour = np.array([
                    [cx, expected_chin_y],
                    [int(cx - w_face * 0.25), int(expected_chin_y - 6)],
                    [int(cx + w_face * 0.25), int(expected_chin_y - 6)],
                    [int(cx - w_face * 0.42), int(expected_chin_y - 25)],
                    [int(cx + w_face * 0.42), int(expected_chin_y - 25)],
                ])
                all_pts = np.vstack([all_pts, chin_contour])
                ymax = expected_chin_y

        face_h = max(1, ymax - ymin)
        hull = cv2.convexHull(all_pts)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
        total_face_area = (hull_mask == 1)
    else:
        face_h = max(1, ymax - ymin)
        total_face_area = face_elements

    total_face_pixels = max(1, int(total_face_area.sum()))

    # Direct occlusion includes:
    # 1. Any non-anatomical intrusion inside the facial convex hull (masks/sunglasses classified as background)
    # 2. Explicit occluder classes (mask, sunglasses)
    hull_intrusions = total_face_area & (~core_face_mask)
    direct_occlusion = hull_intrusions | mask_occ | sunglasses_occ
    occluded_pixels = int(direct_occlusion.sum())

    raw_occlusion_pct = (occluded_pixels / total_face_pixels) * 100.0
    # Clean faces have small (~0-3%) boundary concavities along jawline; threshold them out
    total_occlusion_pct = round(max(0.0, raw_occlusion_pct - 2.5), 2) if raw_occlusion_pct > 3.0 else 0.0
    is_recognizable = (total_occlusion_pct < 50.0)

    # Tri-region facial partition (vertical slicing based on face height)
    # Upper face: top 35% (forehead, eyes, upper bridge)
    # Mid face: middle 30% (nose, upper cheeks)
    # Lower face: bottom 35% (mouth, chin, jawline)
    y_split1 = int(ymin + face_h * 0.35)
    y_split2 = int(ymin + face_h * 0.65)

    # Region masks
    upper_zone = np.zeros((h, w), dtype=bool)
    upper_zone[ymin:y_split1, xmin:xmax] = True

    mid_zone = np.zeros((h, w), dtype=bool)
    mid_zone[y_split1:y_split2, xmin:xmax] = True

    lower_zone = np.zeros((h, w), dtype=bool)
    lower_zone[y_split2:ymax+1, xmin:xmax] = True

    def calc_zone_stats(zone_mask):
        zone_face = int((total_face_area & zone_mask).sum())
        if zone_face == 0:
            return 0.0
        zone_occ = int((direct_occlusion & zone_mask).sum())
        raw_pct = (zone_occ / zone_face) * 100.0
        return round(max(0.0, raw_pct - 2.5), 2) if raw_pct > 3.0 else 0.0

    occ_upper = calc_zone_stats(upper_zone)
    occ_mid   = calc_zone_stats(mid_zone)
    occ_lower = calc_zone_stats(lower_zone)

    # Adaptive region weights for recognition
    # Higher weight given to regions with low occlusion
    raw_w_upper = max(0.05, 1.0 - (occ_upper / 100.0))
    raw_w_mid   = max(0.05, 1.0 - (occ_mid / 100.0))
    raw_w_lower = max(0.05, 1.0 - (occ_lower / 100.0))

    w_sum = raw_w_upper + raw_w_mid + raw_w_lower
    w_upper = round(raw_w_upper / w_sum, 3)
    w_mid   = round(raw_w_mid / w_sum, 3)
    w_lower = round(1.0 - w_upper - w_mid, 3)

    return {
        "total_face_pixels": total_face_pixels,
        "occluded_pixels": occluded_pixels,
        "occlusion_percentage": total_occlusion_pct,
        "is_recognizable": is_recognizable,
        "viability_status": "ACCEPT" if is_recognizable else "REJECT",
        "regional_occlusion": {
            "upper": occ_upper,
            "mid":   occ_mid,
            "lower": occ_lower,
        },
        "adaptive_weights": {
            "upper": w_upper,
            "mid":   w_mid,
            "lower": w_lower,
        },
        "occlusion_breakdown": {
            "mask": round((int(mask_occ.sum()) / total_face_pixels) * 100.0, 2),
            "sunglasses": round((int(sunglasses_occ.sum()) / total_face_pixels) * 100.0, 2),
            "hat": round((int(hat_occ.sum()) / total_face_pixels) * 100.0, 2),
        },
        "region_bounds": {
            "ymin": ymin, "ymax": ymax,
            "xmin": xmin, "xmax": xmax,
            "y_split1": y_split1,
            "y_split2": y_split2,
        },
        "occ_mask": direct_occlusion,
    }


def draw_occlusion_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    occ_info: dict,
) -> np.ndarray:
    """
    Draws facial bounding box, tri-region dividers, and occlusion heat highlights on image.
    """
    h, w = image.shape[:2]
    canvas = image.copy()
    b = occ_info.get("region_bounds")
    if not b:
        return canvas

    ymin, ymax = b["ymin"], b["ymax"]
    xmin, xmax = b["xmin"], b["xmax"]
    y_split1, y_split2 = b["y_split1"], b["y_split2"]

    # Draw region horizontal division lines (subtle dashed/colored)
    cv2.line(canvas, (xmin, y_split1), (xmax, y_split1), (255, 200, 0), 1)
    cv2.line(canvas, (xmin, y_split2), (xmax, y_split2), (255, 200, 0), 1)

    # Highlight occluded pixels with red-orange tint
    occ_pixels = occ_info.get("occ_mask")
    if occ_pixels is None:
        occ_pixels = (mask == 10) | (mask == 11)
    if occ_pixels.any():
        overlay = canvas.copy()
        overlay[occ_pixels] = [255, 60, 60]  # bright red highlight
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    # Outer bounding box
    color = (0, 230, 100) if occ_info["is_recognizable"] else (230, 40, 40)
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

    return canvas
