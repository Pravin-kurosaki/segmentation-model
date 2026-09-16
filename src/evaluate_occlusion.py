"""
evaluate_occlusion.py - Phase 2 Occlusion Evaluation & Visualization Pipeline.

Evaluates test samples under varying occlusion conditions:
  - Clean unoccluded faces
  - Masked faces (surgical, cloth, KN95)
  - Sunglasses faces (aviator, wayfarer, round)
  - Heavily occluded faces (mask + sunglasses)

Produces:
  1. Detailed occlusion statistics and distributions
  2. Multi-panel diagnostic visualization cards saved to results/occlusion_reports/
  3. Metrics JSON summary (results/metrics/occlusion_evaluation.json)
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import occlusion as occ_mod
import occlusion_augmenter as occ_aug
from visualize import mask_to_color, overlay_mask


def generate_diagnostic_card(
    image: np.ndarray,
    pred_mask: np.ndarray,
    occ_info: dict,
    sample_id: str,
    save_path: Path,
):
    """
    Generates a 4-panel diagnostic card:
      1. Original input face
      2. Semantic segmentation mask
      3. Occlusion & region boundary overlay
      4. Quantitative Diagnostic Analytics & Decision Badge
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title(f"Input Face ({sample_id})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Segmentation Mask
    color_mask = mask_to_color(pred_mask, cfg.CLASS_COLORS)
    axes[1].imshow(color_mask)
    axes[1].set_title("13-Class Semantic Segmentation", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Occlusion overlay with boundaries
    overlay = occ_mod.draw_occlusion_overlay(image, pred_mask, occ_info)
    axes[2].imshow(overlay)
    axes[2].set_title("Occlusion & Tri-Region Partition", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # 4. Diagnostic readout panel
    ax_diag = axes[3]
    ax_diag.axis("off")

    tot_pct = occ_info["occlusion_percentage"]
    is_recog = occ_info["is_recognizable"]
    reg = occ_info["regional_occlusion"]
    w = occ_info["adaptive_weights"]

    # Status banner color
    banner_color = "#2E7D32" if is_recog else "#C62828"
    status_text = "STATUS: ACCEPT (<50% OCCLUDED)" if is_recog else "STATUS: REJECT (>=50% OCCLUDED)"

    # Draw card background
    card_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_diag.transAxes,
                              facecolor="#FAFAFA", edgecolor="#CCCCCC", linewidth=1.5, zorder=0)
    ax_diag.add_patch(card_rect)

    # Header banner
    banner_rect = plt.Rectangle((0, 0.84), 1, 0.16, transform=ax_diag.transAxes,
                                facecolor=banner_color, zorder=1)
    ax_diag.add_patch(banner_rect)
    ax_diag.text(0.5, 0.91, status_text, color="white", fontsize=11, fontweight="bold",
                 ha="center", va="center", transform=ax_diag.transAxes, zorder=2)

    # Analytics text
    content_y = 0.74
    ax_diag.text(0.06, content_y, f"Total Occlusion: {tot_pct:.1f}%", fontsize=12, fontweight="bold",
                 color="#D32F2F" if tot_pct >= 50 else "#1B5E20", transform=ax_diag.transAxes)

    # Horizontal progress bar for total occlusion
    bar_bg = plt.Rectangle((0.06, content_y - 0.07), 0.88, 0.045, transform=ax_diag.transAxes,
                           facecolor="#E0E0E0", edgecolor="#BDBDBD")
    bar_fill_color = "#E53935" if tot_pct >= 50 else ("#FB8C00" if tot_pct > 25 else "#43A047")
    bar_fill = plt.Rectangle((0.06, content_y - 0.07), 0.88 * min(1.0, tot_pct / 100.0), 0.045,
                             transform=ax_diag.transAxes, facecolor=bar_fill_color)
    # Threshold marker line at 50%
    ax_diag.add_patch(bar_bg)
    ax_diag.add_patch(bar_fill)
    ax_diag.plot([0.06 + 0.88 * 0.5, 0.06 + 0.88 * 0.5], [content_y - 0.08, content_y - 0.02],
                 color="black", linestyle="--", linewidth=1.5, transform=ax_diag.transAxes)
    ax_diag.text(0.06 + 0.88 * 0.5, content_y - 0.11, "50% Limit", fontsize=8, color="#616161",
                 ha="center", transform=ax_diag.transAxes)

    # Regional breakdown
    y_reg = 0.50
    ax_diag.text(0.06, y_reg, "Regional Occlusion:", fontsize=10, fontweight="bold", color="#333333", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_reg - 0.07, f"- Upper (Eyes/Forehead): {reg['upper']:.1f}%", fontsize=9.5, color="#424242", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_reg - 0.13, f"- Mid (Nose/Cheeks):     {reg['mid']:.1f}%", fontsize=9.5, color="#424242", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_reg - 0.19, f"- Lower (Mouth/Chin):    {reg['lower']:.1f}%", fontsize=9.5, color="#424242", transform=ax_diag.transAxes)

    # Adaptive weights
    y_w = 0.22
    ax_diag.text(0.06, y_w, "Adaptive Recognition Weights:", fontsize=10, fontweight="bold", color="#333333", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_w - 0.07, f"- Upper Face Weight: {w['upper']:.2f} ({int(w['upper']*100)}%)", fontsize=9.5, color="#1565C0", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_w - 0.13, f"- Mid Face Weight:   {w['mid']:.2f} ({int(w['mid']*100)}%)", fontsize=9.5, color="#1565C0", transform=ax_diag.transAxes)
    ax_diag.text(0.10, y_w - 0.19, f"- Lower Face Weight: {w['lower']:.2f} ({int(w['lower']*100)}%)", fontsize=9.5, color="#1565C0", transform=ax_diag.transAxes)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_occlusion_evaluation(n_samples: int = 12):
    """Runs full occlusion estimation evaluation on a set of test faces."""
    print("=" * 65)
    print("  PHASE 2: OCCLUSION ESTIMATION & ADAPTIVE WEIGHTING EVALUATION")
    print("=" * 65)

    test_split_file = cfg.SPLITS_DIR / "test.txt"
    if not test_split_file.exists():
        print(f"Error: {test_split_file} does not exist.")
        return

    with open(test_split_file) as f:
        stems = [line.strip() for line in f if line.strip()][:n_samples]

    reports_dir = cfg.RESULTS_DIR / "occlusion_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\nProcessing {len(stems)} test samples under multiple occlusion conditions...\n")

    for idx, stem in enumerate(stems):
        img_path = cfg.PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = cfg.PROCESSED_MASKS_DIR / f"{stem}.png"

        if not img_path.exists() or not mask_path.exists():
            continue

        raw_img = np.array(Image.open(img_path).convert("RGB"))
        raw_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        # Create 4 test conditions:
        # 1. Clean
        # 2. Masked
        # 3. Sunglasses
        # 4. Heavy (both)
        conditions = [
            ("clean", raw_img, raw_mask),
        ]

        m_img, m_mask = occ_aug.apply_face_mask(raw_img, raw_mask)
        conditions.append(("mask", m_img, m_mask))

        s_img, s_mask = occ_aug.apply_sunglasses(raw_img, raw_mask)
        conditions.append(("sunglasses", s_img, s_mask))

        h_img, h_mask = occ_aug.apply_sunglasses(m_img, m_mask)
        conditions.append(("heavy", h_img, h_mask))

        for cond_name, c_img, c_mask in conditions:
            occ_info = occ_mod.estimate_occlusion(c_mask)
            sample_key = f"{stem}_{cond_name}"

            # Save card
            card_path = reports_dir / f"{sample_key}_card.png"
            generate_diagnostic_card(c_img, c_mask, occ_info, sample_key, card_path)

            results.append({
                "sample_id": sample_key,
                "stem": stem,
                "condition": cond_name,
                "occlusion_pct": occ_info["occlusion_percentage"],
                "is_recognizable": occ_info["is_recognizable"],
                "regional_occlusion": occ_info["regional_occlusion"],
                "adaptive_weights": occ_info["adaptive_weights"],
                "card_path": str(card_path),
            })

            recog_badge = "[ACCEPT]" if occ_info["is_recognizable"] else "[REJECT]"
            print(f"  {sample_key:<20} | Occ: {occ_info['occlusion_percentage']:>5.1f}% | {recog_badge} | "
                  f"Weights (U/M/L): {occ_info['adaptive_weights']['upper']:.2f} / "
                  f"{occ_info['adaptive_weights']['mid']:.2f} / "
                  f"{occ_info['adaptive_weights']['lower']:.2f}")

    # Summary metrics
    summary_path = cfg.RESULTS_DIR / "metrics" / "occlusion_evaluation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Evaluation complete! Saved {len(results)} diagnostic cards.")
    print(f"  Reports directory : {reports_dir}")
    print(f"  Metrics JSON      : {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_occlusion_evaluation(n_samples=4)
