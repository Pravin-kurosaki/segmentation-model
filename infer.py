"""
infer.py - Quickstart Single-Image Inference and Occlusion Diagnostics.

Performs:
  1. Face Semantic Segmentation (13 classes)
  2. Occlusion Estimation (Convex face hull vs occluding pixels)
  3. Tri-Region Breakdown (Upper, Mid, Lower)
  4. Viability Check (<50% occlusion threshold: ACCEPT vs REJECT)
  5. Adaptive Region Weighting for downstream face recognition
  6. Generates 4-panel diagnostic card

Usage Examples:
  # Inference on clean face:
  python infer.py --image dataset/sample/images/00000.png

  # Inference with simulated mask:
  python infer.py --image dataset/sample/images/00000.png --occlude mask

  # Inference with simulated sunglasses:
  python infer.py --image dataset/sample/images/00000.png --occlude sunglasses

  # Inference with heavy occlusion (mask + sunglasses):
  python infer.py --image dataset/sample/images/00000.png --occlude heavy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
import src.config as cfg
from src.model import build_model
import src.occlusion as occ_mod
import src.occlusion_augmenter as occ_aug
from src.evaluate_occlusion import generate_diagnostic_card


def parse_args():
    parser = argparse.ArgumentParser(description="Occlusion-Aware Face Segmentation and Inference")
    parser.add_argument("--image", type=str, default="dataset/sample/images/00000.png",
                        help="Path to input face image")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.BEST_MODEL_PATH),
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="results/inference_output.png",
                        help="Path to save diagnostic output card")
    parser.add_argument("--occlude", type=str, default="none",
                        choices=["none", "mask", "sunglasses", "heavy"],
                        help="Optional synthetic occlusion to simulate on input image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()


def load_inference_model(checkpoint_path: str, device: torch.device):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        msg = f"Model checkpoint not found at: {checkpoint_path}\nPlease download the trained weights from GitHub Releases or train using src/train.py."
        raise FileNotFoundError(msg)

    encoder_name = getattr(cfg, "ENCODER_NAME", "resnet18")
    num_classes = getattr(cfg, "NUM_CLASSES", 13)

    model = build_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        num_classes=num_classes,
        in_channels=3
    )

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Input image not found: {image_path}")
        sys.exit(1)

    target_size = cfg.IMAGE_SIZE if isinstance(cfg.IMAGE_SIZE, tuple) else (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    h_target, w_target = target_size

    # Load and resize image
    raw_img = Image.open(image_path).convert("RGB")
    if raw_img.size != (w_target, h_target):
        raw_img = raw_img.resize((w_target, h_target), Image.Resampling.BILINEAR)
    img_np = np.array(raw_img, dtype=np.uint8)

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}")
    model = load_inference_model(args.checkpoint, device)

    # Preprocessing transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD),
    ])

    simulated_name = "Original (No Augmentation)"
    if args.occlude != "none":
        # Check if corresponding ground-truth mask exists in sample directory
        potential_mask = image_path.parent.parent / "masks" / f"{image_path.stem}.png"
        if potential_mask.exists():
            guide_mask = np.array(Image.open(potential_mask))
            if guide_mask.shape != (h_target, w_target):
                guide_mask = np.array(Image.fromarray(guide_mask).resize((w_target, h_target), Image.Resampling.NEAREST))
        else:
            # Predict initial mask for landmark alignment
            init_t = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
            with torch.no_grad():
                guide_mask = torch.argmax(model(init_t), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if args.occlude == "mask":
            img_np, _ = occ_aug.apply_face_mask(img_np, guide_mask)
            simulated_name = "Synthetic Face Mask"
        elif args.occlude == "sunglasses":
            img_np, _ = occ_aug.apply_sunglasses(img_np, guide_mask)
            simulated_name = "Synthetic Sunglasses"
        elif args.occlude == "heavy":
            m_img, m_mask = occ_aug.apply_face_mask(img_np, guide_mask)
            img_np, _ = occ_aug.apply_sunglasses(m_img, m_mask)
            simulated_name = "Heavy Occlusion (Mask + Sunglasses)"

    # Final forward pass
    tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Compute Occlusion Analytics
    occ_info = occ_mod.estimate_occlusion(pred_mask)

    # Display Report
    total_face = occ_info["total_face_pixels"]
    occ_pixels = occ_info["occluded_pixels"]
    occ_pct = occ_info["occlusion_percentage"]
    status = occ_info["viability_status"]
    is_rec = occ_info["is_recognizable"]
    reg = occ_info["regional_occlusion"]
    weights = occ_info["adaptive_weights"]

    rec_str = "YES - Suitable for Recognition" if is_rec else "NO - High Occlusion Risk"

    print("\n" + "=" * 65)
    print("  OCCLUSION-AWARE FACE SEGMENTATION & ESTIMATION REPORT")
    print("=" * 65)
    print(f"  Input Image          : {image_path}")
    print(f"  Simulated Occlusion  : {simulated_name}")
    print(f"  Total Face Pixels    : {total_face:,}")
    print(f"  Occluded Pixels      : {occ_pixels:,}")
    print(f"  Occlusion Percentage : {occ_pct:.2f}%")
    print(f"  Viability Status     : {status} (Threshold < 50.0%)")
    print(f"  Face Recognizable?   : {rec_str}")
    print("-" * 65)
    print("  Regional Occlusion Breakdown:")
    print(f"    - Upper Face (Eyes/Forehead) : {reg['upper']:6.2f}%")
    print(f"    - Mid Face   (Nose/Cheeks)   : {reg['mid']:6.2f}%")
    print(f"    - Lower Face (Mouth/Chin)    : {reg['lower']:6.2f}%")
    print("-" * 65)
    print("  Downstream Adaptive Region Weights (Sum = 1.000):")
    print(f"    - Upper Face Weight (w_upper) : {weights['upper']:.3f}")
    print(f"    - Mid Face Weight   (w_mid)   : {weights['mid']:.3f}")
    print(f"    - Lower Face Weight (w_lower) : {weights['lower']:.3f}")
    print("=" * 65)

    # Save visual diagnostic card
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_id = image_path.stem
    generate_diagnostic_card(img_np, pred_mask, occ_info, sample_id, out_path)
    print(f"  Diagnostic Card saved: {out_path.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()