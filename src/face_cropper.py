"""
face_cropper.py - Automatic Face Detection & High-Resolution Pre-Cropping Pipeline.

Solves the scale & resolution problem for real-world images (selfies, outdoor photos, CCTV):
  1. Detects face bounding box & landmarks via OpenCV YuNet (assets/yunet.onnx).
  2. Centers and squares the facial bounding box with anatomical margins (hair & chin padding).
  3. Yields a standardized 256x256 high-resolution face canvas matching CelebAMask-HQ proportions.
  4. Preserves transform metadata to map masks back to original coordinates.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = PROJECT_ROOT / "assets" / "yunet.onnx"


class FacePreCropper:
    def __init__(self, model_path: Optional[Path] = None, score_thresh: float = 0.5):
        self.model_path = Path(model_path or YUNET_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YuNet model weights not found at: {self.model_path}")

        self.score_thresh = score_thresh
        self._detector = None
        self._last_input_size = None

    def _get_detector(self, width: int, height: int):
        if self._detector is None or self._last_input_size != (width, height):
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(width, height),
                score_threshold=self.score_thresh,
                nms_threshold=0.3,
                top_k=5
            )
            self._last_input_size = (width, height)
        return self._detector

    def detect_main_face(self, image_np: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects faces in an RGB or BGR numpy image.
        Returns the primary (largest) face detection dict or None.
        """
        h, w = image_np.shape[:2]
        if image_np.ndim == 2:
            img_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 3:
            # Assume RGB input from PIL
            img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image_np[:, :, :3]

        detector = self._get_detector(w, h)
        _, faces = detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            return None

        # Filter and select largest face by area
        best_face = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)[0]
        fx, fy, fw, fh = best_face[0:4].astype(int)
        conf = float(best_face[-1])

        # Extract landmarks: right_eye, left_eye, nose_tip, mouth_right, mouth_left
        landmarks = best_face[4:14].reshape((5, 2))

        return {
            "bbox": (fx, fy, fw, fh),
            "confidence": conf,
            "landmarks": landmarks,
        }

    def crop_face(
        self,
        image_pil: Image.Image,
        target_size: Tuple[int, int] = (256, 256),
        padding_factor: float = 1.45,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Detects and extracts a standardized face crop from an arbitrary resolution PIL image.
        If no face is detected or image is already a close-up portrait, returns the resized original.
        """
        orig_w, orig_h = image_pil.size
        img_np = np.array(image_pil)

        det = self.detect_main_face(img_np)
        if det is None:
            # Fallback: standard center-fit resize
            resized = image_pil.resize(target_size, Image.Resampling.BILINEAR)
            meta = {
                "detected": False,
                "crop_box": (0, 0, orig_w, orig_h),
                "original_size": (orig_w, orig_h),
                "scale": (target_size[0] / orig_w, target_size[1] / orig_h),
            }
            return resized, meta

        fx, fy, fw, fh = det["bbox"]

        # Check if image is already a tight portrait (face fills > 55% of canvas)
        face_area_ratio = (fw * fh) / (orig_w * orig_h)
        if face_area_ratio > 0.55:
            resized = image_pil.resize(target_size, Image.Resampling.BILINEAR)
            meta = {
                "detected": True,
                "already_tight": True,
                "crop_box": (0, 0, orig_w, orig_h),
                "original_size": (orig_w, orig_h),
                "confidence": det["confidence"],
            }
            return resized, meta

        # Compute smart square crop box around the head
        cx = fx + fw // 2
        cy = fy + int(fh * 0.52) # slight downward anchor to ensure full chin/jawline coverage
        crop_extent = int(max(fw, fh) * padding_factor)
        half = crop_extent // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - int(half * 1.05)) # shift slightly up to cover crown hair
        x2 = min(orig_w, x1 + crop_extent)
        y2 = min(orig_h, y1 + crop_extent)

        # Make sure bounding box is square
        bw = x2 - x1
        bh = y2 - y1
        side = min(bw, bh)
        x2 = x1 + side
        y2 = y1 + side

        face_crop = image_pil.crop((x1, y1, x2, y2))
        resized_crop = face_crop.resize(target_size, Image.Resampling.BILINEAR)

        meta = {
            "detected": True,
            "already_tight": False,
            "crop_box": (x1, y1, x2, y2),
            "original_size": (orig_w, orig_h),
            "confidence": det["confidence"],
            "landmarks": det["landmarks"],
        }
        return resized_crop, meta
