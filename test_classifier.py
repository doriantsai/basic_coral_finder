#!/usr/bin/env python3
"""
test_classifier.py
Apply a trained WhiteTabClassifier to images and produce pixel-level
white/other masks.

Usage:
    python3 test_classifier.py [model_path] [image_dir] [output_dir]

Defaults:
    model_path  — most recent .joblib in train/models/
    image_dir   — train/images/
    output_dir  — train/output_masks/

Outputs per image  (written to output_dir):
    <stem>_mask.png      binary mask  (255 = white,  0 = other)
    <stem>_overlay.jpg   original image with white mask highlighted in green
    <stem>_compare.jpg   side-by-side: original | mask | overlay
"""

import cv2
import sys
import numpy as np
from pathlib import Path

import joblib

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DIR      = Path("train")
INFER_SCALE    = 0.15   # same down-scale used by detect_white_squares.py
OVERLAY_COLOR  = (0, 220, 80)   # BGR green for "white" pixels
OVERLAY_ALPHA  = 0.45
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── Feature extraction (must match train_classifier.py exactly) ───────────────

def extract_features(img_bgr):
    """(N, 9) float32: B G R  H S V  L a b — all normalised to [0, 1]."""
    bgr = img_bgr.reshape(-1, 3).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 180.0
    hsv[:, 1:] /= 255.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    lab[:, 0] /= 100.0
    lab[:, 1:] = (lab[:, 1:] + 128) / 255.0

    return np.hstack([bgr, hsv, lab])


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_mask(clf, img_bgr, scale=INFER_SCALE):
    """
    Classify every pixel of img_bgr as white (1) or other (0).

    Inference runs at `scale` fraction of full resolution for speed, then
    the binary mask is upsampled back to full resolution with nearest-neighbour
    interpolation (preserving crisp edges without blurring the class boundary).

    Returns a uint8 mask of the same H×W as img_bgr (255=white, 0=other).
    """
    H, W = img_bgr.shape[:2]
    small  = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    sh, sw = small.shape[:2]

    feats  = extract_features(small)          # (sh*sw, 9)
    labels = clf.predict(feats).astype(np.uint8)   # 0 or 1 per pixel
    mask_s = (labels.reshape(sh, sw) * 255).astype(np.uint8)

    # Upsample to original resolution
    mask = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def make_overlay(img_bgr, mask, color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA):
    """Return img_bgr with `color` blended over pixels where mask == 255."""
    overlay = img_bgr.copy()
    overlay[mask == 255] = color
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def make_compare(img_bgr, mask, overlay, max_w=2400):
    """
    Three-panel side-by-side: original | binary mask | overlay.
    Scaled so the total width fits within max_w pixels.
    """
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_w / (W * 3))
    tw, th = int(W * scale), int(H * scale)

    panels = [
        cv2.resize(img_bgr, (tw, th)),
        cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (tw, th)),
        cv2.resize(overlay, (tw, th)),
    ]
    sep = np.full((th, 4, 3), 60, dtype=np.uint8)

    row = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

    # Panel labels
    for i, lbl in enumerate(["original", "mask (white=255)", "overlay"]):
        cx = i * (tw + 4) + 6
        cv2.putText(row, lbl, (cx, th - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_model(model_dir):
    models = sorted(model_dir.glob("*.joblib"))
    if not models:
        return None
    return models[-1]   # alphabetical = chronological (YYYYMMDD prefix)


def main():
    # Parse arguments with sensible defaults
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    img_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_DIR / "images"
    out_dir    = Path(sys.argv[3]) if len(sys.argv) > 3 else TRAIN_DIR / "output_masks"

    if model_path is None:
        model_path = find_latest_model(TRAIN_DIR / "models")
    if model_path is None or not model_path.exists():
        print("No trained model found.  Run train_classifier.py first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model:  {model_path.name}")
    clf = joblib.load(model_path)

    # ── Process each image ─────────────────────────────────────────────────────
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"No images found in {img_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) at {INFER_SCALE*100:.0f}% scale "
          f"-> {out_dir}/\n")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue

        H, W = img.shape[:2]
        mask    = predict_mask(clf, img)
        overlay = make_overlay(img, mask)
        compare = make_compare(img, mask, overlay)

        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"),    mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out_dir / f"{stem}_compare.jpg"), compare,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        n_white = int((mask == 255).sum())
        pct     = 100.0 * n_white / (H * W)
        print(f"  {img_path.name}:  {n_white:,} / {H*W:,} px white  ({pct:.1f}%)")

    print(f"\nDone.  Results in {out_dir}/")


if __name__ == "__main__":
    main()
