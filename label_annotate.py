#!/usr/bin/env python3
"""
label_annotate.py
Prepare and visualise bounding-box training labels for the white-square
random forest classifier.

─── Workflow ────────────────────────────────────────────────────────────────

STEP 1  Run once to set up training images and label templates:

    python3 label_annotate.py

    * Randomly samples 10 images from images/ into train/images/.
    * For each image, writes a JSON template to train/labels/<stem>.json.
    * If a matching output/<stem>_rois.json exists (from detect_corals_on_white_squares.py),
      its detected squares are pre-loaded as "white" ROIs so you only need to
      add the "other" samples.

STEP 2  Edit train/labels/<stem>.json in any text editor.

    Each entry in "rois" needs a "label" field:
      "white" — box covers white square frame material only
      "other" — box covers anything else (coral, background, calibration disc)

    You can adjust, delete, or add ROIs freely.  The "row" / "col" fields
    from the detector output are kept for reference but are not required.

STEP 3  Run again to produce annotated images:

    python3 label_annotate.py

    Annotated images are written to train/annotated/<stem>_labeled.jpg.
    Re-run any time you edit a JSON file to refresh the preview.

─── JSON format ─────────────────────────────────────────────────────────────

Same structure as detect_corals_on_white_squares.py output, with "label" added per ROI:

{
  "image": "train/images/DSC00150.JPG",
  "image_width": 7008,
  "image_height": 4672,
  "rois": [
    {"label": "white", "row": 0, "col": 0, "x": 1666, "y": 853,  "width": 1013, "height": 773},
    {"label": "white", "row": 0, "col": 1, "x": 2653, "y": 873,  "width": 1006, "height": 733},
    {"label": "other",                     "x": 100,  "y": 100,  "width": 400,  "height": 400}
  ]
}
"""

import cv2
import json
import random
import shutil
import sys
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_N    = 10
IMG_DIR    = Path("images")
OUT_DIR    = Path("output")
TRAIN_DIR  = Path("train")
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SAMPLE_EXTS     = {".jpg", ".jpeg"}   # only camera JPEGs for training

COLORS = {
    "white": (0, 200, 50),    # green
    "other": (0, 60, 220),    # red
    "coral": (0, 165, 255),   # orange
}
FILL_ALPHA = 0.22            # opacity of the shaded fill inside each box
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── Step 1 helpers ────────────────────────────────────────────────────────────

def sample_images(src, dst, n):
    """Copy n randomly chosen images from src to dst."""
    candidates = sorted(p for p in src.iterdir() if p.suffix.lower() in SAMPLE_EXTS)
    if not candidates:
        print(f"  [warn] no images found in {src}")
        return []
    chosen = random.sample(candidates, min(n, len(candidates)))
    copied = []
    for p in sorted(chosen):
        target = dst / p.name
        if not target.exists():
            shutil.copy2(p, target)
            print(f"  copied  {p.name}")
        else:
            print(f"  exists  {p.name}")
        copied.append(target)
    return copied


def make_template(img_path, label_dir, output_json=None):
    """
    Write train/labels/<stem>.json.

    If output_json points to an existing detect_white_squares output file,
    its ROIs are pre-populated with label "white" (they are the detected
    white squares).  Otherwise an empty template is written.
    """
    label_path = label_dir / f"{img_path.stem}.json"
    if label_path.exists():
        return  # don't overwrite user edits

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return
    H, W = img.shape[:2]

    rois = []
    if output_json and output_json.exists():
        with open(output_json) as f:
            det = json.load(f)
        for r in det.get("rois", []):
            roi = {"label": "white"}
            for k in ("row", "col", "x", "y", "width", "height"):
                if k in r:
                    roi[k] = r[k]
            rois.append(roi)
        print(f"  template {img_path.stem}.json  "
              f"({len(rois)} detected squares pre-loaded as 'white')")
    else:
        print(f"  template {img_path.stem}.json  (empty — add ROIs manually)")

    doc = {
        "image": str(img_path),
        "image_width": W,
        "image_height": H,
        "rois": rois
    }
    with open(label_path, "w") as f:
        json.dump(doc, f, indent=2)


# ── Step 3 helpers ────────────────────────────────────────────────────────────

def _resolve_image(label_path, stored_path):
    """Find the actual image file given the path stored in the JSON."""
    p = Path(stored_path)
    if p.exists():
        return p
    # Try relative to the label file's grandparent (the train/ root)
    alt = label_path.parent.parent / "images" / p.name
    if alt.exists():
        return alt
    return None


def annotate(label_path, out_dir):
    """Draw labeled bounding boxes on the image and write to out_dir."""
    with open(label_path) as f:
        data = json.load(f)

    img_path = _resolve_image(label_path, data.get("image", ""))
    if img_path is None:
        print(f"  [skip] image not found for {label_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return

    H, W   = img.shape[:2]
    rois   = [r for r in data.get("rois", []) if "label" in r]

    if not rois:
        print(f"  {label_path.stem}: no labeled ROIs yet — skipping")
        return

    ann    = img.copy()
    thick  = max(3, W // 1400)
    fscale = max(0.7, W / 5000)

    for roi in rois:
        label = roi.get("label", "other")
        x  = int(roi["x"]);  y  = int(roi["y"])
        w  = int(roi["width"]); h = int(roi["height"])
        x2 = min(x + w, W);  y2 = min(y + h, H)
        color = COLORS.get(label, (160, 160, 160))

        # Shaded fill
        overlay = ann.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
        cv2.addWeighted(overlay, FILL_ALPHA, ann, 1 - FILL_ALPHA, 0, ann)

        # Solid border
        cv2.rectangle(ann, (x, y), (x2, y2), color, thick)

        # Label tag above the box
        tag_parts = [label]
        if "row" in roi and "col" in roi:
            tag_parts.append(f"R{roi['row']}C{roi['col']}")
        tag_parts.append(f"{w}x{h}px")
        tag = "  ".join(tag_parts)

        (tw, th), _ = cv2.getTextSize(tag, FONT, fscale, thick)
        tag_y0 = max(0, y - th - 10)
        cv2.rectangle(ann, (x, tag_y0), (x + tw + 8, y), (20, 20, 20), -1)
        cv2.putText(ann, tag, (x + 4, y - 6),
                    FONT, fscale, color, thick, cv2.LINE_AA)

    # Summary bar at the top
    counts  = {lbl: sum(1 for r in rois if r.get("label") == lbl)
               for lbl in ("white", "coral", "other")}
    bar_h   = max(48, H // 50)
    bar     = np.zeros((bar_h, W, 3), dtype=np.uint8)
    summary = (f"{img_path.stem}   |   "
               + "    ".join(f"{lbl}: {n}" for lbl, n in counts.items() if n > 0))
    cv2.putText(bar, summary, (12, bar_h - 12),
                FONT, max(0.8, W / 5500), (0, 220, 255), 2, cv2.LINE_AA)
    ann = np.vstack([bar, ann])

    out_path = out_dir / f"{label_path.stem}_labeled.jpg"
    cv2.imwrite(str(out_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
    summary_parts = [f"{n} {lbl}" for lbl, n in counts.items() if n > 0]
    print(f"  {label_path.stem}: {', '.join(summary_parts)}  ->  {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_DIR
    img_dir    = train_dir / "images"
    label_dir  = train_dir / "labels"
    annot_dir  = train_dir / "annotated"

    for d in (img_dir, label_dir, annot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: sample images if train/images/ is still empty ─────────────────
    existing = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not existing:
        if not IMG_DIR.exists():
            print(f"Source image directory '{IMG_DIR}' not found.")
            sys.exit(1)
        print(f"Sampling {TRAIN_N} images from {IMG_DIR}/ -> {img_dir}/")
        existing = sample_images(IMG_DIR, img_dir, TRAIN_N)
        print()

    # ── Step 2: create JSON templates for any image without one ───────────────
    new_templates = 0
    for img_path in sorted(existing):
        lp = label_dir / f"{img_path.stem}.json"
        if not lp.exists():
            det_json = OUT_DIR / f"{img_path.stem}_rois.json"
            make_template(img_path, label_dir, det_json)
            new_templates += 1

    if new_templates:
        print(f"\n{new_templates} template(s) written to {label_dir}/")
        print("Edit them to add / adjust 'label' fields, then run again.\n")

    # ── Step 3: annotate any labeled JSON ─────────────────────────────────────
    label_files = sorted(label_dir.glob("*.json"))
    annotated   = 0
    for lp in label_files:
        try:
            with open(lp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [error] {lp.name}: {e}")
            continue

        labeled = [r for r in data.get("rois", []) if "label" in r]
        if labeled:
            annotate(lp, annot_dir)
            annotated += 1

    if annotated:
        print(f"\n{annotated} annotated image(s) written to {annot_dir}/")
    elif not new_templates:
        print("No labeled ROIs found.  Add 'label' fields to the JSON files and run again.")


if __name__ == "__main__":
    main()
