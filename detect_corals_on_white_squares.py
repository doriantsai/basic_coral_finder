#!/usr/bin/env python3
"""
detect_corals_on_white_squares.py
Detect white square coral holders and the coral specimen within each holder.

Pipeline:
  Steps 1-10   Locate and grid-assign white square ROIs using the
               WhiteTabClassifier random forest.
  Steps 11-13  Within each ROI, segment the coral using the CoralClassifier
               random forest, apply morphological cleanup, and return the
               coral contour.

Outputs per image:
  <name>_annotated.jpg   original with ROI boxes (green) and coral outlines (orange)
  <name>_rois.csv        row,col,x,y,width,height in full-resolution pixels
  <name>_rois.json       same as JSON for ImageJ / downstream tools
  <name>_contours.csv    coral contour vertices: row,col,point_idx,x,y

Usage:
  python3 detect_corals_on_white_squares.py <image_or_dir> [output_dir]
"""

import cv2
import numpy as np
import csv
import json
import sys
from pathlib import Path

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False


# ── Tunable parameters ────────────────────────────────────────────────────────

PROCESS_SCALE   = 0.15    # down-scale for fast detection
HSV_V_MIN       = 185     # HSV Value lower bound (white fallback only)
HSV_S_MAX       = 65      # HSV Saturation upper bound (white fallback only)
MIN_BLOB_FRAC   = 0.002   # min blob area / image area
MAX_BLOB_FRAC   = 0.10    # max blob area / image area
MAX_BLOB_ASPECT = 2.0     # max bounding-box aspect ratio

# Step 3a-3b white mask morphology (at 15% scale)
DENOISE_KSIZE   = 5       # median blur — removes RF salt-and-pepper
CLOSE_KSIZE     = 15      # morphological close — seals white frame gaps

# Step 10 ROI border
ROI_BORDER_FRAC = 0.15    # fraction of blob size added each side

# Steps 11-12 coral mask morphology (at 15% scale, elliptical kernels)
CORAL_DENOISE_KSIZE = 3   # median blur on raw coral RF mask
CORAL_CLOSE_KSIZE   = 7   # elliptical close — rounds and fills coral shape
CORAL_OPEN_KSIZE    = 3   # elliptical open  — removes small noise patches

BOX_COLOR     = (0, 200, 50)    # BGR green  — ROI bounding boxes
CONTOUR_COLOR = (0, 165, 255)   # BGR orange — coral outline
TEXT_COLOR    = (0, 220, 255)   # BGR yellow — labels
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ── Model loading ─────────────────────────────────────────────────────────────

_MODEL_DIR = Path(__file__).parent / "train" / "models"

_white_model        = None
_white_model_loaded = False
_coral_model        = None
_coral_model_loaded = False


def _load_white_model():
    global _white_model, _white_model_loaded
    if _white_model_loaded:
        return _white_model
    _white_model_loaded = True
    if not _JOBLIB_OK:
        return None
    candidates = sorted(_MODEL_DIR.glob("*WhiteTabClassifier*.joblib"))
    if not candidates:
        return None
    path = candidates[-1]
    _white_model = joblib.load(path)
    print(f"  [Step 2]  White model: {path.name}")
    return _white_model


def _load_coral_model():
    global _coral_model, _coral_model_loaded
    if _coral_model_loaded:
        return _coral_model
    _coral_model_loaded = True
    if not _JOBLIB_OK:
        return None
    candidates = sorted(_MODEL_DIR.glob("*CoralClassifier*.joblib"))
    if not candidates:
        return None
    path = candidates[-1]
    _coral_model = joblib.load(path)
    print(f"  [Step 11] Coral model: {path.name}")
    return _coral_model


def _rf_features(img_bgr):
    """(H*W, 9) float32 — B G R H S V L a b, all in [0, 1]."""
    bgr = img_bgr.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 180.0
    hsv[:, 1:] /= 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    lab[:, 0] /= 100.0
    lab[:, 1:] = (lab[:, 1:] + 128.0) / 255.0
    return np.hstack([bgr, hsv, lab])


# ── Step 2: white pixel mask ──────────────────────────────────────────────────

def get_white_mask(small_bgr):
    """
    Binary mask of white-frame pixels (255) at 15% scale.
    Uses WhiteTabClassifier when available; falls back to HSV threshold.
    """
    model = _load_white_model()
    if model is not None:
        h, w   = small_bgr.shape[:2]
        feats  = _rf_features(small_bgr)
        labels = model.predict(feats).astype(np.uint8)
        return (labels.reshape(h, w) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, HSV_V_MIN), (180, HSV_S_MAX, 255))


# ── Steps 11-12: coral pixel mask per ROI ────────────────────────────────────

def get_coral_mask_for_roi(small_bgr, sx, sy, sw, sh):
    """
    Run the CoralClassifier on a small-image ROI crop and apply morphological
    cleanup to round the coral outline.

    sx, sy, sw, sh — ROI origin and size in 15%-scale (small) pixels.

    Returns (raw_mask, clean_mask) at crop resolution, or (None, None) when no
    CoralClassifier model is available.  raw_mask is the direct RF output;
    clean_mask is after median blur + elliptical close + elliptical open.
    """
    model = _load_coral_model()
    if model is None:
        return None, None

    sh_img, sw_img = small_bgr.shape[:2]
    sx  = max(0, sx);        sy  = max(0, sy)
    ex  = min(sw_img, sx + sw); ey  = min(sh_img, sy + sh)
    crop = small_bgr[sy:ey, sx:ex]
    if crop.size == 0:
        return None, None

    h, w   = crop.shape[:2]
    feats  = _rf_features(crop)
    labels = model.predict(feats).astype(np.uint8)
    raw    = (labels.reshape(h, w) * 255).astype(np.uint8)

    clean = cv2.medianBlur(raw, CORAL_DENOISE_KSIZE) if CORAL_DENOISE_KSIZE >= 3 else raw.copy()
    el_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (CORAL_CLOSE_KSIZE, CORAL_CLOSE_KSIZE))
    el_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (CORAL_OPEN_KSIZE,  CORAL_OPEN_KSIZE))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, el_close)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  el_open)
    return raw, clean


def _contour_from_mask(mask, offset_x, offset_y, mask_w, mask_h, img_shape):
    """
    Extract the largest contour from a binary small-scale mask, translate by
    (offset_x, offset_y) in small-scale pixels, and scale up to full-res coords.

    Returns (N, 2) int32 in full-image coordinates, or None.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    min_area = mask_w * mask_h * 0.05
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    eps = max(1.0, min(mask_w, mask_h) * 0.003)
    cnt = cv2.approxPolyDP(cnt, eps, closed=True)

    pts  = cnt.reshape(-1, 2) + np.array([[offset_x, offset_y]])
    H_img, W_img = img_shape[:2]
    inv  = 1.0 / PROCESS_SCALE
    full = (pts.astype(np.float32) * inv).astype(np.int32)
    full[:, 0] = np.clip(full[:, 0], 0, W_img - 1)
    full[:, 1] = np.clip(full[:, 1], 0, H_img - 1)
    return full


# ── Step 13: coral contour ────────────────────────────────────────────────────

def find_coral_contour(img_bgr, small_bgr, roi_x, roi_y, roi_w, roi_h):
    """
    Detect the coral outline within a white square ROI.

    When a CoralClassifier model is available: runs on the 15%-scale image,
    applies morphological cleanup (Steps 11-12), then finds the contour (Step 13).

    Falls back to Otsu threshold on full-res when no coral model is found
    (i.e. before the model has been trained).

    Returns (N, 2) int32 in full-image pixel coordinates, or None.
    """
    if _load_coral_model() is not None:
        sx = int(roi_x * PROCESS_SCALE)
        sy = int(roi_y * PROCESS_SCALE)
        sw = max(1, int(roi_w * PROCESS_SCALE))
        sh = max(1, int(roi_h * PROCESS_SCALE))
        _, clean = get_coral_mask_for_roi(small_bgr, sx, sy, sw, sh)
        if clean is None:
            return None
        return _contour_from_mask(clean, sx, sy, sw, sh, img_bgr.shape)

    # ── Otsu fallback on full resolution ─────────────────────────────────────
    H_img, W_img = img_bgr.shape[:2]
    x1 = max(0, roi_x);             y1 = max(0, roi_y)
    x2 = min(W_img, roi_x + roi_w); y2 = min(H_img, roi_y + roi_h)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    min_area = (x2 - x1) * (y2 - y1) * 0.05
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    eps = max(2.0, min(x2 - x1, y2 - y1) * 0.003)
    cnt = cv2.approxPolyDP(cnt, eps, closed=True)
    return cnt.reshape(-1, 2).astype(np.int32) + np.array([[x1, y1]], dtype=np.int32)


# ── Utilities ─────────────────────────────────────────────────────────────────

def cluster_1d(values, min_gap):
    """Split sorted values into groups wherever consecutive gap > min_gap."""
    if not values:
        return []
    s = sorted(values)
    groups = [[s[0]]]
    for v in s[1:]:
        if v - groups[-1][-1] > min_gap:
            groups.append([])
        groups[-1].append(v)
    return [int(np.median(g)) for g in groups]


# ── Core blob detection ───────────────────────────────────────────────────────

def _white_in_strip(white_msk, x1, y1, x2, y2):
    """Fraction of pixels in a rectangular strip that are white (0-1)."""
    sh, sw = white_msk.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(sw, x2); y2 = min(sh, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    strip = white_msk[y1:y2, x1:x2]
    return float(strip.sum()) / (255.0 * strip.size)


def holes_in_white(white_msk, img_area):
    """
    Find dark regions enclosed by white contours (RETR_CCOMP hierarchy).
    Requires ≥8% white on all four sides to reject open-edged blobs.
    """
    sh, sw = white_msk.shape
    cnts, hier = cv2.findContours(white_msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    blobs = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        a_frac = (cw * ch) / img_area
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if not (MIN_BLOB_FRAC < a_frac < MAX_BLOB_FRAC and aspect < MAX_BLOB_ASPECT):
            continue
        b = max(int(min(cw, ch) * 0.20), 5)
        sides = [
            _white_in_strip(white_msk, x - b, y,      x,       y + ch),
            _white_in_strip(white_msk, x + cw, y,      x+cw+b,  y + ch),
            _white_in_strip(white_msk, x,      y - b,  x + cw,  y),
            _white_in_strip(white_msk, x,      y + ch, x + cw,  y+ch+b),
        ]
        if min(sides) < 0.08:
            continue
        blobs.append((x, y, cw, ch))
    return blobs


def distinct_cols(blobs):
    """Count distinct column positions in a blob list."""
    if not blobs:
        return 0
    cx    = [x + cw // 2 for x, _, cw, _ in blobs]
    med_w = int(np.median([cw for _, _, cw, _ in blobs]))
    return len(cluster_1d(cx, max(int(med_w * 0.7), 20)))


def detect_blobs(small, white_msk):
    """
    Find coral/marker blobs via two strategies:
    1. Denoise + isotropic close + contour-hierarchy hole detection.
    2. Directional closing fallback when strategy 1 finds < 3 distinct columns.
    """
    sh, sw = small.shape[:2]
    img_area = sh * sw

    wm_clean = cv2.medianBlur(white_msk, DENOISE_KSIZE)
    m1 = cv2.morphologyEx(wm_clean, cv2.MORPH_CLOSE,
                          np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8))
    b1 = holes_in_white(m1, img_area)
    if distinct_cols(b1) >= 3:
        return b1

    best_blobs = b1
    for dv in [20, 30, 40, 50]:
        m2 = cv2.morphologyEx(wm_clean, cv2.MORPH_CLOSE, np.ones((dv, 4), np.uint8))
        m2 = cv2.morphologyEx(m2,       cv2.MORPH_CLOSE, np.ones((4, dv), np.uint8))
        b2 = holes_in_white(m2, img_area)
        if b2:
            areas  = [cw * ch for _, _, cw, ch in b2]
            med_a  = float(np.median(areas))
            if max(areas) > med_a * 4:
                continue
        if distinct_cols(b2) > distinct_cols(best_blobs):
            best_blobs = b2
        if distinct_cols(best_blobs) >= 3:
            break

    return best_blobs


# ── Grid fitting ──────────────────────────────────────────────────────────────

def _prune_edge_centers(centers):
    """
    Remove calibration-disc columns at left/right edges.
    Compares each edge gap to the inner-spacings median; removes if < 0.93×.
    Requires ≥4 centers.
    """
    if len(centers) < 4:
        return centers
    spacings  = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    inner     = spacings[1:-1]
    inner_med = float(np.median(inner))
    threshold = 0.93 * inner_med

    pruned = list(centers)
    if spacings[0] < threshold:
        pruned = pruned[1:]
    if spacings[-1] < threshold:
        pruned = pruned[:-1]
    return pruned


def blobs_to_grid_rois(blobs, sh, sw, scale):
    """
    Cluster blobs into a row×column grid and expand each cell by ROI_BORDER_FRAC.
    Returns [(row, col, x, y, w, h)] in full-resolution pixel coordinates.
    """
    if not blobs:
        return []

    med_w   = int(np.median([cw for _, _, cw, _ in blobs]))
    med_h   = int(np.median([ch for _, _, _, ch in blobs]))
    row_gap = max(int(med_h * 0.6), 18)
    col_gap = max(int(med_w * 0.6), 18)

    cy_all      = sorted([y + ch // 2 for _, y, _, ch in blobs])
    cx_all      = sorted([x + cw // 2 for x, _, cw, _ in blobs])
    row_centers = cluster_1d(cy_all, row_gap)
    col_centers = cluster_1d(cx_all, col_gap)
    col_centers = _prune_edge_centers(col_centers)

    blobs = [b for b in blobs
             if any(abs(b[0] + b[2]//2 - cc) < col_gap * 2 for cc in col_centers)]
    if not blobs:
        return []

    def nearest(val, centers):
        return int(np.argmin([abs(val - c) for c in centers]))

    cell_blobs: dict = {}
    for (x, y, cw, ch) in blobs:
        r = nearest(y + ch // 2, row_centers)
        c = nearest(x + cw // 2, col_centers)
        cell_blobs.setdefault((r, c), []).append((x, y, cw, ch))

    border_y = int(med_h * ROI_BORDER_FRAC)
    border_x = int(med_w * ROI_BORDER_FRAC)
    inv = 1.0 / scale

    result = []
    for (r, c), cell in sorted(cell_blobs.items()):
        x1 = min(x        for x, y, cw, ch in cell)
        y1 = min(y        for x, y, cw, ch in cell)
        x2 = max(x + cw   for x, y, cw, ch in cell)
        y2 = max(y + ch   for x, y, cw, ch in cell)
        rx1 = max(0,  x1 - border_x);  ry1 = max(0,  y1 - border_y)
        rx2 = min(sw, x2 + border_x);  ry2 = min(sh, y2 + border_y)
        fx  = int(rx1 * inv);  fy  = int(ry1 * inv)
        fw  = int((rx2 - rx1) * inv);  fh  = int((ry2 - ry1) * inv)
        result.append((r, c, fx, fy, fw, fh))

    return result


# ── Per-image pipeline ────────────────────────────────────────────────────────

def process_image(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return []

    H, W  = img.shape[:2]
    scale = PROCESS_SCALE
    small = cv2.resize(img, (int(W * scale), int(H * scale)))
    sh, sw = small.shape[:2]

    wm    = get_white_mask(small)                        # Step 2
    blobs = detect_blobs(small, wm)                      # Steps 3-6
    rois  = blobs_to_grid_rois(blobs, sh, sw, scale)    # Steps 7-10

    # ── Steps 11-13: coral contours ───────────────────────────────────────────
    contours = []
    for (row, col, x, y, w, h) in rois:
        pts = find_coral_contour(img, small, x, y, w, h)
        contours.append((row, col, pts))

    stem = Path(img_path).stem

    # ── Annotated image ───────────────────────────────────────────────────────
    ann        = img.copy()
    thick      = max(2, W // 1400)
    font_scale = max(0.5, W / 5000)

    for (row, col, x, y, w, h) in rois:
        cv2.rectangle(ann, (x, y), (x + w, y + h), BOX_COLOR, thick)
        label = f"R{row}C{col}"
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thick)
        cv2.rectangle(ann, (x, y), (x + tw + 6, y + th + 8), (30, 30, 30), -1)
        cv2.putText(ann, label, (x + 3, y + th + 4),
                    FONT, font_scale, TEXT_COLOR, thick, cv2.LINE_AA)
        coord = f"({x},{y})"
        cv2.putText(ann, coord, (x + 3, y + h - 5),
                    FONT, font_scale * 0.65, BOX_COLOR, max(1, thick - 1), cv2.LINE_AA)

    for (_row, _col, pts) in contours:
        if pts is not None:
            cv2.polylines(ann, [pts.reshape(-1, 1, 2)], True, CONTOUR_COLOR, thick)

    ann_path = Path(output_dir) / f"{stem}_annotated.jpg"
    cv2.imwrite(str(ann_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── ROI CSV ───────────────────────────────────────────────────────────────
    csv_path = Path(output_dir) / f"{stem}_rois.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "x", "y", "width", "height",
                         "center_x", "center_y", "image_width", "image_height"])
        for (row, col, x, y, w, h) in rois:
            writer.writerow([row, col, x, y, w, h,
                             x + w // 2, y + h // 2, W, H])

    # ── ROI JSON ──────────────────────────────────────────────────────────────
    json_path = Path(output_dir) / f"{stem}_rois.json"
    roi_list = [{"row": r, "col": c, "x": x, "y": y, "width": w, "height": h}
                for (r, c, x, y, w, h) in rois]
    with open(json_path, "w") as f:
        json.dump({"image": str(img_path), "image_width": W, "image_height": H,
                   "squares_detected": len(rois), "rois": roi_list}, f, indent=2)

    # ── Contours CSV ──────────────────────────────────────────────────────────
    # One row per polygon vertex.  Load in ImageJ with Table.open(); group by
    # (row, col), then: makeSelection("polygon", xArray, yArray); roiManager("add");
    contours_path = Path(output_dir) / f"{stem}_contours.csv"
    with open(contours_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "point_idx", "x", "y"])
        for (row, col, pts) in contours:
            if pts is not None:
                for idx, (px, py) in enumerate(pts):
                    writer.writerow([row, col, idx, int(px), int(py)])

    n_blobs = len(blobs)
    print(f"  {stem}: {len(rois)} squares detected ({n_blobs} inner blobs found)")
    print(f"    -> {ann_path.name}  |  {csv_path.name}  |  {contours_path.name}")
    return rois


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target     = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else target.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    if target.is_dir():
        paths = sorted(p for p in target.iterdir()
                       if p.suffix.lower() in image_exts)
    elif target.is_file():
        paths = [target]
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    if not paths:
        print("No images found.")
        sys.exit(1)

    print(f"Processing {len(paths)} image(s) -> {output_dir}\n")
    for p in paths:
        print(f"[{p.name}]")
        process_image(p, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
