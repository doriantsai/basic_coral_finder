#!/usr/bin/env python3
"""
detect_corals_on_white_squares.py
Detect white square coral holders, the coral specimen within each holder,
and the 8 calibration circles (4 per side panel) used for reflectivity
calibration.

Pipeline:
  Steps 1-10   Locate and grid-assign white square ROIs using the
               WhiteTabClassifier random forest.
  Steps 11-13  Within each ROI, segment the coral using the CoralClassifier
               random forest, apply morphological cleanup, and return the
               coral contour.
  Step 14      Detect the 4 calibration circles on the left side panel and
               the 4 on the right side panel using HoughCircles on a
               CLAHE-enhanced greyscale strip.

Outputs per image:
  <name>_annotated.jpg            original with ROI boxes (green), coral outlines
                                  (orange), and calibration circles (magenta)
  <name>_rois.csv                 row,col,x,y,width,height in full-resolution pixels
  <name>_rois.json                same as JSON for ImageJ / downstream tools
  <name>_contours.csv             coral contour vertices: row,col,point_idx,x,y
  <name>_calibration_rois.csv     side,idx,x,y,width,height,center_x,center_y,radius
  <name>_calibration_rois.json    same as JSON for ImageJ / downstream tools

Usage:
  python3 detect_corals_on_white_squares.py <image_or_dir> [output_dir]
"""

import cv2
import numpy as np
import csv
import json
import sys
import time
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

# Step 14 calibration circle detection
# Search zone per side: image x in [CALIB_X_LO, CALIB_X_HI] (left)
#                                 and [1-CALIB_X_HI, 1-CALIB_X_LO] (right)
# LO skips corner-bracket structures near the image edge.
# HI covers the full grey panel width; x_spread scoring keeps the tight column.
CALIB_X_LO_FRAC    = 0.010   # inner search bound (fraction of small-image width)
CALIB_X_HI_FRAC    = 0.250   # outer search bound (fraction of small-image width)
CALIB_Y_MIN_FRAC   = 0.01    # skip top fraction of image height (frame bolts)
CALIB_Y_MAX_FRAC   = 0.99    # skip bottom fraction of image height
CALIB_R_MIN_FRAC   = 0.020   # min calibration-circle radius / small-image width
CALIB_R_MAX_FRAC   = 0.075   # max calibration-circle radius / small-image width (~1000px dia)
CALIB_CIRCLE_COLOR = (255, 50, 200)  # BGR magenta — calibration circles annotation colour


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


# ── Step 14: calibration circle detection ────────────────────────────────────

def _deduplicate_circles(circles, min_dist):
    """Remove near-duplicate circle centres within min_dist of each other."""
    unique = []
    for c in circles:
        if not any((c[0]-u[0])**2 + (c[1]-u[1])**2 < min_dist**2 for u in unique):
            unique.append(c)
    return unique


def _best_n_circles(candidates, n):
    """
    Pick the n circles from candidates that best form a vertical calibration
    column, scored by vertical spacing evenness + radius consistency +
    horizontal x-alignment.  Uses brute-force combination search (fast for
    small candidate sets).
    """
    import itertools
    if len(candidates) <= n:
        return sorted(candidates, key=lambda t: t[1])

    mean_r_all = float(np.mean([c[2] for c in candidates])) + 1e-6
    best_score = -1.0
    best_group = sorted(candidates, key=lambda t: t[1])[:n]

    for grp in itertools.combinations(candidates, n):
        grp = sorted(grp, key=lambda t: t[1])
        ys  = [c[1] for c in grp]
        rs  = [c[2] for c in grp]
        xs  = [c[0] for c in grp]

        spacings = [ys[i+1] - ys[i] for i in range(n - 1)]
        if min(spacings) < max(rs) * 0.4:   # circles must not nearly overlap
            continue

        sp_cv    = float(np.std(spacings)) / (float(np.mean(spacings)) + 1e-6)
        r_cv     = float(np.std(rs)) / (float(np.mean(rs)) + 1e-6)
        x_spread = float(np.std(xs)) / mean_r_all

        score = 1.0 / (1.0 + sp_cv + r_cv * 1.5 + x_spread * 0.5)
        if score > best_score:
            best_score = score
            best_group = list(grp)

    return best_group


def _extrapolate_column(anchors, n, strip_h, strip_w, blurred, raw_blr, mean_r):
    """
    Complete a partial column of n evenly-spaced same-size circles from k < n anchors.

    Tries all C(n, k) slot assignments for the k anchors, selects the assignment
    whose implied spacing is most consistent and whose predicted positions fit best
    within the strip.  For each unfilled slot whose predicted y is within the strip,
    runs a tight-radius local HoughCircles search in a small window; if that fails,
    inserts the predicted centre with the consensus radius as a best-guess fallback.
    """
    import itertools as _it

    if len(anchors) < 2:
        return list(anchors)

    anchors_s = sorted(anchors, key=lambda c: c[1])
    k        = len(anchors_s)
    ys       = [c[1] for c in anchors_s]
    xs       = [c[0] for c in anchors_s]
    rs       = [c[2] for c in anchors_s]
    mean_x   = float(np.mean(xs))
    use_r    = float(np.mean(rs)) if rs else mean_r

    # Radius window for local searches
    t_lo = max(2, int(use_r * 0.78))
    t_hi = max(t_lo + 2, int(use_r * 1.28))

    # --- Try every possible slot assignment for the k anchors within n slots ---
    best_score = -1.0
    best_y0    = None
    best_sp    = None
    best_slots = None

    for slot_combo in _it.combinations(range(n), k):
        # Infer spacing from each consecutive anchor pair in the slot assignment
        sp_list = []
        for i in range(k - 1):
            ds = slot_combo[i + 1] - slot_combo[i]
            dy = ys[i + 1] - ys[i]
            sp_list.append(dy / ds)

        sp  = float(np.mean(sp_list)) if sp_list else use_r * 2.8
        sp  = max(sp, use_r * 1.5)    # minimum physical separation
        sp_cv = (float(np.std(sp_list)) / (sp + 1e-6)
                 if len(sp_list) > 1 else 0.0)

        y0   = ys[0] - slot_combo[0] * sp
        ypred = [y0 + i * sp for i in range(n)]

        # Count predicted positions inside the strip
        in_strip = sum(1 for yp in ypred if 0 <= yp <= strip_h)
        # Penalise positions that overshoot the strip edges
        overshoot = (max(0.0, -y0) + max(0.0, ypred[-1] - strip_h)) / (sp + 1e-6)

        score = in_strip / n - sp_cv * 0.5 - overshoot * 0.15
        if score > best_score:
            best_score = score
            best_y0    = y0
            best_sp    = sp
            best_slots = slot_combo

    if best_y0 is None:
        return list(anchors_s)

    y_pred = [best_y0 + i * best_sp for i in range(n)]

    # Assign anchors to their designated slots
    result = [None] * n
    for i, slot in enumerate(best_slots):
        result[slot] = anchors_s[i]

    # Fill empty slots with local Hough search → predicted-position fallback
    for slot in range(n):
        if result[slot] is not None:
            continue
        yp = float(y_pred[slot])
        xp = mean_x

        # Skip slot if predicted centre is outside the strip
        if yp < -best_sp * 0.4 or yp > strip_h + best_sp * 0.4:
            continue

        half_y = max(int(best_sp * 0.38), int(use_r * 0.9))
        y_lo = max(0, int(yp - half_y))
        y_hi = min(strip_h, int(yp + half_y))

        found = None
        if y_hi - y_lo >= 6:
            # Use full strip width so crop_r_hi is not limited by a narrow x-window
            crop_r_hi = min(t_hi, (y_hi - y_lo) // 2)
            crop_r_hi = max(crop_r_hi, t_lo + 1)
            for src in (blurred, raw_blr):
                crop = src[y_lo:y_hi, :]
                for p1, p2 in [(40, 20), (30, 15), (20, 10)]:
                    rc = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.0,
                        minDist=max(t_lo, 2), param1=p1, param2=p2,
                        minRadius=t_lo, maxRadius=crop_r_hi)
                    if rc is not None:
                        # Pick candidate closest to predicted centre
                        bc = min(rc[0],
                                 key=lambda c: (c[0] - xp)**2 + (c[1] + y_lo - yp)**2)
                        found = (int(round(bc[0])),
                                 int(round(bc[1])) + y_lo,
                                 int(round(bc[2])))
                        break
                if found:
                    break

        # Fallback: predicted centre with consensus radius
        if found is None:
            found = (int(round(xp)), int(round(max(0, min(strip_h, yp)))), int(round(use_r)))

        result[slot] = found

    return [c for c in result if c is not None]


def _detect_circles_in_strip(strip_gray, r_min, r_max, n_expected=4):
    """
    Detect n_expected circles forming a vertical column within strip_gray.

    Exploits two physical constraints of the calibration target:
      - All circles have the same radius (within ~±20%).
      - Circles are evenly spaced in a vertical column (same x position).

    Pipeline:
      1. Initial Hough sweep on disc-anomaly (tophat+blackhat) and CLAHE-raw
         images with progressively relaxed param2 (30→22→16→12).
      2. Best-n selector: score all C(k,n) subsets by spacing evenness, radius
         consistency, and column alignment (x-spread).  No radius pre-filter —
         pre-filtering by a biased median can eject valid small-radius circles.
      3. Post-selection same-size check: drop any circle >40% off the set mean
         (applied after best-n, when the set mean is reliable).
      4. Tight-radius re-scan: using the consensus radius from step 3, sweep
         Hough again at the known disc size to recover missed same-size discs.
      5. Extrapolation (_extrapolate_column): if < n circles remain, fit the
         column geometry (spacing, x) from anchors and search locally at each
         missing slot; insert predicted centre if local search fails.
    """
    strip_h, strip_w = strip_gray.shape[:2]
    if strip_h < r_min * 2 or strip_w < r_min:
        return []

    # Pre-process both sources (same as before)
    r_est    = (r_min + r_max) // 2
    ksize    = max(3, (r_est // 2) | 1)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat   = cv2.morphologyEx(strip_gray, cv2.MORPH_TOPHAT,   kernel)
    blackhat = cv2.morphologyEx(strip_gray, cv2.MORPH_BLACKHAT, kernel)
    disc_sig = cv2.add(tophat, blackhat)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    blurred  = cv2.GaussianBlur(clahe.apply(disc_sig), (9, 9), 2)
    clahe2   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    raw_blr  = cv2.GaussianBlur(clahe2.apply(strip_gray), (9, 9), 2)

    dedup_dist = max(2, r_min // 3)

    def _hough_sweep(r_lo, r_hi):
        out = []
        for src in (blurred, raw_blr):
            for p1, p2 in [(50, 30), (40, 22), (30, 16), (20, 12)]:
                rc = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=1.0,
                    minDist=max(r_lo, 4), param1=p1, param2=p2,
                    minRadius=max(r_lo, 1), maxRadius=r_hi)
                if rc is not None:
                    for c in rc[0]:
                        out.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
            out = _deduplicate_circles(out, dedup_dist)
            if len(out) >= n_expected:
                break
        return out

    # Phase 1: Wide-radius Hough sweep
    cands = _hough_sweep(r_min, r_max)

    # Canny-edge circularity fallback when Hough finds nothing
    if not cands:
        edges = cv2.Canny(raw_blr, 20, 60)
        kf    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kf)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            circ = 4.0 * np.pi * area / (peri * peri)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            r = int(round(r))
            if r_min <= r <= r_max and circ > 0.55:
                cands.append((int(round(cx)), int(round(cy)), r))
        cands = _deduplicate_circles(cands, dedup_dist)

    if not cands:
        return []

    # Phase 2: Select best n by column-geometry scoring (no radius pre-filter).
    # Cap candidates first: with a wide search strip many false-positive Hough
    # circles can arrive, making C(k,4) blow up.  The real calibration circles
    # always form the tightest x-cluster, so keep only candidates nearest the
    # peak of the x-histogram before scoring.
    _MAX_CANDS = 24
    if len(cands) > _MAX_CANDS:
        xs     = np.array([c[0] for c in cands])
        bin_w  = max(1, r_min)
        hist, edges = np.histogram(xs, bins=np.arange(xs.min(), xs.max() + bin_w + 1, bin_w))
        peak_x = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
        cands  = sorted(cands, key=lambda c: abs(c[0] - peak_x))[:_MAX_CANDS]

    cands.sort(key=lambda t: t[1])
    if len(cands) >= n_expected:
        anchors = _best_n_circles(cands, n_expected)
    else:
        anchors = list(cands)

    # Phase 3: Post-selection same-size validation.
    # Now that best-n has chosen the column, the set mean is reliable.
    # Drop any circle >40% off the mean (clearly wrong detection).
    if len(anchors) >= 2:
        mean_r = float(np.mean([c[2] for c in anchors]))
        good   = [c for c in anchors if abs(c[2] - mean_r) / (mean_r + 1e-6) <= 0.40]
        if len(good) >= 2:
            anchors = good

    # Phase 4: Tight-radius re-scan bootstrapped from the validated anchors.
    # The consensus radius is now reliable; sweep again to recover missed discs.
    cons_r = float(np.mean([c[2] for c in anchors])) if anchors else float(r_est)
    t_lo   = max(r_min, int(cons_r * 0.78))
    t_hi   = min(r_max, int(cons_r * 1.28))
    if t_hi > t_lo:
        extra  = _hough_sweep(t_lo, t_hi)
        extra  = [c for c in extra if abs(c[2] - cons_r) / (cons_r + 1e-6) <= 0.30]
        merged = _deduplicate_circles(anchors + extra, dedup_dist)
        if len(merged) > len(anchors):
            merged.sort(key=lambda t: t[1])
            if len(merged) >= n_expected:
                # Apply same cap before re-scoring
                if len(merged) > _MAX_CANDS:
                    xs     = np.array([c[0] for c in merged])
                    hist, edges = np.histogram(xs, bins=np.arange(xs.min(), xs.max() + bin_w + 1, bin_w))
                    peak_x = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
                    merged = sorted(merged, key=lambda c: abs(c[0] - peak_x))[:_MAX_CANDS]
                    merged.sort(key=lambda t: t[1])
                anchors = _best_n_circles(merged, n_expected)
            else:
                anchors = merged

    # Phase 5: Extrapolate missing positions using column geometry
    if len(anchors) < n_expected:
        cons_r = float(np.mean([c[2] for c in anchors])) if anchors else float(r_est)
        anchors = _extrapolate_column(
            anchors, n_expected, strip_h, strip_w, blurred, raw_blr, cons_r
        )

    return anchors


def _validate_calibration_circles(results, img_h_full):
    """
    Cross-column and same-size post-detection validation.

    Physical constraints:
      1. All 8 calibration circles are the same physical size: any circle whose
         radius deviates >25% from the global median has its radius and bounding
         box replaced with the median value.
      2. Row alignment: L[i] and R[i] should be at the same image height.
         If a pair differs by more than 12% of image height, the circle whose
         radius is further from the global median is snapped to the other's y.

    Modifies results in-place and returns the list.
    """
    if len(results) < 2:
        return results

    L = sorted([c for c in results if c['side'] == 'L'], key=lambda c: c['center_y'])
    R = sorted([c for c in results if c['side'] == 'R'], key=lambda c: c['center_y'])

    all_r = [c['radius'] for c in results]
    med_r = int(round(float(np.median(all_r))))

    # Same-size enforcement: replace outlier radii with global median
    for c in results:
        if abs(c['radius'] - med_r) / (med_r + 1e-6) > 0.25:
            cx, cy   = c['center_x'], c['center_y']
            c['radius'] = med_r
            c['width']  = med_r * 2
            c['height'] = med_r * 2
            c['x']      = cx - med_r
            c['y']      = cy - med_r

    # Cross-column row alignment: L[i] and R[i] should share the same y
    if len(L) == 4 and len(R) == 4:
        y_tol = img_h_full * 0.12
        for lc, rc in zip(L, R):
            if abs(lc['center_y'] - rc['center_y']) <= y_tol:
                continue
            # Trust whichever circle's radius is closer to the global median
            l_err = abs(lc['radius'] - med_r)
            r_err = abs(rc['radius'] - med_r)
            anchor_y = lc['center_y'] if l_err <= r_err else rc['center_y']
            fix      = rc               if l_err <= r_err else lc
            r = fix['radius']
            fix['center_y'] = anchor_y
            fix['y']        = anchor_y - r

    return results


def detect_calibration_circles(small_bgr, scale):
    """
    Detect the 4 calibration discs on the left side panel and 4 on the right.

    Each side is searched within a horizontal zone covering the grey panel:
      - x: [CALIB_X_LO_FRAC × W,  CALIB_X_HI_FRAC × W]         (left panel)
           [(1−CALIB_X_HI_FRAC) × W,  (1−CALIB_X_LO_FRAC) × W]  (right panel)
      LO skips corner-bracket structures near the image edge (4%).
      HI covers the full panel width (25%); the x_spread scoring in
      _best_n_circles keeps the tight calibration column and rejects any
      candidates scattered further inward.
      - y: [CALIB_Y_MIN_FRAC × H, CALIB_Y_MAX_FRAC × H]
      The y-crop excludes frame bolts at the top and off-panel clutter below.

    Detection uses tophat+blackhat morphology to enhance disc-shaped intensity
    deviations (works for both dark and near-white discs against the grey panel),
    followed by progressive HoughCircles and a best-4 selector that scores by
    vertical spacing evenness, radius consistency, and horizontal alignment.

    Returns a list of up to 8 dicts sorted L0..L3 then R0..R3:
        side        'L' or 'R'
        idx         0-3 top-to-bottom within the panel
        x, y        top-left of the bounding box in full-resolution pixels
        width       bounding-box width  (= 2 × radius)
        height      bounding-box height (= 2 × radius)
        center_x    circle centre x in full-resolution pixels
        center_y    circle centre y in full-resolution pixels
        radius      circle radius in full-resolution pixels
    """
    sh, sw   = small_bgr.shape[:2]
    x_lo     = max(0,  int(sw * CALIB_X_LO_FRAC))
    x_hi     = min(sw, int(sw * CALIB_X_HI_FRAC))
    y_min_s  = max(0,  int(sh * CALIB_Y_MIN_FRAC))
    y_max_s  = min(sh, int(sh * CALIB_Y_MAX_FRAC))
    r_min    = max(4,  int(sw * CALIB_R_MIN_FRAC))
    r_max    = max(8,  int(sw * CALIB_R_MAX_FRAC))
    gray     = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    inv      = 1.0 / scale

    results = []
    for side, x0_img, x1_img in [
        ('L', x_lo,      x_hi),
        ('R', sw - x_hi, sw - x_lo),
    ]:
        if x1_img <= x0_img:
            continue
        strip = gray[y_min_s:y_max_s, x0_img:x1_img]
        det   = _detect_circles_in_strip(strip, r_min, r_max, n_expected=4)
        for idx, (cx_s, cy_s, r_s) in enumerate(det):
            cx_f = int((cx_s + x0_img) * inv)
            cy_f = int((cy_s + y_min_s) * inv)
            r_f  = int(r_s * inv)
            d    = r_f * 2
            results.append({
                "side": side, "idx": idx,
                "x": cx_f - r_f, "y": cy_f - r_f,
                "width": d, "height": d,
                "center_x": cx_f, "center_y": cy_f,
                "radius": r_f,
            })

    img_h_full = int(round(sh * inv))
    _validate_calibration_circles(results, img_h_full)
    return results


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

    # ── Step 14: calibration circles ──────────────────────────────────────────
    cal_circles = detect_calibration_circles(small, scale)

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

    for c in cal_circles:
        cx, cy, r = c["center_x"], c["center_y"], c["radius"]
        cv2.circle(ann, (cx, cy), r, CALIB_CIRCLE_COLOR, thick)
        label = f"{c['side']}{c['idx']}"
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thick)
        lx, ly = cx - r, cy - r
        cv2.rectangle(ann, (lx, ly), (lx + tw + 6, ly + th + 8), (30, 30, 30), -1)
        cv2.putText(ann, label, (lx + 3, ly + th + 4),
                    FONT, font_scale, CALIB_CIRCLE_COLOR, thick, cv2.LINE_AA)

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

    # ── Calibration circles CSV ───────────────────────────────────────────────
    cal_csv_path = Path(output_dir) / f"{stem}_calibration_rois.csv"
    with open(cal_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "idx", "x", "y", "width", "height",
                         "center_x", "center_y", "radius",
                         "image_width", "image_height"])
        for c in cal_circles:
            writer.writerow([c["side"], c["idx"],
                             c["x"], c["y"], c["width"], c["height"],
                             c["center_x"], c["center_y"], c["radius"],
                             W, H])

    # ── Calibration circles JSON ──────────────────────────────────────────────
    cal_json_path = Path(output_dir) / f"{stem}_calibration_rois.json"
    with open(cal_json_path, "w") as f:
        json.dump({
            "image": str(img_path), "image_width": W, "image_height": H,
            "calibration_circles_detected": len(cal_circles),
            "calibration_circles": cal_circles,
        }, f, indent=2)

    n_blobs = len(blobs)
    n_cal   = len(cal_circles)
    print(f"  {stem}: {len(rois)} squares, {n_cal}/8 calibration circles"
          f"  ({n_blobs} inner blobs found)")
    print(f"    -> {ann_path.name}  |  {csv_path.name}  |  {contours_path.name}")
    print(f"    -> {cal_csv_path.name}  |  {cal_json_path.name}")
    if n_cal < 8:
        print(f"  [WARNING] Expected 8 calibration circles, got {n_cal}")
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
    t_total = time.time()
    for p in paths:
        print(f"[{p.name}]")
        t0 = time.time()
        process_image(p, output_dir)
        print(f"  time: {time.time() - t0:.1f}s")

    elapsed = time.time() - t_total
    print(f"\nDone.  Total: {elapsed:.1f}s  ({elapsed/len(paths):.1f}s/image)")


if __name__ == "__main__":
    main()
