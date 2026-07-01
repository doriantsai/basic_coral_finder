#!/usr/bin/env python3
"""
detect_calibration_circles.py

Standalone calibration-circle detector for coral rack images.

The 8 calibration discs are mechanically fixed in a 2-column × 4-row grid,
one column on the left panel and one on the right, each within the outer
quarter of the image width.

Detection strategy
------------------
Stage 1 — Per-column anchor-pair search.
  Each column contains 4 solid discs that span the full luminance range (one
  near-black, one near-white, two intermediate grays) by design.  Detection
  exploits this property: for each strip, find the darkest and brightest
  HoughCircles candidate, validate the pair as likely column endpoints
  (x-alignment, y-span, luminance contrast, spatial consistency), then fill
  the two intermediate rows via nearest-neighbour search around the predicted
  positions.  A composite score (luminance range × monotonicity, penalised by
  y-spacing irregularity, radius variance, x-spread, and L/R row-y mismatch)
  selects the best hypothesis from the top-20 dark × top-20 bright candidates.

  Left column is processed first; its detected row y-positions are passed as a
  soft constraint when searching the right column (both share the same physical
  disc heights).

  After selecting the best column, the two intermediate circles' radii are used
  as a per-column reference to cap any 2× over-detected anchor radius.

Usage:
    python3 detect_calibration_circles.py <image_or_dir> [output_dir]

    image_or_dir  a single image file or a directory of images
    output_dir    default: output/

Output per image:
    <stem>_calibration_rois.json
    <stem>_calibration_rois.csv
    <stem>_calibration_annotated.jpg
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

PROCESS_SCALE    = 0.25    # downscale factor for Hough processing

CALIB_X_LO_FRAC  = 0.010   # inner x bound from each edge (fraction of image width)
CALIB_X_HI_FRAC  = 0.250   # outer x bound from each edge
CALIB_Y_MIN_FRAC = 0.01    # skip this fraction of image height at top
CALIB_Y_MAX_FRAC = 0.99    # skip this fraction of image height at bottom
CALIB_R_MIN_FRAC = 0.025   # min circle radius as fraction of image width
CALIB_R_MAX_FRAC = 0.050   # max circle radius

N_ROWS  = 4     # circles per column
N_TOTAL = 8     # 2 columns × 4 rows

CIRCLE_COLOR = (255, 50, 200)   # BGR magenta for annotated output
FONT         = cv2.FONT_HERSHEY_SIMPLEX

LABELS_DIR = Path("calibration_labels") / "labels"


# ── Stage 1: anchor-pair column detector ─────────────────────────────────────

def _dedup(circles, min_dist):
    unique = []
    for c in circles:
        if not any((c[0]-u[0])**2 + (c[1]-u[1])**2 < min_dist**2 for u in unique):
            unique.append(c)
    return unique


def _mean_gray_fixed(strip, cx, cy, r_sample):
    """Mean gray of a disc of fixed radius r_sample//2 centred at (cx, cy)."""
    h, w = strip.shape[:2]
    cx = int(np.clip(round(cx), 0, w - 1))
    cy = int(np.clip(round(cy), 0, h - 1))
    ir = int(max(1, r_sample // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), ir, 255, -1)
    pix = strip[mask > 0]
    return float(np.mean(pix)) if len(pix) > 0 else 128.0


def _hough_cands(strip, r_min, r_max):
    """
    Return all Hough circle candidates after r_min-radius dedup.

    Two pre-processed sources:
      blurred  — CLAHE on tophat+blackhat morphology (disc-anomaly image)
      raw      — CLAHE on the original gray strip

    Eight (param1, param2) combinations give progressive sensitivity.
    """
    r_est  = (r_min + r_max) // 2
    ksize  = max(3, (r_est // 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    th  = cv2.morphologyEx(strip, cv2.MORPH_TOPHAT,   kernel)
    bh  = cv2.morphologyEx(strip, cv2.MORPH_BLACKHAT, kernel)
    ds  = cv2.add(th, bh)
    blr = cv2.GaussianBlur(cv2.createCLAHE(4.0, (4, 4)).apply(ds),    (9, 9), 2)
    raw = cv2.GaussianBlur(cv2.createCLAHE(3.0, (8, 8)).apply(strip), (9, 9), 2)

    cands = []
    for src in (blr, raw):
        for p1, p2 in [(50, 30), (40, 22), (30, 16), (20, 12)]:
            rc = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=1.0,
                                  minDist=max(r_min, 4), param1=p1, param2=p2,
                                  minRadius=max(r_min, 1), maxRadius=r_max)
            if rc is not None:
                for c in rc[0]:
                    cands.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
    return _dedup(cands, r_min)


def _detect_col(strip, r_min, r_max, ref_ys=None):
    """
    Detect the 4-circle calibration column in strip_gray.

    Algorithm
    ---------
    1. Run HoughCircles with 8 parameter combinations on two preprocessed
       sources; deduplicate; exclude candidates within r_max//2 of the strip
       top/bottom edges (panel corners / diagonal struts).
    2. Sample mean gray at each candidate position using a fixed r_min-radius
       circle (avoids dilution from over-detected radii).
    3. For each pair of the top-20 darkest × top-20 brightest candidates:
         a. Check x-alignment (|Δx| ≤ 3.5 × r_max).
         b. Check y-span (35 % to 82 % of strip height).
         c. Check luminance contrast (|Δgray| ≥ 60).
         d. Check mean column x ≥ 30 % of strip width (reject inner-edge noise).
         e. Fill intermediate rows by nearest-neighbour in a tight
            (±1.5 × r_min) x-window.
         f. Score = (gray_range/200) × monotonicity /
                    (1 + sp_cv + r_cv×0.5 + x_cv×2.0 + y_match_err×2.0)
            where y_match_err uses ref_ys from the L column as a soft
            constraint (both columns share the same physical row heights).
    4. Return the column with the highest score as a sorted list of
       (cx, cy, r) tuples in strip-local coordinates, or None.

    Parameters
    ----------
    strip   : uint8 grayscale strip image
    r_min   : minimum circle radius in strip pixels
    r_max   : maximum circle radius in strip pixels
    ref_ys  : list of 4 y-positions from the other column (optional)
    """
    sh, sw = strip.shape[:2]

    raw_cands = _hough_cands(strip, r_min, r_max)
    # Exclude edge candidates: panel corners and diagonal rail cross-sections
    raw_cands = [c for c in raw_cands if r_max // 2 <= c[1] <= sh - r_max // 2]
    if len(raw_cands) < N_ROWS:
        return None

    r_samp = r_min  # fixed sample radius for gray measurement
    cands_g = [(c[0], c[1], c[2], _mean_gray_fixed(strip, c[0], c[1], r_samp))
               for c in raw_cands]

    dark_sorted   = sorted(cands_g, key=lambda c:  c[3])
    bright_sorted = sorted(cands_g, key=lambda c: -c[3])

    y_span_min = 0.35 * sh
    y_span_max = 0.82 * sh
    x_col_min  = 0.30 * sw   # calibration column never at the very inner edge
    N_TEST     = 20           # test top-N dark × top-N bright = 400 pairs

    def _nearest(pred_x, pred_y, step):
        x_win = 1.5 * r_min
        y_win = 0.65 * step
        nearby = [c for c in cands_g
                  if abs(c[0] - pred_x) < x_win and abs(c[1] - pred_y) < y_win]
        if not nearby:
            nearby = [c for c in cands_g
                      if abs(c[0] - pred_x) < 3 * r_max and abs(c[1] - pred_y) < y_win * 2]
        if not nearby:
            g = _mean_gray_fixed(strip, int(pred_x), int(pred_y), r_samp)
            return (int(pred_x), int(pred_y), r_min, g)
        return min(nearby, key=lambda c: (c[0] - pred_x)**2 + (c[1] - pred_y)**2)

    best_score, best_col = -1.0, None

    for dc in dark_sorted[:N_TEST]:
        for bc in bright_sorted[:N_TEST]:
            x_diff = abs(dc[0] - bc[0])
            y_diff = abs(dc[1] - bc[1])
            if x_diff > 3.5 * r_max:
                continue
            if not (y_span_min < y_diff < y_span_max):
                continue
            if abs(dc[3] - bc[3]) < 60:
                continue

            top_c, bot_c = sorted([dc, bc], key=lambda c: c[1])
            step  = (bot_c[1] - top_c[1]) / 3.0
            x_col = (top_c[0] + bot_c[0]) / 2.0

            mid1 = _nearest(x_col, top_c[1] + step,     step)
            mid2 = _nearest(x_col, top_c[1] + 2 * step, step)

            col = sorted([top_c, mid1, mid2, bot_c], key=lambda c: c[1])
            xs  = [c[0] for c in col]
            ys  = [c[1] for c in col]
            gs  = [c[3] for c in col]
            rs  = [c[2] for c in col]
            sp  = [ys[i + 1] - ys[i] for i in range(3)]

            if min(sp) < r_min * 0.5:
                continue
            if np.mean(xs) < x_col_min:
                continue
            x_cv = np.std(xs) / r_min
            if x_cv > 0.60:
                continue

            g_range = (max(gs) - min(gs)) / 200.0
            g_mono  = 1.0 if (gs == sorted(gs) or gs == sorted(gs, reverse=True)) else 0.6
            sp_cv   = float(np.std(sp)) / (float(np.mean(sp)) + 1e-6)
            r_cv    = float(np.std(rs)) / (float(np.mean(rs)) + 1e-6)

            # Soft joint constraint: L and R columns share physical row heights
            y_match = 0.0
            if ref_ys is not None:
                mean_sp = float(np.mean(sp)) + 1e-6
                y_match = sum(abs(ys[i] - ref_ys[i]) for i in range(N_ROWS)) / (N_ROWS * mean_sp)

            score = g_range * g_mono / (1.0 + sp_cv + r_cv * 0.5 + x_cv * 2.0 + y_match * 2.0)
            if score > best_score:
                best_score = score
                best_col   = [(c[0], c[1], c[2]) for c in col]

    return best_col


# ── Main detection function ───────────────────────────────────────────────────

def detect_calibration_circles(small_bgr, scale):
    """
    Detect 8 calibration circles from a downscaled BGR image.

    Returns a list of dicts sorted L0..L3 then R0..R3, each with:
        side, idx, x, y, width, height, center_x, center_y, radius
    All coordinates are in full-resolution pixels.
    """
    sh, sw = small_bgr.shape[:2]
    x_lo   = max(0,  int(sw * CALIB_X_LO_FRAC))
    x_hi   = min(sw, int(sw * CALIB_X_HI_FRAC))
    y_min  = max(0,  int(sh * CALIB_Y_MIN_FRAC))
    y_max  = min(sh, int(sh * CALIB_Y_MAX_FRAC))
    r_min  = max(4,  int(sw * CALIB_R_MIN_FRAC))
    r_max  = max(8,  int(sw * CALIB_R_MAX_FRAC))
    gray   = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    inv    = 1.0 / scale

    def _strip_circles_to_full(col, x0, y0_offset):
        """Convert strip-local (cx,cy,r) list to full-res center dicts."""
        return [{"center_x": int(round((cx + x0) * inv)),
                 "center_y": int(round((cy + y0_offset) * inv)),
                 "radius":   int(round(r * inv))}
                for cx, cy, r in col]

    # Stage 1: detect L column first, then R using L's y-positions
    L_raw = R_raw = []

    L_strip = gray[y_min:y_max, x_lo:x_hi] if x_hi > x_lo else None
    if L_strip is not None:
        L_col = _detect_col(L_strip, r_min, r_max, ref_ys=None)
        if L_col:
            L_raw = _strip_circles_to_full(L_col, x_lo, y_min)
            ref_ys = [c[1] for c in L_col]   # strip-local y-positions

            R_strip = gray[y_min:y_max, sw - x_hi:sw - x_lo] if x_hi > x_lo else None
            if R_strip is not None:
                R_col = _detect_col(R_strip, r_min, r_max, ref_ys=ref_ys)
                if R_col is None:
                    R_col = _detect_col(R_strip, r_min, r_max, ref_ys=None)
                if R_col:
                    R_raw = _strip_circles_to_full(R_col, sw - x_hi, y_min)

    # Radius sanity: anchor discs (top/bottom of each column) can be detected at
    # 2× true radius by Hough.  The two inner circles have accurate radii — use
    # their median as the per-column reference and cap any outlier above 1.3×.
    for col in [L_raw, R_raw]:
        if len(col) >= 4:
            sy = sorted(col, key=lambda c: c['center_y'])
            inner_r = [sy[1]['radius'], sy[2]['radius']]
            ref_r = int(round(float(np.median(inner_r))))
            cap   = int(ref_r * 1.3)
            for c in col:
                if c['radius'] > cap:
                    c['radius'] = ref_r

    results = []
    for side, col in [('L', L_raw), ('R', R_raw)]:
        for idx, c in enumerate(sorted(col, key=lambda c: c['center_y'])):
            r = c['radius']
            results.append({
                "side":     side,
                "idx":      idx,
                "x":        c['center_x'] - r,
                "y":        c['center_y'] - r,
                "width":    r * 2,
                "height":   r * 2,
                "center_x": c['center_x'],
                "center_y": c['center_y'],
                "radius":   r,
            })
    return results


# ── Ground-truth comparison ───────────────────────────────────────────────────

def _load_labels(img_stem):
    lp = LABELS_DIR / f"circles_{img_stem}.json"
    if not lp.exists():
        return None
    try:
        data = json.loads(lp.read_text())
        return [c for c in data.get("circles", [])
                if all(k in c for k in ("center_x", "center_y", "radius"))]
    except (json.JSONDecodeError, KeyError):
        return None


def _compare_with_labels(detected, labeled):
    print(f"  {'ID':<6} {'Det (cx,cy)':>16} {'Lbl (cx,cy)':>16} "
          f"{'Ctr err':>9} {'Det r':>7} {'Lbl r':>7} {'R err':>7}")
    print(f"  {'-'*76}")
    total_ctr, total_r = 0.0, 0.0
    for lc in sorted(labeled, key=lambda c: c['center_y']):
        lx, ly, lr = lc['center_x'], lc['center_y'], lc['radius']
        best = min(detected,
                   key=lambda d: (d['center_x'] - lx)**2 + (d['center_y'] - ly)**2)
        dx, dy, dr = best['center_x'], best['center_y'], best['radius']
        ctr_err = ((dx - lx)**2 + (dy - ly)**2) ** 0.5
        r_err   = abs(dr - lr)
        total_ctr += ctr_err
        total_r   += r_err
        cid = f"{best['side']}{best['idx']}"
        print(f"  {cid:<6} ({dx:5d},{dy:5d})   ({lx:5d},{ly:5d})"
              f"   {ctr_err:7.1f}px   {dr:5d}   {lr:5d}   {r_err:5d}")
    n = len(labeled)
    print(f"  {'Mean':<6} {'':>16} {'':>16} "
          f"  {total_ctr/n:6.1f}px {'':>7} {'':>7} {total_r/n:5.1f}")


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_annotated(img, circles, out_path):
    H, W  = img.shape[:2]
    sc    = min(1400 / W, 900 / H)
    vis   = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    thick = max(1, round(sc * 3))
    fs    = max(0.35, sc * 2.0)

    for c in circles:
        cx = int(round(c['center_x'] * sc))
        cy = int(round(c['center_y'] * sc))
        r  = max(1, int(round(c['radius'] * sc)))
        cv2.circle(vis, (cx, cy), r,  CIRCLE_COLOR, thick)
        cv2.circle(vis, (cx, cy), 3,  CIRCLE_COLOR, -1)
        tag = f"{c['side']}{c['idx']}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, fs, 1)
        lx, ly = cx - r, cy - r
        cv2.rectangle(vis, (lx, ly - th - 6), (lx + tw + 6, ly), (20, 20, 20), -1)
        cv2.putText(vis, tag, (lx + 3, ly - 3), FONT, fs, CIRCLE_COLOR, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 93])


def _save_json(img_path, img_shape, circles, out_path):
    H, W = img_shape[:2]
    with open(out_path, "w") as f:
        json.dump({
            "image":                       str(img_path),
            "image_width":                 W,
            "image_height":                H,
            "calibration_circles_detected": len(circles),
            "calibration_circles":          circles,
        }, f, indent=2)


def _save_csv(circles, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "idx", "x", "y", "width", "height",
                    "center_x", "center_y", "radius"])
        for c in circles:
            w.writerow([c["side"], c["idx"],
                        c["x"], c["y"], c["width"], c["height"],
                        c["center_x"], c["center_y"], c["radius"]])


# ── Per-image pipeline ────────────────────────────────────────────────────────

def process_image(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return

    H, W  = img.shape[:2]
    small = cv2.resize(img, (int(W * PROCESS_SCALE), int(H * PROCESS_SCALE)))

    t0      = time.time()
    circles = detect_calibration_circles(small, PROCESS_SCALE)
    elapsed = time.time() - t0

    stem = Path(img_path).stem
    _save_json(img_path, img.shape,
               circles, output_dir / f"{stem}_calibration_rois.json")
    _save_csv(circles,   output_dir / f"{stem}_calibration_rois.csv")
    _save_annotated(img, circles,
                    output_dir / f"{stem}_calibration_annotated.jpg")

    n     = len(circles)
    L     = sum(1 for c in circles if c['side'] == 'L')
    R     = sum(1 for c in circles if c['side'] == 'R')
    diams = [c['radius'] * 2 for c in circles]
    d_rng = f"dia={min(diams)}-{max(diams)}px" if diams else "none"
    status = "OK  " if n == N_TOTAL else "WARN"
    print(f"  {status}  {n}/{N_TOTAL}  L={L}  R={R}  {d_rng}  ({elapsed:.2f}s)")

    labeled = _load_labels(stem)
    if labeled:
        print(f"  Ground truth: {len(labeled)} labeled circles")
        _compare_with_labels(circles, labeled)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target     = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
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

    print(f"Processing {len(paths)} image(s)  ->  {output_dir}/\n")
    t0 = time.time()
    for p in paths:
        print(f"[{p.name}]")
        process_image(p, output_dir)

    elapsed = time.time() - t0
    print(f"Done.  {elapsed:.1f}s total  ({elapsed/len(paths):.1f}s/image)")


if __name__ == "__main__":
    main()
