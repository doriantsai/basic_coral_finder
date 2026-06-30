#!/usr/bin/env python3
"""
Save step-by-step debug images for each stage of the coral detection pipeline.
One numbered image per step, saved to:

    debug/<image_stem>/01_downscaled.jpg
    debug/<image_stem>/02_white_mask.jpg
    ...
    debug/<image_stem>/13_coral_contours.jpg
    debug/<image_stem>/14_calibration_circles.jpg

Usage:
    python3 debug_pipeline.py <image_or_dir> [debug_dir]
"""

import cv2
import numpy as np
import sys
from pathlib import Path

from detect_corals_on_white_squares import (
    PROCESS_SCALE, HSV_V_MIN, HSV_S_MAX,
    MIN_BLOB_FRAC, MAX_BLOB_FRAC, MAX_BLOB_ASPECT,
    DENOISE_KSIZE, CLOSE_KSIZE, ROI_BORDER_FRAC,
    CORAL_DENOISE_KSIZE, CORAL_CLOSE_KSIZE, CORAL_OPEN_KSIZE,
    CALIB_X_LO_FRAC, CALIB_X_HI_FRAC, CALIB_Y_MIN_FRAC, CALIB_Y_MAX_FRAC,
    CALIB_R_MIN_FRAC, CALIB_R_MAX_FRAC, CALIB_CIRCLE_COLOR,
    get_white_mask, get_coral_mask_for_roi,
    cluster_1d, _white_in_strip, _prune_edge_centers,
    blobs_to_grid_rois, distinct_cols, find_coral_contour,
    detect_calibration_circles,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write(path, img, label):
    """Save a BGR or grayscale image with a legible label bar at the top."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    bar  = np.zeros((28, w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (5, 19), FONT, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([bar, out]),
                [cv2.IMWRITE_JPEG_QUALITY, 93])


def _cell_color(r, c):
    """Return a distinct BGR color for grid cell (r, c)."""
    hue = int(((r * 4 + c) * 43) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 210, 230]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(v) for v in bgr)


def _nearest(val, centers):
    return int(np.argmin([abs(val - c) for c in centers]))


def _overlay_mask(base, mask_crop, ox, oy, color, alpha=0.5):
    """Blend a binary mask crop into base at pixel offset (ox, oy)."""
    h, w   = mask_crop.shape
    bh, bw = base.shape[:2]
    ey, ex = min(oy + h, bh), min(ox + w, bw)
    mh, mw = ey - oy, ex - ox
    if mh <= 0 or mw <= 0:
        return
    region  = base[oy:ey, ox:ex].astype(np.float32)
    m       = mask_crop[:mh, :mw] > 0
    region[m] = region[m] * (1 - alpha) + np.array(color, np.float32) * alpha
    base[oy:ey, ox:ex] = region.clip(0, 255).astype(np.uint8)


# ── Per-image debug pipeline ──────────────────────────────────────────────────

def debug_image(img_path, dbg_root):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return

    H, W  = img.shape[:2]
    scale = PROCESS_SCALE
    stem  = Path(img_path).stem

    dbg = Path(dbg_root) / stem
    dbg.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Downscale ─────────────────────────────────────────────────────
    small   = cv2.resize(img, (int(W * scale), int(H * scale)))
    sh, sw  = small.shape[:2]
    _write(dbg / "01_downscaled.jpg", small,
           f"Step 1 | Downscale  {W}x{H} -> {sw}x{sh}  (scale={scale})")

    # ── Step 2: White mask (RF or HSV fallback) ───────────────────────────────
    wm = get_white_mask(small)
    _write(dbg / "02_white_mask.jpg", wm,
           f"Step 2 | White mask  (WhiteTabClassifier RF, or HSV fallback "
           f"Value>={HSV_V_MIN} Sat<={HSV_S_MAX})")

    # ── Step 3a: Denoise ──────────────────────────────────────────────────────
    wm_clean = cv2.medianBlur(wm, DENOISE_KSIZE)
    _write(dbg / "03a_denoised.jpg", wm_clean,
           f"Step 3a | Median blur {DENOISE_KSIZE}x{DENOISE_KSIZE}  "
           "(removes RF salt-and-pepper noise)")

    # ── Step 3b: Morphological close ──────────────────────────────────────────
    m1 = cv2.morphologyEx(wm_clean, cv2.MORPH_CLOSE,
                          np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8))
    _write(dbg / "03b_morph_close.jpg", m1,
           f"Step 3b | Morph close {CLOSE_KSIZE}x{CLOSE_KSIZE}  "
           "(seals white frame-edge gaps)")

    # ── Step 3c: All inner contours ───────────────────────────────────────────
    img_area = sh * sw
    cnts, hier = cv2.findContours(m1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dbg3c  = small.copy()
    n_inner = 0
    if hier is not None:
        for i, c in enumerate(cnts):
            if hier[0][i][3] < 0:
                continue
            n_inner += 1
            x, y, cw, ch = cv2.boundingRect(c)
            cv2.rectangle(dbg3c, (x, y), (x + cw, y + ch), (200, 200, 0), 1)
    _write(dbg / "03c_all_inner_contours.jpg", dbg3c,
           f"Step 3c | RETR_CCOMP hierarchy  -  {n_inner} inner contours found")

    # ── Step 4: Size and aspect-ratio filter ──────────────────────────────────
    dbg4   = small.copy()
    n_pass4 = 0
    if hier is not None:
        for i, c in enumerate(cnts):
            if hier[0][i][3] < 0:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            a_frac = (cw * ch) / img_area
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            ok     = MIN_BLOB_FRAC < a_frac < MAX_BLOB_FRAC and aspect < MAX_BLOB_ASPECT
            cv2.rectangle(dbg4, (x, y), (x + cw, y + ch),
                          (0, 200, 50) if ok else (0, 50, 200), 1)
            if ok:
                n_pass4 += 1
    _write(dbg / "04_size_shape_filter.jpg", dbg4,
           f"Step 4 | Size & aspect filter  -  green=pass ({n_pass4}), red=fail"
           f"  [area {MIN_BLOB_FRAC*100:.1f}%-{MAX_BLOB_FRAC*100:.0f}%, "
           f"aspect<{MAX_BLOB_ASPECT}]")

    # ── Step 5: 4-direction white border check ────────────────────────────────
    sized_blobs = []
    if hier is not None:
        for i, c in enumerate(cnts):
            if hier[0][i][3] < 0:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            a_frac = (cw * ch) / img_area
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if MIN_BLOB_FRAC < a_frac < MAX_BLOB_FRAC and aspect < MAX_BLOB_ASPECT:
                sized_blobs.append((x, y, cw, ch))

    overlay = small.copy()
    dbg5    = small.copy()
    blobs_s1 = []
    for (x, y, cw, ch) in sized_blobs:
        b = max(int(min(cw, ch) * 0.20), 5)
        sides = [
            _white_in_strip(m1, x - b,  y,       x,        y + ch),
            _white_in_strip(m1, x + cw,  y,       x + cw+b, y + ch),
            _white_in_strip(m1, x,       y - b,   x + cw,   y),
            _white_in_strip(m1, x,       y + ch,  x + cw,   y + ch+b),
        ]
        ok = min(sides) >= 0.08
        strip_col = (100, 220, 100) if ok else (100, 100, 220)
        for sx1, sy1, sx2, sy2 in [
            (max(0, x-b),  y,          x,             y+ch),
            (x+cw,         y,          min(sw,x+cw+b),y+ch),
            (x,            max(0,y-b), x+cw,          y),
            (x,            y+ch,       x+cw,          min(sh,y+ch+b)),
        ]:
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), strip_col, -1)
        if ok:
            blobs_s1.append((x, y, cw, ch))

    cv2.addWeighted(overlay, 0.35, small, 0.65, 0, dbg5)
    for (x, y, cw, ch) in sized_blobs:
        b = max(int(min(cw, ch) * 0.20), 5)
        sides = [
            _white_in_strip(m1, x-b,  y,      x,       y+ch),
            _white_in_strip(m1, x+cw, y,      x+cw+b,  y+ch),
            _white_in_strip(m1, x,    y-b,    x+cw,    y),
            _white_in_strip(m1, x,    y+ch,   x+cw,    y+ch+b),
        ]
        ok = min(sides) >= 0.08
        cv2.rectangle(dbg5, (x, y), (x+cw, y+ch),
                      (0, 200, 50) if ok else (0, 100, 255), 1)
        cv2.putText(dbg5, f"{min(sides)*100:.0f}%", (x+1, y+ch-2),
                    FONT, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    _write(dbg / "05_white_border_check.jpg", dbg5,
           f"Step 5 | 4-way white border check (>=8% each side)  -  "
           f"green=pass ({len(blobs_s1)}), orange=fail  [% = min side]")

    # ── Step 6: Strategy selection ────────────────────────────────────────────
    def _apply_holes(mask):
        c2, h2 = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        if h2 is None:
            return result
        for i, c in enumerate(c2):
            if h2[0][i][3] < 0:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            a_frac = (cw * ch) / img_area
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if not (MIN_BLOB_FRAC < a_frac < MAX_BLOB_FRAC and aspect < MAX_BLOB_ASPECT):
                continue
            b = max(int(min(cw, ch) * 0.20), 5)
            sides = [
                _white_in_strip(mask, x-b, y,    x,       y+ch),
                _white_in_strip(mask, x+cw, y,   x+cw+b,  y+ch),
                _white_in_strip(mask, x,   y-b,  x+cw,    y),
                _white_in_strip(mask, x,   y+ch, x+cw,    y+ch+b),
            ]
            if min(sides) >= 0.08:
                result.append((x, y, cw, ch))
        return result

    if distinct_cols(blobs_s1) >= 3:
        strategy_label = (f"Strategy 1 ({CLOSE_KSIZE}x{CLOSE_KSIZE} close)  -  "
                          f"{distinct_cols(blobs_s1)} distinct columns >= 3, "
                          "no fallback needed")
        final_blobs = blobs_s1
        dbg6 = small.copy()
        for b in blobs_s1:
            cv2.rectangle(dbg6, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                          (0, 200, 50), 1)
        _write(dbg / "06_strategy.jpg", dbg6, f"Step 6 | {strategy_label}")
    else:
        best_blobs  = blobs_s1
        best_mask   = m1
        fallback_dv = None
        for dv in [20, 30, 40, 50]:
            m2 = cv2.morphologyEx(wm_clean, cv2.MORPH_CLOSE,
                                  np.ones((dv, 4), np.uint8))
            m2 = cv2.morphologyEx(m2,       cv2.MORPH_CLOSE,
                                  np.ones((4, dv), np.uint8))
            b2 = _apply_holes(m2)
            if b2:
                areas = [cw * ch for _, _, cw, ch in b2]
                if max(areas) > float(np.median(areas)) * 4:
                    continue
            if distinct_cols(b2) > distinct_cols(best_blobs):
                best_blobs  = b2
                best_mask   = m2
                fallback_dv = dv
            if distinct_cols(best_blobs) >= 3:
                break
        final_blobs    = best_blobs
        strategy_label = (
            f"Fallback Strategy 2  -  directional close {fallback_dv}px kernel"
            if fallback_dv else
            f"Fallback attempted but still only {distinct_cols(best_blobs)} cols"
        )
        m1_bgr = cv2.cvtColor(m1,        cv2.COLOR_GRAY2BGR)
        m2_bgr = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
        sep    = np.full((sh, 4, 3), 128, dtype=np.uint8)
        _write(dbg / "06_fallback_mask.jpg",
               np.hstack([m1_bgr, sep, m2_bgr]),
               f"Step 6 | {strategy_label}  "
               f"[left={CLOSE_KSIZE}x{CLOSE_KSIZE} close, right=directional close]")

    # ── Step 7: 1D clustering ─────────────────────────────────────────────────
    blobs = final_blobs
    dbg7  = small.copy()
    row_centers    = []
    col_centers_raw = []

    if blobs:
        med_w   = int(np.median([cw for _, _, cw, _ in blobs]))
        med_h   = int(np.median([ch for _, _, _, ch in blobs]))
        row_gap = max(int(med_h * 0.6), 18)
        col_gap = max(int(med_w * 0.6), 18)
        cy_all  = sorted([y + ch // 2 for _, y, _, ch in blobs])
        cx_all  = sorted([x + cw // 2 for x, _, cw, _ in blobs])
        row_centers     = cluster_1d(cy_all, row_gap)
        col_centers_raw = cluster_1d(cx_all, col_gap)

        for b in blobs:
            cv2.rectangle(dbg7, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                          (180, 180, 180), 1)
        for ri, rc in enumerate(row_centers):
            cv2.line(dbg7, (0, rc), (sw, rc), (0, 180, 255), 1)
            cv2.putText(dbg7, f"Row {ri}", (3, rc - 3),
                        FONT, 0.32, (0, 180, 255), 1)
        for ci, cc in enumerate(col_centers_raw):
            cv2.line(dbg7, (cc, 0), (cc, sh), (255, 120, 0), 1)
            cv2.putText(dbg7, f"C{ci}", (cc + 2, 14), FONT, 0.32, (255, 120, 0), 1)

    _write(dbg / "07_clustering.jpg", dbg7,
           f"Step 7 | 1D gap clustering  -  {len(row_centers)} row centers (orange), "
           f"{len(col_centers_raw)} col centers (blue)  [before calibration pruning]")

    # ── Step 8a/8b: Calibration-column pruning ────────────────────────────────
    if blobs:
        spacings  = [col_centers_raw[i+1] - col_centers_raw[i]
                     for i in range(len(col_centers_raw) - 1)]
        inner     = spacings[1:-1] if len(spacings) >= 2 else spacings
        inner_med = float(np.median(inner)) if inner else 0.0
        threshold = 0.93 * inner_med

        dbg8a = small.copy()
        for b in blobs:
            cv2.rectangle(dbg8a, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                          (180, 180, 180), 1)
        for cc in col_centers_raw:
            cv2.line(dbg8a, (cc, 0), (cc, sh), (0, 220, 255), 1)
        for i, s in enumerate(spacings):
            mid   = (col_centers_raw[i] + col_centers_raw[i+1]) // 2
            ratio = s / inner_med if inner_med else 0
            cv2.putText(dbg8a, f"{s}px ({ratio:.2f})", (mid - 20, 18),
                        FONT, 0.28, (0, 220, 255), 1)
        _write(dbg / "08a_before_pruning.jpg", dbg8a,
               f"Step 8a | Before pruning  -  {len(col_centers_raw)} cols  "
               f"[inner_median={inner_med:.0f}px, threshold={threshold:.0f}px]")

        col_centers_pruned = _prune_edge_centers(col_centers_raw)
        pruned_set         = set(col_centers_raw) - set(col_centers_pruned)

        dbg8b = small.copy()
        for b in blobs:
            cv2.rectangle(dbg8b, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                          (180, 180, 180), 1)
        for cc in col_centers_raw:
            is_pruned = cc in pruned_set
            color     = (0, 50, 220) if is_pruned else (0, 220, 50)
            cv2.line(dbg8b, (cc, 0), (cc, sh), color, 2)
            cv2.putText(dbg8b, "PRUNED" if is_pruned else "kept",
                        (cc - 18, sh - 6), FONT, 0.3, color, 1)
        _write(dbg / "08b_after_pruning.jpg", dbg8b,
               f"Step 8b | After pruning  -  {len(col_centers_pruned)} cols kept "
               f"(green), {len(pruned_set)} removed (red) = calibration columns")

        col_centers  = col_centers_pruned
        blobs_kept   = [b for b in blobs
                        if any(abs(b[0]+b[2]//2 - cc) < col_gap * 2
                               for cc in col_centers)]

        # ── Step 9: Grid assignment ───────────────────────────────────────────
        dbg9 = small.copy()
        for b in blobs_kept:
            r = _nearest(b[1] + b[3]//2, row_centers)
            c = _nearest(b[0] + b[2]//2, col_centers)
            color = _cell_color(r, c)
            cv2.rectangle(dbg9, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 2)
            cv2.putText(dbg9, f"R{r}C{c}", (b[0]+2, b[1]+b[3]-3),
                        FONT, 0.32, color, 1, cv2.LINE_AA)
        _write(dbg / "09_grid_assignment.jpg", dbg9,
               f"Step 9 | Grid assignment  -  {len(blobs_kept)} blobs -> RxC labels")

        # ── Step 10: Expanded ROIs ────────────────────────────────────────────
        rois  = blobs_to_grid_rois(blobs, sh, sw, scale)
        dbg10 = small.copy()
        for (r, c, fx, fy, fw, fh) in rois:
            sx  = int(fx * scale);  sy  = int(fy * scale)
            sw2 = int(fw * scale);  sh2 = int(fh * scale)
            color = _cell_color(r, c)
            cv2.rectangle(dbg10, (sx, sy), (sx+sw2, sy+sh2), color, 2)
            cv2.putText(dbg10, f"R{r}C{c}", (sx+3, sy+14),
                        FONT, 0.35, color, 1, cv2.LINE_AA)
        _write(dbg / "10_expanded_rois.jpg", dbg10,
               f"Step 10 | Expanded ROIs  -  {len(rois)} squares  "
               f"[blob box + {ROI_BORDER_FRAC*100:.0f}% border each side]")

        # ── Steps 11-12: Coral RF masks ───────────────────────────────────────
        # Compute once; reused for both raw (Step 11) and clean (Step 12) renders.
        coral_data = {}   # (r, c) -> (sx, sy, sw, sh, raw_mask, clean_mask)
        for (r, c, fx, fy, fw, fh) in rois:
            sx = int(fx * scale);          sy = int(fy * scale)
            sw_ = max(1, int(fw * scale)); sh_ = max(1, int(fh * scale))
            raw, clean = get_coral_mask_for_roi(small, sx, sy, sw_, sh_)
            coral_data[(r, c)] = (sx, sy, sw_, sh_, raw, clean)

        has_model = any(v[4] is not None for v in coral_data.values())

        # Step 11: raw coral mask
        dbg11 = small.copy()
        for (r, c, fx, fy, fw, fh) in rois:
            sx, sy, sw_, sh_, raw, _ = coral_data[(r, c)]
            color = _cell_color(r, c)
            cv2.rectangle(dbg11, (sx, sy), (sx+sw_, sy+sh_), color, 1)
            if raw is not None:
                _overlay_mask(dbg11, raw, sx, sy, color)
        label11 = (
            f"Step 11 | Coral RF mask (raw)  -  {len(rois)} ROIs"
            if has_model else
            "Step 11 | Coral RF mask  -  no CoralClassifier model found"
        )
        _write(dbg / "11_coral_raw_mask.jpg", dbg11, label11)

        # Step 12: cleaned coral mask
        dbg12 = small.copy()
        for (r, c, fx, fy, fw, fh) in rois:
            sx, sy, sw_, sh_, _, clean = coral_data[(r, c)]
            color = _cell_color(r, c)
            cv2.rectangle(dbg12, (sx, sy), (sx+sw_, sy+sh_), color, 1)
            if clean is not None:
                _overlay_mask(dbg12, clean, sx, sy, color)
        _write(dbg / "12_coral_clean_mask.jpg", dbg12,
               f"Step 12 | Coral mask: median blur({CORAL_DENOISE_KSIZE}) + "
               f"ellipse close({CORAL_CLOSE_KSIZE}) + open({CORAL_OPEN_KSIZE})")

        # ── Step 13: Coral contours ───────────────────────────────────────────
        dbg13 = small.copy()
        n_found = 0
        for (r, c, fx, fy, fw, fh) in rois:
            sx  = int(fx * scale);  sy  = int(fy * scale)
            sw2 = int(fw * scale);  sh2 = int(fh * scale)
            color = _cell_color(r, c)
            cv2.rectangle(dbg13, (sx, sy), (sx+sw2, sy+sh2), color, 1)
            pts = find_coral_contour(img, small, fx, fy, fw, fh)
            if pts is not None:
                pts_s = (pts.astype(np.float32) * scale).astype(np.int32)
                cv2.polylines(dbg13, [pts_s.reshape(-1, 1, 2)], True, color, 2)
                cx_s = int(pts_s[:, 0].mean())
                cy_s = int(pts_s[:, 1].mean())
                cv2.putText(dbg13, f"R{r}C{c}", (cx_s - 16, cy_s + 4),
                            FONT, 0.32, color, 1, cv2.LINE_AA)
                n_found += 1
        method = "CoralClassifier RF" if has_model else "Otsu fallback"
        _write(dbg / "13_coral_contours.jpg", dbg13,
               f"Step 13 | Coral contours  -  {n_found}/{len(rois)} found  "
               f"[{method}]")

        # ── Step 14: Calibration circles ─────────────────────────────────────
        cal_circles = detect_calibration_circles(small, scale)
        dbg14 = small.copy()
        x_lo_s = max(0,  int(sw * CALIB_X_LO_FRAC))
        x_hi_s = min(sw, int(sw * CALIB_X_HI_FRAC))
        # shade the two search zones
        overlay14 = dbg14.copy()
        cv2.rectangle(overlay14, (x_lo_s, 0), (x_hi_s, sh), (60, 30, 80), -1)
        cv2.rectangle(overlay14, (sw - x_hi_s, 0), (sw - x_lo_s, sh), (60, 30, 80), -1)
        cv2.addWeighted(overlay14, 0.35, dbg14, 0.65, 0, dbg14)
        for c in cal_circles:
            cx_s = int(c["center_x"] * scale)
            cy_s = int(c["center_y"] * scale)
            r_s  = max(1, int(c["radius"] * scale))
            cv2.circle(dbg14, (cx_s, cy_s), r_s, CALIB_CIRCLE_COLOR, 2)
            label = f"{c['side']}{c['idx']}"
            cv2.putText(dbg14, label, (cx_s - r_s + 2, cy_s - r_s - 3),
                        FONT, 0.32, CALIB_CIRCLE_COLOR, 1, cv2.LINE_AA)
        _write(dbg / "14_calibration_circles.jpg", dbg14,
               f"Step 14 | Calibration circles  -  {len(cal_circles)}/8 detected  "
               f"[zone={CALIB_X_LO_FRAC*100:.1f}%-{CALIB_X_HI_FRAC*100:.1f}% from each edge, "
               f"r={CALIB_R_MIN_FRAC*100:.1f}%-{CALIB_R_MAX_FRAC*100:.1f}% of img width]")

    n_sq = len(rois) if blobs else 0
    print(f"  {stem}: {n_sq} squares  ->  {dbg}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target    = Path(sys.argv[1])
    debug_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Processing {len(paths)} image(s) -> {debug_dir}/\n")
    for p in paths:
        print(f"[{p.name}]")
        debug_image(p, debug_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
