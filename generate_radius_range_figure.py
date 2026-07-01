#!/usr/bin/env python3
"""
generate_radius_range_figure.py

Generate an annotated image showing the calibration circle search radius range
(r_min and r_max) overlaid on a detected result.

Usage:
    python3 generate_radius_range_figure.py <image> [output]

    image   path to the input image (e.g. images/DSC00141.JPG)
    output  path for the output JPEG (default: <stem>_radius_range.jpg)

The script runs calibration circle detection on the image, then draws:
  - Magenta circles  — all 8 detected calibration discs
  - Green ring       — r_min at L2 and R2 (smallest accepted disc)
  - Orange dashed    — r_max at L2 and R2 (largest accepted disc)
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from detect_calibration_circles import (
    PROCESS_SCALE,
    CALIB_R_MIN_FRAC,
    CALIB_R_MAX_FRAC,
    detect_calibration_circles,
)


def generate(img_path: Path, out_path: Path, display_width: int = 1400) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    H, W = img.shape[:2]
    sw = int(W * PROCESS_SCALE)
    r_min_full = int(max(4,  int(sw * CALIB_R_MIN_FRAC)) / PROCESS_SCALE)
    r_max_full = int(max(8,  int(sw * CALIB_R_MAX_FRAC)) / PROCESS_SCALE)
    print(f"  r_min = {r_min_full} px  (dia {r_min_full * 2} px)")
    print(f"  r_max = {r_max_full} px  (dia {r_max_full * 2} px)")

    # Run detection
    small   = cv2.resize(img, (sw, int(H * PROCESS_SCALE)))
    circles = detect_calibration_circles(small, PROCESS_SCALE)
    print(f"  Detected {len(circles)}/8 calibration circles")

    by_id = {f"{c['side']}{c['idx']}": c for c in circles}

    # ── build display image ────────────────────────────────────────────────────
    sc  = display_width / W
    vis = cv2.resize(img, (display_width, int(H * sc)), interpolation=cv2.INTER_AREA)

    def s(v): return int(round(v * sc))

    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    MAGENTA = (255,  50, 200)
    GREEN   = ( 50, 210,  50)
    ORANGE  = (  0, 165, 255)
    WHITE   = (255, 255, 255)

    # All detected circles (magenta)
    for c in circles:
        cx, cy, r = s(c['center_x']), s(c['center_y']), s(c['radius'])
        cv2.circle(vis, (cx, cy), r, MAGENTA, 2)
        cv2.circle(vis, (cx, cy), 4, MAGENTA, -1)
        tag = f"{c['side']}{c['idx']}"
        cv2.putText(vis, tag, (cx - r + 4, cy - r - 8),
                    FONT, 0.55, MAGENTA, 2, cv2.LINE_AA)

    # r_min / r_max rings at L2 and R2 (dashed via alternating line segments)
    angles = np.linspace(0, 2 * np.pi, 72)
    for key in ('L2', 'R2'):
        if key not in by_id:
            continue
        c  = by_id[key]
        cx = s(c['center_x']); cy = s(c['center_y'])

        # r_max — orange dashed ring
        pts = [(cx + int(s(r_max_full) * np.cos(a)),
                cy + int(s(r_max_full) * np.sin(a))) for a in angles]
        for i in range(0, len(pts), 2):
            cv2.line(vis, pts[i], pts[(i + 1) % len(pts)], ORANGE, 3)

        # r_min — green dashed ring
        pts = [(cx + int(s(r_min_full) * np.cos(a)),
                cy + int(s(r_min_full) * np.sin(a))) for a in angles]
        for i in range(0, len(pts), 2):
            cv2.line(vis, pts[i], pts[(i + 1) % len(pts)], GREEN, 3)

    # ── legend ─────────────────────────────────────────────────────────────────
    l2r = by_id['L2']['radius'] if 'L2' in by_id else '?'
    r2r = by_id['R2']['radius'] if 'R2' in by_id else '?'
    legend = [
        (MAGENTA, f"Detected circle  (L2: r={l2r}px, R2: r={r2r}px)"),
        (GREEN,   f"r_min = {r_min_full}px  (dia {r_min_full * 2}px)"
                  f" - smallest disc the algorithm accepts"),
        (ORANGE,  f"r_max = {r_max_full}px  (dia {r_max_full * 2}px)"
                  f" - largest disc the algorithm accepts"),
    ]
    LH = 32; LP = 12
    box_w = 820; box_h = len(legend) * LH + LP * 2
    box_x = (display_width - box_w) // 2
    box_y = vis.shape[0] - box_h - 18
    overlay = vis.copy()
    cv2.rectangle(overlay, (box_x - LP, box_y - LP),
                  (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    for i, (color, text) in enumerate(legend):
        ty = box_y + i * LH + LH // 2 + LP
        cv2.rectangle(vis, (box_x, ty - 11), (box_x + 28, ty + 8), color, -1)
        cv2.putText(vis, text, (box_x + 36, ty + 5),
                    FONT, 0.52, WHITE, 1, cv2.LINE_AA)

    # ── title ──────────────────────────────────────────────────────────────────
    title = (f"{img_path.name}  |  Calibration circle search radius range"
             f"  (PROCESS_SCALE={PROCESS_SCALE})")
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.65, 2)
    tx = (display_width - tw) // 2; ty = 34
    cv2.rectangle(vis, (tx - 10, ty - th - 8), (tx + tw + 10, ty + 8),
                  (20, 20, 20), -1)
    cv2.putText(vis, title, (tx, ty), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

    # ── save ───────────────────────────────────────────────────────────────────
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_path = Path(sys.argv[1])
    out_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                else img_path.parent / f"{img_path.stem}_radius_range.jpg")

    print(f"[{img_path.name}]")
    generate(img_path, out_path)


if __name__ == "__main__":
    main()
