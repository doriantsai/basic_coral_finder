#!/usr/bin/env python3
"""
calibration_circle_labeler.py
Interactive GUI for annotating calibration circles on coral rack images.

Two annotation types:
  circle  — drag from the circle centre outward to set the radius (magenta)
  other   — drag to draw a bounding box over non-calibration regions (red)

Output saved per image under the output directory:
  labels/circles_<stem>.json     circle annotations: center_x, center_y, radius
  labels/boxes_<stem>.json       bounding boxes: circle-derived squares (label="circle")
                                 plus drawn "other" boxes (label="other")
  annotated/<stem>_annotated.jpg image with all annotations drawn on it
  images/<stem>.<ext>            copy of the source image for training

Usage:
    python3 calibration_circle_labeler.py [images_dir [output_dir]]

    images_dir  default: calibration_labels/images/
    output_dir  default: calibration_labels/

Controls:
    c              set active class to CIRCLE  (drag centre → edge)
    o              set active class to OTHER   (drag bounding box)
    left-drag      draw annotation in the active class
    right-click    delete the annotation under the cursor
    z              undo last annotation
    n              save and go to next image
    p              save and go to previous image
    s              save current image's labels
    r r            press twice to clear all annotations for this image
    q / Esc        save and quit
"""

import cv2
import json
import math
import shutil
import sys
import time
import numpy as np
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DISPLAY_W  = 1400
DISPLAY_H  = 860
INFO_H     = 52
MIN_RADIUS = 8    # minimum circle radius in display pixels before committing
MIN_BOX_PX = 10   # minimum box side in display pixels before committing

COLORS = {
    "circle":  (255,  50, 200),   # BGR magenta
    "other":   (  0,  80, 220),   # BGR red
    "preview": (  0, 220, 255),   # BGR yellow — live drawing
}
FILL_ALPHA = 0.15
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Labeller class ────────────────────────────────────────────────────────────

class CalibLabeler:

    def __init__(self, img_dir: Path, out_dir: Path):
        self.img_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.img_paths:
            raise SystemExit(f"No images found in {img_dir}")

        self.out_dir       = out_dir
        self.labels_dir    = out_dir / "labels"
        self.annotated_dir = out_dir / "annotated"
        self.training_dir  = out_dir / "images"
        for d in (self.labels_dir, self.annotated_dir, self.training_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.n             = len(self.img_paths)
        self.idx           = 0
        self.current_class = "circle"

        # All annotations for the current image, in draw order.
        # Circles: {"type":"circle", "label":"circle",
        #           "center_x":int, "center_y":int, "radius":int}
        # Boxes:   {"type":"box", "label":"other",
        #           "x":int, "y":int, "width":int, "height":int}
        # All coordinates are full-resolution pixels.
        self.annotations: list[dict] = []

        # Mouse drawing state (display-pixel coordinates)
        self.drawing    = False
        self.draw_start = None   # (x, y) in display pixels
        self.draw_cur   = None   # (x, y) in display pixels

        # Image / display state
        self.full_img = None
        self.disp_img = None
        self.scale    = 1.0
        self.disp_w   = DISPLAY_W
        self.disp_h   = DISPLAY_H

        # Status feedback
        self.status_msg  = ""
        self.status_time = 0.0
        self.dirty       = False
        self._reset_armed = False


    # ── I/O ───────────────────────────────────────────────────────────────────

    def _circles_path(self):
        return self.labels_dir / f"circles_{self.img_paths[self.idx].stem}.json"

    def _boxes_path(self):
        return self.labels_dir / f"boxes_{self.img_paths[self.idx].stem}.json"

    def _annotated_path(self):
        return self.annotated_dir / f"{self.img_paths[self.idx].stem}_annotated.jpg"

    def _training_img_path(self):
        p = self.img_paths[self.idx]
        return self.training_dir / p.name

    def load_image(self):
        p = self.img_paths[self.idx]
        img = cv2.imread(str(p))
        if img is None:
            self._status(f"Cannot read {p.name} — skipping")
            return
        self.full_img = img
        H, W = img.shape[:2]
        self.scale  = min(DISPLAY_W / W, DISPLAY_H / H)
        self.disp_w = int(W * self.scale)
        self.disp_h = int(H * self.scale)
        self.disp_img = cv2.resize(img, (self.disp_w, self.disp_h),
                                   interpolation=cv2.INTER_AREA)
        self._load_labels()
        self.drawing     = False
        self.draw_start  = self.draw_cur = None
        self.dirty       = False
        self._reset_armed = False
        cv2.setWindowTitle("Calibration Labeler",
                           f"Calibration Labeler  [{self.idx+1}/{self.n}]  {p.name}")

    def _load_labels(self):
        self.annotations = []

        cp = self._circles_path()
        if cp.exists():
            try:
                data = json.loads(cp.read_text())
                for c in data.get("circles", []):
                    if all(k in c for k in ("center_x", "center_y", "radius")):
                        self.annotations.append({
                            "type":     "circle",
                            "label":    "circle",
                            "center_x": int(c["center_x"]),
                            "center_y": int(c["center_y"]),
                            "radius":   int(c["radius"]),
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                self._status(f"Could not parse {cp.name}")

        bp = self._boxes_path()
        if bp.exists():
            try:
                data = json.loads(bp.read_text())
                for b in data.get("boxes", []):
                    if (b.get("label") == "other" and
                            all(k in b for k in ("x", "y", "width", "height"))):
                        self.annotations.append({
                            "type":   "box",
                            "label":  "other",
                            "x":      int(b["x"]),
                            "y":      int(b["y"]),
                            "width":  int(b["width"]),
                            "height": int(b["height"]),
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    def save_labels(self):
        if self.full_img is None:
            return
        H, W = self.full_img.shape[:2]
        p = self.img_paths[self.idx]

        circles = [a for a in self.annotations if a["type"] == "circle"]
        boxes   = [a for a in self.annotations if a["type"] == "box"]

        # circles_<stem>.json
        cp = self._circles_path()
        with open(cp, "w") as f:
            json.dump({
                "image":        str(p),
                "image_width":  W,
                "image_height": H,
                "circles": [
                    {"label":    c["label"],
                     "center_x": c["center_x"],
                     "center_y": c["center_y"],
                     "radius":   c["radius"]}
                    for c in circles
                ],
            }, f, indent=2)

        # boxes_<stem>.json — square boxes from circles + drawn "other" boxes
        all_boxes = []
        for c in circles:
            r = c["radius"]
            all_boxes.append({
                "label":  "circle",
                "x":      c["center_x"] - r,
                "y":      c["center_y"] - r,
                "width":  r * 2,
                "height": r * 2,
            })
        for b in boxes:
            all_boxes.append({
                "label":  b["label"],
                "x":      b["x"],
                "y":      b["y"],
                "width":  b["width"],
                "height": b["height"],
            })

        bp = self._boxes_path()
        with open(bp, "w") as f:
            json.dump({
                "image":        str(p),
                "image_width":  W,
                "image_height": H,
                "boxes": all_boxes,
            }, f, indent=2)

        # annotated preview
        self._save_annotated()

        # training image copy (no re-encode — preserves original quality)
        src = self.img_paths[self.idx]
        tp  = self._training_img_path()
        if src.resolve() != tp.resolve():
            shutil.copy2(src, tp)

        self.dirty = False
        self._status(
            f"Saved  {len(circles)} circles, {len(boxes)} other  ->  "
            f"labels/  annotated/  images/"
        )


    def _save_annotated(self):
        """Render all annotations onto the display-scale image and save to annotated/."""
        if self.full_img is None:
            return
        base    = self.disp_img.copy()
        overlay = base.copy()

        circles = [a for a in self.annotations if a["type"] == "circle"]
        boxes   = [a for a in self.annotations if a["type"] == "box"]

        for a in circles:
            cx, cy = self._to_disp(a["center_x"], a["center_y"])
            r = max(1, round(a["radius"] * self.scale))
            cv2.circle(overlay, (cx, cy), r, COLORS["circle"], -1)
        for a in boxes:
            bx,  by  = self._to_disp(a["x"],             a["y"])
            bx2, by2 = self._to_disp(a["x"] + a["width"], a["y"] + a["height"])
            cv2.rectangle(overlay, (bx, by), (bx2, by2), COLORS["other"], -1)

        cv2.addWeighted(overlay, FILL_ALPHA, base, 1 - FILL_ALPHA, 0, base)

        for i, a in enumerate(circles):
            cx, cy = self._to_disp(a["center_x"], a["center_y"])
            r = max(1, round(a["radius"] * self.scale))
            cv2.circle(base, (cx, cy), r, COLORS["circle"], 2)
            cv2.circle(base, (cx, cy), 3, COLORS["circle"], -1)
            tag = f"C{i}  r={a['radius']}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
            lx, ly = cx - r, cy - r
            cv2.rectangle(base, (lx, ly - th - 6), (lx + tw + 6, ly), (20, 20, 20), -1)
            cv2.putText(base, tag, (lx + 3, ly - 3), FONT, 0.44, COLORS["circle"], 1, cv2.LINE_AA)

        for a in boxes:
            bx,  by  = self._to_disp(a["x"],             a["y"])
            bx2, by2 = self._to_disp(a["x"] + a["width"], a["y"] + a["height"])
            cv2.rectangle(base, (bx, by), (bx2, by2), COLORS["other"], 2)
            tag = "OTHER"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
            cv2.rectangle(base, (bx, by - th - 6), (bx + tw + 6, by), (20, 20, 20), -1)
            cv2.putText(base, tag, (bx + 3, by - 3), FONT, 0.44, COLORS["other"], 1, cv2.LINE_AA)

        cv2.imwrite(str(self._annotated_path()), base,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])


    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _to_full(self, dx, dy):
        """Display pixel -> full-resolution pixel (clamped to image bounds)."""
        if self.full_img is None:
            return 0, 0
        H, W = self.full_img.shape[:2]
        return (max(0, min(W, round(dx / self.scale))),
                max(0, min(H, round(dy / self.scale))))

    def _to_disp(self, fx, fy):
        """Full-resolution pixel -> display pixel."""
        return round(fx * self.scale), round(fy * self.scale)


    # ── Mouse callback ────────────────────────────────────────────────────────

    def mouse_cb(self, event, x, y, flags, _):
        y -= INFO_H   # offset for the info bar at the top

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing    = True
            self.draw_start = (x, y)
            self.draw_cur   = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.draw_cur = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.draw_start
            x2, y2 = x, y

            if self.current_class == "circle":
                r_disp = math.hypot(x2 - x1, y2 - y1)
                if r_disp >= MIN_RADIUS:
                    fcx, fcy = self._to_full(x1, y1)
                    fr = max(1, round(r_disp / self.scale))
                    self.annotations.append({
                        "type":     "circle",
                        "label":    "circle",
                        "center_x": fcx,
                        "center_y": fcy,
                        "radius":   fr,
                    })
                    self.dirty = True
            else:
                if abs(x2 - x1) >= MIN_BOX_PX and abs(y2 - y1) >= MIN_BOX_PX:
                    fx1, fy1 = self._to_full(min(x1, x2), min(y1, y2))
                    fx2, fy2 = self._to_full(max(x1, x2), max(y1, y2))
                    self.annotations.append({
                        "type":   "box",
                        "label":  "other",
                        "x":      fx1,
                        "y":      fy1,
                        "width":  fx2 - fx1,
                        "height": fy2 - fy1,
                    })
                    self.dirty = True

            self.draw_start = self.draw_cur = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            fx, fy = self._to_full(x, y)
            for i in range(len(self.annotations) - 1, -1, -1):
                a = self.annotations[i]
                if a["type"] == "circle":
                    if math.hypot(fx - a["center_x"], fy - a["center_y"]) <= a["radius"]:
                        self.annotations.pop(i)
                        self.dirty = True
                        self._status("Deleted circle")
                        break
                else:
                    if (a["x"] <= fx <= a["x"] + a["width"] and
                            a["y"] <= fy <= a["y"] + a["height"]):
                        self.annotations.pop(i)
                        self.dirty = True
                        self._status(f"Deleted other box")
                        break


    # ── Rendering ─────────────────────────────────────────────────────────────

    def _status(self, msg: str, duration: float = 2.5):
        self.status_msg  = msg
        self.status_time = time.monotonic() + duration

    def _build_frame(self):
        base    = self.disp_img.copy()
        overlay = base.copy()

        circles = [a for a in self.annotations if a["type"] == "circle"]
        boxes   = [a for a in self.annotations if a["type"] == "box"]

        # Semi-transparent fills
        for a in circles:
            cx, cy = self._to_disp(a["center_x"], a["center_y"])
            r = max(1, round(a["radius"] * self.scale))
            cv2.circle(overlay, (cx, cy), r, COLORS["circle"], -1)
        for a in boxes:
            bx,  by  = self._to_disp(a["x"],             a["y"])
            bx2, by2 = self._to_disp(a["x"] + a["width"], a["y"] + a["height"])
            cv2.rectangle(overlay, (bx, by), (bx2, by2), COLORS["other"], -1)

        cv2.addWeighted(overlay, FILL_ALPHA, base, 1 - FILL_ALPHA, 0, base)

        # Outlines and labels
        for i, a in enumerate(circles):
            cx, cy = self._to_disp(a["center_x"], a["center_y"])
            r = max(1, round(a["radius"] * self.scale))
            cv2.circle(base, (cx, cy), r, COLORS["circle"], 2)
            cv2.circle(base, (cx, cy), 3, COLORS["circle"], -1)
            tag = f"C{i}  r={a['radius']}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
            lx, ly = cx - r, cy - r
            cv2.rectangle(base, (lx, ly - th - 6), (lx + tw + 6, ly),
                          (20, 20, 20), -1)
            cv2.putText(base, tag, (lx + 3, ly - 3),
                        FONT, 0.44, COLORS["circle"], 1, cv2.LINE_AA)

        for a in boxes:
            bx,  by  = self._to_disp(a["x"],             a["y"])
            bx2, by2 = self._to_disp(a["x"] + a["width"], a["y"] + a["height"])
            cv2.rectangle(base, (bx, by), (bx2, by2), COLORS["other"], 2)
            tag = "OTHER"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
            cv2.rectangle(base, (bx, by - th - 6), (bx + tw + 6, by),
                          (20, 20, 20), -1)
            cv2.putText(base, tag, (bx + 3, by - 3),
                        FONT, 0.44, COLORS["other"], 1, cv2.LINE_AA)

        # Live drawing preview
        if self.drawing and self.draw_start and self.draw_cur:
            x1, y1 = self.draw_start
            x2, y2 = self.draw_cur
            if self.current_class == "circle":
                r = max(1, round(math.hypot(x2 - x1, y2 - y1)))
                cv2.circle(base, (x1, y1), r, COLORS["preview"], 2)
                cv2.circle(base, (x1, y1), 3, COLORS["preview"], -1)
                cv2.line(base, (x1, y1), (x2, y2), COLORS["preview"], 1)
            else:
                cv2.rectangle(base,
                              (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)),
                              COLORS["preview"], 2)

        # Info bar
        bar = np.full((INFO_H, self.disp_w, 3), 28, dtype=np.uint8)

        p     = self.img_paths[self.idx]
        dirty = " *" if self.dirty else ""
        left  = (f"[{self.idx+1}/{self.n}]  {p.name}{dirty}   |   "
                 f"{len(circles)} circles  {len(boxes)} other")
        cv2.putText(bar, left, (8, 34), FONT, 0.56, (210, 210, 210), 1, cv2.LINE_AA)

        cls_col = COLORS[self.current_class]
        cls_txt = f"Class: {self.current_class.upper()}"
        (tw, _), _ = cv2.getTextSize(cls_txt, FONT, 0.65, 2)
        cv2.putText(bar, cls_txt, (self.disp_w - tw - 10, 36),
                    FONT, 0.65, cls_col, 2, cv2.LINE_AA)

        hint = ("[c]ircle=drag-centre-to-edge  [o]ther=drag-box  "
                "right-click=delete  |  [z]undo  [r]eset  [n]ext  [p]rev  [s]ave  [q]uit")
        cv2.putText(bar, hint, (8, 16), FONT, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

        if time.monotonic() < self.status_time:
            cv2.putText(bar, self.status_msg,
                        (self.disp_w // 4, 34),
                        FONT, 0.54, (0, 220, 255), 1, cv2.LINE_AA)

        return np.vstack([bar, base])


    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        win = "Calibration Labeler"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_W, DISPLAY_H + INFO_H)
        cv2.setMouseCallback(win, self.mouse_cb)

        self.load_image()
        self._status(
            "Ready  —  [c]ircle: drag centre→edge   [o]ther: drag box", 4
        )

        while True:
            if self.full_img is None:
                time.sleep(0.02)
                continue

            cv2.imshow(win, self._build_frame())
            key = cv2.waitKey(15) & 0xFF

            if key == ord('c'):
                self.current_class = "circle"
                self._status("Active class -> CIRCLE  (drag centre → edge)", 1.5)

            elif key == ord('o'):
                self.current_class = "other"
                self._status("Active class -> OTHER  (drag bounding box)", 1.5)

            elif key == ord('z'):
                if self.annotations:
                    removed = self.annotations.pop()
                    self.dirty = True
                    self._status(f"Undo — removed {removed['type']}", 1.5)

            elif key == ord('r'):
                if self._reset_armed:
                    n = len(self.annotations)
                    self.annotations = []
                    self.dirty = True
                    self._reset_armed = False
                    self._status(f"Cleared {n} annotation(s)", 2)
                else:
                    self._reset_armed = True
                    self._status(
                        "Press [r] again to clear ALL annotations for this image", 3
                    )

            elif key == ord('s'):
                self.save_labels()

            elif key == ord('n'):
                self.save_labels()
                if self.idx < self.n - 1:
                    self.idx += 1
                    self.load_image()
                else:
                    self._status("Already at last image", 2)

            elif key == ord('p'):
                self.save_labels()
                if self.idx > 0:
                    self.idx -= 1
                    self.load_image()
                else:
                    self._status("Already at first image", 2)

            elif key in (ord('q'), 27):   # q or Esc
                if self.dirty:
                    self.save_labels()
                break

        cv2.destroyAllWindows()
        print("Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    img_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("calibration_labels/images")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("calibration_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    print(f"Images:    {img_dir}")
    print(f"Labels:    {out_dir}/labels/")
    print(f"Annotated: {out_dir}/annotated/")
    print(f"Training:  {out_dir}/images/")
    print(f"Controls: [c]ircle / [o]ther  drag=draw  right-click=delete  "
          f"[z]undo  [r]eset  [n]ext  [p]rev  [s]ave  [q]uit\n")

    CalibLabeler(img_dir, out_dir).run()


if __name__ == "__main__":
    main()
