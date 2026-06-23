#!/usr/bin/env python3
"""
bbox_labeler.py
Interactive GUI for drawing bounding-box training labels on images.

Usage:
    python3 bbox_labeler.py [train_dir]

Controls:
    w              set active class to WHITE  (green boxes)
    o              set active class to OTHER  (red boxes)
    left-drag      draw a new bounding box in the active class
    right-click    delete the box under the cursor
    z              undo last drawn box
    n              save and go to next image
    p              save and go to previous image
    s              save current image's labels to JSON
    r              clear ALL boxes for this image
    q / Esc        save and quit

Labels are written to train/labels/<stem>.json in the same format as
detect_corals_on_white_squares.py output, with a "label" field added per ROI.
Existing boxes from the JSON (e.g. pre-loaded from the detector) are shown
on load and can be edited, deleted, or added to.
"""

import cv2
import json
import sys
import time
import numpy as np
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_DIR    = Path("train")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DISPLAY_W    = 1400   # max display width  (window can be resized by the user)
DISPLAY_H    = 860    # max display height (leaves room for the info bar)
INFO_H       = 52     # height of the top info bar in pixels
MIN_BOX_PX   = 10     # minimum box side length in display pixels

COLORS = {
    "white":   (0, 220, 80),    # green
    "other":   (0, 80, 220),    # red
    "coral":   (0, 165, 255),   # orange
    "preview": (0, 220, 255),   # yellow — box being drawn
}
FILL_ALPHA = 0.18
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ── Labeller class ────────────────────────────────────────────────────────────

class BoxLabeler:

    def __init__(self, img_dir: Path, label_dir: Path):
        self.img_paths  = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.img_paths:
            raise SystemExit(f"No images found in {img_dir}")

        self.label_dir     = label_dir
        self.n             = len(self.img_paths)
        self.idx           = 0
        self.current_class = "white"

        # Boxes for current image: list of dicts matching the JSON ROI format.
        # All coordinates are stored in FULL-RESOLUTION pixels.
        self.boxes: list[dict] = []

        # Mouse drawing state (display-pixel coordinates)
        self.drawing    = False
        self.draw_start = None   # (x, y)
        self.draw_cur   = None   # (x, y)

        # Image / display state
        self.full_img   = None
        self.disp_img   = None   # pre-scaled BGR image shown in the window
        self.scale      = 1.0
        self.disp_w     = DISPLAY_W
        self.disp_h     = DISPLAY_H

        # Status feedback
        self.status_msg  = ""
        self.status_time = 0.0   # time.monotonic() when message was set
        self.dirty       = False  # unsaved changes


    # ── Image / label I/O ─────────────────────────────────────────────────────

    def _label_path(self):
        return self.label_dir / f"{self.img_paths[self.idx].stem}.json"

    def load_image(self):
        p = self.img_paths[self.idx]
        img = cv2.imread(str(p))
        if img is None:
            self._status(f"Cannot read {p.name} — skipping")
            return
        self.full_img = img
        H, W = img.shape[:2]
        self.scale   = min(DISPLAY_W / W, DISPLAY_H / H)
        self.disp_w  = int(W * self.scale)
        self.disp_h  = int(H * self.scale)
        self.disp_img = cv2.resize(img, (self.disp_w, self.disp_h),
                                   interpolation=cv2.INTER_AREA)
        self._load_labels()
        self.drawing    = False
        self.draw_start = None
        self.draw_cur   = None
        self.dirty      = False
        cv2.setWindowTitle("BBox Labeler",
                           f"BBox Labeler  [{self.idx+1}/{self.n}]  {p.name}")

    def _load_labels(self):
        lp = self._label_path()
        self.boxes = []
        if not lp.exists():
            return
        try:
            with open(lp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self._status(f"JSON error in {lp.name}: {e}")
            return
        for r in data.get("rois", []):
            if "label" in r and all(k in r for k in ("x","y","width","height")):
                entry = {k: r[k] for k in ("label","x","y","width","height")}
                for opt in ("row", "col"):
                    if opt in r:
                        entry[opt] = r[opt]
                self.boxes.append(entry)

    def save_labels(self):
        if self.full_img is None:
            return
        lp = self._label_path()
        H, W = self.full_img.shape[:2]
        doc = {
            "image":        str(self.img_paths[self.idx]),
            "image_width":  W,
            "image_height": H,
            "rois":         self.boxes,
        }
        with open(lp, "w") as f:
            json.dump(doc, f, indent=2)
        self.dirty = False
        n_w = sum(1 for b in self.boxes if b["label"] == "white")
        n_o = sum(1 for b in self.boxes if b["label"] == "other")
        self._status(f"Saved  {lp.name}  ({n_w} white, {n_o} other)")


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
        # Offset y by INFO_H because the mouse reports coords inside the window
        # which includes the info bar at the top.
        y -= INFO_H

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
            # Require a minimum drag distance
            if abs(x2 - x1) >= MIN_BOX_PX and abs(y2 - y1) >= MIN_BOX_PX:
                fx1, fy1 = self._to_full(min(x1, x2), min(y1, y2))
                fx2, fy2 = self._to_full(max(x1, x2), max(y1, y2))
                self.boxes.append({
                    "label":  self.current_class,
                    "x":      fx1,  "y":      fy1,
                    "width":  fx2 - fx1,
                    "height": fy2 - fy1,
                })
                self.dirty = True
            self.draw_start = self.draw_cur = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete the topmost box that contains this point
            fx, fy = self._to_full(x, y)
            for i in range(len(self.boxes) - 1, -1, -1):
                b = self.boxes[i]
                if (b["x"] <= fx <= b["x"] + b["width"] and
                        b["y"] <= fy <= b["y"] + b["height"]):
                    removed = self.boxes.pop(i)
                    self.dirty = True
                    self._status(f"Deleted {removed['label']} box")
                    break


    # ── Rendering ─────────────────────────────────────────────────────────────

    def _status(self, msg: str, duration: float = 2.5):
        self.status_msg  = msg
        self.status_time = time.monotonic() + duration

    def _build_frame(self):
        """Return the composited frame (info bar + annotated image)."""
        # ── image layer ───────────────────────────────────────────────────────
        base    = self.disp_img.copy()
        overlay = base.copy()

        for box in self.boxes:
            color = COLORS.get(box["label"], (160, 160, 160))
            bx,  by  = self._to_disp(box["x"],              box["y"])
            bx2, by2 = self._to_disp(box["x"] + box["width"],
                                      box["y"] + box["height"])
            cv2.rectangle(overlay, (bx, by), (bx2, by2), color, -1)

        cv2.addWeighted(overlay, FILL_ALPHA, base, 1 - FILL_ALPHA, 0, base)

        for box in self.boxes:
            color = COLORS.get(box["label"], (160, 160, 160))
            bx,  by  = self._to_disp(box["x"],              box["y"])
            bx2, by2 = self._to_disp(box["x"] + box["width"],
                                      box["y"] + box["height"])
            cv2.rectangle(base, (bx, by), (bx2, by2), color, 2)
            tag = box["label"].upper()
            if "row" in box and "col" in box:
                tag += f"  R{box['row']}C{box['col']}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
            cv2.rectangle(base, (bx, by - th - 6), (bx + tw + 6, by),
                          (20, 20, 20), -1)
            cv2.putText(base, tag, (bx + 3, by - 3),
                        FONT, 0.44, color, 1, cv2.LINE_AA)

        # live preview of box being drawn
        if self.drawing and self.draw_start and self.draw_cur:
            x1, y1 = self.draw_start
            x2, y2 = self.draw_cur
            cv2.rectangle(base,
                          (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)),
                          COLORS["preview"], 2)

        # ── info bar ─────────────────────────────────────────────────────────
        bar = np.full((INFO_H, self.disp_w, 3), 28, dtype=np.uint8)

        # left: image counter + filename + dirty marker
        p     = self.img_paths[self.idx]
        n_w   = sum(1 for b in self.boxes if b["label"] == "white")
        n_o   = sum(1 for b in self.boxes if b["label"] == "other")
        n_c   = sum(1 for b in self.boxes if b["label"] == "coral")
        dirty = " *" if self.dirty else ""
        left  = f"[{self.idx+1}/{self.n}]  {p.name}{dirty}   |   {n_w}W  {n_c}C  {n_o}O"
        cv2.putText(bar, left, (8, 34), FONT, 0.56, (210, 210, 210), 1, cv2.LINE_AA)

        # right: current class label
        cls_col = COLORS[self.current_class]
        cls_txt = f"Class: {self.current_class.upper()}"
        (tw, _), _ = cv2.getTextSize(cls_txt, FONT, 0.65, 2)
        cv2.putText(bar, cls_txt, (self.disp_w - tw - 10, 36),
                    FONT, 0.65, cls_col, 2, cv2.LINE_AA)

        # bottom hint row
        hint = ("[w]hite  [c]oral  [o]ther  |  left-drag=draw  right-click=delete  "
                "|  [z]undo  [r]eset  [n]ext  [p]rev  [s]ave  [q]uit")
        cv2.putText(bar, hint, (8, 16), FONT, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

        # status message (fades after duration)
        if time.monotonic() < self.status_time:
            cv2.putText(bar, self.status_msg,
                        (self.disp_w // 4, 34),
                        FONT, 0.54, (0, 220, 255), 1, cv2.LINE_AA)

        return np.vstack([bar, base])


    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        win = "BBox Labeler"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_W, DISPLAY_H + INFO_H)
        cv2.setMouseCallback(win, self.mouse_cb)

        self.load_image()
        self._status("Ready  —  [w]hite / [o]ther to choose class, then drag to draw", 4)

        while True:
            if self.full_img is None:
                time.sleep(0.02)
                continue

            cv2.imshow(win, self._build_frame())
            key = cv2.waitKey(15) & 0xFF

            if key == ord('w'):
                self.current_class = "white"
                self._status("Active class -> WHITE", 1.5)

            elif key == ord('c'):
                self.current_class = "coral"
                self._status("Active class -> CORAL", 1.5)

            elif key == ord('o'):
                self.current_class = "other"
                self._status("Active class -> OTHER", 1.5)

            elif key == ord('z'):
                if self.boxes:
                    removed = self.boxes.pop()
                    self.dirty = True
                    self._status(f"Undo — removed {removed['label']} box", 1.5)

            elif key == ord('r'):
                # Reset all boxes (simple keypress confirmation — press r twice)
                if getattr(self, "_reset_armed", False):
                    n = len(self.boxes)
                    self.boxes = []
                    self.dirty = True
                    self._reset_armed = False
                    self._status(f"Cleared {n} box(es)", 2)
                else:
                    self._reset_armed = True
                    self._status("Press [r] again to clear ALL boxes for this image", 3)

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
    train_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_DIR
    img_dir   = train_dir / "images"
    label_dir = train_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    print(f"Images:  {img_dir}")
    print(f"Labels:  {label_dir}")
    print(f"Controls: [w]hite / [c]oral / [o]ther  left-drag=draw  right-click=delete  "
          f"[z]undo  [n]ext  [p]rev  [s]ave  [q]uit\n")

    BoxLabeler(img_dir, label_dir).run()


if __name__ == "__main__":
    main()
