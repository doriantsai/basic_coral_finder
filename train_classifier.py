#!/usr/bin/env python3
"""
train_classifier.py
Train a random forest pixel classifier to distinguish white square frame
material from everything else (coral, background, calibration discs).

Usage:
    python3 train_classifier.py [train_dir]

Reads:   train/labels/*.json  (ROIs must have "label": "white" or "label": "other")
Writes:  train/models/YYYYMMDD_WhiteTabClassifier.joblib
         train/models/YYYYMMDD_WhiteTabClassifier_info.txt

NOTE — why .joblib, not .csv:
    A random forest is a collection of decision trees, each containing
    hundreds of branching rules stored in a binary tree structure.  This
    cannot be expressed as tabular rows and columns.  The .joblib format
    is scikit-learn's recommended binary serialisation for fitted models.

Features per pixel (9 total, all normalised to [0, 1]):
    B  G  R          raw BGR channel values
    H  S  V          HSV colour space
    L  a  b          CIE L*a*b* colour space

Multi-colourspace features give the classifier redundant representations of
colour, making it robust to variation in illumination and camera settings.
"""

import cv2
import json
import sys
import numpy as np
from datetime import date
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DIR   = Path("train")
MAX_PX_ROI  = 8_000    # pixels sampled per ROI (keeps RAM usage manageable)
N_TREES     = 200      # number of decision trees in the forest
MAX_DEPTH   = 20       # max tree depth (None = unlimited; 20 avoids overfitting)
TEST_FRAC   = 0.20     # fraction of pixels held out for validation report
RANDOM_SEED = 42

FEATURE_NAMES = ["B", "G", "R", "H", "S", "V", "L", "a", "b"]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(img_bgr):
    """
    Return an (N, 9) float32 feature matrix for every pixel in img_bgr.
    All values are normalised to [0, 1].
    """
    bgr = img_bgr.reshape(-1, 3).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 180.0    # H: 0-180 -> 0-1
    hsv[:, 1:] /= 255.0   # S, V: 0-255 -> 0-1

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    lab[:, 0] /= 100.0                      # L: 0-100 -> 0-1
    lab[:, 1:] = (lab[:, 1:] + 128) / 255.0 # a, b: -128..127 -> 0-1

    return np.hstack([bgr, hsv, lab])


# ── Data collection ───────────────────────────────────────────────────────────

def collect_training_data(label_dir, img_dir, rng):
    """
    Read all labeled JSON files and extract pixel features + labels.
    Returns (X, y) arrays, or (None, None) if no usable data found.
    """
    X_parts, y_parts = [], []

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        print(f"  [error] no JSON files in {label_dir}")
        return None, None

    for lp in label_files:
        with open(lp) as f:
            data = json.load(f)

        # Resolve image path (stored path or fallback to img_dir)
        img_path = Path(data.get("image", ""))
        if not img_path.exists():
            img_path = img_dir / img_path.name
        if not img_path.exists():
            print(f"  [skip] image not found for {lp.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] cannot read {img_path}")
            continue

        H, W   = img.shape[:2]
        rois   = [r for r in data.get("rois", []) if "label" in r]
        if not rois:
            print(f"  [skip] {lp.stem} — no labeled ROIs yet")
            continue

        n_w = sum(1 for r in rois if r["label"] == "white")
        n_o = sum(1 for r in rois if r["label"] == "other")
        print(f"  {lp.stem}:  {n_w} white ROI(s),  {n_o} other ROI(s)")

        for roi in rois:
            label  = 1 if roi["label"] == "white" else 0
            x  = max(0, int(roi["x"]));       y  = max(0, int(roi["y"]))
            x2 = min(W, x + int(roi["width"])); y2 = min(H, y + int(roi["height"]))
            patch  = img[y:y2, x:x2]
            if patch.size == 0:
                continue

            feats = extract_features(patch)          # (N_pixels, 9)

            # Sub-sample large ROIs so no single box dominates
            if len(feats) > MAX_PX_ROI:
                idx   = rng.choice(len(feats), MAX_PX_ROI, replace=False)
                feats = feats[idx]

            X_parts.append(feats)
            y_parts.append(np.full(len(feats), label, dtype=np.int8))

    if not X_parts:
        return None, None

    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_DIR
    label_dir = train_dir / "labels"
    img_dir   = train_dir / "images"
    model_dir = train_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Collect pixel features from all labeled ROIs ────────────────────────
    print(f"Reading labels from {label_dir}/\n")
    X, y = collect_training_data(label_dir, img_dir, rng)

    if X is None:
        print("\nNo training data found.  "
              "Add 'label' fields to the JSON files in train/labels/ and re-run.")
        sys.exit(1)

    n_white = int((y == 1).sum())
    n_other = int((y == 0).sum())
    print(f"\nDataset: {len(X):,} pixels total  "
          f"({n_white:,} white,  {n_other:,} other)")

    if n_white == 0 or n_other == 0:
        print("\n[error] Both classes ('white' and 'other') must have at least one "
              "labeled ROI.  Add 'other' boxes to the JSON files and re-run.")
        sys.exit(1)

    # ── 2. Train / validation split ────────────────────────────────────────────
    sss   = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                   random_state=RANDOM_SEED)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print(f"\nTraining RandomForest  "
          f"({N_TREES} trees, max_depth={MAX_DEPTH}, class_weight='balanced') ...")
    clf = RandomForestClassifier(
        n_estimators   = N_TREES,
        max_depth      = MAX_DEPTH,
        class_weight   = "balanced",   # handles unequal white/other sample counts
        n_jobs         = -1,           # use all CPU cores
        random_state   = RANDOM_SEED,
    )
    clf.fit(X_tr, y_tr)

    # ── 4. Validation report ───────────────────────────────────────────────────
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred,
                                   target_names=["other", "white"],
                                   digits=4)
    print("\nValidation report (held-out 20% of pixels):")
    print(report)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    print("Feature importances:")
    for name, imp in importances:
        bar = "#" * int(imp * 40)
        print(f"  {name:>3}  {imp:.4f}  {bar}")

    # ── 5. Save model ──────────────────────────────────────────────────────────
    today      = date.today().strftime("%Y%m%d")
    model_stem = f"{today}_WhiteTabClassifier"
    model_path = model_dir / f"{model_stem}.joblib"
    info_path  = model_dir / f"{model_stem}_info.txt"

    joblib.dump(clf, model_path)

    info = (
        f"Model:          {model_stem}\n"
        f"Date:           {today}\n"
        f"Format:         joblib (scikit-learn RandomForestClassifier)\n"
        f"n_estimators:   {N_TREES}\n"
        f"max_depth:      {MAX_DEPTH}\n"
        f"class_weight:   balanced\n"
        f"Features (9):   {', '.join(FEATURE_NAMES)}\n"
        f"Training pixels:{len(X_tr):>10,}  (white: {int((y_tr==1).sum()):,},"
        f"  other: {int((y_tr==0).sum()):,})\n"
        f"Held-out pixels:{len(X_te):>10,}\n\n"
        f"Validation report:\n{report}\n"
        f"Feature importances:\n"
        + "".join(f"  {n}: {v:.4f}\n" for n, v in importances)
    )
    info_path.write_text(info)

    print(f"\nModel saved ->  {model_path}")
    print(f"Info saved  ->  {info_path}")


if __name__ == "__main__":
    main()
