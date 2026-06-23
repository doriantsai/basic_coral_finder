#!/usr/bin/env python3
"""
classifier_utils.py
Shared feature extraction, data collection, and training utilities.

Not run directly — imported by train_white_classifier.py and
train_coral_classifier.py.
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

FEATURE_NAMES = ["B", "G", "R", "H", "S", "V", "L", "a", "b"]
MAX_PX_ROI   = 8_000   # pixels sampled per ROI
N_TREES      = 200
MAX_DEPTH    = 20
TEST_FRAC    = 0.20
RANDOM_SEED  = 42


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(img_bgr):
    """(N, 9) float32: B G R  H S V  L a b — all normalised to [0, 1]."""
    bgr = img_bgr.reshape(-1, 3).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 180.0
    hsv[:, 1:] /= 255.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    lab[:, 0] /= 100.0
    lab[:, 1:] = (lab[:, 1:] + 128.0) / 255.0

    return np.hstack([bgr, hsv, lab])


# ── Data collection ───────────────────────────────────────────────────────────

def collect_training_data(label_dir, img_dir, rng, positive, negative):
    """
    Read all labeled JSON files and extract pixel features + binary labels.

    positive — label strings mapped to class 1  (e.g. ["white"])
    negative — label strings mapped to class 0  (e.g. ["other", "coral"])

    ROIs whose label is in neither list are silently skipped, so the same JSON
    files can be shared by both classifiers regardless of which labels are present.

    Returns (X, y) float32/int8 arrays, or (None, None) if nothing usable.
    """
    all_labels = set(positive) | set(negative)
    X_parts, y_parts = [], []

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        print(f"  [error] no JSON files in {label_dir}")
        return None, None

    for lp in label_files:
        with open(lp) as f:
            data = json.load(f)

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

        H, W = img.shape[:2]
        rois = [r for r in data.get("rois", []) if r.get("label") in all_labels]
        if not rois:
            print(f"  [skip] {lp.stem} — no matching labeled ROIs")
            continue

        counts = {lbl: sum(1 for r in rois if r["label"] == lbl)
                  for lbl in sorted(all_labels)}
        print(f"  {lp.stem}:  "
              + "  ".join(f"{lbl}={n}" for lbl, n in counts.items() if n > 0))

        for roi in rois:
            y_val = 1 if roi["label"] in positive else 0
            x  = max(0, int(roi["x"]));        y  = max(0, int(roi["y"]))
            x2 = min(W, x + int(roi["width"])); y2 = min(H, y + int(roi["height"]))
            patch = img[y:y2, x:x2]
            if patch.size == 0:
                continue
            feats = extract_features(patch)
            if len(feats) > MAX_PX_ROI:
                idx   = rng.choice(len(feats), MAX_PX_ROI, replace=False)
                feats = feats[idx]
            X_parts.append(feats)
            y_parts.append(np.full(len(feats), y_val, dtype=np.int8))

    if not X_parts:
        return None, None
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Training + evaluation ─────────────────────────────────────────────────────

def train_and_report(X, y, positive_name, negative_name):
    """
    Stratified train/val split, fit RandomForest, print report + importances.

    Returns (clf, report_str, importances, (X_tr, y_tr, X_te, y_te)).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                 random_state=RANDOM_SEED)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"\nTraining RandomForest  "
          f"({N_TREES} trees, max_depth={MAX_DEPTH}, class_weight='balanced') ...")
    clf = RandomForestClassifier(
        n_estimators = N_TREES,
        max_depth    = MAX_DEPTH,
        class_weight = "balanced",
        n_jobs       = -1,
        random_state = RANDOM_SEED,
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred,
                                   target_names=[negative_name, positive_name],
                                   digits=4)
    print("\nValidation report (held-out 20% of pixels):")
    print(report)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name:>3}  {imp:.4f}  {'#' * int(imp * 40)}")

    return clf, report, importances, (X_tr, y_tr, X_te, y_te)


def save_model(clf, model_dir, model_stem,
               X_tr, y_tr, X_te, y_te, report, importances):
    """Save .joblib + _info.txt; return (model_path, info_path)."""
    model_path = model_dir / f"{model_stem}.joblib"
    info_path  = model_dir / f"{model_stem}_info.txt"

    joblib.dump(clf, model_path)

    info = (
        f"Model:          {model_stem}\n"
        f"Format:         joblib (scikit-learn RandomForestClassifier)\n"
        f"n_estimators:   {N_TREES}\n"
        f"max_depth:      {MAX_DEPTH}\n"
        f"class_weight:   balanced\n"
        f"Features (9):   {', '.join(FEATURE_NAMES)}\n"
        f"Training pixels:{len(X_tr):>10,}\n"
        f"Held-out pixels:{len(X_te):>10,}\n\n"
        f"Validation report:\n{report}\n"
        f"Feature importances:\n"
        + "".join(f"  {n}: {v:.4f}\n" for n, v in importances)
    )
    info_path.write_text(info)

    print(f"\nModel saved ->  {model_path}")
    print(f"Info saved  ->  {info_path}")
    return model_path, info_path


# ── Top-level pipeline (called from the thin wrapper scripts) ─────────────────

def main_train(positive_labels, negative_labels, model_stem_suffix, train_dir):
    """
    Complete training pipeline for one binary classifier.

    positive_labels   — ROI label strings → class 1  (e.g. ["white"])
    negative_labels   — ROI label strings → class 0  (e.g. ["other", "coral"])
    model_stem_suffix — filename component (e.g. "WhiteTabClassifier")
    train_dir         — pathlib.Path to the train/ directory
    """
    label_dir = train_dir / "labels"
    img_dir   = train_dir / "images"
    model_dir = train_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    pos_name = "/".join(positive_labels)
    neg_name = "/".join(negative_labels)

    print(f"Reading labels from {label_dir}/")
    print(f"  positive (class 1): {pos_name}")
    print(f"  negative (class 0): {neg_name}\n")

    X, y = collect_training_data(label_dir, img_dir, rng,
                                 positive_labels, negative_labels)
    if X is None:
        print("\nNo training data found.  "
              "Add labeled ROIs to the JSON files and re-run.")
        sys.exit(1)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\nDataset: {len(X):,} pixels total  "
          f"({n_pos:,} {pos_name},  {n_neg:,} {neg_name})")

    if n_pos == 0 or n_neg == 0:
        print(f"\n[error] Both classes must have at least one labeled ROI.")
        sys.exit(1)

    clf, report, importances, (X_tr, y_tr, X_te, y_te) = train_and_report(
        X, y, pos_name, neg_name)

    today      = date.today().strftime("%Y%m%d")
    model_stem = f"{today}_{model_stem_suffix}"
    save_model(clf, model_dir, model_stem,
               X_tr, y_tr, X_te, y_te, report, importances)
