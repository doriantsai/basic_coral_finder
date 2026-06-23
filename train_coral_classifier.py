#!/usr/bin/env python3
"""
train_coral_classifier.py
Train a RandomForest to identify coral pixels (class 1) against all other
material — white frame and background (class 0).

Usage:
    python3 train_coral_classifier.py [train_dir]

Reads:   train/labels/*.json  (ROIs labeled "white", "other", "coral")
         Files without any "coral" ROIs are skipped automatically.
Writes:  train/models/YYYYMMDD_CoralClassifier.joblib
         train/models/YYYYMMDD_CoralClassifier_info.txt
"""

import sys
from pathlib import Path
from classifier_utils import main_train

if __name__ == "__main__":
    main_train(
        positive_labels   = ["coral"],
        negative_labels   = ["white", "other"],
        model_stem_suffix = "CoralClassifier",
        train_dir         = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train"),
    )
