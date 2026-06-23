#!/usr/bin/env python3
"""
train_white_classifier.py
Train a RandomForest to identify white square frame pixels (class 1) against
all other material — coral and background (class 0).

Usage:
    python3 train_white_classifier.py [train_dir]

Reads:   train/labels/*.json  (ROIs labeled "white", "other", "coral")
Writes:  train/models/YYYYMMDD_WhiteTabClassifier.joblib
         train/models/YYYYMMDD_WhiteTabClassifier_info.txt
"""

import sys
from pathlib import Path
from classifier_utils import main_train

if __name__ == "__main__":
    main_train(
        positive_labels   = ["white"],
        negative_labels   = ["other", "coral"],
        model_stem_suffix = "WhiteTabClassifier",
        train_dir         = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train"),
    )
