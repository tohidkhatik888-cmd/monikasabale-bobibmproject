"""
MalwareGuard AI — Model Training Script

Usage
-----
    python scripts/train.py --data-dir /path/to/labeled/pe/files \
                            --classifier both \
                            --output-dir data/models

Directory layout expected:
    data-dir/
        benign/   ← PE files labelled 0
        malware/  ← PE files labelled 1

Outputs saved to output-dir:
    random_forest.joblib
    xgboost.joblib
    scaler.joblib
    feature_names.joblib
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.feature_extractor import extract_features, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk data_dir/benign and data_dir/malware, extract features.
    Returns (X, y).
    """
    X_rows, y_rows = [], []

    for label, subfolder in [(0, "benign"), (1, "malware")]:
        folder = data_dir / subfolder
        if not folder.exists():
            print(f"[WARN] Subfolder not found: {folder}")
            continue
        files = list(folder.iterdir())
        print(f"[INFO] {subfolder}: {len(files)} files")
        for fpath in files:
            if not fpath.is_file():
                continue
            try:
                feat = extract_features(fpath)
                X_rows.append(feat)
                y_rows.append(label)
            except Exception as exc:
                print(f"[SKIP] {fpath.name}: {exc}")

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train) -> XGBClassifier:
    scale_pos_weight = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test, name: str) -> None:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["benign", "malware"]))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MalwareGuard AI classifiers")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root directory with benign/ and malware/ subdirs")
    parser.add_argument("--classifier", default="both", choices=["random_forest", "xgboost", "both"])
    parser.add_argument("--output-dir", default="data/models", type=Path)
    parser.add_argument("--test-size", default=0.2, type=float)
    args = parser.parse_args()

    print("[INFO] Loading dataset …")
    X, y = load_dataset(args.data_dir)
    if len(X) == 0:
        print("[ERROR] No samples loaded. Exiting.")
        sys.exit(1)

    print(f"[INFO] Total samples: {len(X)}  (benign={np.sum(y==0)}, malware={np.sum(y==1)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    joblib.dump(FEATURE_NAMES, args.output_dir / "feature_names.joblib")
    print("[INFO] Scaler + feature names saved.")

    if args.classifier in ("random_forest", "both"):
        print("[INFO] Training Random Forest …")
        rf = train_random_forest(X_train_s, y_train)
        evaluate(rf, X_test_s, y_test, "Random Forest")
        joblib.dump(rf, args.output_dir / "random_forest.joblib")
        print(f"[INFO] Random Forest saved to {args.output_dir / 'random_forest.joblib'}")

    if args.classifier in ("xgboost", "both"):
        print("[INFO] Training XGBoost …")
        xgb = train_xgboost(X_train_s, y_train)
        evaluate(xgb, X_test_s, y_test, "XGBoost")
        joblib.dump(xgb, args.output_dir / "xgboost.joblib")
        print(f"[INFO] XGBoost saved to {args.output_dir / 'xgboost.joblib'}")

    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
