"""
preprocess.py
Preprocesses raw CICIDS2017 CSV files into train/val/test .npy tensors.

Usage:
    python scripts/preprocess.py --input data/raw/ --output data/processed/
    python scripts/preprocess.py --input data/raw/ --output data/processed/ --window 50

Steps:
    1. Load all CSV files from --input directory
    2. Clean inf/NaN values
    3. Encode labels to integer indices
    4. Apply SMOTE to training set to balance class distribution
    5. Fit MinMaxScaler on training features (saved to data/processed/scaler.pkl)
    6. Create sliding windows for LSTM input
    7. Save X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
"""

import argparse
import os
import glob
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_engineering import (
    extract_features, encode_labels, create_sliding_windows,
    clean_dataframe, fit_scaler, CICIDS_FEATURES
)

CLASS_NAMES = [
    "BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris",
    "DoS Slowhttptest", "Heartbleed", "FTP-Patator", "SSH-Patator",
    "Web Attack Brute Force", "Web Attack XSS", "Web Attack Sql Injection",
    "Infiltration", "Bot", "PortScan", "DDoS"
]


def load_all_csvs(raw_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
            f"Place the CSV files in: {raw_dir}"
        )
    logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    dfs = []
    for fpath in csv_files:
        logger.info(f"  Loading {os.path.basename(fpath)}...")
        df = pd.read_csv(fpath, low_memory=False)
        # Standardise column names: strip spaces
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined):,}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="data/raw/",
                        help="Directory with raw CICIDS2017 CSV files")
    parser.add_argument("--output", type=str, default="data/processed/",
                        help="Output directory for processed .npy files")
    parser.add_argument("--window", type=int, default=50,
                        help="Sliding window size for LSTM sequences")
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--no_smote", action="store_true",
                        help="Disable SMOTE oversampling")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    scaler_path = os.path.join(args.output, "scaler.pkl")

    # ── 1. Load ─────────────────────────────────────────────────────────
    df = load_all_csvs(args.input)

    # ── 2. Labels ───────────────────────────────────────────────────────
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    logger.info(f"Label column: '{label_col}'")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}")

    y_all = encode_labels(df[label_col], CLASS_NAMES)
    df = clean_dataframe(df)

    # ── 3. Features ──────────────────────────────────────────────────────
    available_features = [f for f in CICIDS_FEATURES if f in df.columns]
    logger.info(f"Using {len(available_features)}/{len(CICIDS_FEATURES)} features")
    X_raw = df[available_features].values.astype(np.float32)

    # ── 4. Train/Val/Test Split (stratified, before windowing) ──────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_all, test_size=args.test_split, stratify=y_all, random_state=42
    )
    val_size_adjusted = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    logger.info(f"Split — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    # ── 5. Fit Scaler on Training Data ───────────────────────────────────
    scaler = fit_scaler(X_train, scaler_path)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── 6. SMOTE on Training Set ─────────────────────────────────────────
    if not args.no_smote:
        try:
            from imblearn.over_sampling import SMOTE
            logger.info("Applying SMOTE to training set...")
            smote = SMOTE(sampling_strategy="minority", random_state=42, n_jobs=-1)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE — train size: {len(X_train):,}")
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            logger.warning("Install with: pip install imbalanced-learn")

    # ── 7. Sliding Windows ───────────────────────────────────────────────
    logger.info(f"Creating sliding windows (size={args.window})...")
    X_train_w, y_train_w = create_sliding_windows(X_train, y_train, args.window)
    X_val_w,   y_val_w   = create_sliding_windows(X_val,   y_val,   args.window)
    X_test_w,  y_test_w  = create_sliding_windows(X_test,  y_test,  args.window)

    logger.info(f"Windowed shapes — train: {X_train_w.shape} | val: {X_val_w.shape} | test: {X_test_w.shape}")

    # ── 8. Save ──────────────────────────────────────────────────────────
    np.save(os.path.join(args.output, "X_train.npy"), X_train_w)
    np.save(os.path.join(args.output, "y_train.npy"), y_train_w)
    np.save(os.path.join(args.output, "X_val.npy"),   X_val_w)
    np.save(os.path.join(args.output, "y_val.npy"),   y_val_w)
    np.save(os.path.join(args.output, "X_test.npy"),  X_test_w)
    np.save(os.path.join(args.output, "y_test.npy"),  y_test_w)

    logger.info(f"All arrays saved to {args.output}")
    logger.info("Preprocessing complete. Run: python train.py")


if __name__ == "__main__":
    main()
