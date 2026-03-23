"""
src/preprocess.py
=================
Preprocessing pipeline for the Early Fraud Signal System.

IS557 Applied ML — UIUC | Rahul, Anisha, Ananyaa

Usage (run from main/code/):
    python src/preprocess.py
    — or —
    from src.preprocess import full_pipeline
    X_train, X_test, y_train, y_test = full_pipeline("../Base.csv")
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "device_os",
    "source",
]

SCALE_COLS = [
    "days_since_request",
    "session_length_in_minutes",
    "intended_balcon_amount",
]

TARGET = "fraud_bool"


# ---------------------------------------------------------------------------
# 1. load_data
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load Base.csv from disk and print a diagnostic summary.

    Parameters
    ----------
    filepath : str
        Path to Base.csv (e.g. "../Base.csv" when running from main/code/).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all 32 original features.
    """
    print(f"[load_data] Loading: {filepath}")
    df = pd.read_csv(filepath)

    print(f"  Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("\n  Dtypes:")
    print(df.dtypes.value_counts().to_string())

    fraud_counts = df[TARGET].value_counts()
    fraud_rate = df[TARGET].mean() * 100
    print(f"\n  Class distribution ({TARGET}):")
    print(f"    Legitimate (0): {fraud_counts[0]:,}")
    print(f"    Fraud      (1): {fraud_counts[1]:,}")
    print(f"    Fraud rate    : {fraud_rate:.2f}%")

    return df


# ---------------------------------------------------------------------------
# 2. audit_missing_values
# ---------------------------------------------------------------------------

def audit_missing_values(df: pd.DataFrame) -> dict:
    """
    Audit NaN values and sentinel -1 values in known sentinel columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially processed dataframe.

    Returns
    -------
    dict
        Summary with keys 'nan_counts' and 'sentinel_counts'.
    """
    print("\n[audit_missing_values]")

    # --- NaN audit ---
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        print("  No NaN values found in any column.")
    else:
        print(f"  Columns with NaN ({len(nan_cols)} found):")
        print(nan_cols.to_string())

    # --- Sentinel -1 audit ---
    sentinel_counts = {}
    print(f"\n  Sentinel value (-1) audit:")
    for col in SENTINEL_COLS:
        if col in df.columns:
            count = (df[col] == -1).sum()
            pct = count / len(df) * 100
            sentinel_counts[col] = count
            print(f"    {col}: {count:,} rows ({pct:.1f}%) have -1")
        else:
            print(f"    {col}: NOT FOUND in dataframe")

    summary = {
        "nan_counts": nan_counts.to_dict(),
        "sentinel_counts": sentinel_counts,
    }
    return summary


# ---------------------------------------------------------------------------
# 3. engineer_features
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw columns.

    New features created:
        has_prev_address       - 1 if prev_address_months_count != -1
        is_dirty_device        - 1 if device_fraud_count > 0
        velocity_ratio         - velocity_6h / (velocity_4w + 1)
        credit_to_income_ratio - proposed_credit_limit / (income + 1)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after loading (sentinel values still present).

    Returns
    -------
    pd.DataFrame
        Dataframe with 4 new engineered columns appended.
    """
    print("\n[engineer_features]")
    df = df.copy()

    df["has_prev_address"] = (df["prev_address_months_count"] != -1).astype(int)
    df["is_dirty_device"] = (df["device_fraud_count"] > 0).astype(int)
    df["velocity_ratio"] = df["velocity_6h"] / (df["velocity_4w"] + 1)
    df["credit_to_income_ratio"] = df["proposed_credit_limit"] / (df["income"] + 1)

    new_cols = [
        "has_prev_address",
        "is_dirty_device",
        "velocity_ratio",
        "credit_to_income_ratio",
    ]
    for col in new_cols:
        print(f"  Created '{col}': mean={df[col].mean():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

    return df


# ---------------------------------------------------------------------------
# 4. encode_categoricals
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns and drop the originals.

    Columns encoded: payment_type, employment_status, housing_status,
                     device_os, source.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw categorical columns present.

    Returns
    -------
    pd.DataFrame
        Dataframe with categorical columns replaced by one-hot dummies.
    """
    print("\n[encode_categoricals]")
    df = df.copy()

    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    cols_missing = [c for c in CATEGORICAL_COLS if c not in df.columns]

    if cols_missing:
        print(f"  WARNING: columns not found, skipping: {cols_missing}")

    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False, dtype=int)
    after_cols = df.shape[1]

    print(f"  Encoded {len(cols_present)} categorical columns")
    print(f"  Total columns: {before_cols} -> {after_cols}")

    return df


# ---------------------------------------------------------------------------
# 5. scale_features
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: MinMaxScaler = None,
) -> tuple:
    """
    MinMax-scale the three continuous skewed columns.

    Columns scaled: days_since_request, session_length_in_minutes,
                    intended_balcon_amount.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with SCALE_COLS present.
    fit : bool
        If True, fit a new scaler on df and return it.
        If False, apply the provided scaler (transform only).
    scaler : MinMaxScaler or None
        Required when fit=False. Ignored when fit=True.

    Returns
    -------
    tuple
        (transformed dataframe, fitted scaler)
    """
    print("\n[scale_features]")
    df = df.copy()

    cols_present = [c for c in SCALE_COLS if c in df.columns]

    if fit:
        scaler = MinMaxScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
        print(f"  Fitted and scaled: {cols_present}")
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        df[cols_present] = scaler.transform(df[cols_present])
        print(f"  Transformed (no fit): {cols_present}")

    return df, scaler


# ---------------------------------------------------------------------------
# 6. temporal_split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple:
    """
    Split data temporally: train on months 0-5, test on months 6-7.

    The 'month' column is dropped after splitting.
    Target column (fraud_bool) is separated from features.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed dataframe including the 'month' column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) as pd.DataFrames / pd.Series.
    """
    print("\n[temporal_split]")

    train_df = df[df["month"] <= 5].copy()
    test_df = df[df["month"] >= 6].copy()

    # Drop month column after split
    train_df = train_df.drop(columns=["month"])
    test_df = test_df.drop(columns=["month"])

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    train_fraud_rate = y_train.mean() * 100
    test_fraud_rate = y_test.mean() * 100

    print(f"  Train set : {X_train.shape[0]:,} rows | fraud rate: {train_fraud_rate:.2f}%")
    print(f"  Test  set : {X_test.shape[0]:,} rows | fraud rate: {test_fraud_rate:.2f}%")
    print(f"  Features  : {X_train.shape[1]} columns")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 7. full_pipeline
# ---------------------------------------------------------------------------

def full_pipeline(filepath: str) -> tuple:
    """
    Run the complete preprocessing pipeline end-to-end.

    Steps (in order):
        1. load_data
        2. audit_missing_values
        3. engineer_features
        4. encode_categoricals
        5. temporal_split  (split BEFORE scaling to prevent data leakage)
        6. scale_features  (fit on train only, transform both splits)
        7. Save scaler  -> models/scaler.pkl
        8. Save splits  -> outputs/X_train.csv, X_test.csv,
                                   outputs/y_train.csv, y_test.csv

    Parameters
    ----------
    filepath : str
        Path to Base.csv.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("  FRAUD DETECTION — PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1 — Load
    df = load_data(filepath)

    # Step 2 — Audit
    audit_missing_values(df)

    # Step 3 — Feature engineering
    df = engineer_features(df)

    # Step 4 — Encode categoricals
    df = encode_categoricals(df)

    # Step 5 — Temporal split (BEFORE scaling to prevent leakage)
    X_train, X_test, y_train, y_test = temporal_split(df)

    # Step 6 — Scale (fit on train only, transform both)
    X_train, scaler = scale_features(X_train, fit=True)
    X_test, _ = scale_features(X_test, fit=False, scaler=scaler)

    # Step 7 — Save scaler
    os.makedirs("models", exist_ok=True)
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n[full_pipeline] Scaler saved -> {scaler_path}")

    # Step 8 — Save splits
    os.makedirs("outputs", exist_ok=True)
    X_train.to_csv("outputs/X_train.csv", index=False)
    X_test.to_csv("outputs/X_test.csv", index=False)
    y_train.to_csv("outputs/y_train.csv", index=False)
    y_test.to_csv("outputs/y_test.csv", index=False)
    print("[full_pipeline] Splits saved -> outputs/")

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape} | fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  y_test  : {y_test.shape} | fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run from main/code/: python src/preprocess.py
    X_train, X_test, y_train, y_test = full_pipeline("../Base.csv")
