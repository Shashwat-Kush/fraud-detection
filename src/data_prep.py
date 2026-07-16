"""Shared dataset preparation used by both the tabular and graph pipelines."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Columns that must never reach the model: labels, identifiers and raw
# PaySim columns superseded by engineered features.
LEAKAGE_AND_ID_COLS = [
    "step",
    "isFlaggedFraud",
    "is_fraud",
    "isFraud",
    "transaction_id",
    "account_id",
    "merchant_category",
    "type",
    "nameOrig",
    "nameDest",
    "timestamp",
]


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df chronologically by row position.

    Assumes df is already sorted ascending by timestamp.
    """
    assert 0 < train_frac < 1
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, OneHotEncoder]:
    """One-hot encode merchant_category (fit on train only) and drop
    label/identifier columns."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[["merchant_category"]])

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        cat = pd.DataFrame(
            encoder.transform(df[["merchant_category"]]),
            columns=encoder.get_feature_names_out(),
            index=df.index,
        )
        return pd.concat([df, cat], axis=1).drop(
            columns=LEAKAGE_AND_ID_COLS, errors="ignore"
        )

    return (
        encode(train_df),
        train_df["is_fraud"],
        encode(test_df),
        test_df["is_fraud"],
        encoder,
    )
