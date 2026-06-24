from pathlib import Path

import pandas as pd


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve original row order
    df = df.reset_index(drop=False)

    # Sort for rolling calculations
    df = df.sort_values(["account_id", "timestamp"])

    rolling = df.groupby("account_id").rolling(
        window="24h",
        on="timestamp",
        closed="left",
    )

    df["txn_count_24h"] = rolling["amount"].count().fillna(0).astype(int).to_numpy()

    df["amount_sum_24h"] = rolling["amount"].sum().fillna(0).to_numpy()

    # Restore original chronological order
    df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_path = project_root / "data" / "raw" / "historical.parquet"

    output_path = project_root / "data" / "processed" / "historical_features.parquet"

    df = pd.read_parquet(input_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    original_transaction_ids = df["transaction_id"].copy()

    featured_df = build_rolling_features(df)

    assert featured_df["transaction_id"].equals(original_transaction_ids), (
        "Row order changed during feature generation."
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    featured_df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved {len(featured_df):,} rows to {output_path}")
