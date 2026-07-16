import uuid
import pandas as pd
from pathlib import Path

import config

BASE_TIME = pd.Timestamp("2024-01-01 00:00:00")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    raw_path = project_root / "data" / "raw" / "transactions.csv"

    df = pd.read_csv(raw_path)

    df["account_id"] = df["nameOrig"]

    df["merchant_category"] = df["type"]

    df["is_fraud"] = df["isFraud"]

    df["timestamp"] = BASE_TIME + pd.to_timedelta(df["step"], unit="h")

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    split_idx = int(len(df) * config.TRAIN_FRAC)

    historical_df = df.iloc[:split_idx]
    streaming_df = df.iloc[split_idx:]

    historical_df.to_parquet(
        project_root / "data" / "raw" / "historical.parquet",
        index=False,
    )

    streaming_df.to_parquet(
        project_root / "data" / "raw" / "streaming.parquet",
        index=False,
    )

    print(f"Historical rows: {len(historical_df)}")
    print(f"Streaming rows: {len(streaming_df)}")


if __name__ == "__main__":
    main()
