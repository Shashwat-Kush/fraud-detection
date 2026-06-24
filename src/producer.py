from confluent_kafka import Producer
from pathlib import Path
import pandas as pd
import json
import time
import sys


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery Failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] offset {msg.offset()}")


conf = {
    "bootstrap.servers": "localhost:9092",
}

producer = Producer(conf)

project_root = Path(__file__).resolve().parent.parent

streaming_path = project_root / "data" / "raw" / "streaming.parquet"

df = pd.read_parquet(streaming_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

try:
    for _, row in df.iterrows():
        payload = {
            "transaction_id": str(row["transaction_id"]),
            "account_id": str(row["account_id"]),
            "amount": float(row["amount"]),
            "merchant_category": str(row["merchant_category"]),
            "timestamp": row["timestamp"].isoformat(),
            "is_fraud": bool(row["is_fraud"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
        }

        print(f"Produced: {payload['transaction_id']} - {payload['account_id']}")

        try:
            producer.produce(
                topic="transactions",
                key=payload["account_id"].encode("utf-8"),
                value=json.dumps(payload).encode("utf-8"),
                callback=delivery_report,
            )

            producer.poll(0)

        except BufferError:
            producer.poll(1)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nShutting down...")
    sys.exit(0)

finally:
    print("Flushing producer...")
    producer.flush()
