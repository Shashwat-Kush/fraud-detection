from confluent_kafka import Consumer, KafkaError, KafkaException
import json
from src.schemas import Transaction, APITransaction
from pydantic import ValidationError
from collections import defaultdict
from datetime import timedelta
import requests
import time

conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fraud_detection_group",
    "auto.offset.reset": "earliest",
}
account_history = defaultdict(list)
consumer = Consumer(conf)

consumer.subscribe(["transactions"])
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        elif msg.error():
            err_code = msg.error().code()
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            elif err_code == KafkaError.UNKNOWN_TOPIC_OR_PART:
                print("Topic 'transactions' not found. Waiting for producer...")
                time.sleep(1)
                continue
            raise KafkaException(msg.error())
        else:
            try:
                string_data = msg.value().decode("utf-8")
                data = json.loads(string_data)
                txn = Transaction(**data)
                history = account_history[txn.account_id]
                window_start = txn.timestamp - timedelta(hours=24)
                history[:] = [
                    record for record in history if record["timestamp"] >= window_start
                ]
                txn_count_24h = len(history)
                amount_sum_24h = sum(record["amount"] for record in history)
                api_txn = APITransaction(
                    transaction_id=txn.transaction_id,
                    account_id=txn.account_id,
                    amount=txn.amount,
                    merchant_category=txn.merchant_category,
                    timestamp=txn.timestamp,
                    oldbalanceOrg=txn.oldbalanceOrg,
                    newbalanceOrig=txn.newbalanceOrig,
                    oldbalanceDest=txn.oldbalanceDest,
                    newbalanceDest=txn.newbalanceDest,
                    txn_count_24h=txn_count_24h,
                    amount_sum_24h=amount_sum_24h,
                )
                payload = api_txn.model_dump(mode="json")
                history.append(
                    {
                        "timestamp": txn.timestamp,
                        "amount": txn.amount,
                    }
                )
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/score", json=payload, timeout=5
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to score transaction {txn.transaction_id}: {e}")
                    continue
                result = response.json()
                print(
                    f"txn_id={txn.transaction_id} "
                    f"txn_count_24h={txn_count_24h} "
                    f"amount_sum_24h={amount_sum_24h:.2f} "
                    f"is_fraud={result['is_fraud']} "
                    f"fraud_probability={result['fraud_probability']:.4f}"
                )
            except json.JSONDecodeError as e:
                print(f"Invalid Message {e}")
                continue
            except ValidationError as e:
                print(f"Invalid Message {e}")
                continue
except KeyboardInterrupt:
    print("Shutting Down...")
finally:
    consumer.close()
