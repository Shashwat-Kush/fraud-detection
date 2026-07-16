from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import APITransaction, Transaction

KAFKA_PAYLOAD = {
    "transaction_id": "t1",
    "account_id": "A",
    "receiver_id": "B",
    "amount": 10.5,
    "merchant_category": "PAYMENT",
    "timestamp": "2024-01-01T00:00:00",
    "is_fraud": 0,
    "oldbalanceOrg": 100.0,
    "newbalanceOrig": 89.5,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 10.5,
}


def test_transaction_parses_kafka_payload():
    txn = Transaction(**KAFKA_PAYLOAD)

    assert isinstance(txn.timestamp, datetime)
    assert txn.amount == 10.5


def test_api_transaction_requires_rolling_features():
    payload = {k: v for k, v in KAFKA_PAYLOAD.items() if k != "is_fraud"}

    with pytest.raises(ValidationError):
        APITransaction(**payload)

    txn = APITransaction(**payload, txn_count_24h=3, amount_sum_24h=42.0)
    assert txn.txn_count_24h == 3
