from pydantic import BaseModel
from datetime import datetime


class Transaction(BaseModel):
    transaction_id: str
    account_id: str
    receiver_id: str
    amount: float
    merchant_category: str
    timestamp: datetime
    is_fraud: int
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class APITransaction(BaseModel):
    transaction_id: str
    account_id: str
    receiver_id: str
    amount: float
    merchant_category: str
    timestamp: datetime
    txn_count_24h: int
    amount_sum_24h: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
