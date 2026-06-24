import random
from datetime import datetime, timedelta
from src.schemas import Transaction
from dataclasses import dataclass


# population ->1000accounts, 8-10 categories, 95% normal and 5 % fraud transaction with large amount, rapid succession, unusual category, odd hours transaction
# creating customer personas first. then assign personas to accounts and then define account profiles and then normal transaction genration, then define trsaction fraud rules and then think about the transaction rate

NORMAL_TXN_PCT = 0.95
FRAUD_TXN_PCT = 0.05
population = 1000

merchant_categories = [
    "Grocery",
    "Restaurant",
    "Fuel",
    "Electronics",
    "Travel",
    "Shopping",
    "Healthcare",
]
personas = ["student", "professional", "traveller", "family"]
persona_distribution = {
    "student": 0.40,
    "professional": 0.35,
    "family": 0.15,
    "traveller": 0.10,
}

persona_metadata = {
    "student": {
        "pref_cats": ["Restaurant", "Shopping"],
        "avg_amount_range": [200, 1500],
        "active_hours": [10, 23],
        "avg_txns_per_day": [2, 8],
    },
    "professional": {
        "pref_cats": ["Fuel", "Restaurant", "Shopping", "Electronics"],
        "avg_amount_range": [500, 8000],
        "active_hours": [8, 22],
        "avg_txns_per_day": [1, 5],
    },
    "traveller": {
        "pref_cats": ["Travel", "Restaurant", "Fuel"],
        "avg_amount_range": [1000, 15000],
        "active_hours": [6, 23],
        "avg_txns_per_day": [3, 10],
    },
    "family": {
        "pref_cats": ["Grocery", "Healthcare", "Shopping"],
        "avg_amount_range": [500, 12000],
        "active_hours": [7, 22],
        "avg_txns_per_day": [2, 7],
    },
}

#!account_id, persona, pref_cats,range,active hours,avg trxs/day, avg_amount transaction


@dataclass
class Account:
    account_id: str
    persona: str
    pref_cats: list[str]
    avg_amount: float
    active_hours: list[int]
    avg_trxns: list[int]


def create_account(
    account_num: int, persona_distribution: dict, persona_metadata: dict
) -> Account:
    samples = random.choices(
        list(persona_distribution.keys()), weights=list(persona_distribution.values())
    )
    selected_persona = samples[0]
    selected_meta = persona_metadata.get(selected_persona)
    amount_range = selected_meta["avg_amount_range"]
    low, high = amount_range[0], amount_range[1]
    mean, sd = (low + high) / 2, (high - low) / 6
    avg_amount = round(random.gauss(mean, sd), 2)
    avg_amount = max(low, min(avg_amount, high))
    pref_cats = selected_meta["pref_cats"]
    active_hours = selected_meta["active_hours"]
    avg_txns_per_day = selected_meta["avg_txns_per_day"]
    account_id = f"ACC{account_num:06d}"
    return Account(
        account_id=account_id,
        persona=selected_persona,
        pref_cats=pref_cats,
        avg_amount=avg_amount,
        active_hours=active_hours,
        avg_trxns=avg_txns_per_day,
    )


def generate_accounts(
    population: int, persona_distribution: dict, persona_metadata: dict
) -> list[Account]:
    accounts = []
    for i in range(1, population + 1):
        accounts.append(create_account(i, persona_distribution, persona_metadata))
    return accounts


def generate_normal_transaction(
    txn_num: int, account: Account, timestamp: datetime
) -> Transaction:
    rand_category = random.choice(account.pref_cats)
    mean, std = account.avg_amount, account.avg_amount * 0.2
    amount = round(max(1, random.gauss(mean, std)), 2)
    txn_id = f"TXN{txn_num:08d}"
    return Transaction(
        transaction_id=txn_id,
        account_id=account.account_id,
        amount=amount,
        merchant_category=rand_category,
        timestamp=timestamp,
        is_fraud=0,
    )


def generate_fraud_transaction(
    txn_num: int,
    account: Account,
    timestamp: datetime,
    merchant_categories: list[str],
    persona_metadata: dict,
) -> Transaction:
    txn_id = f"TXN{txn_num:08d}"
    fraud_types = ["large_amount", "odd_hour", "unusual_category"]
    fraud = random.choice(fraud_types)
    if fraud == "large_amount":
        rand_category = random.choice(account.pref_cats)
        amount = round(account.avg_amount * random.uniform(5, 15), 2)

    elif fraud == "odd_hour":
        amount = round(account.avg_amount, 2)
        rand_category = random.choice(account.pref_cats)
        timestamp = timestamp.replace(
            hour=random.randint(0, 5),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )
    else:
        amount = round(account.avg_amount, 2)
        left_cats = list(
            set(merchant_categories)
            - set(persona_metadata[account.persona]["pref_cats"])
        )
        rand_category = random.choice(left_cats)

    return Transaction(
        transaction_id=txn_id,
        account_id=account.account_id,
        amount=amount,
        merchant_category=rand_category,
        timestamp=timestamp,
        is_fraud=1,
    )


def generate_transaction(
    txn_num: int,
    account: Account,
    timestamp: datetime,
    merchant_categories: list[str],
    persona_metadata: dict,
) -> Transaction:
    probability = [NORMAL_TXN_PCT, FRAUD_TXN_PCT]
    choices = ["normal", "fraud"]

    choice = random.choices(
        choices,
        probability,
    )[0]
    if choice == "normal":
        return generate_normal_transaction(txn_num, account, timestamp)
    else:
        return generate_fraud_transaction(
            txn_num, account, timestamp, merchant_categories, persona_metadata
        )


def transaction_generator(accounts: list[Account]):
    txn_num = 1
    current_time = datetime.now() - timedelta(days=30)

    while True:
        account = random.choice(accounts)
        txn = generate_transaction(
            txn_num,
            account,
            current_time,
            merchant_categories,
            persona_metadata,
        )
        yield txn
        current_time += timedelta(minutes=random.randint(1, 30))
        txn_num += 1
