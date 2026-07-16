import pandas as pd

from src.data_prep import chronological_split, prepare_datasets


def make_df(n=10):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "account_id": [f"acc_{i}" for i in range(n)],
            "transaction_id": [str(i) for i in range(n)],
            "merchant_category": ["PAYMENT", "TRANSFER"] * (n // 2),
            "amount": [float(i) for i in range(n)],
            "txn_count_24h": [0] * n,
            "amount_sum_24h": [0.0] * n,
            "is_fraud": [0, 1] * (n // 2),
        }
    )


def test_chronological_split_keeps_time_order():
    df = make_df(10)
    train, test = chronological_split(df, 0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_labels_and_ids_never_reach_features():
    df = make_df(10)
    train, test = chronological_split(df, 0.8)
    x_train, y_train, x_test, _, _ = prepare_datasets(train, test)

    for leaked in ("is_fraud", "transaction_id", "account_id", "timestamp"):
        assert leaked not in x_train.columns
        assert leaked not in x_test.columns

    assert y_train.tolist() == train["is_fraud"].tolist()


def test_unknown_category_at_test_time_is_ignored():
    df = make_df(10)
    train, test = chronological_split(df, 0.8)
    test = test.assign(merchant_category="NEVER_SEEN")

    x_train, _, x_test, _, _ = prepare_datasets(train, test)

    onehot_cols = [c for c in x_test.columns if c.startswith("merchant_category_")]
    assert x_train.columns.equals(x_test.columns)
    assert (x_test[onehot_cols] == 0).all().all()
