import pandas as pd

from src.features import build_rolling_features


def make_df(rows):
    df = pd.DataFrame(rows, columns=["account_id", "timestamp", "amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_current_transaction_excluded_from_own_window():
    # closed="left" must exclude the current transaction, otherwise the
    # feature leaks the target row into itself.
    df = make_df(
        [
            ("A", "2024-01-01 00:00", 100.0),
            ("A", "2024-01-01 01:00", 50.0),
        ]
    )

    out = build_rolling_features(df)

    assert out["txn_count_24h"].tolist() == [0, 1]
    assert out["amount_sum_24h"].tolist() == [0.0, 100.0]


def test_window_expires_after_24_hours():
    df = make_df(
        [
            ("A", "2024-01-01 00:00", 100.0),
            ("A", "2024-01-02 01:00", 50.0),
        ]
    )

    out = build_rolling_features(df)

    assert out["txn_count_24h"].tolist() == [0, 0]


def test_accounts_do_not_share_state():
    df = make_df(
        [
            ("A", "2024-01-01 00:00", 100.0),
            ("B", "2024-01-01 00:30", 999.0),
            ("A", "2024-01-01 01:00", 50.0),
        ]
    )

    out = build_rolling_features(df)

    assert out["txn_count_24h"].tolist() == [0, 0, 1]
    assert out["amount_sum_24h"].tolist() == [0.0, 0.0, 100.0]


def test_original_row_order_preserved():
    df = make_df(
        [
            ("B", "2024-01-01 00:00", 1.0),
            ("A", "2024-01-01 00:10", 2.0),
            ("B", "2024-01-01 00:20", 3.0),
        ]
    )

    out = build_rolling_features(df)

    assert out["account_id"].tolist() == ["B", "A", "B"]
    assert out["amount"].tolist() == [1.0, 2.0, 3.0]
