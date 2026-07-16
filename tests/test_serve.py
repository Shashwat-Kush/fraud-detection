import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

import config
from serve import RECEIVER_EMB_COLS, SENDER_EMB_COLS, build_features

TRANSACTION = {
    "transaction_id": "t1",
    "account_id": "A",
    "receiver_id": "UNKNOWN",
    "amount": 10.5,
    "merchant_category": "PAYMENT",
    "txn_count_24h": 2,
    "amount_sum_24h": 20.0,
    "oldbalanceOrg": 100.0,
    "newbalanceOrig": 89.5,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 10.5,
}


@pytest.fixture
def encoder():
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(pd.DataFrame({"merchant_category": ["PAYMENT", "TRANSFER"]}))
    return enc


@pytest.fixture
def feature_cols(encoder):
    return (
        ["amount", "txn_count_24h", "amount_sum_24h"]
        + list(encoder.get_feature_names_out())
        + SENDER_EMB_COLS
        + RECEIVER_EMB_COLS
    )


def test_known_and_unknown_accounts(encoder, feature_cols):
    embeddings = np.ones((1, config.EMBEDDING_DIM), dtype=np.float32)

    df = build_features(
        transaction=TRANSACTION,
        encoder=encoder,
        feature_cols=feature_cols,
        account_to_id={"A": 0},
        graph_embeddings=embeddings,
    )

    assert list(df.columns) == feature_cols
    # Known sender gets its embedding, unknown receiver falls back to zeros.
    assert (df[SENDER_EMB_COLS].to_numpy() == 1.0).all()
    assert (df[RECEIVER_EMB_COLS].to_numpy() == 0.0).all()
    assert df.loc[0, "merchant_category_PAYMENT"] == 1.0


def test_missing_feature_column_raises(encoder, feature_cols):
    embeddings = np.zeros((1, config.EMBEDDING_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="Missing required feature columns"):
        build_features(
            transaction=TRANSACTION,
            encoder=encoder,
            feature_cols=feature_cols + ["not_a_real_feature"],
            account_to_id={},
            graph_embeddings=embeddings,
        )
