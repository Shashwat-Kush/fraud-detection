from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

import config
from src.data_prep import chronological_split, load_features, prepare_datasets


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "historical_features.parquet"

EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "graph_embeddings.npy"

ACCOUNT_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "account_to_id.pkl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_features.parquet"
TEST_FEATURES_PATH = OUTPUT_DIR / "test_features.parquet"

TRAIN_LABELS_PATH = OUTPUT_DIR / "y_train.npy"
TEST_LABELS_PATH = OUTPUT_DIR / "y_test.npy"

ENCODER_PATH = OUTPUT_DIR / "graph_encoder.pkl"

FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.json"


SENDER_EMB_COLS = [f"sender_emb_{i}" for i in range(config.EMBEDDING_DIM)]

RECEIVER_EMB_COLS = [f"receiver_emb_{i}" for i in range(config.EMBEDDING_DIM)]


def load_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    """Load GraphSAGE embeddings and account index mapping."""

    embeddings = np.load(EMBEDDING_PATH)

    with open(ACCOUNT_MAP_PATH, "rb") as f:
        account_to_id = pickle.load(f)

    return embeddings, account_to_id


def create_embedding_frame(
    account_series: pd.Series,
    embeddings: np.ndarray,
    account_to_id: dict[str, int],
    column_names: list[str],
) -> pd.DataFrame:
    """
    Convert account ids into GraphSAGE embeddings.

    Accounts missing from the graph receive zero embeddings.
    """

    embedding_dim = embeddings.shape[1]

    indices = account_series.map(account_to_id)

    embedding_matrix = np.zeros(
        (len(account_series), embedding_dim),
        dtype=np.float32,
    )

    valid_mask = indices.notna()

    if valid_mask.any():
        embedding_matrix[valid_mask] = embeddings[
            indices[valid_mask].astype(int).to_numpy()
        ]

    return pd.DataFrame(
        embedding_matrix,
        columns=column_names,
        index=account_series.index,
    )


def append_graph_embeddings(
    X: pd.DataFrame,
    original_df: pd.DataFrame,
    embeddings: np.ndarray,
    account_to_id: dict[str, int],
) -> pd.DataFrame:
    """
    Append sender and receiver GraphSAGE embeddings
    to an already-prepared tabular feature matrix.
    """

    sender_df = create_embedding_frame(
        original_df["nameOrig"],
        embeddings,
        account_to_id,
        SENDER_EMB_COLS,
    )

    receiver_df = create_embedding_frame(
        original_df["nameDest"],
        embeddings,
        account_to_id,
        RECEIVER_EMB_COLS,
    )

    return pd.concat(
        [
            X,
            sender_df,
            receiver_df,
        ],
        axis=1,
        copy=False,
    )


def main() -> None:
    print("Loading historical features...")
    df = load_features(FEATURE_PATH)

    train_df, test_df = chronological_split(df, config.TRAIN_FRAC)

    print(f"Train rows : {len(train_df):,}")
    print(f"Test rows  : {len(test_df):,}")

    x_train, y_train, x_test, y_test, encoder = prepare_datasets(
        train_df,
        test_df,
    )

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    print("Appending GraphSAGE embeddings...")

    embeddings, account_to_id = load_embeddings()

    x_train = append_graph_embeddings(
        x_train,
        train_df,
        embeddings,
        account_to_id,
    )

    x_test = append_graph_embeddings(
        x_test,
        test_df,
        embeddings,
        account_to_id,
    )

    assert x_train.columns.equals(x_test.columns), "Train/Test feature mismatch!"

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    assert not x_train.isnull().any().any()
    assert not x_test.isnull().any().any()

    print("Saving datasets...")

    x_train.to_parquet(TRAIN_FEATURES_PATH, index=False)
    x_test.to_parquet(TEST_FEATURES_PATH, index=False)

    np.save(TRAIN_LABELS_PATH, y_train.to_numpy())
    np.save(TEST_LABELS_PATH, y_test.to_numpy())

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(list(x_train.columns), f, indent=2)

    print(f"Train features : {TRAIN_FEATURES_PATH}")
    print(f"Test features  : {TEST_FEATURES_PATH}")
    print(f"Feature count  : {len(x_train.columns)}")


if __name__ == "__main__":
    main()
