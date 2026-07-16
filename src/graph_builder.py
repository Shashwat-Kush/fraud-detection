from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config


def load_graph_window(
    parquet_path: str | Path = "data/raw/historical.parquet",
) -> pd.DataFrame:
    """Load the transactions the graph may be built from.

    Only the training window is used: the graph's weak node labels come from
    is_fraud, so including test-period transactions would leak future labels
    into the embeddings.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=["nameOrig", "nameDest", "amount", "is_fraud", "timestamp"],
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.iloc[: int(len(df) * config.TRAIN_FRAC)]


def build_edge_index(
    df: pd.DataFrame,
) -> tuple[torch.LongTensor, dict[str, int]]:
    orig_unique = pd.unique(df["nameOrig"])
    dest_unique = pd.unique(df["nameDest"])

    all_accounts = pd.Index(orig_unique).union(pd.Index(dest_unique))
    categories = pd.Categorical(all_accounts)
    account_to_id = dict(zip(categories.categories, range(len(categories.categories))))
    src = pd.Categorical(df["nameOrig"], categories=categories.categories).codes
    dst = pd.Categorical(df["nameDest"], categories=categories.categories).codes

    edge_index = torch.from_numpy(np.vstack((src, dst))).long()
    return edge_index, account_to_id


def build_node_features(
    df: pd.DataFrame,
    account_to_id: dict[str, int],
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    n_nodes = len(account_to_id)

    out_degree = np.zeros(n_nodes, dtype=np.float32)
    in_degree = np.zeros(n_nodes, dtype=np.float32)

    total_out_amount = np.zeros(n_nodes, dtype=np.float32)
    total_in_amount = np.zeros(n_nodes, dtype=np.float32)

    labels = np.zeros(n_nodes, dtype=np.int64)

    # -------------------------
    # Outgoing stats
    # -------------------------

    out_stats = df.groupby("nameOrig", sort=False).agg(
        out_degree=("nameOrig", "size"),
        total_out_amount=("amount", "sum"),
    )

    out_node_ids = out_stats.index.map(account_to_id).to_numpy(dtype=np.int64)

    out_degree[out_node_ids] = out_stats["out_degree"].to_numpy(dtype=np.float32)

    total_out_amount[out_node_ids] = out_stats["total_out_amount"].to_numpy(
        dtype=np.float32
    )

    # -------------------------
    # Incoming stats
    # -------------------------

    in_stats = df.groupby("nameDest", sort=False).agg(
        in_degree=("nameDest", "size"),
        total_in_amount=("amount", "sum"),
    )

    in_node_ids = in_stats.index.map(account_to_id).to_numpy(dtype=np.int64)

    in_degree[in_node_ids] = in_stats["in_degree"].to_numpy(dtype=np.float32)

    total_in_amount[in_node_ids] = in_stats["total_in_amount"].to_numpy(
        dtype=np.float32
    )

    # -------------------------
    # Weak node labels
    # -------------------------

    fraud_txns = df.loc[df["is_fraud"] == 1]

    fraud_accounts = pd.unique(
        np.concatenate(
            [
                fraud_txns["nameOrig"].to_numpy(),
                fraud_txns["nameDest"].to_numpy(),
            ]
        )
    )

    fraud_node_ids = (
        pd.Index(fraud_accounts).map(account_to_id).to_numpy(dtype=np.int64)
    )

    labels[fraud_node_ids] = 1

    # -------------------------
    # Build x
    # -------------------------

    x = np.column_stack(
        [
            out_degree,
            in_degree,
            total_out_amount,
            total_in_amount,
        ]
    )
    x = np.log1p(x)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std

    return torch.from_numpy(x).float(), torch.from_numpy(labels).long()


if __name__ == "__main__":
    df = load_graph_window()
    edge_index, account_to_id = build_edge_index(df)
    x, y = build_node_features(df, account_to_id)

    n_nodes = len(account_to_id)

    print(f"Nodes: {n_nodes:,}")
    print(f"Edges: {edge_index.shape[1]:,}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")

    assert x.shape[0] == n_nodes
    assert y.shape[0] == n_nodes

    print(f"Fraud nodes: {y.sum().item():,}")
    print(f"Fraud rate: {(y.float().mean().item() * 100):.4f}%")
