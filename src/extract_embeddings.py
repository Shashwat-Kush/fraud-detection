from pathlib import Path
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.graph_builder import build_edge_index, build_node_features, load_graph_window
from src.graph_model import FraudGraphSAGE


if __name__ == "__main__":
    df = load_graph_window()
    edge_index, account_to_id = build_edge_index(df)
    x, _ = build_node_features(df, account_to_id)

    data = Data(
        x=x,
        edge_index=edge_index,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(
        "data/processed/graphsage.pt",
        map_location=device,
    )

    model = FraudGraphSAGE(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=False,
    )

    graph_embeddings = torch.empty(
        (x.size(0), checkpoint["hidden_channels"]),
        dtype=torch.float32,
    )

    start_idx = 0

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            batch = batch.to(device)

            embeddings, _ = model(
                batch.x,
                batch.edge_index,
            )

            embeddings = embeddings[: batch.batch_size].cpu()

            end_idx = start_idx + batch.batch_size

            graph_embeddings[start_idx:end_idx] = embeddings

            start_idx = end_idx

            if i % 100 == 0:
                print(f"Processed {i} batches")

    assert graph_embeddings.shape[0] == x.shape[0]
    assert graph_embeddings.shape[1] == checkpoint["hidden_channels"]

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = output_dir / "graph_embeddings.npy"

    np.save(embeddings_path, graph_embeddings.numpy())

    with open(
        output_dir / "account_to_id.pkl",
        "wb",
    ) as f:
        pickle.dump(account_to_id, f)

    print(f"Embeddings shape: {tuple(graph_embeddings.shape)}")
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved account mapping to {output_dir / 'account_to_id.pkl'}")
