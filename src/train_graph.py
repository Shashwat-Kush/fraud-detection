from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import config
from src.graph_builder import build_edge_index, build_node_features, load_graph_window
from src.graph_model import FraudGraphSAGE


if __name__ == "__main__":
    df = load_graph_window()
    edge_index, account_to_id = build_edge_index(df)
    x, y = build_node_features(df, account_to_id)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
    )

    train_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=True,
    )

    num_positive = y.sum().item()
    num_negative = len(y) - num_positive

    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
    )

    print(f"Positive nodes: {num_positive:,}")
    print(f"Negative nodes: {num_negative:,}")
    print(f"pos_weight: {pos_weight.item():.2f}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    model = FraudGraphSAGE(
        in_channels=4,
        hidden_channels=config.EMBEDDING_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device),
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.003,
    )

    num_epochs = 20

    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            embeddings, logits = model(
                batch.x,
                batch.edge_index,
            )

            loss = criterion(
                logits[: batch.batch_size].squeeze(),
                batch.y[: batch.batch_size].float(),
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:2d}/{num_epochs} | "
            f"Loss: {avg_epoch_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "graphsage.pt"

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "in_channels": 4,
            "hidden_channels": config.EMBEDDING_DIM,
        },
        model_path,
    )

    print(f"\nModel saved to {model_path}")
