from torch import nn
from torch_geometric.nn import SAGEConv
import torch


class FraudGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embeddings = self.conv2(x, edge_index)
        logits = self.classifier(embeddings)

        return embeddings, logits
