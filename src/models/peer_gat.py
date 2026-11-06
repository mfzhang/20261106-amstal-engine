"""
CPU-Pinned 2-Layer Dynamic Peer Graph Attention Network (QARE-MM v1.3 PRD Section 5.10)
Consumes batched 10-node subgraphs representing target equity, 8 GICS peers, and sector ETF.
Aggregates cross-sectional peer spillover into a 32-dimensional graph latent representation.
Explicitly pinned to CPU threads to bypass Apple Silicon MPS kernel fallback overhead.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv


class PeerGAT(nn.Module):
    """
    2-Layer Graph Attention Network (GAT) with Multi-Head Attention and LayerNorm.
    Extracts 32-dimensional root target node embedding via batch.ptr[:-1] slicing.
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_classes: int = 3,
        dropout: float = 0.15,
        num_threads: int = 4,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_threads = num_threads
        self.random_state = random_state

        torch.manual_seed(self.random_state)
        try:
            torch.set_num_threads(self.num_threads)
        except RuntimeError:
            pass

        # Layer 1: GATConv 15 -> 32
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,  # Self loops already constructed in graph_builder.py
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: GATConv 32 -> 32
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        # Projection head for standalone classification / Gate 2 benchmarks
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.temperature_: float = 1.0
        self.is_fitted_: bool = False

    def forward(
        self,
        data: Union[Batch, Data],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes GAT message passing across all nodes in the batched graph.
        Returns:
        1. h_graph: (Batch, 32) root target node representation sliced via batch.ptr[:-1]
        2. logits: (Batch, 3) classification logits from target representation
        """
        x = data.x
        if x.shape[1] > self.in_channels:
            x = x[:, : self.in_channels]
        elif x.shape[1] < self.in_channels:
            pad = torch.zeros(
                (x.shape[0], self.in_channels - x.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Pass 1: GATConv + LN + ELU + Dropout
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.ln1(h)
        h = self.act1(h)
        h = self.drop1(h)

        # Pass 2: GATConv + Residual + LN + ELU + Dropout
        h_res = h
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.ln2(h + h_res)
        h = self.act2(h)
        h = self.drop2(h)

        # Extract root target node v0 embeddings for each subgraph in batch
        if hasattr(data, "ptr") and data.ptr is not None and len(data.ptr) > 1:
            root_indices = data.ptr[:-1]
            h_target = h[root_indices]
        else:
            # Single Data instance: root target node is always at index 0
            h_target = h[0:1]

        logits = self.classifier(h_target)
        return h_target, logits

    def extract_latent(self, data: Union[Batch, Data]) -> np.ndarray:
        """Extracts 32-dimensional root target node embedding as NumPy array (Batch, 32)."""
        self.eval()
        with torch.no_grad():
            h_target, _ = self.forward(data)
            return h_target.cpu().numpy().astype(np.float32)

    def predict_proba(self, data: Union[Batch, Data]) -> np.ndarray:
        """Returns normalized class probabilities (Batch, 3)."""
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(data)
            scaled = logits / max(self.temperature_, 1e-4)
            probs = torch.softmax(scaled, dim=-1)
            return probs.cpu().numpy().astype(np.float64)

    def fit(
        self,
        train_batches: List[Union[Batch, Data]],
        y_train: List[int],
        epochs: int = 5,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
    ) -> "PeerGAT":
        """Fits PeerGAT model on list of training subgraphs and class labels."""
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        y_tensor = torch.tensor(y_train, dtype=torch.long)

        for _ in range(epochs):
            total_loss = 0.0
            idx = 0
            for batch_data in train_batches:
                optimizer.zero_grad()
                _, logits = self.forward(batch_data)
                batch_size = logits.shape[0]
                y_batch = y_tensor[idx : idx + batch_size]
                idx += batch_size

                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.is_fitted_ = True
        return self
