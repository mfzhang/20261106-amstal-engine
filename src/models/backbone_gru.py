"""
CPU-Pinned 1-Layer GRU with Multi-Head Self-Attention
Hidden size = 32, consumes (Batch, 126, 15) microstructure sequence tensor.
"""

from typing import Optional
import numpy as np

class AttentionGRUBackbone:
    """Compact sequence model explicitly pinned to CPU threads for Apple Silicon M5 Air."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_classes: int = 3,
        num_threads: int = 4,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_threads = num_threads
        self.model = None

        # Pin CPU threads to prevent thread contention / launch overhead
        try:
            import torch
            torch.set_num_threads(self.num_threads)
            self._init_torch_model()
        except ImportError:
            pass

    def _init_torch_model(self) -> None:
        import torch
        import torch.nn as nn

        class GRUAttentionNet(nn.Module):
            def __init__(self, in_d: int, h_d: int, n_heads: int, n_cls: int):
                super().__init__()
                self.gru = nn.GRU(in_d, h_d, batch_first=True)
                self.attn = nn.MultiheadAttention(h_d, n_heads, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(h_d, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_cls),
                )

            def forward(self, x):
                # x: (B, T, in_d)
                gru_out, _ = self.gru(x)  # (B, T, h_d)
                attn_out, _ = self.attn(gru_out, gru_out, gru_out)
                pooled = attn_out[:, -1, :]  # (B, h_d)
                logits = self.head(pooled)
                return logits

        self.model = GRUAttentionNet(
            self.input_dim, self.hidden_dim, self.num_heads, self.num_classes
        )
        self.model.to("cpu")
