"""
CPU-Pinned 1-Layer Attention-GRU Backbone (QARE-MM v1.3)
Hidden size = 32, consumes (Batch, 126, 15) microstructure sequence tensor.
Explicitly pinned to CPU threads to bypass Apple Silicon MPS kernel fallback overhead.
Outputs:
1. Latent Representation: (Batch, 32) for multimodal fusion
2. Calibrated Class Probabilities: (Batch, 3) for ternary triple-barrier classification
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


class AttentionGRUBackbone:
    """
    1-Layer GRU with 2-Head Self-Attention and LayerNorm explicitly pinned to CPU threads.
    Extracts 32-dimensional latent feature vector and predicts ternary triple-barrier probabilities.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_classes: int = 3,
        num_threads: int = 4,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_threads = num_threads
        self.random_state = random_state

        self.torch_available = False
        self.model = None
        self.temperature_: float = 1.0
        self.is_fitted_: bool = False

        self._init_network()

    def _init_network(self) -> None:
        try:
            import torch
            import torch.nn as nn

            torch.manual_seed(self.random_state)
            try:
                torch.set_num_threads(self.num_threads)
            except RuntimeError:
                pass

            class GRUSelfAttentionNet(nn.Module):
                def __init__(self, in_d: int, h_d: int, n_heads: int, n_cls: int):
                    super().__init__()
                    self.gru = nn.GRU(in_d, h_d, batch_first=True, num_layers=1)
                    self.attn = nn.MultiheadAttention(h_d, n_heads, batch_first=True)
                    self.ln = nn.LayerNorm(h_d)
                    self.head = nn.Sequential(
                        nn.Linear(h_d, 32),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(32, n_cls),
                    )

                def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (Batch, T, input_dim)
                    gru_out, _ = self.gru(x)  # (Batch, T, hidden_dim)
                    attn_out, _ = self.attn(gru_out, gru_out, gru_out)  # (Batch, T, hidden_dim)
                    norm_out = self.ln(gru_out + attn_out)  # Residual connection + LayerNorm
                    pooled = norm_out[:, -1, :]  # Last time step pooling: (Batch, hidden_dim)
                    return pooled

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    pooled = self.forward_latent(x)
                    logits = self.head(pooled)  # (Batch, num_classes)
                    return logits

            self.model = GRUSelfAttentionNet(
                self.input_dim, self.hidden_dim, self.num_heads, self.num_classes
            )
            self.model.to("cpu")
            self.torch_available = True
        except ImportError:
            self.torch_available = False

    def extract_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts pooled 32-dimensional latent vector (Batch, 32) from the attention-GRU backbone
        for multimodal concatenation into (B, 66).
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=0)

        if self.torch_available and self.model is not None:
            import torch

            self.model.eval()
            with torch.no_grad():
                tensor_in = torch.from_numpy(X_arr).to("cpu")
                latent = self.model.forward_latent(tensor_in).cpu().numpy()
            return latent
        else:
            # Deterministic linear projection fallback
            # Pool across time and project to 32 dimensions
            pooled = np.mean(X_arr, axis=1)  # (Batch, input_dim)
            if pooled.shape[1] < self.hidden_dim:
                pad = np.zeros((len(pooled), self.hidden_dim - pooled.shape[1]))
                return np.hstack([pooled, pad])
            return pooled[:, : self.hidden_dim]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "AttentionGRUBackbone":
        """
        Trains the CPU-pinned Attention-GRU using AdamW with sample uniqueness weighting.
        """
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int64)
        weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else np.ones(len(y_tr), dtype=np.float32)
        )

        if self.torch_available and self.model is not None:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

            dataset = TensorDataset(
                torch.from_numpy(X_tr),
                torch.from_numpy(y_tr),
                torch.from_numpy(weights),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss(reduction="none")

            for epoch in range(epochs):
                for b_x, b_y, b_w in loader:
                    optimizer.zero_grad()
                    logits = self.model(b_x)
                    loss_unweighted = criterion(logits, b_y)
                    weighted_loss = torch.sum(loss_unweighted * b_w) / (torch.sum(b_w) + 1e-8)
                    weighted_loss.backward()
                    optimizer.step()

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax class probabilities (Batch, 3)."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=0)

        if self.torch_available and self.model is not None:
            import torch

            self.model.eval()
            with torch.no_grad():
                tensor_in = torch.from_numpy(X_arr).to("cpu")
                logits = self.model(tensor_in).cpu().numpy()
            probs = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
            return probs
        else:
            # Fallback uniform/simple softmax
            n = len(X_arr)
            probs = np.full((n, self.num_classes), 1.0 / self.num_classes, dtype=np.float64)
            return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels (Batch,) in {0, 1, 2}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit_temperature(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Fits post-hoc temperature parameter T > 0 on validation fold."""
        raw_probs = self.predict_proba(X_val)
        logits = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
        y_v = np.asarray(y_val, dtype=int)
        n_samples = len(y_v)

        def nll_objective(T_val: float) -> float:
            if T_val <= 0:
                return 1e9
            scaled = logits / T_val
            log_denom = logsumexp(scaled, axis=1)
            target_logits = scaled[np.arange(n_samples), y_v]
            log_likelihood = target_logits - log_denom
            return float(-np.mean(log_likelihood))

        res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
        self.temperature_ = float(res.x)
        return self.temperature_

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities using fitted temperature."""
        raw_probs = self.predict_proba(X)
        if abs(self.temperature_ - 1.0) < 1e-5:
            return raw_probs

        logits = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
        scaled = logits / self.temperature_
        log_denom = logsumexp(scaled, axis=1, keepdims=True)
        return np.exp(scaled - log_denom)
