"""
Modality-Masked Fusion Network with HMM Sigmoid Gate (QARE-MM v1.3 PRD Section 5.10)
Fuses:
1. Temporal / Cross-Modal Representation h_temporal_or_cross: (Batch, 32)
2. Dynamic Rolling Peer GAT h_graph: (Batch, 32) from PeerGAT
3. Signed Smart-Money Divergence D_t: (Batch, 2) [D_t^+, D_t^-] (Zero-masked [0, 0] in Phase 2)
Concatenated into the institutional locked (Batch, 66) latent vector.
Applies HMM Sigmoid Gate Linear(3 -> 66) + regime multiplier kappa in [0.25, 0.50],
followed by Classification Head: Linear(66 -> 32) + LayerNorm + SiLU + Dropout(0.3) + Linear(32 -> 3).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
import torch.nn as nn


class ModalityMaskedFusionNet(nn.Module):
    """
    Multimodal Fusion Network enforcing the locked (B, 66) latent vector,
    HMM Sigmoid Gating Linear(3 -> 66), and Linear(66 -> 32) -> Linear(32 -> 3) classification head.
    """

    TEMPORAL_DIM = 32
    GRAPH_DIM = 32
    EVENT_DIM = 12
    ALT_DATA_DIM = 2
    HMM_STATES = 3
    FUSED_DIM = 66  # 32 + 32 + 2
    NUM_CLASSES = 3

    def __init__(
        self,
        dropout: float = 0.30,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        # Phase 3: Alt-Data Event Projection: Linear(12 -> 32) + LN + GELU + Dropout(0.2)
        self.event_proj = nn.Sequential(
            nn.Linear(self.EVENT_DIM, self.TEMPORAL_DIM),
            nn.LayerNorm(self.TEMPORAL_DIM),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        # Phase 3: Cross-Modal Multi-Head Attention (Alt-Data Query Q vs Temporal Sequence K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.TEMPORAL_DIM,
            num_heads=2,
            batch_first=True,
            dropout=0.20,
        )
        self.ln_cross = nn.LayerNorm(self.TEMPORAL_DIM)

        # HMM Sigmoid Gate (PRD Section 5.10): Linear(3 -> 66) + Sigmoid
        self.hmm_gate = nn.Sequential(
            nn.Linear(self.HMM_STATES, self.FUSED_DIM),
            nn.Sigmoid(),
        )

        # Classification Head Step 1: Linear(66 -> 32) (bias=False preserves homogeneous kappa scaling)
        self.proj_66_to_32 = nn.Linear(self.FUSED_DIM, self.TEMPORAL_DIM, bias=False)

        # Classification Head Step 2: LayerNorm(32) -> SiLU -> Dropout(0.3) -> Linear(32 -> 3)
        self.head = nn.Sequential(
            nn.LayerNorm(self.TEMPORAL_DIM),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.TEMPORAL_DIM, self.NUM_CLASSES),
        )

        self.temperature_: float = 1.0
        self.is_fitted_: bool = False

    def forward(
        self,
        h_temporal: torch.Tensor,
        h_graph: torch.Tensor,
        kappa: Union[float, torch.Tensor] = 0.50,
        d_t: Optional[torch.Tensor] = None,
        event_features: Optional[torch.Tensor] = None,
        temporal_seq: Optional[torch.Tensor] = None,
        hmm_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes multimodal fusion and HMM sigmoid gating:
        In Phase 2 Mode (event_features=None or Masks=0.0, d_t=None):
          Uses pooled h_temporal directly and zero-masks smart-money channels.
        In Phase 3 Mode (event_features with Masks=1.0 and d_t present):
          Executes Cross-Modal Attention with event Query Q and temporal Key/Value K, V
          and incorporates explicit signed smart-money divergence [D_t^+, D_t^-].
        Returns:
        1. logits: (Batch, 3) classification logits
        2. h_gated: (Batch, 32) gated 66->32 latent representation
        3. h_fused: (Batch, 66) locked multimodal feature vector
        """
        batch_size = h_temporal.shape[0]
        if h_graph.shape[0] != batch_size:
            raise ValueError(
                f"Batch dimension mismatch: h_temporal has {batch_size} samples, h_graph has {h_graph.shape[0]}"
            )

        # Phase 3 Cross-Modal Attention
        h_temp_eff = h_temporal
        if event_features is not None:
            if event_features.shape[0] != batch_size or event_features.shape[1] != self.EVENT_DIM:
                raise ValueError(
                    f"Expected event_features of shape ({batch_size}, {self.EVENT_DIM}), got {event_features.shape}"
                )

            # Channels 9..11 are Modality Masks (PRD Section 5.10):
            # 0.0 = Missing (Phase 2 Mode), 1.0 = Active (Phase 3 Mode)
            masks = event_features[:, 9:12]
            any_active = torch.any(masks > 0.01)

            if any_active:
                # Project event features (B, 12) -> (B, 1, 32)
                e_proj = self.event_proj(event_features).unsqueeze(1)
                if temporal_seq is not None:
                    # Cross-attention: Query = e_proj (B, 1, 32), Key/Value = temporal_seq (B, T, 32)
                    attn_out, _ = self.cross_attn(query=e_proj, key=temporal_seq, value=temporal_seq)
                    h_temp_eff = self.ln_cross(attn_out.squeeze(1) + h_temporal)
                else:
                    # Direct residual blending with temporal pooled state
                    h_temp_eff = self.ln_cross(e_proj.squeeze(1) + h_temporal)

        # Smart-Money Divergence channels [64..65]
        if d_t is None:
            d_t = torch.zeros((batch_size, self.ALT_DATA_DIM), dtype=torch.float32, device=h_temporal.device)
        elif d_t.shape[1] != self.ALT_DATA_DIM:
            raise ValueError(f"Expected d_t of shape ({batch_size}, {self.ALT_DATA_DIM}), got {d_t.shape}")

        # Assemble locked (B, 66) vector: [h_temporal_effective (32) || h_graph (32) || D_t (2)]
        h_fused = torch.cat([h_temp_eff, h_graph, d_t], dim=-1)

        # Modulate by macro regime multiplier kappa in [0.25, 0.50] and optional HMM Sigmoid Gate Linear(3 -> 66)
        if isinstance(kappa, (int, float)):
            kappa_t = torch.tensor(kappa, dtype=torch.float32, device=h_temporal.device)
        else:
            kappa_t = torch.as_tensor(kappa, dtype=torch.float32, device=h_temporal.device)
            if kappa_t.ndim == 1:
                kappa_t = kappa_t.unsqueeze(-1)

        if hmm_probs is not None:
            p_hmm = torch.as_tensor(hmm_probs, dtype=torch.float32, device=h_temporal.device)
            if p_hmm.ndim == 1:
                p_hmm = p_hmm.unsqueeze(0).expand(batch_size, -1)
            gate_66 = self.hmm_gate(p_hmm)
            h_fused_gated = h_fused * gate_66 * kappa_t
        else:
            h_fused_gated = h_fused * kappa_t

        # Linear(66 -> 32) projection into gated representation
        h_gated = self.proj_66_to_32(h_fused_gated)

        # LayerNorm(32) -> SiLU -> Dropout(0.3) -> Linear(32 -> 3)
        logits = self.head(h_gated)
        return logits, h_gated, h_fused

    def predict_proba(
        self,
        h_temporal: Union[np.ndarray, torch.Tensor],
        h_graph: Union[np.ndarray, torch.Tensor],
        kappa: Union[float, np.ndarray, torch.Tensor] = 0.50,
        d_t: Optional[Union[np.ndarray, torch.Tensor]] = None,
        event_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temporal_seq: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hmm_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Computes uncalibrated softmax probabilities (Batch, 3)."""
        self.eval()
        with torch.no_grad():
            t_temp = torch.as_tensor(h_temporal, dtype=torch.float32)
            t_graph = torch.as_tensor(h_graph, dtype=torch.float32)
            if t_temp.ndim == 1:
                t_temp = t_temp.unsqueeze(0)
            if t_graph.ndim == 1:
                t_graph = t_graph.unsqueeze(0)
            t_dt = torch.as_tensor(d_t, dtype=torch.float32) if d_t is not None else None
            if t_dt is not None and t_dt.ndim == 1:
                t_dt = t_dt.unsqueeze(0)
            t_ev = torch.as_tensor(event_features, dtype=torch.float32) if event_features is not None else None
            if t_ev is not None and t_ev.ndim == 1:
                t_ev = t_ev.unsqueeze(0)
            t_seq = torch.as_tensor(temporal_seq, dtype=torch.float32) if temporal_seq is not None else None
            t_hmm = torch.as_tensor(hmm_probs, dtype=torch.float32) if hmm_probs is not None else None

            logits, _, _ = self.forward(
                t_temp,
                t_graph,
                kappa=kappa,
                d_t=t_dt,
                event_features=t_ev,
                temporal_seq=t_seq,
                hmm_probs=t_hmm,
            )
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy().astype(np.float64)

    def fit_temperature(
        self,
        val_logits: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Fits post-hoc temperature parameter T > 0 on validation fold logits
        to minimize negative log-likelihood (Brier/ECE loss).
        """
        logits = np.asarray(val_logits, dtype=np.float64)
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

    def predict_calibrated_proba(
        self,
        h_temporal: Union[np.ndarray, torch.Tensor],
        h_graph: Union[np.ndarray, torch.Tensor],
        kappa: Union[float, np.ndarray, torch.Tensor] = 0.50,
        d_t: Optional[Union[np.ndarray, torch.Tensor]] = None,
        event_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temporal_seq: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hmm_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Computes temperature-calibrated softmax probabilities (Batch, 3)."""
        self.eval()
        with torch.no_grad():
            t_temp = torch.as_tensor(h_temporal, dtype=torch.float32)
            t_graph = torch.as_tensor(h_graph, dtype=torch.float32)
            if t_temp.ndim == 1:
                t_temp = t_temp.unsqueeze(0)
            if t_graph.ndim == 1:
                t_graph = t_graph.unsqueeze(0)
            t_dt = torch.as_tensor(d_t, dtype=torch.float32) if d_t is not None else None
            if t_dt is not None and t_dt.ndim == 1:
                t_dt = t_dt.unsqueeze(0)
            t_ev = torch.as_tensor(event_features, dtype=torch.float32) if event_features is not None else None
            if t_ev is not None and t_ev.ndim == 1:
                t_ev = t_ev.unsqueeze(0)
            t_seq = torch.as_tensor(temporal_seq, dtype=torch.float32) if temporal_seq is not None else None
            t_hmm = torch.as_tensor(hmm_probs, dtype=torch.float32) if hmm_probs is not None else None

            logits, _, _ = self.forward(
                t_temp,
                t_graph,
                kappa=kappa,
                d_t=t_dt,
                event_features=t_ev,
                temporal_seq=t_seq,
                hmm_probs=t_hmm,
            )
            scaled = logits / max(self.temperature_, 1e-4)
            probs = torch.softmax(scaled, dim=-1)
            return probs.cpu().numpy().astype(np.float64)

    def fit(
        self,
        h_temporal: np.ndarray,
        h_graph: np.ndarray,
        y_train: np.ndarray,
        kappa: Union[float, np.ndarray] = 0.50,
        d_t: Optional[np.ndarray] = None,
        event_features: Optional[np.ndarray] = None,
        temporal_seq: Optional[np.ndarray] = None,
        hmm_probs: Optional[np.ndarray] = None,
        epochs: int = 15,
        lr: float = 0.005,
        batch_size: int = 32,
    ) -> "ModalityMaskedFusionNet":
        """Fits ModalityMaskedFusionNet on multimodal embeddings and labels."""
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t_temp = torch.tensor(h_temporal, dtype=torch.float32)
        t_graph = torch.tensor(h_graph, dtype=torch.float32)
        t_y = torch.tensor(y_train, dtype=torch.long)
        t_dt = torch.tensor(d_t, dtype=torch.float32) if d_t is not None else None
        t_ev = torch.tensor(event_features, dtype=torch.float32) if event_features is not None else None
        t_seq = torch.tensor(temporal_seq, dtype=torch.float32) if temporal_seq is not None else None
        t_hmm = torch.tensor(hmm_probs, dtype=torch.float32) if hmm_probs is not None else None
        t_kappa = torch.tensor(kappa, dtype=torch.float32) if isinstance(kappa, (np.ndarray, list, torch.Tensor)) else None

        n = len(t_temp)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                batch_idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                b_dt = t_dt[batch_idx] if t_dt is not None else None
                b_ev = t_ev[batch_idx] if t_ev is not None else None
                b_seq = t_seq[batch_idx] if t_seq is not None else None
                b_hmm = t_hmm[batch_idx] if t_hmm is not None and t_hmm.ndim == 2 else t_hmm
                b_kappa = t_kappa[batch_idx] if t_kappa is not None else kappa
                logits, _, _ = self.forward(
                    t_temp[batch_idx],
                    t_graph[batch_idx],
                    kappa=b_kappa,
                    d_t=b_dt,
                    event_features=b_ev,
                    temporal_seq=b_seq,
                    hmm_probs=b_hmm,
                )
                loss = criterion(logits, t_y[batch_idx])
                loss.backward()
                optimizer.step()

        self.is_fitted_ = True
        return self
