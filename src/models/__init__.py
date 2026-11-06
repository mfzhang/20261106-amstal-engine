"""
Model Backbones & Macro Volatility Regime Estimators
"""

import os
# Ensure safe OpenMP multi-threading coexistence on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch  # Pre-initialize PyTorch runtime before LightGBM OpenMP initialization
except ImportError:
    pass

from src.models.backbone_gru import AttentionGRUBackbone
from src.models.backbone_lgbm import LightGBMBackbone
from src.models.regime_hmm import RegimeHMM

__all__ = ["LightGBMBackbone", "AttentionGRUBackbone", "RegimeHMM"]
