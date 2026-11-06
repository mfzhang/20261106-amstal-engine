"""
Risk Management, Kelly Sizing & Portfolio Exposure Engine (QARE-MM v1.3)
"""

from src.risk.calibration import TemperatureScaler
from src.risk.kelly_engine import (
    MultiOutcomeKellySizer,
    PortfolioExposureNormalizer,
    ExecutionRecommendation,
)

__all__ = [
    "TemperatureScaler",
    "MultiOutcomeKellySizer",
    "PortfolioExposureNormalizer",
    "ExecutionRecommendation",
]
