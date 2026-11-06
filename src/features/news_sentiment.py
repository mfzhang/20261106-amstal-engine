"""
FinBERT News Sentiment Scorer & Continuous Event Decay Engine (QARE-MM v1.3 PRD Section 5.6 & 5.10)
Scores financial headlines into [-1.0, +1.0], enforces >16:00 America/New_York Point-in-Time
rollover, applies continuous exponential decay (tau = 3.0 trading days), and computes
Signed Smart-Money Divergence [D_t^+, D_t^-].
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import pandas as pd

from src.data.market_loader import MarketDataLoader
from src.features.event_decay import EventDecayKernel


@dataclass(frozen=True)
class HeadlineObservation:
    """Represents a timestamped financial news headline."""
    headline: str
    published_at: str  # ISO-8601 timestamp or YYYY-MM-DD
    sentiment_override: Optional[float] = None

    @property
    def effective_date(self) -> str:
        """
        Computes Point-in-Time effective trading date after applying the
        >16:00 America/New_York -> t+1 rollover rule.
        """
        if "T" in self.published_at or ":" in self.published_at:
            ts = MarketDataLoader.apply_pit_cutoff(self.published_at)
            return ts.strftime("%Y-%m-%d")
        return self.published_at.strip()[:10]


class FinBertSentimentEngine:
    """
    Scores financial news headlines into [-1.0, +1.0] (P(positive) - P(negative)),
    aggregates decayed sentiment with tau = 3.0d half-life, and computes Smart-Money Divergence.
    """

    TAU_DAYS: float = EventDecayKernel.TAU_NEWS  # 3.0 trading days

    POSITIVE_LEXICON = {
        "beat", "beats", "exceeds", "surges", "soars", "record", "upgrade", "upgraded",
        "outperform", "buyback", "dividend", "growth", "bullish", "breakthrough", "approval",
        "approved", "partnership", "profit", "profitable", "raises", "raised", "strong",
        "expansion", "accelerates", "accretive", "awarded", "wins", "rally",
    }
    NEGATIVE_LEXICON = {
        "miss", "misses", "plummets", "plunges", "downgrade", "downgraded", "underperform",
        "lawsuit", "subpoena", "investigation", "fraud", "default", "bankruptcy", "layoffs",
        "cuts", "lowered", "weak", "warning", "recall", "breach", "deficit", "loss",
        "losses", "slump", "probe", "delay", "delayed", "halted", "headwinds", "fine",
    }

    def __init__(self, use_transformers: bool = False) -> None:
        self.use_transformers = use_transformers
        self._pipeline = None
        if self.use_transformers:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    top_k=None,
                )
            except Exception:
                self._pipeline = None

    def score_headline(self, text: str) -> float:
        """
        Scores a financial headline into [-1.0, +1.0]:
        Uses local FinBERT pipeline if initialized, otherwise applies deterministic
        Loughran-McDonald financial catalyst lexicon scoring.
        """
        if not text or not text.strip():
            return 0.0

        if self._pipeline is not None:
            try:
                res = self._pipeline(text[:512])[0]
                probs = {item["label"].lower(): float(item["score"]) for item in res}
                return float(probs.get("positive", 0.0) - probs.get("negative", 0.0))
            except Exception:
                pass

        words = [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
        pos_hits = sum(1 for w in words if w in self.POSITIVE_LEXICON)
        neg_hits = sum(1 for w in words if w in self.NEGATIVE_LEXICON)
        net = pos_hits - neg_hits
        if net == 0:
            return 0.0
        return float(math.tanh(net * 0.65))

    def aggregate_decayed_sentiment(
        self,
        headlines: List[HeadlineObservation],
        as_of_date: Union[str, pd.Timestamp],
        max_age_cap: float = 63.0,
    ) -> Tuple[float, float, float]:
        """
        Aggregates Point-in-Time news headlines as of as_of_date using exponential decay (tau = 3d).
        Strictly ignores headlines whose effective_date > as_of_date (post-16:00 ET rollover).
        Returns: (decayed_sentiment_score, most_recent_age_days, modality_mask)
        where modality_mask = 1.0 if active news present, 0.0 if missing (PRD Section 5.10).
        """
        if not headlines:
            return 0.0, 0.0, 0.0

        curr_dt = pd.Timestamp(as_of_date).tz_localize(None)
        weighted_sum = 0.0
        weight_total = 0.0
        min_elapsed: Optional[float] = None

        for item in headlines:
            eff_dt = pd.Timestamp(item.effective_date).tz_localize(None)
            if eff_dt > curr_dt:
                continue

            elapsed = float((curr_dt - eff_dt).days)
            if elapsed > max_age_cap:
                continue

            raw_s = (
                float(item.sentiment_override)
                if item.sentiment_override is not None
                else self.score_headline(item.headline)
            )
            decay_weight = EventDecayKernel.compute_decay(1.0, elapsed, self.TAU_DAYS)
            weighted_sum += raw_s * decay_weight
            weight_total += 1.0

            if min_elapsed is None or elapsed < min_elapsed:
                min_elapsed = elapsed

        if weight_total == 0.0 or min_elapsed is None:
            return 0.0, 0.0, 0.0

        agg_score = float(max(-1.0, min(1.0, weighted_sum / weight_total)))
        return agg_score, float(min_elapsed), 1.0

    @staticmethod
    def compute_smart_money_divergence(
        sentiment: float,
        residual_return: float,
        volume: float,
        adv20: float,
    ) -> Tuple[float, float]:
        """
        Computes Signed Smart-Money Divergence [D_t^+, D_t^-] (PRD Section 5.10).
        """
        return EventDecayKernel.compute_smart_money_divergence(
            sentiment=sentiment,
            residual_return=residual_return,
            volume=volume,
            adv20=adv20,
        )
