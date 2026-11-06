"""
SEC 10-Q / 10-K Redline Diff Engine & Hybrid LLM Catalyst Scorer (QARE-MM v1.3 PRD Section 4.4 & 5.6)
Computes structural text redlines for Item 1A (Risk Factors) and Item 7 / Item 2 (MD&A)
across consecutive quarterly/annual filings using Python difflib.
Supports:
1. One-time historical backfill via OpenAI-compatible Batch API (Llama-3.1-8B-Instruct, temp=0.0)
2. 100% local ongoing inference via Apple Silicon Ollama (llama3.1:8b)
3. Deterministic domain-lexicon structural redline scoring fallback for offline/CI execution
Enforces >16:00 America/New_York Point-in-Time rollover and tau = 15d exponential decay.
"""

from dataclasses import dataclass, field
import difflib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.data.market_loader import MarketDataLoader
from src.features.event_decay import EventDecayKernel


@dataclass(frozen=True)
class SecRedlineResult:
    """Normalized output of a 10-Q/10-K Item 1A & MD&A redline comparison."""
    item1a_risk_delta: float
    mda_catalyst_score: float
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    similarity_ratio: float = 1.0
    effective_date: str = ""
    inference_mode: str = "local_deterministic"


class SecRedlineAnalyzer:
    """
    Extracts Item 1A and MD&A redline diffs via difflib and scores risk deltas and catalysts
    with continuous half-life decay (tau = 15 trading days).
    """

    TAU_DAYS: float = EventDecayKernel.TAU_SEC_10Q  # 15.0 trading days

    RISK_ESCALATION_TERMS = {
        "subpoena", "investigation", "litigation", "default", "covenant", "impairment",
        "material weakness", "restatement", "going concern", "cyberattack", "breach",
        "shortage", "sanctions", "tariff", "insolvency", "bankruptcy", "delisting",
        "adverse", "uncertainty", "decline", "termination", "penalty", "fine",
    }
    RISK_MITIGATION_TERMS = {
        "resolved", "dismissed", "settled", "remediated", "eliminated", "Compliance",
        "recovered", "stabilized", "insured", "hedged",
    }
    MDA_POSITIVE_TERMS = {
        "record revenue", "margin expansion", "operating leverage", "backlog growth",
        "exceeded", "strong demand", "raised guidance", "accelerated", "accretive",
        "free cash flow growth", "market share gains", "outperformed", "upside",
    }
    MDA_NEGATIVE_TERMS = {
        "margin compression", "headwinds", "lowered guidance", "deceleration",
        "inventory buildup", "restructuring charges", "softness", "challenging macro",
        "elongated sales cycles", "attrition", "churn", "write-down", "shortfall",
    }

    @staticmethod
    def split_clauses(text: str) -> List[str]:
        """Splits filing prose into normalized sentence clauses for redline comparison."""
        if not text or not text.strip():
            return []
        cleaned = re.sub(r"\s+", " ", text.strip())
        clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+", cleaned) if c.strip()]
        return clauses

    @classmethod
    def compute_difflib_redline(
        cls,
        prev_text: str,
        curr_text: str,
    ) -> Tuple[List[str], List[str], float]:
        """
        Computes sentence-level added and removed lines between consecutive filings
        and returns (added_lines, removed_lines, similarity_ratio).
        """
        prev_clauses = cls.split_clauses(prev_text)
        curr_clauses = cls.split_clauses(curr_text)

        if not prev_clauses and not curr_clauses:
            return [], [], 1.0

        matcher = difflib.SequenceMatcher(None, prev_clauses, curr_clauses)
        similarity = float(matcher.ratio())

        added: List[str] = []
        removed: List[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "insert"):
                added.extend(curr_clauses[j1:j2])
            if tag in ("replace", "delete"):
                removed.extend(prev_clauses[i1:i2])

        return added, removed, similarity

    @classmethod
    def score_item1a_deterministic(
        cls,
        added_lines: List[str],
        removed_lines: List[str],
    ) -> float:
        """
        Scores Item 1A Risk Factor delta in [-1.0, +1.0]:
        - Newly added risk escalations produce negative delta (heightened risk).
        - Removed risk factors or added mitigations produce positive delta (reduced risk).
        """
        added_text = " ".join(added_lines).lower()
        removed_text = " ".join(removed_lines).lower()

        raw_score = 0.0
        for term in cls.RISK_ESCALATION_TERMS:
            if term in added_text:
                raw_score -= 0.45
            if term in removed_text:
                raw_score += 0.30

        for term in cls.RISK_MITIGATION_TERMS:
            if term.lower() in added_text:
                raw_score += 0.35

        # Structural bloat penalty: large expansion of risk disclosures
        net_clause_expansion = len(added_lines) - len(removed_lines)
        raw_score -= 0.05 * net_clause_expansion

        return float(math.tanh(raw_score))

    @classmethod
    def score_mda_deterministic(
        cls,
        added_lines: List[str],
        removed_lines: List[str],
    ) -> float:
        """
        Scores Item 7 / Item 2 MD&A Catalyst delta in [-1.0, +1.0]:
        - Added positive operational catalysts increase score toward +1.0.
        - Added negative operational headwinds decrease score toward -1.0.
        """
        added_text = " ".join(added_lines).lower()
        removed_text = " ".join(removed_lines).lower()

        raw_score = 0.0
        for term in cls.MDA_POSITIVE_TERMS:
            if term in added_text:
                raw_score += 0.45
            if term in removed_text:
                raw_score -= 0.25

        for term in cls.MDA_NEGATIVE_TERMS:
            if term in added_text:
                raw_score -= 0.45
            if term in removed_text:
                raw_score += 0.25

        return float(math.tanh(raw_score))

    @classmethod
    def score_filing_diff(
        cls,
        prev_item1a: str,
        curr_item1a: str,
        prev_mda: str,
        curr_mda: str,
        acceptance_datetime: str = "",
        filing_date: str = "",
        mode: str = "local_deterministic",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        batch_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> SecRedlineResult:
        """
        Computes Item 1A and MD&A redline deltas and scores them via the configured engine:
        - 'local_ollama': Queries local Apple Silicon Ollama (llama3.1:8b, temp=0.0)
        - 'batch_api': Queries high-throughput OpenAI-compatible endpoint for historical backfill
        - 'local_deterministic': Fast deterministic domain-lexicon redline scorer (also automatic fallback)
        """
        added_1a, removed_1a, sim_1a = cls.compute_difflib_redline(prev_item1a, curr_item1a)
        added_mda, removed_mda, sim_mda = cls.compute_difflib_redline(prev_mda, curr_mda)

        effective_dt = filing_date
        if acceptance_datetime and acceptance_datetime.strip():
            effective_dt = MarketDataLoader.apply_pit_cutoff(acceptance_datetime).strftime("%Y-%m-%d")

        used_mode = "local_deterministic"
        risk_delta: Optional[float] = None
        mda_score: Optional[float] = None

        if mode == "local_ollama":
            try:
                import requests

                prompt = (
                    "Return strict JSON {\"item1a_risk_delta\": float, \"mda_catalyst_score\": float} "
                    f"in [-1.0, 1.0]. Item1A Added: {added_1a[:5]} | MD&A Added: {added_mda[:5]}"
                )
                resp = requests.post(
                    ollama_url,
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                    timeout=2.0,
                )
                if resp.status_code == 200:
                    payload = json.loads(resp.json().get("response", "{}"))
                    risk_delta = float(max(-1.0, min(1.0, float(payload["item1a_risk_delta"]))))
                    mda_score = float(max(-1.0, min(1.0, float(payload["mda_catalyst_score"]))))
                    used_mode = "local_ollama"
            except Exception:
                risk_delta = None
                mda_score = None

        elif mode == "batch_api" and batch_api_url and api_key:
            try:
                import requests

                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                prompt = (
                    "Return strict JSON {\"item1a_risk_delta\": float, \"mda_catalyst_score\": float} "
                    f"in [-1.0, 1.0]. Item1A Added: {added_1a[:5]} | MD&A Added: {added_mda[:5]}"
                )
                resp = requests.post(
                    batch_api_url,
                    headers=headers,
                    json={
                        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                    },
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    payload = json.loads(content)
                    risk_delta = float(max(-1.0, min(1.0, float(payload["item1a_risk_delta"]))))
                    mda_score = float(max(-1.0, min(1.0, float(payload["mda_catalyst_score"]))))
                    used_mode = "batch_api"
            except Exception:
                risk_delta = None
                mda_score = None

        if risk_delta is None or mda_score is None:
            risk_delta = cls.score_item1a_deterministic(added_1a, removed_1a)
            mda_score = cls.score_mda_deterministic(added_mda, removed_mda)
            used_mode = "local_deterministic"

        combined_sim = 0.5 * (sim_1a + sim_mda)
        return SecRedlineResult(
            item1a_risk_delta=risk_delta,
            mda_catalyst_score=mda_score,
            added_lines=added_1a + added_mda,
            removed_lines=removed_1a + removed_mda,
            similarity_ratio=float(combined_sim),
            effective_date=effective_dt,
            inference_mode=used_mode,
        )

    @classmethod
    def compute_decayed_scores(
        cls,
        redline: SecRedlineResult,
        as_of_date: Union[str, pd.Timestamp],
    ) -> Tuple[float, float, float]:
        """
        Applies continuous exponential decay (tau = 15d) as of as_of_date.
        Returns (decayed_risk_delta, decayed_mda_score, elapsed_days).
        Strictly returns (0.0, 0.0, 0.0) if redline.effective_date > as_of_date (Point-in-Time).
        """
        if not redline.effective_date:
            return 0.0, 0.0, 0.0

        curr_dt = pd.Timestamp(as_of_date).tz_localize(None)
        eff_dt = pd.Timestamp(redline.effective_date).tz_localize(None)
        if eff_dt > curr_dt:
            return 0.0, 0.0, 0.0

        elapsed = float((curr_dt - eff_dt).days)
        decayed_risk = EventDecayKernel.compute_decay(redline.item1a_risk_delta, elapsed, cls.TAU_DAYS)
        decayed_mda = EventDecayKernel.compute_decay(redline.mda_catalyst_score, elapsed, cls.TAU_DAYS)
        return decayed_risk, decayed_mda, elapsed
