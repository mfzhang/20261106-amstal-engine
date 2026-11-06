"""
SEC EDGAR XBRL & Filing Client
Enforces token bucket rate limits (<=10 req/s), mandatory User-Agent compliance,
deterministic US-GAAP XBRL tag fallback hierarchy, and Form 4 XML cluster scoring.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET

import pandas as pd

from src.data.market_loader import MarketDataLoader


@dataclass(frozen=True)
class StandardInsiderTransaction:
    """Standardized SEC Form 4 insider transaction record (PRD Section 4.2)."""
    filing_date: str
    acceptance_datetime: str
    transaction_date: str
    insider_name: str
    relation: str
    transaction_code: str
    is_10b5_1: bool
    shares: float
    price: float

    @property
    def dollar_value(self) -> float:
        return float(self.shares * self.price)

    @property
    def effective_date(self) -> str:
        """
        Returns the Point-in-Time effective trading date after applying the
        >16:00 America/New_York -> t+1 rollover rule.
        """
        if self.acceptance_datetime and self.acceptance_datetime.strip():
            ts = MarketDataLoader.apply_pit_cutoff(self.acceptance_datetime)
            return ts.strftime("%Y-%m-%d")
        return self.filing_date


@dataclass(frozen=True)
class StandardIncomeStatement:
    """Standardized Point-in-Time financial statement record (PRD Section 4.2)."""
    period_end: str
    filed_date: str
    acceptance_datetime: str
    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    eps_diluted: float


class XBRLTagResolver:
    """
    Deterministic US-GAAP XBRL Tag Fallback Hierarchy (architecture.md Section 4.2).
    Resolves tag drift across 10-year reporting histories and enforces Point-in-Time
    filing acceptance timestamps (>16:00 America/New_York shifted to t+1).
    """

    REVENUE_TAGS: List[str] = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ]
    GROSS_PROFIT_TAGS: List[str] = [
        "GrossProfit",
    ]
    COST_OF_REVENUE_TAGS: List[str] = [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ]
    OPERATING_INCOME_TAGS: List[str] = [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ]
    NET_INCOME_TAGS: List[str] = [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ]
    EPS_DILUTED_TAGS: List[str] = [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "EarningsPerShareBasic",
    ]

    @classmethod
    def resolve_tag(
        cls,
        facts: Dict[str, Any],
        hierarchy: List[str],
        default: float = 0.0,
    ) -> float:
        """
        Resolves the first non-null numeric value from a US-GAAP tag hierarchy.
        Supports both unprefixed tags ('Revenues') and namespace-prefixed tags ('us-gaap:Revenues').
        """
        if not facts:
            return default

        for tag in hierarchy:
            for candidate_key in (tag, f"us-gaap:{tag}", f"us-gaap_{tag}"):
                if candidate_key in facts and facts[candidate_key] is not None:
                    try:
                        val = float(facts[candidate_key])
                        if not math.isnan(val) and not math.isinf(val):
                            return val
                    except (TypeError, ValueError):
                        continue
        return default

    @classmethod
    def parse_income_statement_pit(
        cls,
        period_end: str,
        facts: Dict[str, Any],
        acceptance_datetime: str = "",
        filed_date: str = "",
    ) -> StandardIncomeStatement:
        """
        Normalizes a raw XBRL fact dictionary into an immutable StandardIncomeStatement.
        Enforces the >16:00 America/New_York -> t+1 rollover rule on acceptance_datetime.
        """
        effective_filed = filed_date
        if acceptance_datetime and acceptance_datetime.strip():
            eff_ts = MarketDataLoader.apply_pit_cutoff(acceptance_datetime)
            effective_filed = eff_ts.strftime("%Y-%m-%d")
        elif not effective_filed:
            effective_filed = period_end

        revenue = cls.resolve_tag(facts, cls.REVENUE_TAGS, default=0.0)
        gross_profit = cls.resolve_tag(facts, cls.GROSS_PROFIT_TAGS, default=float("nan"))
        if math.isnan(gross_profit):
            cost_of_rev = cls.resolve_tag(facts, cls.COST_OF_REVENUE_TAGS, default=float("nan"))
            if not math.isnan(cost_of_rev) and revenue != 0.0:
                gross_profit = revenue - cost_of_rev
            else:
                gross_profit = 0.0

        operating_income = cls.resolve_tag(facts, cls.OPERATING_INCOME_TAGS, default=0.0)
        net_income = cls.resolve_tag(facts, cls.NET_INCOME_TAGS, default=0.0)
        eps_diluted = cls.resolve_tag(facts, cls.EPS_DILUTED_TAGS, default=0.0)

        return StandardIncomeStatement(
            period_end=period_end,
            filed_date=effective_filed,
            acceptance_datetime=acceptance_datetime or effective_filed,
            revenue=float(revenue),
            gross_profit=float(gross_profit),
            operating_income=float(operating_income),
            net_income=float(net_income),
            eps_diluted=float(eps_diluted),
        )

    @staticmethod
    def filter_pit_statements(
        statements: List[StandardIncomeStatement],
        as_of_date: Union[str, pd.Timestamp],
    ) -> List[StandardIncomeStatement]:
        """
        Filters financial statements strictly to those filed on or before as_of_date.
        If a fiscal period_end was later restated, retains only the version known as of as_of_date.
        Returns statements sorted chronologically by period_end.
        """
        cutoff = pd.Timestamp(as_of_date).tz_localize(None)
        known_by_period: Dict[str, StandardIncomeStatement] = {}

        for stmt in statements:
            filed_ts = pd.Timestamp(stmt.filed_date).tz_localize(None)
            if filed_ts <= cutoff:
                existing = known_by_period.get(stmt.period_end)
                if existing is None or pd.Timestamp(stmt.filed_date) >= pd.Timestamp(existing.filed_date):
                    known_by_period[stmt.period_end] = stmt

        return sorted(known_by_period.values(), key=lambda s: s.period_end)


class Form4Parser:
    """Parses SEC EDGAR Form 4 XML filings and computes insider cluster intensity."""

    ROLE_WEIGHTS = {
        "CEO": 2.5,
        "CFO": 2.5,
        "COO": 2.0,
        "PRESIDENT": 2.0,
        "DIRECTOR": 1.2,
        "OFFICER": 1.0,
        "10% OWNER": 0.8,
    }

    @classmethod
    def parse_xml(
        cls,
        xml_content: str,
        filing_date: str = "",
        acceptance_datetime: str = "",
    ) -> List[StandardInsiderTransaction]:
        """Parses Form 4 XML into standardized insider transaction objects."""
        transactions: List[StandardInsiderTransaction] = []
        if not xml_content or not xml_content.strip():
            return transactions

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return transactions

        # Extract reporting owner name and relationship
        insider_name = "UNKNOWN"
        name_node = root.find(".//rptOwnerName")
        if name_node is not None:
            inner_name = name_node.find("name")
            if inner_name is not None and inner_name.text:
                insider_name = inner_name.text.strip()
            elif name_node.text and name_node.text.strip():
                insider_name = name_node.text.strip()
            else:
                insider_name = "".join(name_node.itertext()).strip()

        relation = "OFFICER"
        officer_title_node = root.find(".//officerTitle")
        director_node = root.find(".//isDirector")
        ten_pct_node = root.find(".//isTenPercentOwner")

        if officer_title_node is not None and officer_title_node.text and officer_title_node.text.strip():
            title_upper = officer_title_node.text.strip().upper()
            if "CEO" in title_upper or "CHIEF EXECUTIVE" in title_upper:
                relation = "CEO"
            elif "CFO" in title_upper or "CHIEF FINANCIAL" in title_upper:
                relation = "CFO"
            elif "PRESIDENT" in title_upper:
                relation = "PRESIDENT"
            elif "COO" in title_upper or "OPERATING" in title_upper:
                relation = "COO"
            else:
                relation = "OFFICER"
        elif director_node is not None and director_node.text and director_node.text.strip() in ("1", "true", "True"):
            relation = "DIRECTOR"
        elif ten_pct_node is not None and ten_pct_node.text and ten_pct_node.text.strip() in ("1", "true", "True"):
            relation = "10% OWNER"

        # Check for Rule 10b5-1 indication
        is_10b5_1 = False
        aff_node = root.find(".//affirmation10b51")
        if aff_node is not None and aff_node.text and aff_node.text.strip() in ("1", "true", "True"):
            is_10b5_1 = True
        else:
            # Check footnotes for 10b5-1 mentions
            for fn in root.findall(".//footnote"):
                if fn.text and "10b5-1" in fn.text.lower():
                    is_10b5_1 = True
                    break

        # Effective filing date if acceptance_datetime is provided
        effective_filing_date = filing_date
        if not effective_filing_date and acceptance_datetime and acceptance_datetime.strip():
            effective_filing_date = MarketDataLoader.apply_pit_cutoff(acceptance_datetime).strftime("%Y-%m-%d")

        # Parse non-derivative transactions
        for tx_node in root.findall(".//nonDerivativeTransaction"):
            code_node = tx_node.find(".//transactionCoding/transactionCode")
            code = code_node.text.strip().upper() if code_node is not None and code_node.text else "P"

            date_node = tx_node.find(".//transactionDate/value")
            tx_date = date_node.text.strip() if date_node is not None and date_node.text else effective_filing_date

            shares_node = tx_node.find(".//transactionAmounts/transactionShares/value")
            try:
                shares = float(shares_node.text.strip()) if shares_node is not None and shares_node.text else 0.0
            except ValueError:
                shares = 0.0

            price_node = tx_node.find(".//transactionAmounts/transactionPricePerShare/value")
            try:
                price = float(price_node.text.strip()) if price_node is not None and price_node.text else 0.0
            except ValueError:
                price = 0.0

            acq_node = tx_node.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
            acq_disp = acq_node.text.strip().upper() if acq_node is not None and acq_node.text else "A"

            # If transactionCode is S or disposed 'D', mark as sale
            if acq_disp == "D" and code != "S":
                code = "S"

            transactions.append(
                StandardInsiderTransaction(
                    filing_date=effective_filing_date,
                    acceptance_datetime=acceptance_datetime,
                    transaction_date=tx_date,
                    insider_name=insider_name,
                    relation=relation,
                    transaction_code=code,
                    is_10b5_1=is_10b5_1,
                    shares=shares,
                    price=price,
                )
            )

        return transactions

    @classmethod
    def compute_cluster_score(
        cls,
        transactions: List[StandardInsiderTransaction],
        as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> float:
        """
        Computes signed insider cluster conviction score:
        - Open market purchases ('P') produce positive edge: + w_role * ln(1 + dollar_value)
        - Open market sales ('S') produce negative edge: - 0.5 * w_role * ln(1 + dollar_value)
        - 10b5-1 pre-scheduled transactions are discounted by 80%
        - If as_of_date is supplied, strictly filters out transactions whose Point-in-Time
          effective_date is after as_of_date.
        Returns normalized score in [-1.0, +1.0].
        """
        if not transactions:
            return 0.0

        cutoff_ts = pd.Timestamp(as_of_date).tz_localize(None) if as_of_date is not None else None

        net_raw = 0.0
        for tx in transactions:
            if cutoff_ts is not None and tx.effective_date:
                if pd.Timestamp(tx.effective_date).tz_localize(None) > cutoff_ts:
                    continue

            if tx.transaction_code not in ("P", "S"):
                continue  # Skip option exercises / gifts / awards

            w_role = cls.ROLE_WEIGHTS.get(tx.relation, 1.0)
            val = max(tx.dollar_value, 0.0)
            log_vol = math.log1p(val)

            # 10b5-1 pre-planned plans carry lower informational edge
            discount = 0.20 if tx.is_10b5_1 else 1.0

            if tx.transaction_code == "P":
                net_raw += w_role * log_vol * discount
            elif tx.transaction_code == "S":
                # Executive sales often routine; weight at 50% relative to buying
                net_raw -= 0.5 * w_role * log_vol * discount

        # Soft-sign compress into [-1.0, +1.0] (threshold ~$500k cluster)
        score = math.tanh(net_raw / 25.0)
        return float(score)


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket Rate Limiter enforcing SEC EDGAR <= 10 req/s quota."""

    def __init__(self, rate: float = 10.0, capacity: float = 10.0) -> None:
        self.rate = rate  # tokens added per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Blocks until requested tokens are available. Returns time waited."""
        with self.lock:
            total_waited = 0.0
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return total_waited

                sleep_time = (tokens - self.tokens) / self.rate
                time.sleep(sleep_time)
                total_waited += sleep_time


class SecEdgarClient:
    """Compliant SEC EDGAR client with pre-flight User-Agent verification and rate limiting."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        rate_limit_per_sec: float = 10.0,
    ) -> None:
        self.user_agent = user_agent or os.environ.get("SEC_EDGAR_USER_AGENT", "")
        self._validate_user_agent(self.user_agent)
        self.limiter = TokenBucketRateLimiter(rate=rate_limit_per_sec, capacity=rate_limit_per_sec)

    def acquire_slot(self) -> float:
        """Acquires a request slot under the SEC 10 req/s limit. Returns wait time."""
        return self.limiter.acquire(1.0)

    @staticmethod
    def _validate_user_agent(ua: str) -> None:
        """Validates that User-Agent declares organization and valid contact email."""
        if not ua:
            raise ValueError(
                "SEC_EDGAR_USER_AGENT must be configured (format: 'OrgName contact@domain.com')."
            )
        # Check for valid email format within string
        if not re.search(r"[\w\.-]+@[\w\.-]+\.\w+", ua):
            raise ValueError(
                f"SEC User-Agent '{ua}' missing a valid contact email address."
            )
