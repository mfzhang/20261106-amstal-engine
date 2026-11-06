"""
SEC EDGAR XBRL & Filing Client
Enforces token bucket rate limits (<=10 req/s) and mandatory User-Agent compliance.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import os
import re
import threading
import time
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


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
        if officer_title_node is not None and officer_title_node.text:
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
        elif root.find(".//isDirector") is not None and root.find(".//isDirector").text in ("1", "true", "True"):
            relation = "DIRECTOR"
        elif root.find(".//isTenPercentOwner") is not None and root.find(".//isTenPercentOwner").text in ("1", "true", "True"):
            relation = "10% OWNER"

        # Check for Rule 10b5-1 indication
        is_10b5_1 = False
        aff_node = root.find(".//affirmation10b51")
        if aff_node is not None and aff_node.text in ("1", "true", "True"):
            is_10b5_1 = True
        else:
            # Check footnotes for 10b5-1 mentions
            for fn in root.findall(".//footnote"):
                if fn.text and "10b5-1" in fn.text.lower():
                    is_10b5_1 = True
                    break

        # Parse non-derivative transactions
        for tx_node in root.findall(".//nonDerivativeTransaction"):
            code_node = tx_node.find(".//transactionCoding/transactionCode")
            code = code_node.text.strip().upper() if code_node is not None and code_node.text else "P"

            date_node = tx_node.find(".//transactionDate/value")
            tx_date = date_node.text.strip() if date_node is not None and date_node.text else filing_date

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
                    filing_date=filing_date,
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
    ) -> float:
        """
        Computes signed insider cluster conviction score:
        - Open market purchases ('P') produce positive edge: + w_role * ln(1 + dollar_value)
        - Open market sales ('S') produce negative edge: - 0.5 * w_role * ln(1 + dollar_value)
        - 10b5-1 pre-scheduled transactions are discounted by 80%
        Returns normalized score in [-1.0, +1.0].
        """
        if not transactions:
            return 0.0

        net_raw = 0.0
        for tx in transactions:
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
