"""
SEC EDGAR XBRL & Filing Client
Enforces token bucket rate limits (<=10 req/s) and mandatory User-Agent compliance.
"""

import os
import re
import threading
import time
from typing import Optional


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
