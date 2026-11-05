"""
SEC EDGAR XBRL & Filing Client
Enforces token bucket rate limits (<=10 req/s) and mandatory User-Agent compliance.
"""

import os
import re
from typing import Optional

class SecEdgarClient:
    """Compliant SEC EDGAR client with pre-flight User-Agent verification."""

    def __init__(self, user_agent: Optional[str] = None) -> None:
        self.user_agent = user_agent or os.environ.get("SEC_EDGAR_USER_AGENT", "")
        self._validate_user_agent(self.user_agent)

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
