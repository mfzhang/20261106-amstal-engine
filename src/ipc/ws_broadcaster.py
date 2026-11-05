"""
WebSocket Streaming Broadcaster with 100ms Tick Conflation
Dispatches live market quotes, volatility regimes, and QARE-MM v1.3 signal frames to amstal-terminal.
"""

import json
from typing import Dict, Any, List

class WebSocketBroadcaster:
    """Manages streaming client subscriptions and 100ms conflation batches."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8194) -> None:
        self.host = host
        self.port = port
        self.pending_updates: Dict[str, Dict[str, Any]] = {}

    def queue_tick(self, ticker: str, data: Dict[str, Any]) -> None:
        """Conflates tick updates in memory before broadcast."""
        if ticker not in self.pending_updates:
            self.pending_updates[ticker] = {}
        self.pending_updates[ticker].update(data)

    def flush_conflated_frame(self) -> str:
        """Flushes aggregated state changes into a single conflated UTF-8 JSON payload."""
        frame = {
            "type": "CONFLATED_STREAM_FRAME",
            "tickers": self.pending_updates,
        }
        self.pending_updates = {}
        return json.dumps(frame)
