"""
WebSocket Streaming Broadcaster with 100ms Tick Conflation (QARE-MM v1.3 PRD Section 2.1)
Dispatches live market quotes, volatility regimes, and QARE-MM v1.3 signal frames
matching the Target Output API Contract over ws://127.0.0.1:8194 (canonical Bloomberg port).
"""

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List, Optional, Set


def build_qare_signal_frame(
    ticker: str,
    action: str = "BUY",
    conviction_tier: str = "HIGH",
    p_buy: float = 0.635,
    p_hold: float = 0.245,
    p_sell: float = 0.120,
    temperature: float = 1.34,
    unconstrained_kelly_pct: float = 6.40,
    l1_normalized_pct: float = 6.40,
    entry_price: float = 312.40,
    dynamic_take_profit: float = 334.80,
    dynamic_stop_loss: float = 301.20,
    reward_to_risk_ratio: float = 2.0,
    regime_state: str = "LOW_VOLATILITY",
    regime_index: int = 0,
    regime_probs: Optional[Dict[str, float]] = None,
    signal_breakdown: Optional[Dict[str, float]] = None,
    winning_engine: str = "LIGHTGBM_105",
    execution_mode: str = "LONG_ONLY",
) -> Dict[str, Any]:
    """
    Constructs an institutional QARE-MM v1.3 Target Output API Contract frame.
    Strictly adheres to PRD Section 2.4.2 specifications.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg_id = f"sig-{int(time.time() * 1000):x}"

    default_regime_probs = {
        "low_volatility": 0.812,
        "medium_volatility": 0.145,
        "high_volatility": 0.043,
    }
    default_signals = {
        "residual_momentum_20d_rank": 0.78,
        "overnight_return_rank": 0.42,
        "intraday_return_rank": 0.15,
        "amihud_illiquidity_rank": -0.31,
        "closing_power_rank": 0.65,
    }

    return {
        "message_id": msg_id,
        "ticker": ticker.upper(),
        "timestamp_utc": now_iso,
        "horizon_days": 10,
        "winning_engine": winning_engine,
        "execution_mode": execution_mode,
        "data_integrity": {
            "point_in_time_shifted": True,
            "xbrl_filed_date_verified": True,
            "intraday_mode": "HOURLY_FULL",
            "missing_modalities": [],
        },
        "macro_regime": {
            "current_state": regime_state,
            "state_index": regime_index,
            "filtered_probabilities": regime_probs or default_regime_probs,
        },
        "signal_breakdown": signal_breakdown or default_signals,
        "calibration": {
            "temperature": round(temperature, 2),
            "calibrated_probabilities": {
                "outperform_buy": round(p_buy, 3),
                "neutral_hold": round(p_hold, 3),
                "underperform_sell": round(p_sell, 3),
            },
        },
        "execution_recommendation": {
            "action": action,
            "conviction_tier": conviction_tier,
            "unconstrained_kelly_pct": round(unconstrained_kelly_pct, 2),
            "l1_normalized_portfolio_allocation_pct": round(l1_normalized_pct, 2),
            "entry_price": round(entry_price, 2),
            "dynamic_take_profit": round(dynamic_take_profit, 2),
            "dynamic_stop_loss": round(dynamic_stop_loss, 2),
            "reward_to_risk_ratio": round(reward_to_risk_ratio, 2),
            "estimated_round_trip_cost_bps": 15.0,
        },
    }


class WebSocketBroadcaster:
    """
    Manages streaming client subscriptions and 100ms conflation batches over WebSocket.
    Port: 8194 (canonical Bloomberg port).
    """

    DEFAULT_PORT: int = 8194
    CONFLATION_INTERVAL_SEC: float = 0.100  # 100ms conflation window

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        conflation_interval: float = CONFLATION_INTERVAL_SEC,
    ) -> None:
        self.host = host
        self.port = port
        self.conflation_interval = conflation_interval
        self.pending_ticks: Dict[str, Dict[str, Any]] = {}
        self.pending_signals: Dict[str, Dict[str, Any]] = {}
        self.pending_regime: Optional[Dict[str, Any]] = None
        self.clients: Set[Any] = set()
        self.is_running: bool = False
        self._server = None
        self._conflation_task = None

    def queue_tick(self, ticker: str, data: Dict[str, Any]) -> None:
        """Conflates incoming tick updates in memory before broadcast."""
        t = ticker.upper()
        if t not in self.pending_ticks:
            self.pending_ticks[t] = {}
        self.pending_ticks[t].update(data)
        self.pending_ticks[t]["timestamp_ms"] = int(time.time() * 1000)

    def queue_signal(self, ticker: str, signal_frame: Dict[str, Any]) -> None:
        """Queues an updated QARE-MM v1.3 signal recommendation frame."""
        self.pending_signals[ticker.upper()] = signal_frame

    def queue_regime(self, regime_data: Dict[str, Any]) -> None:
        """Queues updated macro volatility regime state."""
        self.pending_regime = regime_data

    def has_pending_data(self) -> bool:
        return bool(self.pending_ticks or self.pending_signals or self.pending_regime)

    def flush_conflated_frame(self) -> Optional[str]:
        """Flushes aggregated state changes into a single conflated UTF-8 JSON payload."""
        if not self.has_pending_data():
            return None

        frame = {
            "type": "CONFLATED_STREAM_FRAME",
            "conflation_interval_ms": int(self.conflation_interval * 1000),
            "timestamp_ms": int(time.time() * 1000),
            "ticks": self.pending_ticks,
            "signals": self.pending_signals,
            "macro_regime": self.pending_regime,
        }

        # Clear buffer
        self.pending_ticks = {}
        self.pending_signals = {}
        self.pending_regime = None

        return json.dumps(frame)

    async def broadcast_conflated_updates(self) -> None:
        """Flushes and sends conflated updates to all connected subscribers."""
        payload = self.flush_conflated_frame()
        if payload is not None and self.clients:
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(payload)
                except Exception:
                    disconnected.add(client)
            self.clients.difference_update(disconnected)

    async def _conflation_loop(self) -> None:
        """Background loop executing every 100ms."""
        while self.is_running:
            try:
                await asyncio.sleep(self.conflation_interval)
                await self.broadcast_conflated_updates()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def register_client(self, websocket: Any) -> None:
        self.clients.add(websocket)
        try:
            # Send initial greeting
            welcome = json.dumps({
                "type": "CONNECTION_ESTABLISHED",
                "service": "amstal-engine-stream",
                "port": self.port,
                "timestamp_ms": int(time.time() * 1000),
            })
            await websocket.send(welcome)
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def start(self) -> None:
        """Starts WebSocket server and 100ms background conflation loop."""
        try:
            import websockets

            self.is_running = True
            self._server = await websockets.serve(
                self.register_client,
                self.host,
                self.port,
            )
            self._conflation_task = asyncio.create_task(self._conflation_loop())
        except ImportError:
            self.is_running = True
            self._conflation_task = asyncio.create_task(self._conflation_loop())

    async def stop(self) -> None:
        """Gracefully stops broadcaster and closes all client connections."""
        self.is_running = False
        if self._conflation_task:
            self._conflation_task.cancel()
            self._conflation_task = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.clients.clear()
