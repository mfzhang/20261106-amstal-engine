"""
Amstal Engine Daemon Entrypoint
Spawns the UDS JSON-RPC 2.0 listener (/tmp/amstal.sock) and WebSocket broadcaster (:8194).
"""

import asyncio
import datetime
import logging
import random
import signal
import sys
from typing import Optional

from src.ipc.live_providers import LiveInferenceProvider
from src.ipc.uds_server import JsonRpcUdsServer
from src.ipc.ws_broadcaster import WebSocketBroadcaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amstal.engine")


class AmstalEngineDaemon:
    """Coordinates lifecycle of Amstal IPC control plane and streaming servers."""

    def __init__(
        self,
        socket_path: str = "/tmp/amstal.sock",
        ws_host: str = "127.0.0.1",
        ws_port: int = 8194,
    ) -> None:
        self.live_provider = LiveInferenceProvider()
        if self.live_provider.is_ready():
            logger.info("Initializing JsonRpcUdsServer with live trained model providers.")
            self.uds_server = JsonRpcUdsServer(
                socket_path=socket_path,
                signal_provider=self.live_provider.get_signal,
                portfolio_provider=self.live_provider.get_portfolio,
                regime_provider=self.live_provider.get_macro_regime,
                security_master_provider=self.live_provider.get_security_master,
                historical_prices_provider=self.live_provider.get_historical_prices,
                analyst_ratings_provider=self.live_provider.get_analyst_ratings,
            )
        else:
            logger.warning("No production weights found. Initializing JsonRpcUdsServer in reference stub mode.")
            self.uds_server = JsonRpcUdsServer(socket_path=socket_path)

        self.ws_broadcaster = WebSocketBroadcaster(
            host=ws_host,
            port=ws_port,
            rpc_handler=self.uds_server.process_message,
        )
        self._stop_event: Optional[asyncio.Event] = None
        self._stream_task: Optional[asyncio.Task] = None

    async def _stream_feed_loop(self) -> None:
        """Periodically queues conflated tick, signal, and regime updates for connected WebSocket clients."""
        watchlist = ["NVDA", "MSFT", "AAPL", "CRWD", "SPY", "TSLA"]
        step = 0
        last_closed_broadcast = 0.0
        while self._stop_event and not self._stop_event.is_set():
            try:
                if self.ws_broadcaster.clients:
                    utc_now = datetime.datetime.now(datetime.timezone.utc)
                    # Check market hours: Mon-Fri (0-4), 13:30 to 21:00 UTC
                    is_market_open = utc_now.weekday() < 5 and (13.5 <= utc_now.hour + utc_now.minute / 60.0 <= 21.0)
                    now_ts = time.time()

                    if not is_market_open:
                        # Off-market hours: Throttle broadcast to once every 15s to conserve CPU
                        if now_ts - last_closed_broadcast >= 15.0:
                            regime = self.live_provider.get_macro_regime()
                            self.ws_broadcaster.queue_regime(regime)
                            for sym in watchlist:
                                sig = self.live_provider.get_signal(sym)
                                self.ws_broadcaster.queue_signal(sym, sig)
                            last_closed_broadcast = now_ts
                        await asyncio.sleep(5.0)
                        continue

                    # Active session: stream conflated updates
                    regime = self.live_provider.get_macro_regime()
                    self.ws_broadcaster.queue_regime(regime)
                    active_pool = list(dict.fromkeys(watchlist + list(self.uds_server.active_tickers)))
                    sym = active_pool[step % len(active_pool)]
                    sig = self.live_provider.get_signal(sym)
                    self.ws_broadcaster.queue_signal(sym, sig)
                    entry_p = float(sig.get("execution_recommendation", {}).get("entry_price", 138.25))
                    self.ws_broadcaster.queue_tick(
                        sym,
                        {
                            "ticker": sym,
                            "price": entry_p,
                            "change": 0.0,
                            "change_pct": 0.0,
                            "volume": 42500000,
                        },
                    )
                    step += 1
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(2.0)
            except Exception as e:
                logger.debug(f"Stream feed loop warning: {e}")
                await asyncio.sleep(2.0)

    async def start(self) -> None:
        logger.info("Initializing Amstal Quantitative Alpha & Risk Engine (QARE-MM v1.3)...")
        self._stop_event = asyncio.Event()

        # Start UDS Control Plane
        await self.uds_server.start()
        logger.info(f"UDS Control Plane listening on {self.uds_server.socket_path}")

        # Start WebSocket Broadcaster
        await self.ws_broadcaster.start()
        logger.info(
            f"WebSocket Broadcaster listening on ws://{self.ws_broadcaster.host}:{self.ws_broadcaster.port}"
        )

        self._stream_task = asyncio.create_task(self._stream_feed_loop())

        logger.info("Engine daemon fully active. Awaiting terminal connections...")
        await self._stop_event.wait()

    async def stop_async(self) -> None:
        """Asynchronously and cleanly shuts down UDS server and WebSocket broadcaster."""
        logger.info("Initiating graceful teardown of Amstal Engine daemon...")
        if self._stop_event:
            self._stop_event.set()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self.uds_server.stop()
        if self.ws_broadcaster:
            await self.ws_broadcaster.stop()
        logger.info("Amstal Engine daemon teardown complete.")

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self.uds_server.stop()
        if self.ws_broadcaster:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.ws_broadcaster.stop())
            except RuntimeError:
                pass


async def _run_daemon() -> None:
    daemon = AmstalEngineDaemon()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, daemon.stop)
        except NotImplementedError:
            pass

    try:
        await daemon.start()
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop_async()


def main() -> int:
    try:
        asyncio.run(_run_daemon())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

