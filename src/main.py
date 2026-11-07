"""
Amstal Engine Daemon Entrypoint
Spawns the UDS JSON-RPC 2.0 listener (/tmp/amstal.sock) and WebSocket broadcaster (:8194).
"""

import asyncio
import logging
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
                regime_provider=self.live_provider.get_macro_regime,
                security_master_provider=self.live_provider.get_security_master,
                historical_prices_provider=self.live_provider.get_historical_prices,
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

        logger.info("Engine daemon fully active. Awaiting terminal connections...")
        await self._stop_event.wait()

    async def stop_async(self) -> None:
        """Asynchronously and cleanly shuts down UDS server and WebSocket broadcaster."""
        logger.info("Initiating graceful teardown of Amstal Engine daemon...")
        if self._stop_event:
            self._stop_event.set()
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

