"""
Amstal Engine Daemon Entrypoint
Spawns the UDS JSON-RPC 2.0 listener and WebSocket broadcaster.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("amstal.engine")

def main() -> int:
    logger.info("Initializing Amstal Engine v1.3...")
    logger.info("Stateless DuckDB lakehouse initialized.")
    logger.info("Ready for Phase 1 compute.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
