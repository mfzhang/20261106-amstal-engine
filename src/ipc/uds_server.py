"""
Unix Domain Socket JSON-RPC 2.0 Server (QARE-MM v1.3 PRD Section 2.1)
Handles mnemonic execution requests, synchronous signal queries,
and 2,000ms terminal heartbeat pings from amstal-terminal over /tmp/amstal.sock.
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Callable, Dict, Optional, Tuple


def parse_mnemonic_command(raw_input: str) -> Dict[str, Any]:
    """
    Parses Bloomberg-style RPN mnemonic commands:
    Syntax: [TICKER] [MARKET] <ASSET CLASS> [MNEMONIC] <GO>
    Examples:
      - 'NVDA US <EQUITY> FA <GO>'
      - 'AAPL US <EQUITY> GP <GO>'
      - 'CRWD PORT <GO>'
      - 'HMM <GO>'
    """
    cleaned = raw_input.strip()
    if not cleaned:
        return {"raw_input": "", "is_valid": False, "error": "Empty command"}

    tokens = cleaned.split()
    # Normalize ending <GO> or GO
    has_go = False
    if tokens and tokens[-1].upper() in ("<GO>", "GO"):
        has_go = True
        tokens = tokens[:-1]

    identifier = "SPY"
    market = "US"
    asset_class = "EQUITY"
    mnemonic = "GP"
    parameters: Dict[str, Any] = {}

    # Extract <ASSET CLASS> if present (e.g. <EQUITY>, <INDEX>, <CORP>)
    asset_class_match = re.search(r"<([A-Za-z]+)>", cleaned)
    if asset_class_match:
        asset_class = asset_class_match.group(1).upper()
        # Remove the <ASSET_CLASS> token from token list
        tokens = [t for t in tokens if not re.match(r"^<[A-Za-z]+>$", t)]

    if len(tokens) == 1:
        # e.g. "HMM", "DES", "PORT", "FA"
        tok = tokens[0].upper()
        if tok in ("HMM", "PORT", "GP", "FA", "DES", "HELP", "TOP"):
            mnemonic = tok
        else:
            identifier = tok
    elif len(tokens) == 2:
        # e.g. "NVDA FA", "CRWD PORT", "AAPL GP"
        identifier = tokens[0].upper()
        mnemonic = tokens[1].upper()
    elif len(tokens) >= 3:
        # e.g. "NVDA US FA"
        identifier = tokens[0].upper()
        if tokens[1].upper() in ("US", "LN", "GY", "JP", "HK", "IN"):
            market = tokens[1].upper()
            mnemonic = tokens[2].upper()
        else:
            mnemonic = tokens[-1].upper()

    return {
        "raw_input": raw_input,
        "is_valid": True,
        "has_go": has_go,
        "identifier": identifier,
        "market": market,
        "asset_class": asset_class,
        "mnemonic": mnemonic,
        "parameters": parameters,
    }


class JsonRpcUdsServer:
    """
    Unix Domain Socket JSON-RPC 2.0 Server.
    Provides low-latency IPC control plane for amstal-terminal.
    """

    def __init__(
        self,
        socket_path: str = "/tmp/amstal.sock",
        signal_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        portfolio_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        regime_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.socket_path = socket_path
        self.signal_provider = signal_provider
        self.portfolio_provider = portfolio_provider
        self.regime_provider = regime_provider
        self.server: Optional[asyncio.AbstractServer] = None
        self.is_running: bool = False

    def process_message(self, raw_json_str: str) -> str:
        """Parses and dispatches a JSON-RPC 2.0 request string synchronously."""
        try:
            req = json.loads(raw_json_str)
        except json.JSONDecodeError as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": None,
            })

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if req.get("jsonrpc") != "2.0" or not method:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing jsonrpc=2.0 or method",
                },
                "id": req_id,
            })

        # Method dispatch
        if method == "ping":
            return json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "status": "PONG",
                    "client_timestamp": params.get("timestamp"),
                    "server_timestamp": int(time.time() * 1000),
                },
                "id": req_id,
            })

        elif method == "execute_command":
            raw_input = params.get("raw_input", "")
            parsed = parse_mnemonic_command(raw_input)
            # Merge client-provided parsed parameters if available
            if "parsed" in params and isinstance(params["parsed"], dict):
                parsed.update(params["parsed"])

            return json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "status": "COMMAND_PROCESSED",
                    "raw_input": raw_input,
                    "parsed": parsed,
                    "timestamp": int(time.time() * 1000),
                },
                "id": req_id,
            })

        elif method == "get_signal":
            ticker = params.get("ticker", "SPY").upper()
            if self.signal_provider:
                signal_data = self.signal_provider(ticker)
            else:
                signal_data = {
                    "ticker": ticker,
                    "message": "Default signal stub",
                    "action": "HOLD",
                }
            return json.dumps({
                "jsonrpc": "2.0",
                "result": signal_data,
                "id": req_id,
            })

        elif method == "get_portfolio":
            if self.portfolio_provider:
                portfolio_data = self.portfolio_provider()
            else:
                portfolio_data = {"allocations": {}, "gross_exposure": 0.0}
            return json.dumps({
                "jsonrpc": "2.0",
                "result": portfolio_data,
                "id": req_id,
            })

        elif method == "get_macro_regime":
            if self.regime_provider:
                regime_data = self.regime_provider()
            else:
                regime_data = {
                    "current_state": "LOW_VOLATILITY",
                    "state_index": 0,
                    "kappa": 0.50,
                }
            return json.dumps({
                "jsonrpc": "2.0",
                "result": regime_data,
                "id": req_id,
            })

        elif method == "shutdown":
            self.stop()
            return json.dumps({
                "jsonrpc": "2.0",
                "result": {"status": "SHUTDOWN_INITIATED"},
                "id": req_id,
            })

        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id,
            })

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handles an incoming connection on the Unix Domain Socket."""
        while self.is_running:
            try:
                line = await reader.readline()
                if not line:
                    break
                raw_str = line.decode("utf-8").strip()
                if not raw_str:
                    continue
                response = self.process_message(raw_str)
                writer.write((response + "\n").encode("utf-8"))
                await writer.drain()
            except asyncio.CancelledError:
                break
            except Exception:
                break
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def start(self) -> None:
        """Starts asynchronous UDS server listening at socket_path."""
        # Clean up stale socket file if present
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path,
        )
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError:
            pass
        self.is_running = True

    def stop(self) -> None:
        """Stops the server and cleanly unlinks the socket file."""
        self.is_running = False
        if self.server:
            self.server.close()
            self.server = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
