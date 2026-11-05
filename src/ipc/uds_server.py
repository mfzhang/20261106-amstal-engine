"""
Unix Domain Socket JSON-RPC 2.0 Server
Handles mnemonic execution requests and heartbeat pings from amstal-terminal.
"""

import json
from typing import Dict, Any, Optional

class JsonRpcUdsServer:
    """Processes incoming JSON-RPC 2.0 frames over Unix Domain Socket."""

    def __init__(self, socket_path: str = "/tmp/amstal.sock") -> None:
        self.socket_path = socket_path

    def process_message(self, raw_json_str: str) -> str:
        """Parses and dispatches a JSON-RPC 2.0 request string."""
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
                "error": {"code": -32600, "message": "Invalid Request: missing jsonrpc=2.0 or method"},
                "id": req_id,
            })

        # Method dispatch
        if method == "ping":
            return json.dumps({
                "jsonrpc": "2.0",
                "result": {"status": "PONG", "timestamp": params.get("timestamp")},
                "id": req_id,
            })
        elif method == "execute_command":
            raw_input = params.get("raw_input", "")
            return json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "status": "COMMAND_RECEIVED",
                    "raw_input": raw_input,
                    "parsed": params.get("parsed", {}),
                },
                "id": req_id,
            })
        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id,
            })
