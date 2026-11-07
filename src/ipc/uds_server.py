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
from typing import Any, Callable, Dict, List, Optional, Tuple


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
        security_master_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        peer_graph_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        edgar_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        news_provider: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        historical_prices_provider: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.socket_path = socket_path
        self.signal_provider = signal_provider
        self.portfolio_provider = portfolio_provider
        self.regime_provider = regime_provider
        self.security_master_provider = security_master_provider
        self.peer_graph_provider = peer_graph_provider
        self.edgar_provider = edgar_provider
        self.news_provider = news_provider
        self.historical_prices_provider = historical_prices_provider
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

        elif method == "get_security_master":
            ticker = params.get("ticker", "SPY").upper()
            if self.security_master_provider:
                master_data = self.security_master_provider(ticker)
            else:
                master_data = {
                    "ticker": ticker,
                    "company_name": f"{ticker} Inc.",
                    "primary_exchange": "NASDAQ",
                    "cik": "0001045810",
                    "figi": f"BBG000{ticker}001",
                    "gics_sector": "Information Technology",
                    "gics_sub_industry": "Semiconductors",
                    "beta_market_60d": 1.25,
                    "beta_sector_60d": 1.08,
                    "adv_30d_shares": 45000000,
                    "market_cap_billions": 3250.5,
                    "business_summary": f"{ticker} designs high-performance accelerated computing platforms for enterprise AI workloads.",
                }
            return json.dumps({
                "jsonrpc": "2.0",
                "result": master_data,
                "id": req_id,
            })

        elif method == "get_peer_graph":
            ticker = params.get("ticker", "SPY").upper()
            if self.peer_graph_provider:
                graph_data = self.peer_graph_provider(ticker)
            else:
                graph_data = {
                    "target_ticker": ticker,
                    "nodes": [
                        {"id": ticker, "role": "TARGET", "role_code": 0.0, "label": ticker},
                        {"id": "AMD", "role": "GICS_PEER", "role_code": 0.33, "label": "AMD"},
                        {"id": "AVGO", "role": "GICS_PEER", "role_code": 0.33, "label": "AVGO"},
                        {"id": "QCOM", "role": "GICS_PEER", "role_code": 0.33, "label": "QCOM"},
                        {"id": "INTC", "role": "GICS_PEER", "role_code": 0.33, "label": "INTC"},
                        {"id": "TSM", "role": "LEAD_LAG", "role_code": 0.66, "label": "TSM", "cross_corr_5d": 0.68},
                        {"id": "ASML", "role": "LEAD_LAG", "role_code": 0.66, "label": "ASML", "cross_corr_5d": 0.61},
                        {"id": "MSFT", "role": "LEAD_LAG", "role_code": 0.66, "label": "MSFT", "cross_corr_5d": 0.54},
                        {"id": "AMZN", "role": "LEAD_LAG", "role_code": 0.66, "label": "AMZN", "cross_corr_5d": 0.51},
                        {"id": "XLK", "role": "SECTOR_ETF", "role_code": 1.0, "label": "XLK"},
                    ],
                    "edges": [
                        {"source": ticker, "target": "AMD", "attention_weight": 0.165},
                        {"source": ticker, "target": "AVGO", "attention_weight": 0.142},
                        {"source": ticker, "target": "QCOM", "attention_weight": 0.118},
                        {"source": ticker, "target": "INTC", "attention_weight": 0.085},
                        {"source": "TSM", "target": ticker, "attention_weight": 0.185, "lead_days": 5},
                        {"source": "ASML", "target": ticker, "attention_weight": 0.134, "lead_days": 5},
                        {"source": "MSFT", "target": ticker, "attention_weight": 0.098, "lead_days": 5},
                        {"source": "AMZN", "target": ticker, "attention_weight": 0.073, "lead_days": 5},
                        {"source": "XLK", "target": ticker, "attention_weight": 0.220},
                    ],
                }
            return json.dumps({
                "jsonrpc": "2.0",
                "result": graph_data,
                "id": req_id,
            })

        elif method == "get_edgar_filings":
            ticker = params.get("ticker", "SPY").upper()
            if self.edgar_provider:
                edgar_data = self.edgar_provider(ticker)
            else:
                edgar_data = {
                    "ticker": ticker,
                    "item1a_similarity": 0.88,
                    "mda_similarity": 0.74,
                    "item1a_sentiment_drift": -0.04,
                    "mda_catalyst_score": 0.28,
                    "redlines": [
                        {
                            "section": "Item 1A Risk Factors",
                            "status": "MODIFIED",
                            "added_clauses": [
                                "Export controls and geopolitical tensions surrounding high-bandwidth AI accelerators could impact regional supply chain fluidity.",
                                "Increased reliance on specialized third-party fabrication for advanced sub-3nm nodes introduces single-source disruption liabilities.",
                            ],
                            "removed_clauses": [
                                "Legacy semiconductor packaging capacity constraints historically observed in fiscal 2022.",
                            ],
                        },
                        {
                            "section": "Item 7 MD&A",
                            "status": "MODIFIED",
                            "added_clauses": [
                                "Data Center revenue expanded by 154% year-over-year driven by compute architecture adoption across cloud hyperscalers.",
                            ],
                            "removed_clauses": [],
                        },
                    ],
                    "form4_transactions": [
                        {
                            "filing_date": "2024-09-15",
                            "insider_name": "Huang Jen Hsun",
                            "relationship": "CEO & President",
                            "transaction_code": "S",
                            "shares": 120000,
                            "price": 119.50,
                            "value": 14340000,
                            "is_10b5_1": True,
                        },
                        {
                            "filing_date": "2024-08-20",
                            "insider_name": "Stevens Mark A",
                            "relationship": "Director",
                            "transaction_code": "P",
                            "shares": 25000,
                            "price": 124.80,
                            "value": 3120000,
                            "is_10b5_1": False,
                        },
                    ],
                }
            return json.dumps({
                "jsonrpc": "2.0",
                "result": edgar_data,
                "id": req_id,
            })

        elif method == "get_news_feed":
            ticker = params.get("ticker", "SPY").upper()
            if self.news_provider:
                news_data = self.news_provider(ticker)
            else:
                news_data = [
                    {
                        "headline": f"{ticker} Secures Strategic Hyperscale Cloud Infrastructure Supply Agreement",
                        "source": "Bloomberg News",
                        "published_at": "2024-11-06T14:30:00Z",
                        "finbert_score": 0.82,
                        "sentiment_label": "BULLISH",
                        "decay_half_life_days": 3.0,
                        "decayed_sentiment": 0.65,
                    },
                    {
                        "headline": f"Tech Sector Momentum Continues as {ticker} Expands Edge AI Architecture",
                        "source": "Reuters",
                        "published_at": "2024-11-05T11:15:00Z",
                        "finbert_score": 0.64,
                        "sentiment_label": "BULLISH",
                        "decay_half_life_days": 3.0,
                        "decayed_sentiment": 0.40,
                    },
                    {
                        "headline": f"Semiconductor Supply Chain Logistics Show Mild Short-Term Bottlenecks",
                        "source": "Dow Jones Newswires",
                        "published_at": "2024-11-04T09:45:00Z",
                        "finbert_score": -0.31,
                        "sentiment_label": "BEARISH",
                        "decay_half_life_days": 3.0,
                        "decayed_sentiment": -0.15,
                    },
                ]
            return json.dumps({
                "jsonrpc": "2.0",
                "result": news_data,
                "id": req_id,
            })

        elif method == "get_historical_prices":
            ticker = params.get("ticker", "SPY").upper()
            count = int(params.get("count", 30))
            if self.historical_prices_provider:
                prices_data = self.historical_prices_provider(ticker, count)
            else:
                prices_data = [
                    {
                        "date": "2024-11-06",
                        "open": 136.50,
                        "high": 139.20,
                        "low": 135.80,
                        "close": 138.25,
                        "volume": 42500000,
                        "overnight_return": 0.008,
                        "intraday_return": 0.0128,
                        "total_return": 0.021,
                    },
                    {
                        "date": "2024-11-05",
                        "open": 134.80,
                        "high": 136.10,
                        "low": 133.90,
                        "close": 135.40,
                        "volume": 38100000,
                        "overnight_return": 0.004,
                        "intraday_return": 0.0045,
                        "total_return": 0.0085,
                    },
                ]
            return json.dumps({
                "jsonrpc": "2.0",
                "result": prices_data,
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
