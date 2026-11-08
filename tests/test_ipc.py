"""
Unit Tests for Local IPC Servers & Target Output Streaming Engine (QARE-MM v1.3)
Verifies:
1. Mnemonic Command Grammar RPN Parsing ([TICKER] [MARKET] <ASSET CLASS> [MNEMONIC] <GO>)
2. Unix Domain Socket JSON-RPC 2.0 Server (Heartbeat PING/PONG, execute_command, get_signal)
3. QARE-MM v1.3 Target Output API Contract Frame Schema Compliance
4. WebSocket 100ms Conflation Buffer and Payload Serialization
"""

import json
import pytest

from src.ipc.uds_server import JsonRpcUdsServer, parse_mnemonic_command
from src.ipc.ws_broadcaster import WebSocketBroadcaster, build_qare_signal_frame


def test_mnemonic_command_parser() -> None:
    # 1. Full Bloomberg RPN syntax
    cmd1 = parse_mnemonic_command("NVDA US <EQUITY> FA <GO>")
    assert cmd1["is_valid"] is True
    assert cmd1["identifier"] == "NVDA"
    assert cmd1["market"] == "US"
    assert cmd1["asset_class"] == "EQUITY"
    assert cmd1["mnemonic"] == "FA"
    assert cmd1["has_go"] is True

    # 2. Short form ticker + mnemonic
    cmd2 = parse_mnemonic_command("CRWD PORT <GO>")
    assert cmd2["is_valid"] is True
    assert cmd2["identifier"] == "CRWD"
    assert cmd2["mnemonic"] == "PORT"
    assert cmd2["has_go"] is True

    # 3. Macro regime command
    cmd3 = parse_mnemonic_command("HMM <GO>")
    assert cmd3["is_valid"] is True
    assert cmd3["mnemonic"] == "HMM"
    assert cmd3["has_go"] is True

    # 4. Empty or invalid command
    cmd4 = parse_mnemonic_command("")
    assert cmd4["is_valid"] is False


def test_jsonrpc_uds_server_ping() -> None:
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    ping_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "ping",
        "params": {"timestamp": 1730825600000},
        "id": "ping-1",
    })

    resp_str = server.process_message(ping_req)
    resp = json.loads(resp_str)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "ping-1"
    assert resp["result"]["status"] == "PONG"
    assert resp["result"]["client_timestamp"] == 1730825600000
    assert "server_timestamp" in resp["result"]


def test_jsonrpc_uds_server_execute_command() -> None:
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    cmd_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "execute_command",
        "params": {
            "raw_input": "AAPL US <EQUITY> GP <GO>",
            "client_timestamp": "2026-11-05T20:00:00.120Z",
        },
        "id": "cmd-42",
    })

    resp = json.loads(server.process_message(cmd_req))

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "cmd-42"
    assert resp["result"]["status"] == "COMMAND_PROCESSED"
    assert resp["result"]["parsed"]["identifier"] == "AAPL"
    assert resp["result"]["parsed"]["mnemonic"] == "GP"


def test_jsonrpc_uds_server_error_handling() -> None:
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    # 1. Parse error
    resp1 = json.loads(server.process_message("{bad_json"))
    assert resp1["error"]["code"] == -32700

    # 2. Invalid Request (missing jsonrpc version)
    resp2 = json.loads(server.process_message(json.dumps({"method": "ping"})))
    assert resp2["error"]["code"] == -32600

    # 3. Method not found
    resp3 = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "unknown_action",
        "id": "err-1",
    })))
    assert resp3["error"]["code"] == -32601


def test_qare_signal_frame_schema() -> None:
    frame = build_qare_signal_frame(
        ticker="CRWD",
        action="BUY",
        conviction_tier="HIGH",
        p_buy=0.635,
        p_hold=0.245,
        p_sell=0.120,
        temperature=1.34,
        unconstrained_kelly_pct=6.40,
        l1_normalized_pct=6.40,
        entry_price=312.40,
        dynamic_take_profit=334.80,
        dynamic_stop_loss=301.20,
        reward_to_risk_ratio=2.0,
    )

    # Validate PRD Section 2.4.2 Target Output API Contract
    assert frame["ticker"] == "CRWD"
    assert frame["horizon_days"] == 10
    assert frame["winning_engine"] == "LIGHTGBM_105"
    assert frame["execution_mode"] == "LONG_ONLY"

    # Data integrity audit block
    assert frame["data_integrity"]["point_in_time_shifted"] is True
    assert frame["data_integrity"]["xbrl_filed_date_verified"] is True
    assert frame["data_integrity"]["intraday_mode"] == "HOURLY_FULL"

    # Macro regime block
    assert frame["macro_regime"]["current_state"] == "LOW_VOLATILITY"
    assert frame["macro_regime"]["state_index"] == 0
    assert "filtered_probabilities" in frame["macro_regime"]

    # Signal breakdown block: all 13 keys from PRD Section 2.4.2
    assert len(frame["signal_breakdown"]) == 13
    assert "decayed_finbert_sentiment" in frame["signal_breakdown"]
    assert "decayed_form4_insider_cluster" in frame["signal_breakdown"]
    assert "decayed_pead_shock" in frame["signal_breakdown"]
    assert "peer_gat_spillover" in frame["signal_breakdown"]

    # Calibration block
    assert frame["calibration"]["temperature"] == 1.34
    cal_probs = frame["calibration"]["calibrated_probabilities"]
    assert cal_probs["outperform_buy"] == 0.635
    assert cal_probs["neutral_hold"] == 0.245
    assert cal_probs["underperform_sell"] == 0.120

    # Execution recommendation block
    exec_rec = frame["execution_recommendation"]
    assert exec_rec["action"] == "BUY"
    assert exec_rec["conviction_tier"] == "HIGH"
    assert exec_rec["unconstrained_kelly_pct"] == 6.40
    assert exec_rec["l1_normalized_portfolio_allocation_pct"] == 6.40
    assert exec_rec["entry_price"] == 312.40
    assert exec_rec["dynamic_take_profit"] == 334.80
    assert exec_rec["dynamic_stop_loss"] == 301.20
    assert exec_rec["reward_to_risk_ratio"] == 2.0
    assert exec_rec["estimated_round_trip_cost_bps"] == 15.0


def test_websocket_broadcaster_conflation() -> None:
    broadcaster = WebSocketBroadcaster(host="127.0.0.1", port=8194)

    assert broadcaster.has_pending_data() is False
    assert broadcaster.flush_conflated_frame() is None

    # Queue multiple tick updates for the same ticker (conflation)
    broadcaster.queue_tick("NVDA", {"price": 140.0, "volume": 1000})
    broadcaster.queue_tick("NVDA", {"price": 141.5, "volume": 1500})  # Overwrites price

    # Queue a signal
    sig_frame = build_qare_signal_frame("NVDA")
    broadcaster.queue_signal("NVDA", sig_frame)

    # Queue macro regime
    broadcaster.queue_regime({"current_state": "LOW_VOLATILITY", "kappa": 0.50})

    assert broadcaster.has_pending_data() is True

    # Flush 100ms conflated frame
    payload = broadcaster.flush_conflated_frame()
    assert payload is not None

    frame = json.loads(payload)
    assert frame["type"] == "CONFLATED_STREAM_FRAME"
    assert frame["ticks"]["NVDA"]["price"] == 141.5  # Conflated to latest price
    assert frame["ticks"]["NVDA"]["volume"] == 1500
    assert frame["signals"]["NVDA"]["ticker"] == "NVDA"
    assert frame["macro_regime"]["current_state"] == "LOW_VOLATILITY"

    # Subsequent flush must be empty
    assert broadcaster.has_pending_data() is False
    assert broadcaster.flush_conflated_frame() is None


def test_jsonrpc_uds_server_security_master_and_peer_graph() -> None:
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    # 1. get_security_master
    sm_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_security_master",
        "params": {"ticker": "NVDA"},
        "id": "sm-1",
    })
    sm_resp = json.loads(server.process_message(sm_req))
    assert sm_resp["jsonrpc"] == "2.0"
    assert sm_resp["id"] == "sm-1"
    assert sm_resp["result"]["ticker"] == "NVDA"
    assert "gics_sector" in sm_resp["result"]
    assert "beta_market_60d" in sm_resp["result"]

    # 2. get_peer_graph
    pg_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_peer_graph",
        "params": {"ticker": "NVDA"},
        "id": "pg-1",
    })
    pg_resp = json.loads(server.process_message(pg_req))
    assert pg_resp["jsonrpc"] == "2.0"
    assert pg_resp["id"] == "pg-1"
    assert pg_resp["result"]["target_ticker"] == "NVDA"
    assert len(pg_resp["result"]["nodes"]) == 10
    assert len(pg_resp["result"]["edges"]) >= 9
    assert pg_resp["result"]["nodes"][0]["role"] == "TARGET"


def test_jsonrpc_uds_server_edgar_news_and_historical_prices() -> None:
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    # 1. get_edgar_filings
    edgar_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_edgar_filings",
        "params": {"ticker": "NVDA"},
        "id": "edgar-1",
    })
    edgar_resp = json.loads(server.process_message(edgar_req))
    assert edgar_resp["jsonrpc"] == "2.0"
    assert edgar_resp["id"] == "edgar-1"
    assert "item1a_similarity" in edgar_resp["result"]
    assert "redlines" in edgar_resp["result"]
    assert "form4_transactions" in edgar_resp["result"]
    assert len(edgar_resp["result"]["form4_transactions"]) > 0

    # 2. get_news_feed
    news_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_news_feed",
        "params": {"ticker": "NVDA"},
        "id": "news-1",
    })
    news_resp = json.loads(server.process_message(news_req))
    assert news_resp["jsonrpc"] == "2.0"
    assert news_resp["id"] == "news-1"
    assert isinstance(news_resp["result"], list)
    assert len(news_resp["result"]) >= 2
    assert "finbert_score" in news_resp["result"][0]
    assert "decayed_sentiment" in news_resp["result"][0]

    # 3. get_historical_prices
    hp_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_historical_prices",
        "params": {"ticker": "NVDA", "count": 20},
        "id": "hp-1",
    })
    hp_resp = json.loads(server.process_message(hp_req))
    assert hp_resp["jsonrpc"] == "2.0"
    assert hp_resp["id"] == "hp-1"
    assert isinstance(hp_resp["result"], list)
    assert "overnight_return" in hp_resp["result"][0]
    assert "intraday_return" in hp_resp["result"][0]


def test_websocket_jsonrpc_dispatch() -> None:
    class MockWebSocket:
        def __init__(self, incoming_messages: list[str]) -> None:
            self.incoming = incoming_messages
            self.sent: list[str] = []
            self._idx = 0

        async def send(self, data: str) -> None:
            self.sent.append(data)

        def __aiter__(self) -> "MockWebSocket":
            self._idx = 0
            return self

        async def __anext__(self) -> str:
            if self._idx < len(self.incoming):
                msg = self.incoming[self._idx]
                self._idx += 1
                return msg
            raise StopAsyncIteration

    async def _run_test() -> None:
        server = JsonRpcUdsServer(socket_path="/tmp/test_amstal_ws.sock")
        broadcaster = WebSocketBroadcaster(
            host="127.0.0.1",
            port=8199,
            rpc_handler=server.process_message,
        )

        ping_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "ping",
            "params": {"timestamp": 12345},
            "id": "ws-rpc-ping",
        })
        cmd_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "execute_command",
            "params": {"raw_input": "NVDA US <EQUITY> DES <GO>"},
            "id": "ws-rpc-cmd",
        })

        mock_ws = MockWebSocket([ping_req, cmd_req])
        await broadcaster.register_client(mock_ws)

        assert len(mock_ws.sent) == 3

        # 1. First message is connection greeting
        greeting = json.loads(mock_ws.sent[0])
        assert greeting["type"] == "CONNECTION_ESTABLISHED"
        assert greeting["port"] == 8199

        # 2. Second message is ping response
        resp = json.loads(mock_ws.sent[1])
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "ws-rpc-ping"
        assert resp["result"]["status"] == "PONG"
        assert resp["result"]["client_timestamp"] == 12345

        # 3. Third message is command response
        cmd_resp = json.loads(mock_ws.sent[2])
        assert cmd_resp["jsonrpc"] == "2.0"
        assert cmd_resp["id"] == "ws-rpc-cmd"
        assert cmd_resp["result"]["status"] == "COMMAND_PROCESSED"
        assert cmd_resp["result"]["parsed"]["identifier"] == "NVDA"
        assert cmd_resp["result"]["parsed"]["mnemonic"] == "DES"

    import asyncio
    asyncio.run(_run_test())


def test_jsonrpc_uds_server_analyst_ratings_and_copilot_query() -> None:
    # 1. Default providers
    server = JsonRpcUdsServer(socket_path="/tmp/test_amstal.sock")

    # A. get_analyst_ratings (default)
    anr_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_analyst_ratings",
        "params": {"ticker": "NVDA"},
        "id": "anr-1",
    })
    anr_resp = json.loads(server.process_message(anr_req))
    assert anr_resp["jsonrpc"] == "2.0"
    assert anr_resp["id"] == "anr-1"
    anr_res = anr_resp["result"]

    # Consensus price target
    assert "consensus_price_target" in anr_res
    cpt = anr_res["consensus_price_target"]
    assert cpt["mean"] > 0
    assert cpt["high"] >= cpt["mean"] >= cpt["low"]
    assert cpt["current_price"] > 0
    assert "upside_pct" in cpt

    # Recommendation distribution breakdown
    assert "recommendation_distribution" in anr_res
    rd = anr_res["recommendation_distribution"]
    assert rd["total_analysts"] == (
        rd["strong_buy"] + rd["buy"] + rd["hold"] + rd["underperform"] + rd["sell"]
    )
    assert 1.0 <= rd["consensus_score"] <= 5.0
    assert rd["consensus_rating"] in ("BUY", "STRONG_BUY", "HOLD", "SELL")

    # Historical revisions
    assert "historical_revisions" in anr_res
    revs = anr_res["historical_revisions"]
    assert len(revs) == 4
    assert [r["period"] for r in revs] == ["Current", "1M Ago", "2M Ago", "3M Ago"]
    assert all("mean_target" in r and "buy_pct" in r for r in revs)

    # Direct top-level accessors
    assert anr_res["mean"] == cpt["mean"]
    assert anr_res["consensus_score"] == rd["consensus_score"]

    # B. copilot_query - Deterministic Tool Routing
    # Case 1: Kelly / Sizing / Allocation
    cq_kelly = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "What is the optimal Kelly sizing and conviction for NVDA?", "ticker": "NVDA"},
        "id": "cq-1",
    })))
    assert cq_kelly["result"]["tool_routed"] == "kelly_sizer"
    assert "Kelly" in cq_kelly["result"]["response"]
    assert "Take-Profit" in cq_kelly["result"]["response"]

    # Case 2: Regime / Volatility / HMM
    cq_regime = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "Show macro volatility regime and HMM probabilities"},
        "id": "cq-2",
    })))
    assert cq_regime["result"]["tool_routed"] == "regime_hmm"
    assert "Macro Volatility Regime" in cq_regime["result"]["response"]
    assert "probabilities" in cq_regime["result"]["response"]

    # Case 3: SEC / Insider / Form 4
    cq_edgar = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "Any SEC Form 4 insider cluster transactions?", "ticker": "NVDA"},
        "id": "cq-3",
    })))
    assert cq_edgar["result"]["tool_routed"] == "sec_edgar"
    assert "Item 1A" in cq_edgar["result"]["response"]
    assert "Form 4" in cq_edgar["result"]["response"]

    # Case 4: Peer / Competitor / SPLC
    cq_peer = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "Analyze competitor peer graph and supply chain spillover", "ticker": "NVDA"},
        "id": "cq-4",
    })))
    assert cq_peer["result"]["tool_routed"] == "peer_gat"
    assert "Peer GAT" in cq_peer["result"]["response"]
    assert "edges" in cq_peer["result"]["response"]

    # Case 5: Default General Quant
    cq_default = json.loads(server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "Tell me about model methodology"},
        "id": "cq-5",
    })))
    assert cq_default["result"]["tool_routed"] == "general_quant"
    assert "zero-lookahead" in cq_default["result"]["response"]

    # 2. Custom Providers Verification
    custom_server = JsonRpcUdsServer(
        socket_path="/tmp/test_custom_amstal.sock",
        analyst_ratings_provider=lambda t: {
            "ticker": t,
            "mean": 200.0,
            "consensus_rating": "STRONG_BUY",
            "custom": True,
        },
        copilot_query_provider=lambda q, t: {
            "query": q,
            "ticker": t,
            "tool_routed": "custom_llm",
            "response": f"Custom copilot answer for {t}: {q}",
            "custom": True,
        },
    )

    custom_anr = json.loads(custom_server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "get_analyst_ratings",
        "params": {"ticker": "CRWD"},
        "id": "c-anr",
    })))
    assert custom_anr["result"]["custom"] is True
    assert custom_anr["result"]["mean"] == 200.0

    custom_cq = json.loads(custom_server.process_message(json.dumps({
        "jsonrpc": "2.0",
        "method": "copilot_query",
        "params": {"query": "custom prompt", "ticker": "CRWD"},
        "id": "c-cq",
    })))
    assert custom_cq["result"]["custom"] is True
    assert custom_cq["result"]["tool_routed"] == "custom_llm"


def test_negative_b_lower_barrier_orientation_and_live_providers() -> None:
    """
    Regression test for BUG-8, BUG-4, BUG-5, BUG-6, BUG-11, BUG-15, BUG-16:
    Verifies that negative b_lower still produces dynamic_stop_loss < entry_price < dynamic_take_profit
    and positive reward_to_risk_ratio, and that LiveInferenceProvider produces valid calibrated
    posteriors, transition_matrix, kappa, and percentage historical returns.
    """
    from src.risk.kelly_engine import MultiOutcomeKellySizer
    from src.ipc.live_providers import LiveInferenceProvider

    sizer = MultiOutcomeKellySizer()
    rec = sizer.compute_recommendation(
        p_buy=0.65,
        p_hold=0.25,
        p_sell=0.10,
        b_upper=0.04,
        b_lower=-0.02,
        entry_price=137.13,
        kappa=0.80,
        execution_mode="LONG_ONLY",
    )
    assert rec.dynamic_stop_loss < rec.entry_price < rec.dynamic_take_profit
    assert rec.reward_to_risk_ratio == pytest.approx(2.0, rel=1e-3)

    provider = LiveInferenceProvider()
    mr = provider.get_macro_regime()
    assert "transition_matrix" in mr
    fp = mr["filtered_probabilities"]
    expected_kappa = round(0.85 * (1.0 - 0.80 * fp["high_volatility"]) * (1.0 - 0.20 * fp["medium_volatility"]), 2)
    assert mr["kappa"] == pytest.approx(expected_kappa, abs=0.02)

    sig = provider.get_signal("NVDA")
    cal = sig["calibration"]["calibrated_probabilities"]
    assert 0.01 < cal["outperform_buy"] < 0.99
    assert 0.01 < cal["neutral_hold"] < 0.99
    assert 0.01 < cal["underperform_sell"] < 0.99
    assert pytest.approx(cal["outperform_buy"] + cal["neutral_hold"] + cal["underperform_sell"], abs=0.01) == 1.0
    s_rec = sig["execution_recommendation"]
    assert s_rec["dynamic_stop_loss"] < s_rec["entry_price"] < s_rec["dynamic_take_profit"]
    assert s_rec["reward_to_risk_ratio"] > 0

    hp = provider.get_historical_prices("NVDA", count=10)
    assert len(hp) == 10
    assert "vwap" in hp[0]

    # Verify BA resolves from lakehouse with authentic pricing (~176.55)
    ba_sig = provider.get_signal("BA")
    ba_rec = ba_sig["execution_recommendation"]
    assert ba_rec["entry_price"] == pytest.approx(176.55, abs=0.5)
    assert ba_rec["dynamic_stop_loss"] < ba_rec["entry_price"] < ba_rec["dynamic_take_profit"]

    ba_sm = provider.get_security_master("BA")
    assert ba_sm["company_name"] == "Boeing Co."
    assert ba_sm["gics_sector"] == "Industrials"
    assert ba_sm["high_52w"] == 262.22
    assert ba_sm["low_52w"] == 137.03

    ba_anr = provider.get_analyst_ratings("BA")
    assert ba_anr["current_price"] == pytest.approx(176.55, abs=0.5)
    assert ba_anr["target_mean"] > ba_anr["current_price"]





