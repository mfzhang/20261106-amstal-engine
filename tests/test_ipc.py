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

