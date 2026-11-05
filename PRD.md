# Product Requirements Document (PRD): Amstal System
## Multi-Modal Quantitative Alpha & Risk Engine (`amstal-engine`) + Institutional Terminal (`amstal-terminal`)

---

## Document Metadata
| Field | Value |
| :--- | :--- |
| **System Name** | **Amstal** |
| **Subsystems** | `amstal-engine` (Quantitative Core v1.3) & `amstal-terminal` (Workstation UI) |
| **Document Version** | 2.0 (Unified System, Deep Terminal-Engine IPC, Bloomberg & OpenBB Reverse-Engineered) |
| **Target Hardware** | Apple Silicon M5 Air (24 GB Unified Memory, Fanless Architecture) & Modern Workstations |
| **Target Universe** | 250 Liquid US Mid/Large-Cap Equities ($>\$25\text{M}$ ADV, S&P 500 ranks 100–500 + S&P 400 MidCap, ex-Financials & Utilities) |
| **Lookback Horizon** | Rolling 126 Trading Days ($T = 126$, ~6 Calendar Months) |
| **Forecast Horizon** | 5 to 20 Trading Days (Residualized Triple-Barrier with Calendar-Date Purging) |
| **Frictional Hurdle** | 15 bps (0.15%) round-trip penalty per trade (slippage, spread, and brokerage) |
| **License** | MIT License |

---

## 1. Executive Summary & Vision

**Amstal** is an institutional-grade, AI-native personal financial intelligence workstation and quantitative execution platform. It bridges the gap between institutional proprietary trading technology and high-ergonomics personal desktop workstations.

The system is architected into two deeply integrated subsystems:
1. **`amstal-engine`**: A high-performance quantitative alpha generation and portfolio risk engine (QARE-MM v1.3). It ingests multi-modal market data (Point-in-Time prices, SEC EDGAR XBRL filings, Form 4 insider transactions, 10-Q/10-K redline diffs, and rolling statistical peer graphs), strips macro and sector beta to isolate idiosyncratic alpha, evaluates market regime states via forward-filtered Gaussian HMMs, and computes multi-outcome fractional Kelly allocations net of execution friction.
2. **`amstal-terminal`**: A low-latency desktop workstation reverse-engineered from the **Bloomberg Terminal** and **OpenBB Platform/Workspace**. Built on Tauri 2.0 (Rust) and Svelte 5, it features a 4-panel dockable workspace, an RPN mnemonic command engine (`<GO>` syntax), cross-panel symbol synchronization, real-time conflated streaming, QMK mechanical keyboard hardware integration, and a local agentic financial copilot.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     AMSTAL UNIFIED SYSTEM TOPOLOGY                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                         AMSTAL TERMINAL (`amstal-terminal`)                                 │
  │  • Tauri 2.0 (Rust shell) + Svelte 5 / SolidJS + TailwindCSS (Sub-millisecond DOM)          │
  │  • 4-Tile Dockable Workspace (GoldenLayout / FlexLayout) with Cross-Panel Linking           │
  │  • TradingView Lightweight Charts (WebGL/Canvas) & Financial Visualizations                 │
  │  • Mnemonic Omnibar REPL (`AAPL US <EQUITY> DES <GO>`) + Hybrid Fuzzy Search (`cmdk`)      │
  │  • Starboard Keyboard Integration (QMK/VIA firmware, Color-coded keys, Rotary scrubber)    │
  │  • Local Copilot & Chat Interface (Ollama / LanceDB document QA)                            │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │
                                                 │ Deep Local IPC Bus
                                                 │ 1. Unix Domain Sockets (Sub-ms query/response)
                                                 │ 2. WebSocket (`ws://127.0.0.1:8194` Pub/Sub)
                                                 │ 3. Shared Memory / Arrow Flight (Bulk Parquet)
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           AMSTAL ENGINE (`amstal-engine`)                                   │
  │  • Ingestion & Ticker Plant: OpenBB-inspired provider plugins (Polygon, SEC, FRED, Tiingo)  │
  │  • Normalization & Storage: DuckDB in-process OLAP + Partitioned Parquet Lakehouse          │
  │  • Alpha Engine (QARE-MM v1.3): Rolling 60d OLS Beta Residualization (Pure Idiosyncratic)   │
  │  • Alt-Data NLP: Local Llama-3.1-8B SEC 10-Q diffs, FinBERT sentiment, Form 4 insider decay │
  │  • Graph Engine: Dynamic rolling GICS & Lead-Lag PyG Graph Attention Network (GAT)          │
  │  • Macro Regime Gate: Forward-filtered 3-State Gaussian HMM (SPY + VIX)                     │
  │  • Risk & Execution: Purged Triple-Barrier Labeling, Temperature Scaling & Kelly Optimizer  │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Terminal-Engine Integration Specification

The integration between `amstal-terminal` and `amstal-engine` mirrors the high-reliability client-server architecture of Bloomberg's `blp.exe` and `bbcomm.exe` daemon, optimized for zero-overhead local workstation execution.

### 2.1 The Local Multi-Channel IPC Bus

```
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    IPC PROTOCOL SPECIFICATION                                   |
+───────────────────────────┬───────────────────────────┬─────────────────────────────────────────+
| Channel                   | Transport Mechanism       | Primary Responsibilities                |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Control Plane**         | Unix Domain Sockets (UDS) | Mnemonic commands, synchronous queries, |
|                           | Windows Named Pipes       | backtest execution triggers, health     |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Market Data Streaming** | Local WebSocket           | Conflated real-time ticks (100ms UI),   |
|                           | `ws://127.0.0.1:8194`     | order book depth, live regime changes,  |
|                           | (Canonical BBG Port)      | asynchronous alert notifications        |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Bulk Data Transfer**    | Apache Arrow Flight IPC / | Multi-year historical intraday bars,    |
|                           | Shared Memory (mmap)      | 250-stock cross-sectional matrices,     |
|                           |                           | DuckDB query results (>100K rows)       |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
```

### 2.2 Process Lifecycle Management & Heartbeat Protocol
1. **Managed Child Daemon**: Upon launch, `amstal-terminal` checks for an active `amstal-engine` daemon at `127.0.0.1:8194`. If absent, the Tauri Rust backend automatically launches the engine binary/virtualenv in a background daemon process.
2. **Heartbeat & Self-Healing**:
   * The terminal issues a ping every 2,000 ms over UDS: `{"type": "PING", "timestamp": 1730825600000}`.
   * If the engine fails to respond to 3 consecutive pings, the terminal flags the UI with a `"DAEMON_OFFLINE"` amber indicator and attempts an automated zero-downtime hot-restart.
3. **Graceful Teardown**: Closing the terminal sends a `SIGTERM` signal through UDS, prompting `amstal-engine` to flush pending DuckDB writes, checkpoint Parquet stores, and cleanly release socket bindings.

### 2.3 Bi-Directional Message Schemas

#### 1. Terminal to Engine: Mnemonic Command Execution
```json
{
  "message_id": "req-9842a1",
  "client_timestamp": "2026-11-05T20:00:00.120Z",
  "command": {
    "raw_input": "NVDA US <EQUITY> FA <GO>",
    "parsed": {
      "identifier": "NVDA",
      "market": "US",
      "asset_class": "EQUITY",
      "mnemonic": "FA",
      "parameters": {
        "statement_type": "INCOME_STATEMENT",
        "period": "QUARTERLY",
        "years": 5,
        "point_in_time": true
      }
    }
  }
}
```

#### 2. Engine to Terminal: Real-Time Signal & Allocation Stream
Adheres strictly to the QARE-MM Target Output API Contract:
```json
{
  "message_id": "sig-0012f4",
  "ticker": "CRWD",
  "timestamp_utc": "2026-11-05T20:00:00Z",
  "horizon_days": 10,
  "winning_engine": "ENSEMBLE_LGBM_GRU32",
  "data_integrity": {
    "point_in_time_shifted": true,
    "xbrl_filed_date_verified": true,
    "missing_modalities": []
  },
  "macro_regime": {
    "current_state": "LOW_VOL_BULL",
    "state_index": 0,
    "filtered_probabilities": {
      "bull": 0.812,
      "sideways": 0.145,
      "crash": 0.043
    }
  },
  "signal_breakdown": {
    "residual_momentum_20d_rank": 0.78,
    "overnight_vs_intraday_skew": 0.42,
    "amihud_illiquidity_z": -0.31,
    "closing_power_z": 1.15,
    "decayed_finbert_sentiment": 0.54,
    "decayed_form4_insider_cluster": 0.82,
    "decayed_pead_shock": 0.35,
    "decayed_10q_legal_risk_delta": -0.04,
    "peer_gat_spillover": 0.49,
    "smart_money_divergence_pos": 0.0,
    "smart_money_divergence_neg": 0.19
  },
  "calibration": {
    "temperature": 1.34,
    "calibrated_probabilities": {
      "outperform_buy": 0.635,
      "neutral_hold": 0.245,
      "underperform_sell": 0.120
    }
  },
  "execution_recommendation": {
    "action": "BUY",
    "conviction_tier": "HIGH",
    "recommended_portfolio_allocation_pct": 6.40,
    "entry_price": 312.40,
    "dynamic_take_profit": 334.80,
    "dynamic_stop_loss": 301.20,
    "reward_to_risk_ratio": 2.0,
    "estimated_round_trip_cost_bps": 15.0
  }
}
```

---

## 3. Features Reverse-Engineered from Bloomberg Terminal

### 3.1 The Mnemonic Command Grammar
The command system parses traditional Bloomberg Reverse Polish Notation (RPN), augmented with modern fuzzy matching:
$$\text{Syntax: } [\text{TICKER}] \quad [\text{MARKET}] \quad \langle\text{ASSET CLASS}\rangle \quad [\text{MNEMONIC}] \quad \langle\text{GO}\rangle$$

* **Autocomplete & Mnemonic Lexicon**:
  * Typing `AAPL` brings up the security selector dropdown.
  * Pressing `<EQUITY>` or `F8` pins the asset context.
  * Pressing `<GO>` or `Enter` executes the command in the currently focused tile.
* **Hybrid Omnibar**:
  * If a command matches an existing mnemonic (e.g., `DES`, `FA`, `HP`), the system routes directly to that module.
  * If the input is natural language (e.g., *"compare gross margins of AMD vs NVDA"*), the Omnibar seamlessly delegates to the local LLM copilot.

### 3.2 Canonical Bloomberg Functions Replicated in Amstal

| Function | Bloomberg Equivalent | Amstal Implementation & Data Source |
| :--- | :--- | :--- |
| **`DES <GO>`** | Description | Security master factsheet, business overview, GICS industry classification, market cap, beta, dividend yield. |
| **`FA <GO>`** | Financial Analysis | Point-in-Time 10-year financial statements sourced from SEC EDGAR XBRL `companyfacts`, standardized by `filed` dates. |
| **`HP <GO>`** | Historical Price | Interactive tabular historical price, volume, split/dividend adjustments, and overnight vs. intraday return splits. |
| **`GP <GO>`** | Graph Price | High-performance TradingView Lightweight Charts with multi-timeframe candlesticks, intraday VWAP, volume profiles, and ATR bands. |
| **`ANR <GO>`** | Analyst Recommendations | Consensus price targets, recommendation distributions (Buy/Hold/Sell), and revision trajectories. |
| **`SPLC <GO>`** | Supply Chain / Peer Map | Interactive dynamic rolling peer graph showing GICS sub-industry peers and 5-day lead-lag cross-correlation networks. |
| **`PORT <GO>`** | Portfolio Management | Live portfolio tracking, multi-asset risk factor attribution, and Multi-Outcome Fractional Kelly position sizing. |
| **`HMM <GO>`** | Macro Volatility Regime | Forward-filtered 3-state Gaussian HMM dashboard visualizing real-time probabilities for Low-Vol Bull, Sideways, and Crash regimes. |
| **`NEWS <GO>`** | Real-Time News Feed | Live financial headline aggregator with real-time FinBERT sentiment scoring and continuous half-life decay ($\tau=3\text{d}$). |
| **`EDGAR <GO>`**| SEC Filings Viewer | SEC 10-K/10-Q filing browser with automated Item 1A / MD&A `difflib` redlines and Form 4 insider cluster analysis. |

### 3.3 Hardware Starboard Emulation (QMK / VIA)
Amstal provides native firmware configuration files for custom mechanical keyboards (75%/80% layouts):
* **Top Row (F1–F10 Macros)**: `<GOVT>`, `<CORP>`, `<EQUITY>`, `<CURNCY>`, `<CMDTY>`, `<INDEX>`, `<CLIENT>`.
* **Execution Keys**:
  * `Enter` mapped to green `<GO>` keycode.
  * `Esc` mapped to red `<CANCEL>` keycode.
  * `F12` mapped to `<HELP>` (press twice to summon the local AI copilot).
* **Rotary Encoder**:
  * Turn: Scrub historical timeseries candlesticks forward/backward.
  * Press + Turn: Zoom in/out of timeframe resolution (1m -> 5m -> 1h -> 1d).

### 3.4 4-Panel Dockable Workspace (Launchpad)
* Replicates Bloomberg's canonical 4-panel multi-monitor workstation using `GoldenLayout` / `FlexLayout`.
* **Channel Synchronization**: Panels support linked color channels (Blue, Yellow, Green, Red). Linking Panel 1 (Chart) and Panel 2 (Financials) to the Blue Channel means entering `AAPL <GO>` in Panel 1 automatically updates Panel 2 to Apple's financials.

---

## 4. Features Reverse-Engineered from OpenBB

### 4.1 Modular Provider Architecture
Inspired by OpenBB's plug-in architecture, `amstal-engine` decouples data fetching into independent, hot-swappable providers:
* **`amstal-provider-polygon`**: High-frequency hourly/daily OHLCV and market quotes.
* **`amstal-provider-edgar`**: SEC EDGAR XBRL company facts, Form 4 insider transactions, and 10-Q/10-K submissions.
* **`amstal-provider-fred`**: Federal Reserve Economic Data (Treasury yields, CPI, GDP, monetary aggregates).
* **`amstal-provider-yfinance`**: Fallback end-of-day price histories and basic ticker profiles.
* **`amstal-provider-binance`**: Real-time 24/7 crypto L2 order-book feeds for cross-market liquidity testing.

### 4.2 Pydantic v2 Unified Schema Normalization
Eliminates vendor-specific schema discrepancies by standardizing incoming payloads into immutable Pydantic v2 data models:
* `StandardCandle`: `(timestamp, open, high, low, close, volume, vwap)`
* `StandardIncomeStatement`: `(period_end, filed_date, revenue, gross_profit, operating_income, net_income, eps)`
* `StandardInsiderTransaction`: `(filing_date, transaction_date, insider_name, relation, transaction_code, shares, price)`

### 4.3 In-Process Columnar Lakehouse (DuckDB + Parquet)
* Ingested market data and computed features are stored in a local directory of partitioned Apache Parquet files: `data/processed/{domain}/{ticker}.parquet`.
* **DuckDB Core**: Runs embedded directly inside `amstal-engine`. It executes analytical OLAP queries across 20 years of tick/bar data across all 250 universe stocks in **under 15 milliseconds**, bypassing external database server setup entirely.

### 4.4 Local Agentic Copilot (Ollama + LanceDB)
* Replicates and expands upon OpenBB Copilot:
  * Runs fully offline on Apple Silicon via **Ollama** (`Llama-3.1-8B-Instruct` Q4_K_M or Q8_0).
  * Stores chunked SEC 10-K/10-Q filings and earnings transcripts in a local **LanceDB** vector store.
  * Supports deterministic function calling to query engine metrics (`get_kelly_allocation`, `get_regime_state`, `get_insider_sentiment`).

---

## 5. Quantitative Alpha & Risk Engine Specification (QARE-MM v1.3)

### 5.1 Market- & Sector-Residualized Returns
Isolates pure idiosyncratic equity alpha by running rolling 60-day OLS beta regressions against `SPY` and the stock's corresponding GICS Sector ETF:
$$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$
$$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, t}\right)$$

### 5.2 Cross-Sectional Daily Rank Normalization
Enforces stationary distributions across market regimes by ranking every feature $x_{i,t}$ across the active universe ($M_t \approx 250$) into $[-1.0, +1.0]$:
$$\tilde{x}_{i, t} = 2 \cdot \left(\frac{\text{rank}(x_{i, t}) - 1}{M_t - 1}\right) - 1.0$$

### 5.3 Intraday Microstructure & Overnight Decomposition
* **Overnight Log-Return:** $r_{\text{overnight}, t} = \ln(P_{\text{open}, t} / P_{\text{close}, t-1})$
* **Intraday Session Log-Return:** $r_{\text{intraday}, t} = \ln(P_{\text{close}, t} / P_{\text{open}, t})$
* **Realized Hourly Volatility:** $\sigma_{\text{intraday}, t} = \sqrt{\sum_{h=1}^7 r_{t, h}^2}$
* **Closing Power:** $\text{CP}_t = \ln\left(\frac{V_{t, \text{hour 7}} + 10^{-5}}{V_{t, \text{hour 1}} + 10^{-5}}\right)$
* **Close-to-VWAP Deviation:** $\Delta \text{VWAP}_t = \frac{P_{t, \text{close}} - \text{VWAP}_t}{\text{VWAP}_t}$

### 5.4 Continuous Event-Decay Kernel
Asynchronous catalysts decay exponentially over elapsed trading days ($\Delta d$):
$$X_{\text{event}}(t) = X_0 \cdot \exp\left(-\frac{\ln(2)}{\tau} \cdot \Delta d\right)$$
* FinBERT News Sentiment: $\tau_{\text{news}} = 3.0\text{ days}$
* PEAD Earnings Shock: $\tau_{\text{pead}} = 10.0\text{ days}$
* SEC 10-Q/10-K Redline Diffs: $\tau_{\text{legal}} = 15.0\text{ days}$
* SEC Form 4 Insider Clusters: $\tau_{\text{form4}} = 20.0\text{ days}$

### 5.5 Variance-Sorted Forward-Filtered HMM
Fits a 3-state Gaussian HMM on daily `SPY` returns and `VIX` levels:
1. Computes forward-filtered probabilities only: $\alpha_t(k) = P(x_{1:t}, S_t = k)$. Backward-pass smoothing is strictly banned.
2. Sorts states by return variance component $\Sigma_{0,rr} < \Sigma_{1,rr} < \Sigma_{2,rr}$ to enforce permutation invariance:
   `[State 0: Low-Vol Bull, State 1: Sideways, State 2: High-Vol Crash]`.

### 5.6 Purged Triple-Barrier Labeling & Sample Uniqueness
Ternary labels $y_{i, t} \in \{0: \text{Buy}, 1: \text{Hold}, 2: \text{Sell}\}$ on cumulative residual returns $\sum \epsilon$:
* Upper Barrier: $+2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* Lower Barrier: $-1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* Vertical Barrier: $H = 10\text{ trading days}$
* **Purging**: Cross-ticker calendar-date purge eliminates samples resolving past the training fold cutoff date.
* **Uniqueness Weighting**: Observations weighted by average inverse concurrency $u_{i,t}$ over their active lifespan.

### 5.7 Temperature Calibration & Multi-Outcome Kelly Allocation
1. **Calibration**: Fits scalar $T > 0$ via Negative Log-Likelihood minimization on an independent purged validation set.
2. **Multi-Outcome Kelly Sizer**: Solves for optimal allocation $f^*$ net of 15 bps round-trip friction ($c = 0.0015$):
   $$\max_{f \in [0, f_{\max}]} \mathbb{E}[\ln(1 + f \cdot R_{\text{net}})]$$
   Scaled by regime factor $\kappa \in [0.25, 0.50]$ (Quarter/Half Kelly) with a 10% single-stock ceiling (15% absolute hard maximum).

---

## 6. Phased Implementation Roadmap & Falsification Gates

```
Phase 0: Lock Unified Specification v2.0 ──► [GATE 0: Complete]
          │
          ▼
Phase 1: Residualized Backbone & IPC     ──► [GATE 1: Passes Shuffled/Lagged Checks;
                                                      UDS/WS Stream Verified; IC > 0.025]
          │
          ▼
Phase 2: Regime HMM & Dynamic Peer GAT   ──► [GATE 2: HMM Reduces Drawdown >= 15%;
                                                      Peer GAT Improves OOS Sharpe]
          │
          ▼
Phase 3: SEC EDGAR & Alt-Data Pipeline   ──► [GATE 3: Form 4 + 10-Q Diffs Add
                                                      Incremental IC > 0.015 Post-Cutoff]
          │
          ▼
Phase 4: Calibration, Kelly & Terminal   ──► [GATE 4: Brier Drops > 10%;
                                                      Full 4-Panel Terminal Synchronized]
```

### Stop Rules
1. **No Modality Stacking Before Verification**: Do not implement GAT or local LLM pipelines until Phase 1 proves predictive edge on pure market/microstructure data.
2. **Model Agnosticism**: If Sample-Weighted LightGBM/CatBoost outperforms the Attention-GRU out-of-sample, prune the neural sequence branch.
3. **Sharpe Ratio Cap**: Backtest Sharpe $> 2.5$ or win rate $> 65\%$ on 10-day swing horizons flags an automatic timestamp leak until disproven.

---

## 7. Hardware & Thermal Ergonomics (Apple Silicon M5 Air 24 GB)

* **Memory Budget**:
  * macOS Core: ~4.6 GB
  * Amstal Terminal (Tauri/Webview): ~150 MB (vs 900 MB for Electron)
  * Amstal Engine & LightGBM/PyTorch: ~2.8 GB
  * DuckDB buffer cache: ~2.0 GB
  * Headroom: ~14.4 GB available
* **Thermal Throttling Defense**:
  * Sustained load on the fanless M5 Air causes thermal throttling after ~12–15 minutes.
  * Latent dimensions are restricted to `hidden_size = 32`.
  * Heavy LLM diff scoring runs offline as a checkpointed batch job pre-materializing scores into Parquet files. Training runs exclusively on pre-computed Parquet data in seconds.

---

## 8. Directory Layout

```
amstal/
├── PRD.md                             # Master Unified PRD (This Document)
├── architecture.md                    # Master End-to-End Technical Architecture
├── amstal-engine/                     # Quantitative Alpha & Risk Engine
│   ├── PRD.md                         # Engine-specific requirements
│   ├── README.md                      # Quant specs and mathematical formulations
│   ├── LICENSE                        # MIT License
│   ├── pyproject.toml                 # Poetry/Pip build config
│   ├── data/                          # Data store (git-ignored)
│   │   ├── raw/                       # Daily/hourly OHLCV, SEC raw XML/HTML
│   │   ├── processed/                 # Partitioned Parquet Lakehouse
│   │   └── universe/                  # Liquid 250-stock annual constituent lists
│   ├── src/
│   │   ├── data/
│   │   │   ├── market_loader.py       # Ingests OHLCV, SPY, and sector ETFs
│   │   │   ├── sec_edgar.py           # Form 4 parser & XBRL fundamentals
│   │   │   └── universe.py            # Liquid universe selector & PIT filters
│   │   ├── features/
│   │   │   ├── residualizer.py        # Rolling 60d OLS beta residualization
│   │   │   ├── microstructure.py      # Overnight/intraday, VWAP skew, closing power
│   │   │   ├── sec_diff.py            # Item 1A/MD&A redlines & Llama-3.1 scoring
│   │   │   ├── news_sentiment.py      # FinBERT scoring & continuous event decay
│   │   │   └── graph_builder.py       # Monthly rolling GICS + Lead-Lag PyG graphs
│   │   ├── models/
│   │   │   ├── backbone_gru.py        # 1-Layer GRU/LSTM + 2-Head Self-Attention
│   │   │   ├── backbone_lgbm.py       # Sample-weighted LightGBM/CatBoost
│   │   │   ├── peer_gat.py            # 2-Layer PyG GAT with root extraction
│   │   │   ├── regime_hmm.py          # Forward-filtered 3-state Gaussian HMM
│   │   │   └── fusion_net.py          # Cross-modal attention, HMM gate & head
│   │   ├── label/
│   │   │   ├── triple_barrier.py      # Residualized triple-barrier labeling
│   │   │   └── weights.py             # Sample uniqueness & calendar-date purge
│   │   ├── risk/
│   │   │   ├── calibration.py         # Temperature scaling via NLL optimization
│   │   │   └── kelly_engine.py        # Multi-outcome fractional Kelly optimizer
│   │   ├── ipc/
│   │   │   ├── socket_server.py       # UDS / Named Pipe control plane listener
│   │   │   └── stream_server.py       # WebSocket real-time pub/sub broadcaster (:8194)
│   │   └── backtest/
│   │       ├── walk_forward.py        # Purged cross-validation engine
│   │       └── metrics.py             # IC, Sharpe, Drawdown, and Sanity tests
│   └── tests/                         # Unit tests and Gate 1-4 falsification checks
└── amstal-terminal/                   # Institutional Desktop Terminal
    ├── src-tauri/                     # Rust desktop shell
    │   ├── Cargo.toml
    │   └── src/
    │       ├── main.rs                # Application entrypoint & daemon manager
    │       ├── ipc_client.rs          # UDS / Arrow Flight IPC connector
    │       └── keyboard.rs            # Global hotkey & QMK USB listener
    ├── src/                           # Svelte 5 / SolidJS UI
    │   ├── App.svelte                 # Master 4-tile workspace
    │   ├── components/
    │   │   ├── Omnibar.svelte         # Mnemonic REPL & fuzzy command palette
    │   │   ├── WorkspaceGrid.svelte   # GoldenLayout dockable multi-panel tiles
    │   │   ├── ChartTile.svelte       # TradingView Lightweight Charts
    │   │   ├── FinancialsTile.svelte  # Point-in-Time XBRL statement viewer
    │   │   ├── RegimeTile.svelte      # Forward-filtered HMM regime meter
    │   │   └── CopilotTile.svelte     # Local Ollama / LanceDB financial chat
    │   └── stores/                    # Channel linking & real-time tick stores
    ├── package.json
    └── tailwind.config.js
```
