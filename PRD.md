# Product Requirements Document (PRD): Amstal System
## Multi-Modal Quantitative Alpha & Risk Engine (`amstal-engine`) + Institutional Terminal (`amstal-terminal`)

---

## Document Metadata
| Field | Value |
| :--- | :--- |
| **System Name** | **Amstal** |
| **Subsystems** | `amstal-engine` (Quantitative Core v1.3) & `amstal-terminal` (Workstation UI) |
| **Document Version** | 2.1 (Hardened Specification: All Mathematical, Systems & Operational Decisions Locked) |
| **Target Hardware** | Apple Silicon M5 Air (24 GB Unified Memory, Fanless Architecture) & Modern Workstations |
| **Target Universe** | 250 Liquid US Mid/Large-Cap Equities ($>\$25\text{M}$ ADV, S&P 500 ranks 100–500 + S&P 400 MidCap, ex-Financials & Utilities, Point-in-Time Survivor-Bias-Free) |
| **Lookback Horizon** | Rolling 126 Trading Days ($T = 126$, ~6 Calendar Months) |
| **Forecast Horizon** | 5 to 20 Trading Days (10-Day Vertical Residualized Triple-Barrier with Calendar-Date Purging) |
| **Walk-Forward Cadence** | Rolling 3Y Train (756d) $\to$ 10d Purge $\to$ 6M Val (126d) $\to$ 10d Purge $\to$ 3M OOS Test (63d, Quarterly Re-fit) |
| **Execution Mode** | Configurable: Default `LONG_ONLY` Cash ($f_i \in [0, 0.10]$) for Portfolio PnL; Cross-Sectional Rank IC & `LONG_SHORT` Stat-Arb Diagnostics |
| **Frictional Hurdle** | 15 bps (0.15%) round-trip penalty per trade (slippage, spread, and brokerage) |
| **License** | MIT License |

---

## 1. Executive Summary & Vision

**Amstal** is an institutional-grade, AI-native personal financial intelligence workstation and quantitative execution platform. It bridges the gap between institutional proprietary trading technology and high-ergonomics personal desktop workstations.

The system is architected into two deeply integrated subsystems:
1. **`amstal-engine`**: A high-performance quantitative alpha generation and portfolio risk engine (QARE-MM v1.3). It ingests multi-modal market data (Point-in-Time prices, SEC EDGAR XBRL filings, Form 4 insider transactions, 10-Q/10-K redline diffs, and rolling statistical peer graphs), strips macro and sector beta via 2-stage sequential OLS orthogonalization to isolate idiosyncratic alpha, evaluates market regime states via forward-filtered Gaussian HMMs, and computes multi-outcome fractional Kelly allocations net of execution friction with portfolio-level $L_1$ gross exposure normalization.
2. **`amstal-terminal`**: A low-latency desktop workstation reverse-engineered from the **Bloomberg Terminal** and **OpenBB Platform/Workspace**. Built on Tauri 2.0 (Rust) and Svelte 5 (Runes + `paneforge` reactive split-pane grid), it features a 4-panel color-linked workspace, an RPN mnemonic command engine (`<GO>` syntax), real-time 100ms conflated streaming, and a local agentic financial copilot.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     AMSTAL UNIFIED SYSTEM TOPOLOGY                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                         AMSTAL TERMINAL (`amstal-terminal`)                                 │
  │  • Tauri 2.0 (Rust shell) + Svelte 5 Runes + `paneforge` / CSS Grid (Sub-millisecond DOM)   │
  │  • 4-Tile Reactive Workspace with Multi-Color Channel Linking (Blue/Amber/Green/Red)        │
  │  • TradingView Lightweight Charts (WebGL/Canvas) & Point-in-Time Financial Tables           │
  │  • Mnemonic Omnibar REPL (`AAPL US <EQUITY> DES <GO>`) + Hybrid Fuzzy Search (`cmdk`)      │
  │  • Local Copilot & Chat Interface (Ollama `Llama-3.1-8B` + LanceDB document QA)             │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │
                                                 │ Streamlined Local IPC & Storage Bus
                                                 │ 1. Unix Domain Sockets (`/tmp/amstal.sock` RPC)
                                                 │ 2. WebSocket (`ws://127.0.0.1:8194` 100ms Pub/Sub)
                                                 │ 3. Stateless `:memory:` DuckDB over Atomic Parquet
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           AMSTAL ENGINE (`amstal-engine`)                                   │
  │  • Ingestion & Ticker Plant: OpenBB-inspired provider plugins (Polygon, SEC, FRED, yfinance)│
  │  • Storage: Stateless `:memory:` DuckDB + Partitioned Parquet with Atomic POSIX `os.replace`│
  │  • Alpha Engine (QARE-MM v1.3): 2-Stage Sequential OLS Orthogonalization (SPY -> Sector)    │
  │  • Alt-Data NLP: Hybrid LLM / Local Ollama 10-Q diffs, FinBERT sentiment, Form 4 decay      │
  │  • Graph Engine: Dynamic rolling GICS & Lead-Lag PyG GAT (Explicitly CPU-pinned)            │
  │  • Macro Regime Gate: Forward-filtered 3-State Gaussian HMM (SPY + VIX, variance-sorted)    │
  │  • Risk & Execution: Purged Triple-Barrier, Temperature Scaling & L1-Normalized Kelly Sizer │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Terminal-Engine Integration Specification

The integration between `amstal-terminal` and `amstal-engine` mirrors the high-reliability client-server architecture of Bloomberg's `blp.exe` and `bbcomm.exe` daemon, streamlined for zero-overhead local workstation execution without redundant serialization layers.

### 2.1 The Streamlined Local IPC & Storage Bus

```
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    IPC & STORAGE SPECIFICATION                                  |
+───────────────────────────┬───────────────────────────┬─────────────────────────────────────────+
| Channel                   | Transport Mechanism       | Primary Responsibilities                |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Control Plane & RPC**   | Unix Domain Sockets (UDS) | Mnemonic commands, synchronous queries, |
|                           | `/tmp/amstal.sock` /      | backtest execution triggers, health     |
|                           | Windows Named Pipes       | pings (2,000ms), process lifecycle      |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Market Data Streaming** | Local WebSocket           | Conflated real-time ticks (100ms UI),   |
|                           | `ws://127.0.0.1:8194`     | order book depth, live regime changes,  |
|                           | (Canonical BBG Port)      | asynchronous alert notifications        |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
| **Lock-Free In-Process**  | Stateless `:memory:`      | Multi-year historical bars & 250-stock  |
| **Analytical OLAP**       | `duckdb-rs` / `polars-rs` | feature matrices read directly from     |
|                           | over Atomic Parquet files | immutable `.parquet` files (<15ms)      |
+───────────────────────────┼───────────────────────────┼─────────────────────────────────────────+
```

### 2.2 Lock-Free Storage Concurrency Protocol (Stateless DuckDB + Atomic Rename)
To prevent single-writer database lock collisions (`IO Error: Could not set lock on file`) between Python (`amstal-engine`) and Rust (`amstal-terminal`):
1. **No Persistent `.duckdb` Lock File**: Neither Python nor Tauri Rust ever opens a persistent on-disk `.duckdb` database file. Both processes instantiate stateless, in-memory `duckdb.connect(":memory:")` sessions that execute vectorized SQL queries directly over partitioned Apache Parquet files (`read_parquet('data/processed/...')`).
2. **POSIX Atomic Write-and-Rename**: Whenever `amstal-engine` writes or updates a Parquet partition, it writes first to a temporary sibling file (`{ticker}.parquet.tmp`), flushes to disk, and invokes POSIX atomic `os.replace("{ticker}.parquet.tmp", "{ticker}.parquet")`. Concurrent Tauri `duckdb-rs` readers never encounter partial writes, corrupted footers, or file locks.

### 2.3 Process Lifecycle Management & Heartbeat Protocol
1. **Managed Child Daemon**: Upon launch, `amstal-terminal` checks for an active `amstal-engine` listener at `/tmp/amstal.sock` and `127.0.0.1:8194`. If absent, the Tauri Rust backend automatically spawns the engine daemon process.
2. **Heartbeat & Self-Healing**:
   * The terminal issues a ping every 2,000 ms over UDS: `{"jsonrpc": "2.0", "method": "ping", "params": {"timestamp": 1730825600000}, "id": "ping-1"}`.
   * If the engine fails to respond to 3 consecutive pings, the terminal flags the UI with a `"DAEMON_OFFLINE"` amber indicator and triggers an automated zero-downtime hot-restart.
3. **Graceful Teardown**: Closing the terminal sends a `SIGTERM` signal through UDS, prompting `amstal-engine` to complete any in-flight `.parquet.tmp` atomic renames and cleanly unlink `/tmp/amstal.sock`.

### 2.4 Bi-Directional Message Schemas

#### 1. Terminal to Engine: Mnemonic Command Execution (JSON-RPC 2.0)
```json
{
  "jsonrpc": "2.0",
  "method": "execute_command",
  "params": {
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
    },
    "client_timestamp": "2026-11-05T20:00:00.120Z"
  },
  "id": "req-9842a1"
}
```

#### 2. Engine to Terminal: Real-Time Signal & Allocation Stream
Adheres strictly to the QARE-MM v1.3 Target Output API Contract:
```json
{
  "message_id": "sig-0012f4",
  "ticker": "CRWD",
  "timestamp_utc": "2026-11-05T20:00:00Z",
  "horizon_days": 10,
  "winning_engine": "ENSEMBLE_LGBM_GRU32",
  "execution_mode": "LONG_ONLY",
  "data_integrity": {
    "point_in_time_shifted": true,
    "xbrl_filed_date_verified": true,
    "intraday_mode": "HOURLY_FULL",
    "missing_modalities": []
  },
  "macro_regime": {
    "current_state": "LOW_VOLATILITY",
    "state_index": 0,
    "filtered_probabilities": {
      "low_volatility": 0.812,
      "medium_volatility": 0.145,
      "high_volatility": 0.043
    }
  },
  "signal_breakdown": {
    "residual_momentum_20d_rank": 0.78,
    "overnight_return_rank": 0.42,
    "intraday_return_rank": 0.15,
    "amihud_illiquidity_rank": -0.31,
    "closing_power_rank": 0.65,
    "decayed_finbert_sentiment": 0.54,
    "decayed_form4_insider_cluster": 0.82,
    "decayed_pead_shock": 0.35,
    "decayed_10q_legal_risk_delta": -0.04,
    "decayed_mda_catalyst_score": 0.28,
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
    "unconstrained_kelly_pct": 6.40,
    "l1_normalized_portfolio_allocation_pct": 6.40,
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
  * If a command matches an existing mnemonic (e.g., `DES`, `FA`, `GP`, `PORT`, `HMM`), the system routes directly to that module.
  * If the input is natural language (e.g., *"compare gross margins of AMD vs NVDA"*), the Omnibar seamlessly delegates to the local LLM copilot.

### 3.2 Canonical Bloomberg Functions & Two-Tier Phased Rollout

To avoid frontend scope creep and ensure the core quantitative workflow launches cleanly after Gate 4, `amstal-terminal` modules are structured into two phased delivery tiers:

#### Tier 1: Core Quantitative Workstation MVP (Phase 4A)
| Function | Bloomberg Equivalent | Amstal Implementation & Data Source |
| :--- | :--- | :--- |
| **`GP <GO>`** | Graph Price | High-performance TradingView Lightweight Charts with multi-timeframe candlesticks, intraday VWAP, ATR barriers, and overnight/intraday return panes. |
| **`PORT <GO>`** | Portfolio & Kelly Risk | Live QARE-MM v1.3 signal breakdown, calibrated probabilities, Multi-Outcome Fractional Kelly sizing, and $L_1$-normalized portfolio weights. |
| **`HMM <GO>`** | Macro Volatility Regime | Forward-filtered 3-state Gaussian HMM dashboard visualizing real-time probabilities for Low-Volatility, Medium-Volatility, and High-Volatility/Turbulent regimes. |
| **`DES <GO>`** | Description | Security master factsheet, business overview, CIK/FIGI identifiers, GICS classification, market cap, 60d orthogonalized betas, dividend yield. |
| **`FA <GO>`** | Financial Analysis | Point-in-Time 10-year financial statements sourced via `edgartools` + deterministic US-GAAP XBRL tag fallback hierarchy, keyed strictly by `filed` date. |

#### Tier 2: Alt-Data & Peer Graph Suite (Phase 4B)
| Function | Bloomberg Equivalent | Amstal Implementation & Data Source |
| :--- | :--- | :--- |
| **`SPLC <GO>`** | Supply Chain / Peer Map | Interactive dynamic rolling peer graph showing GICS sub-industry peers and 5-day lead-lag cross-correlation networks from `peer_gat.py`. |
| **`EDGAR <GO>`**| SEC Filings Viewer | SEC 10-K/10-Q filing browser with automated Item 1A / MD&A `difflib` redlines and Form 4 open-market insider cluster analysis. |
| **`NEWS <GO>`** | Real-Time News Feed | Financial headline aggregator with local FinBERT sentiment scoring and continuous exponential half-life decay ($\tau=3\text{d}$). |
| **`HP <GO>`** | Historical Price | Interactive tabular historical price, volume, split/dividend adjustments, and overnight vs. intraday return decomposition. |
| **`ANR <GO>`** | Analyst Recommendations | Consensus price targets, recommendation distributions (Buy/Hold/Sell), and revision trajectories. |

### 3.3 4-Panel Reactive Split-Pane Workspace (`paneforge` + Svelte 5 Runes)
* Replicates Bloomberg's canonical 4-panel multi-monitor workstation using native Svelte 5 reactive runes (`$state`, `$derived`) and `paneforge` / CSS Grid (avoiding legacy imperative DOM managers like GoldenLayout).
* **Color Channel Synchronization**: Panels support linked color channels (Blue, Amber, Green, Red). Linking Panel 1 (Chart) and Panel 2 (Financials) to the Blue Channel means entering `AAPL <GO>` in Panel 1 automatically updates Panel 2 to Apple's Point-in-Time financials.

---

## 4. Features Reverse-Engineered from OpenBB

### 4.1 Modular Provider Architecture
Inspired by OpenBB's plug-in architecture, `amstal-engine` decouples data fetching into independent, hot-swappable providers:
* **`amstal-provider-polygon`**: High-frequency hourly/daily OHLCV, VWAP, and market quotes.
* **`amstal-provider-edgar`**: Built on `edgartools` with a deterministic US-GAAP XBRL tag fallback hierarchy (handling tag drift across `Revenues`, `SalesRevenueNet`, `RevenueFromContractWithCustomerExcludingAssessedTax`), Form 4 XML insider transactions, and 10-Q/10-K text extraction.
* **`amstal-provider-fred`**: Federal Reserve Economic Data (Treasury yields, CPI, GDP, VIX, monetary aggregates).
* **`amstal-provider-yfinance`**: Free end-of-day price histories and basic ticker profiles for zero-cost 10-year daily backtesting.
* **`amstal-provider-binance`**: Real-time 24/7 crypto L2 order-book feeds for live streaming pipeline verification.

### 4.2 Pydantic v2 Unified Schema Normalization
Eliminates vendor-specific schema discrepancies by standardizing incoming payloads into immutable Pydantic v2 data models:
* `StandardCandle`: `(timestamp, open, high, low, close, volume, vwap, is_intraday)`
* `StandardIncomeStatement`: `(period_end, filed_date, revenue, gross_profit, operating_income, net_income, eps_diluted)`
* `StandardInsiderTransaction`: `(filing_date, acceptance_datetime, transaction_date, insider_name, relation, transaction_code, is_10b5_1, shares, price)`

### 4.3 Stateless In-Process Columnar Lakehouse (DuckDB + Parquet)
* Ingested market data and computed features are stored in partitioned Apache Parquet files: `data/processed/{domain}/{ticker}.parquet`.
* **Stateless `:memory:` DuckDB**: Executes analytical OLAP queries across 20 years of price/feature data across all 250 universe stocks in **under 15 milliseconds** with zero file-lock contention.

### 4.4 Hybrid Backfill & Local Agentic Copilot (Ollama + LanceDB)
* **One-Time Historical SEC Diff Backfill**: To avoid ~40 hours of continuous compute on the fanless M5 Air when scoring ~8,000 historical 10-Q/10-K redline diffs (`250 stocks * 4 quarters * 8 years`), `sec_diff.py` supports an optional high-throughput batch API adapter (Groq / DeepInfra / Together running `Llama-3.1-8B-Instruct` at `temperature=0`, strict JSON schema) to materialize the historical Parquet cache once.
* **100% Local Ongoing & Interactive Inference**: All daily/quarterly ongoing SEC filings and interactive terminal Copilot queries run 100% locally on Apple Silicon via **Ollama** (`Llama-3.1-8B-Instruct` Q4_K_M / Q8_0) paired with an embedded **LanceDB** vector store and deterministic tool calling (`get_kelly_allocation`, `get_regime_state`, `get_insider_sentiment`).

---

## 5. Quantitative Alpha & Risk Engine Specification (QARE-MM v1.3)

### 5.1 Point-in-Time Universe Selection & Survivorship Bias Mitigation
* **Universe Constituents**: 250 liquid US mid-to-large-cap equities selected from S&P 500 (ranks 100–500) and S&P MidCap 400, excluding Financials (GICS 40) and Utilities (GICS 55).
* **Survivorship-Bias-Free Annual Snapshots**:
  * Stored in `data/universe/constituents_{YYYY}.parquet`.
  * Constructed from historical S&P 500/400 index joiners/leavers tables combined with trailing 90-day median daily dollar volume ($>\$25\text{M}$ ADV) as of January 1st of each walk-forward year $YYYY$.
  * Explicitly retains delisted, acquired, and bankrupt tickers (e.g., `SIVB`, `FRC`, `ATVI`) during the years they were active constituents, incorporating delisting returns on terminal dates so backtest IC and Sharpe are never inflated by survivor bias.

### 5.2 Market- & Sector-Residualized Returns (Sequential 2-Stage OLS Orthogonalization)
Because the market ETF (`SPY`) and GICS Sector ETFs (`ETF_s`, e.g., `XLK`, `XLV`) frequently exhibit correlations $\rho \ge 0.80$, unconstrained single-stage multivariable OLS suffers from severe multi-collinearity, causing beta coefficients to swing wildly with unstable signs. Amstal mathematically eliminates this via a **2-Stage Sequential OLS Orthogonalization** over rolling 60-day historical windows $[t-60, t-1]$:

1. **Stage 1: Sector ETF Orthogonalization against SPY**:
   Regress sector return on market return over $[t-60, t-1]$:
   $$r_{\text{ETF}_s, t} = \hat{\alpha}_{s, t-1} + \hat{\beta}_{s, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \nu_{s, t}$$
   Extract the orthogonal sector residual:
   $$r_{\text{ETF}_s, \perp, t} = \nu_{s, t} = r_{\text{ETF}_s, t} - \left(\hat{\alpha}_{s, t-1} + \hat{\beta}_{s, \text{mkt}, t-1} \cdot r_{\text{SPY}, t}\right)$$
   By construction: $\text{Cov}(r_{\text{SPY}, t}, r_{\text{ETF}_s, \perp, t}) = 0$.

2. **Stage 2: Stock Residualization against SPY and Orthogonal Sector**:
   Regress total stock log-return on market and orthogonal sector residual over $[t-60, t-1]$ with an intercept:
   $$r_{i, \tau} = \alpha_{i, t-1} + \beta_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, \tau} + \beta_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, \perp, \tau} + e_{i, \tau}$$
   Out-of-sample residualization on day $t$ strips strictly the systematic factor exposures (never subtracting the stock's idiosyncratic drift $\alpha$):
   $$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$
   $$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, \perp, t}\right)$$

This guarantees stable regression coefficients ($\hat{\beta}_{i, \text{mkt}}$ remains the true total market beta, while $\hat{\beta}_{i, \text{sec}}$ captures pure industry exposure orthogonal to `SPY`), yielding a mathematically robust idiosyncratic residual return $\epsilon_{i, t}$ that isolates true stock-specific alpha.
* **Numerical Stability & Strict Out-of-Sample Projection**:
  * Both Stage 1 and Stage 2 OLS regressions for day $t$ are fit strictly over the historical window $\tau \in [t-60, t-1]$ (where $\nu_{s, \tau}$ and $r_{\text{SPY}, \tau}$ are orthogonal inside the window, decomposing Stage 2 into two independent univariate projections) and applied out-of-sample to day $t$.
  * Denominator variances include a ridge floor ($\text{Var}(x) + 10^{-8}$) and betas are clipped to $\hat{\beta}_{i, \text{mkt}} \in [-1.0, +4.0]$ and $\hat{\beta}_{i, \text{sec}} \in [-3.0, +3.0]$ to prevent division-by-zero or corporate-action spikes during trading halts.

### 5.3 Cross-Sectional Daily Rank Normalization
Enforces stationary distributions across market regimes by ranking every feature $x_{i,t}$ across the active universe ($M_t \approx 250$) on day $t$ into $[-1.0, +1.0]$:
$$\tilde{x}_{i, t} = 2 \cdot \left(\frac{\text{rank}(x_{i, t}) - 1}{M_{t, \text{valid}} - 1}\right) - 1.0 \in [-1.0, +1.0]$$
* **NaN & Tie Safeguard**: Uses `scipy.stats.rankdata(x, method='average', nan_policy='omit')` where $M_{t, \text{valid}}$ is the count of non-NaN universe assets on day $t$ for that channel. Any missing values are imputed post-ranking to `0.0` (the neutral cross-sectional median) so a single missing bar never contaminates the 250-stock cross-section.

### 5.4 Hybrid Dual-Mode Intraday Microstructure & Overnight Decomposition `(Batch, 126, 15)`
Informed institutional repricing concentrates overnight and at the closing cross, while retail liquidity shocks dominate intraday. To allow 10-year walk-forward backtests to run immediately on free daily data while exploiting full hourly bars when Polygon/Alpaca credentials are present, channels `12–14` support an automatic **Hybrid Dual-Mode Fallback**:

| Channel | Feature Name | Primary Formula (Intraday Hourly Bars Available) | Automatic Fallback Proxy (Daily OHLCV Only) |
| :---: | :--- | :--- | :--- |
| `0` | Daily Log-Return | $r_t = \ln(C_t / C_{t-1})$ | Identical |
| `1` | Idiosyncratic Residual Return | $\epsilon_t$ via 2-Stage Sequential OLS | Identical |
| `2` | Overnight Log-Return | $\ln(O_t / C_{t-1})$ | Identical |
| `3` | Intraday Session Log-Return | $\ln(C_t / O_t)$ | Identical |
| `4–6`| Residual Momentum (5d, 20d, 60d) | $\sum_{k=0}^{L-1} \epsilon_{t-k}$ for $L \in \{5, 20, 60\}$ | Identical |
| `7` | Relative Volume Ratio | $V_t / \bar{V}_{20}$ | Identical |
| `8` | Amihud Illiquidity Z-Score | 60d Z-score of $\|r_t\| / (C_t \cdot V_t)$ | Identical |
| `9` | ATR Volatility Ratio | $\text{ATR}_{14, t} / C_t$ | Identical |
| `10`| Idiosyncratic Volatility (20d) | $\sqrt{252 \cdot \frac{1}{20}\sum_{k=0}^{19} \epsilon_{t-k}^2}$ | Identical |
| `11`| Parkinson Volatility | $(\ln(H_t / L_t))^2 / (4 \ln 2)$ | Identical |
| `12`| Realized Intraday Volatility | $\sigma_{\text{intraday}, t} = \sqrt{\sum_{h=1}^7 r_{t, h}^2}$ | Rogers-Satchell Volatility: $\sqrt{\ln(H_t/C_t)\ln(H_t/O_t) + \ln(L_t/C_t)\ln(L_t/O_t)}$ |
| `13`| Closing Power | $\ln\left(\frac{V_{t, \text{hour 7}} + 10^{-5}}{V_{t, \text{hour 1}} + 10^{-5}}\right)$ | Volume-Scaled CLV Skew: $\left(\frac{2C_t - H_t - L_t}{H_t - L_t + 10^{-8}}\right) \cdot \ln\left(\frac{V_t}{\bar{V}_{20}} + 10^{-5}\right)$ |
| `14`| Close-to-VWAP Deviation | $(C_t - \text{VWAP}_t) / \text{VWAP}_t$ | Close-to-Typical-Price Deviation: $(C_t - \text{TP}_t)/\text{TP}_t$, where $\text{TP}_t = \frac{H_t+L_t+C_t}{3}$ |

### 5.5 Structured Multi-Scale Tabular Summary for LightGBM `(Batch, 105)`
At Gate 1, Sample-Weighted LightGBM/CatBoost competes directly against the 1-Layer Attention-GRU (`hidden_size = 32`). While the GRU consumes the 3D tensor `(Batch, 126, 15)`, `backbone_lgbm.py` maps each of the 15 rank-normalized channels over the 126-day lookback window into **7 structured multi-scale summary statistics** ($15 \times 7 = 105$ features):
1. Current cross-sectional value: $\tilde{x}_{i, t}$
2. Short-term EWMA ($\text{span} = 5\text{d}$): $\text{EWMA}_5(\tilde{x})_{i, t}$
3. Monthly EWMA ($\text{span} = 21\text{d}$): $\text{EWMA}_{21}(\tilde{x})_{i, t}$
4. Quarterly EWMA ($\text{span} = 63\text{d}$): $\text{EWMA}_{63}(\tilde{x})_{i, t}$
5. Half-Year EWMA ($\text{span} = 126\text{d}$): $\text{EWMA}_{126}(\tilde{x})_{i, t}$
6. Monthly Volatility (21-day rolling standard deviation): $\text{Std}_{21}(\tilde{x})_{i, t}$
7. Monthly Trend Velocity (21-day OLS linear regression slope): $\text{Slope}_{21}(\tilde{x})_{i, t}$

### 5.6 Continuous Event-Decay Kernel `(Batch, 12)`
Asynchronous catalysts decay exponentially over elapsed trading days ($\Delta d$) since their Point-in-Time acceptance timestamp ($>16:00\text{ America/New_York} \to t+1$):
$$X_{\text{event}}(t) = X_0 \cdot \exp\left(-\frac{\ln(2)}{\tau} \cdot \Delta d\right)$$

| Index | Feature Name | Source / Kernel |
| :---: | :--- | :--- |
| `0-1` | Decayed FinBERT Sentiment & Age | News headlines passed through local FinBERT ($\tau = 3\text{d}$) |
| `2-3` | Decayed Form 4 Insider Score & Age | SEC EDGAR Form 4 open-market cluster purchases/sales ex-10b5-1 ($\tau = 20\text{d}$) |
| `4-5` | Decayed PEAD Earnings Shock & Age | Post-earnings announcement volume-scaled residual shock ($\tau = 10\text{d}$) |
| `6` | Decayed 10-Q Item 1A Risk Delta | `difflib` Item 1A redline delta scored via `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `7` | Decayed MD&A Catalyst Score | `difflib` Item 7/Item 2 redline delta scored via `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `8` | SEC Filing Staleness Age | Time elapsed since SEC acceptance timestamp ($>16:00\text{ America/New_York} \to t+1$) |
| `9-11`| Modality Missingness Masks | Explicit binary indicators (news, insider, 10-Q) |

### 5.7 Dynamic Rolling Peer Graph Features `(Total_Nodes, 10)`
- **10 Nodes per Subgraph:** Target ($v_0$), 4 GICS sub-industry peers ($v_1 \dots v_4$), 4 rolling 5-day lead-lag cross-correlation peers ($v_5 \dots v_8$), and the GICS Sector ETF ($v_9$).
| Index | Node Feature Description |
| :---: | :--- |
| `0` | 20-day residual return $\sum \epsilon_{t-19:t}$ |
| `1` | 5-day residual return $\sum \epsilon_{t-4:t}$ |
| `2` | 60-day rolling Beta vs SPY ETF |
| `3` | Point-in-Time Earnings Yield ($\text{TTM EPS}_{\text{filed}} / \text{Price}_t$) peer Z-score |
| `4` | Point-in-Time YoY Quarterly Revenue Growth ($\text{Rev}_{\text{filed}}$) peer Z-score |
| `5` | Point-in-Time Operating Margin ($\text{OpInc}_{\text{filed}} / \text{Rev}_{\text{filed}}$) peer Z-score |
| `6` | 20-day Realized Idiosyncratic Volatility peer Z-score |
| `7` | 5-day Lead-Lag cross-correlation coefficient with Target Node ($v_0$) |
| `8` | Normalized Market Cap: $\ln(\text{MarketCap}_t) / 10$ |
| `9` | Categorical Node Role: $0.0 = \text{Target}, 0.33 = \text{GICS Peer}, 0.66 = \text{Lead-Lag Peer}, 1.0 = \text{Sector ETF}$ |

### 5.8 Variance-Sorted Forward-Filtered HMM (`regime_hmm.py`)
Fits a 3-state Gaussian HMM (10 random EM restarts) on standardized daily `SPY` log-returns and `VIX` levels:
1. **Fold-Scoped Standardization**: The `StandardScaler` for `SPY` and `VIX` is fit strictly on the 756-day training fold and applied out-of-sample to the validation and test folds.
2. **Custom Log-Space Forward Filter (No `hmmlearn.predict_proba` Smoother Leak)**: Because `hmmlearn.GaussianHMM.predict_proba(X)` executes the forward-backward Baum-Welch smoother (leaking future states $x_{t+1:T}$), `regime_hmm.py` explicitly implements the causal log-space forward recursion using `scipy.special.logsumexp`:
   $$\log \tilde{\alpha}_t(k) = \log P(x_t \mid S_t = k) + \text{logsumexp}_{j=0}^2 \left(\log \alpha_{t-1}(j) + \log A_{j, k}\right), \quad p_t(k) = \exp\left(\log \tilde{\alpha}_t(k) - \text{logsumexp}_{j=0}^2 \log \tilde{\alpha}_t(j)\right)$$
   This prevents both forward-looking bias and floating-point underflow over long time horizons.
3. **State Ordering by Return Variance**: Sorts states by return variance component $\Sigma_{0,rr} < \Sigma_{1,rr} < \Sigma_{2,rr}$ to enforce permutation invariance across walk-forward folds:
   `[State 0: Low-Volatility, State 1: Medium-Volatility, State 2: High-Volatility/Turbulent]`.

### 5.9 Purged Triple-Barrier Labeling, Uniqueness Weighting & Walk-Forward Geometry
Ternary labels $y_{i, t} \in \{0: \text{Buy}, 1: \text{Hold}, 2: \text{Sell}\}$ are generated on cumulative residual return paths $C_{\epsilon, i}(t, k) = \sum_{\Delta=1}^{k} \epsilon_{i, t+\Delta}$ for $k \in \{1 \dots 10\}$:
* **1D Residual Volatility Scale ($\sigma_{\text{ATR}(\epsilon), i, t}$)**: Because $\epsilon_{i, t}$ is a 1D residual series rather than an OHLC bar, its horizon-scaled volatility unit is defined causally over $[t-19, t]$ as:
  $$\sigma_{\text{ATR}(\epsilon), i, t} = \sqrt{10} \cdot \sqrt{\frac{1}{20} \sum_{m=0}^{19} \epsilon_{i, t-m}^2}$$
* **Upper Barrier:** $+2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$ (Label `0: Buy/Outperform` at first $k \le 10$ where $C_{\epsilon, i}(t, k) \ge +2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$)
* **Lower Barrier:** $-1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$ (Label `2: Sell/Underperform` at first $k \le 10$ where $C_{\epsilon, i}(t, k) \le -1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$)
* **Vertical Barrier:** $H = 10\text{ trading days}$ (Label `1: Hold/Neutral` if neither horizontal barrier is breached by $k = 10$, with $t_{\text{end}} = t + 10$)
* **Intra-Ticker Sample Uniqueness Weighting**: Observations are weighted by average inverse **intra-ticker** concurrency $u_{i,t} = \frac{1}{t_{\text{end}} - t + 1} \sum_{\tau=t}^{t_{\text{end}}} \frac{1}{c_{i,\tau}}$, where $c_{i, \tau} \in [1, 10]$ counts active overlapping barrier windows for ticker $i$ on day $\tau$. Weights $u_{i, t}$ are mean-normalized across each training fold ($\frac{1}{N_{\text{train}}}\sum u_{i, t} = 1.0$) to preserve gradient scale.
* **Locked Walk-Forward Fold Geometry (`walk_forward.py`)**:
  ```
  [ 246d Causal Feature Warm-Up ] ──► [ Rolling 3-Year Train (756d) ] ──► [ 10d Purge ] ──► [ 6-Month Val (126d) ] ──► [ 10d Purge ] ──► [ 3-Month OOS Test (63d) ]
  ```
  * **Causal Feature Pre-Computation vs. Fold-Scoped Parametric Fitting**: Purely causal rolling features (`60d OLS` + `60d Momentum/Amihud` + `126d Sequence Window` = $246\text{d}$ total warm-up burn-in) are computed continuously along the causal timeline so Validation ($126\text{d}$) and OOS Test ($63\text{d}$) folds never lose their initial rows to warm-up `NaN`s. Conversely, all parametric models (`LightGBM`, `GRU`, `GAT`, `HMM`, `TemperatureScaler`, and `Kelly` payoff centering) are fit strictly inside their purged fold boundaries.
  * **Purge Rule:** Any observation across all 250 tickers whose barrier resolution timestamp $t_{\text{end}}$ extends past the split cutoff date ($t_{\text{end}} > D_{\text{cutoff}}$) is strictly purged.
  * **Validation Split Purpose:** Used exclusively for model early stopping, LightGBM/GRU ensemble log-likelihood weighting, and fitting the post-hoc Temperature Scaler ($T > 0$) without touching the 3-month Out-of-Sample test fold.
  * **Re-fit Cadence:** Steps forward by 63 trading days (quarterly re-fit).

### 5.10 Neural Architecture Topology & The `(B, 66)` Fused Tensor

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AMSTAL NEURAL TOPOLOGY (v1.3)                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Dual-Mode Price & Microstructure]   [SEC EDGAR & NLP Alt-Data]       [Rolling Peer & GICS Graph]
      Shape: (Batch, 126, 15)              Shape: (Batch, 12)             Nodes: (Total_Nodes, 10)
  (Also summarized -> (Batch, 105)                 │                     Edges: (2, Total_Edges)
     for Sample-Weighted LightGBM)                 │                                │
                 │                                  ▼                                ▼
                 ▼                      ┌───────────────────────┐          ┌───────────────────┐
     ┌───────────────────────┐          │ Linear(12 -> 32)      │          │ 2-Layer PyG GAT   │
     │ 1-Layer GRU / LSTM    │          │ + LayerNorm           │          │ (CPU-Pinned, 32d) │
     │ + 2-Head Self-Attn    │          │ + GELU + Dropout(0.2) │          │ Root extracted    │
     │ (CPU-Pinned, d = 32)  │          └───────────┬───────────┘          │ via batch.ptr[:-1]│
     └───────────┬───────────┘                      │ (B, 1, 32)           └─────────┬─────────┘
                 │ (B, 126, 32)                     │                                │
                 └────────────────┬─────────────────┘                                │
                                  ▼                                                  │
                   ┌─────────────────────────────┐                                   │
                   │ Modality-Masked Fusion      │                                   │
                   │ • Phase 2 (Masks=0):        │                                   │
                   │   Temporal Pooled (B, 32)   │                                   │
                   │ • Phase 3 (Masks=1):        │                                   │
                   │   Cross-Modal Attn (B, 32)  │                                   │
                   └──────────────┬──────────────┘                                   │
                                  │ (B, 32)                                          │ (B, 32)
                                  └──────────────────────┬───────────────────────────┘
                                                         ▼
                                    Concat: [ (B, 32) || (B, 32) || [D+, D-] (B, 2) ]
                                                      = (B, 66)
                                                         │
    [Filtered HMM Probabilities]                         ▼
        Shape: (Batch, 3) ───────────────────► ┌───────────────────┐
                                               │ Sigmoid Gate      │
                                               │ Linear(3 -> 66)   │
                                               └─────────┬─────────┘
                                                         │ (B, 66)
                                                         ▼
                                               ┌───────────────────┐
                                               │ Classification    │
                                               │ Linear(66 -> 32)  │
                                               │ + LayerNorm + SiLU│
                                               │ + Dropout(0.3)    │
                                               │ + Linear(32 -> 3) │
                                               └─────────┬─────────┘
                                                         │ Raw Logits: (B, 3)
                                                         ▼
                                               ┌───────────────────┐
                                               │ Temperature Scale │
                                               │ Softmax(z / T)    │
                                               └─────────┬─────────┘
                                                         │ Calibrated Probabilities: (B, 3)
                                                         ▼
                                         L1-Normalized Multi-Outcome Kelly Sizer
```

The $(B, 66)$ latent vector entering the HMM Sigmoid Gate in `fusion_net.py` is locked across Phases 2 and 3 as:
$$\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{temporal\_or\_cross}}^{(32)} \,\|\, \mathbf{h}_{\text{graph}}^{(32)} \,\|\, \mathbf{D}_t^{(2)}] \in \mathbb{R}^{(B, 66)}$$
1. **Channels `0..31` — Temporal / Cross-Modal Representation $(B, 32)$**:
   * **Phase 2 Mode (Alt-Data Missingness Masks = `0.0`)**: Bypasses Cross-Modal Attention and passes the 2-head self-attention pooled temporal representation $\mathbf{h}_{\text{temporal}} \in \mathbb{R}^{(B, 32)}$ directly.
   * **Phase 3 Mode (Alt-Data Active)**: Uses the Alt-Data vector $(B, 1, 32)$ as Query $Q$ and Temporal sequence $(B, 126, 32)$ as Key/Value $K, V$ with residual connection $t_{126}$.
2. **Channels `32..63` — Dynamic Rolling Peer GAT Output $(B, 32)$**:
   * Root target node $v_0$ (always indexed at local node `0` of each 10-node subgraph) is extracted via `batch.ptr[:-1]`.
   * **Bidirectional Edge + Self-Loop Topology**: Subgraph edges in `graph_builder.py` are constructed bidirectionally between peers/ETF ($v_1 \dots v_9$) and target ($v_0$) with self-loops enabled (`add_self_loops=True`), ensuring $v_0$ aggregates incoming messages from all 9 peer nodes in PyG `GATConv` (using pure `torch-geometric>=2.5.0` native PyTorch scatter ops without legacy C++ `torch-scatter` compilation).
3. **Channels `64..65` — Signed Smart-Money Divergence $[D_t^+, D_t^-] \in \mathbb{R}^{(B, 2)}$**:
   * **Phase 2 Mode**: Zero-masked to `[0.0, 0.0]` prior to FinBERT activation.
   * **Phase 3 Mode**: Decomposes volume-weighted sentiment divergence $D_t = S_{\text{FinBERT}, t} \times \left(-\frac{\epsilon_t \cdot V_t}{\bar{V}_{20}}\right)$ into explicit institutional distribution and accumulation channels:
     $$D_t^+ = \max(D_t, 0) \quad \text{(Media Euphoria into Institutional Selling / Distribution)}$$
     $$D_t^- = \max(-D_t, 0) \quad \text{(Media Panic into Institutional Buying / Accumulation)}$$

### 5.11 Temperature Calibration, Configurable Execution Mode & Portfolio-Level $L_1$ Kelly Sizing
1. **Temperature Scaling (Post-Hoc Calibration)**:
   Fits scalar $T > 0$ via Negative Log-Likelihood minimization on the purged 126-day validation split:
   $$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_{j=0}^2 \exp(z_j / T)}$$
2. **Configurable Execution Mode (`LONG_ONLY` Default vs. `LONG_SHORT` Diagnostics)**:
   * **Default Portfolio Mode (`execution_mode = "LONG_ONLY"`)**: Constrains single-stock allocation to $f_i \in [0, f_{\max}]$ with $f_{\max} = 0.10$. Optimal long allocation $f_i^*$ is determined strictly by the Kelly objective: when expected marginal net return is non-positive ($\left.\frac{d\mathbb{E}[\ln(1 + f R_{\text{net}})]}{df}\right|_{f=0} \le 0$), the optimizer yields $f_i^* = 0.0$ (holding/exiting into cash, never opening margin short positions). Discrete UI action labels (`BUY`, `HOLD`, `SELL`) are formatted separately from position state and probability rankings.
   * **Diagnostic Stat-Arb Mode (`execution_mode = "LONG_SHORT"`)**: Evaluates cross-sectional Rank Information Coefficient (IC) and theoretical dollar-neutral Long-Short portfolio Sharpe in `metrics.py`.
3. **Single-Stock Multi-Outcome Kelly Optimization & Base-Rate Payoff Centering**:
   Because the asymmetric $+2\sigma / -1\sigma$ barriers have unequal unconditional hit rates ($\bar{p}_{\text{buy}}, \bar{p}_{\text{hold}}, \bar{p}_{\text{sell}}$), raw outcome payoffs $R_{\text{upper}}, R_{\text{lower}}, R_{\text{vert}}$ (estimated from validation class means) are centered against the validation prior $\bar{\mathbf{p}}$ to define centered payoff variables:
   $$\mu_{\bar{p}} = \bar{p}_{\text{buy}} R_{\text{upper}} + \bar{p}_{\text{sell}} R_{\text{lower}} + \bar{p}_{\text{hold}} R_{\text{vert}}$$
   $$\tilde{b}_{\text{upper}} = R_{\text{upper}} - \mu_{\bar{p}}, \quad \tilde{b}_{\text{lower}} = R_{\text{lower}} - \mu_{\bar{p}}, \quad \tilde{b}_{\text{vert}} = R_{\text{vert}} - \mu_{\bar{p}} \implies \sum_k \bar{p}_k \tilde{b}_k = 0$$
   This mathematically guarantees **Gate 4 Criterion #2**: any zero-skill or uninformative model outputting prior probabilities ($\hat{\mathbf{p}} = \bar{\mathbf{p}}$) has expected marginal net return $\left.\frac{d\mathbb{E}[\ln(1 + f R_{\text{net}})]}{df}\right|_{f=0} = \sum_k \bar{p}_k \tilde{b}_k - c = -0.0015 < 0$, yielding strictly $f_i^* = 0.0\%$ allocation.
   For informative signals ($\sum_k \hat{p}_k \tilde{b}_k > c$), `kelly_engine.py` solves the concave log-wealth objective net of 15 bps friction ($c = 0.0015$) using centered payoffs:
   $$\max_{f \in [0, f_{\max}]} \mathbb{E}[\ln(1 + f \cdot R_{\text{net}})] = \hat{p}_{\text{buy}} \ln(1 + f (\tilde{b}_{\text{upper}} - c)) + \hat{p}_{\text{sell}} \ln(1 - f (|\tilde{b}_{\text{lower}}| + c)) + \hat{p}_{\text{hold}} \ln(1 + f (\tilde{b}_{\text{vert}} - c))$$
   Scaled by macro regime multiplier $\kappa \in [0.25, 0.50]$ (Quarter/Half Kelly) with a single-stock ceiling of $10\%$ ($15\%$ absolute hard cap).
4. **Portfolio-Level $L_1$ Gross Exposure Normalization**:
   To prevent concurrent buy signals across the 250-stock universe from exceeding 100% gross portfolio capital:
   $$w_i = \frac{f_i^* \cdot \kappa}{\max\left(1.0, \sum_{j=1}^N |f_j^* \cdot \kappa|\right)}$$
   This guarantees $\sum_{i=1}^N |w_i| \le 1.0$ (100% gross equity exposure), eliminating leverage blowups while preserving relative conviction weights.

---

## 6. Phased Implementation Roadmap & Falsification Gates

```
Phase 0: Lock Unified Specification v2.1 ──► [GATE 0: Complete — All Math & Decisions Locked]
          │
          ▼
Phase 1: Residualized Backbone           ──► [GATE 1: Passes Shuffled/Lagged Sanity Checks;
         (2-Stage OLS, 15ch + 105ch LGBM)             Beats Baseline Net of 15 bps OOS; IC > 0.025]
          │
          ▼
Phase 2: Regime HMM & Dynamic Peer GAT   ──► [GATE 2: HMM Reduces OOS Drawdown >= 15%;
         (CPU-Pinned, Modality-Masked)                Peer GAT Improves OOS Log-Likelihood / Sharpe]
          │
          ▼
Phase 3: SEC EDGAR & Alt-Data Pipeline   ──► [GATE 3: Form 4, PEAD, FinBERT & 10-Q Diffs
         (edgartools + Hybrid LLM)                    Add Incremental IC > 0.015 Post-Cutoff]
          │
          ▼
Phase 4: Calibration, Kelly & Terminal   ──► [GATE 4: Brier Drops > 10% OOS; Zero-Skill -> 0.0%;
         (Tier 1 Core UI -> Tier 2 Suite)             Tier 1 4-Panel Svelte 5 Terminal Verified]
```

### Non-Negotiable Stop Rules
1. **No Modality Stacking Before Verification**: Do not implement GAT, SEC scrapers, or Tauri UI components until Phase 1 passes Gate 1 on pure market/microstructure data.
2. **Model Agnosticism**: If Sample-Weighted LightGBM/CatBoost (`(Batch, 105)`) outperforms the Attention-GRU (`(Batch, 126, 15)`) out-of-sample, prune the neural sequence branch or weight strictly by validation log-likelihood.
3. **Sharpe Ratio Cap**: Any backtest yielding annualized Sharpe $> 2.5$ or win rate $> 65\%$ on 10-day swing horizons flags an automatic timestamp leak until disproven.

---

## 7. Hardware & Thermal Ergonomics (Apple Silicon M5 Air 24 GB)

* **Memory Budget**:
  * macOS Core: ~4.6 GB
  * Amstal Terminal (Tauri 2.0 Webview): ~150 MB (vs 900 MB for Electron)
  * Amstal Engine & LightGBM/PyTorch: ~2.8 GB
  * Stateless `:memory:` DuckDB Buffer: ~2.0 GB
  * Available Headroom: ~15.0 GB
* **CPU Execution Pinning for PyG & Sequence Models**:
  * PyTorch Geometric (PyG) scatter/gather operations in `GATConv` lack native Apple Metal (MPS) kernels and incur expensive CPU fallback copies.
  * For compact subgraphs (10 nodes per stock) and `hidden_size = 32` GRUs, CPU execution is faster than MPS due to zero kernel launch overhead.
  * `peer_gat.py` and `backbone_gru.py` are explicitly pinned to CPU threads via `torch.set_num_threads(4)`.
* **Thermal Throttling Defense**:
  * Sustained full-load compute on the fanless M5 Air causes thermal throttling after ~12–15 minutes.
  * With `hidden_size = 32`, CPU thread pinning, and pre-computed Parquet caches, a full walk-forward fold across 250 stocks completes in under 30 seconds, staying strictly inside the burst thermal envelope.

---

## 8. Legal, Regulatory & Licensing Boundaries

1. **Market Data Redistribution TOS**:
   * Raw and processed vendor files (`data/*.parquet`) are strictly `.gitignored`. Users must supply their own API keys for Polygon.io, Tiingo, or Alpaca. Distributing cached vendor data violates redistribution agreements.
2. **SEC EDGAR Rate-Limiting & Compliance**:
   * All requests to `sec.gov` strictly enforce a token-bucket rate limiter capped at $\le 10\text{ requests/sec}$.
   * Every HTTP request in `sec_edgar.py` must declare a User-Agent header configured via the `SEC_EDGAR_USER_AGENT` environment variable or `config.toml` conforming to SEC guidelines (format: `User-Agent: <OrganizationName> <ValidContactEmail>`, e.g., `User-Agent: AmstalResearch dev@yourdomain.com`). Requests without a valid declared contact email are blocked pre-flight.
3. **Bloomberg Trademark & Trade Dress Disclaimer**:
   * Amstal is an independent open-source project unaffiliated with, nor endorsed by, Bloomberg Finance L.P. Mnemonic command conventions (`DES`, `FA`, `GP`, `<GO>`) are used solely for workflow interoperability.
4. **Non-Investment Advice Disclaimer**:
   * This software is provided strictly for quantitative research, academic study, and educational simulation under the MIT License ("AS IS", without warranty). Outputs (including `BUY`/`SELL` classifications and Kelly allocations) do not constitute financial or investment advice.
5. **Cross-Border Trading / FEMA & LRS Guidelines (India)**:
   * Under Reserve Bank of India (RBI) Liberalised Remittance Scheme (LRS) and FEMA regulations, resident Indian individuals are prohibited from using LRS remittances for overseas margin, leveraged, or derivative trading. Live execution for Indian residents must use `execution_mode = "LONG_ONLY"` (cash-settled equity delivery).

---

## 9. Directory Layout

```
amstal/
├── PRD.md                             # Master Unified PRD (This Document)
├── architecture.md                    # Master End-to-End Technical Architecture
├── README.md                          # Workspace Overview & Monorepo Guide
├── scripts/
│   └── audit_specs.py                 # Automated Spec Audit & Invariant Feedback Loop
├── amstal-engine/                     # Quantitative Alpha & Risk Engine
│   ├── PRD.md                         # Engine-synced PRD
│   ├── architecture.md                # Engine-synced Technical Architecture
│   ├── README.md                      # Quant specs and mathematical formulations
│   ├── LICENSE                        # MIT License
│   ├── pyproject.toml                 # Python build & dependency configuration
│   ├── data/                          # Data store (strictly git-ignored)
│   │   ├── raw/                       # Daily/hourly OHLCV, SEC raw XML/HTML
│   │   ├── processed/                 # Partitioned Parquet Lakehouse (atomic os.replace)
│   │   └── universe/                  # PIT 250-stock annual constituent & delisting tables
│   ├── src/
│   │   ├── data/
│   │   │   ├── market_loader.py       # Hybrid hourly/daily OHLCV loader (Polygon/yfinance)
│   │   │   ├── sec_edgar.py           # edgartools XBRL hierarchy & Form 4 XML parser
│   │   │   └── universe.py            # Survivor-bias-free PIT annual universe selector
│   │   ├── features/
│   │   │   ├── residualizer.py        # 2-Stage Sequential OLS beta orthogonalization
│   │   │   ├── microstructure.py      # 15ch sequence + Dual-Mode fallback + 105ch LGBM summary
│   │   │   ├── sec_diff.py            # Item 1A/MD&A redlines (Hybrid Batch API / Local Ollama)
│   │   │   ├── news_sentiment.py      # FinBERT scoring & continuous event decay
│   │   │   └── graph_builder.py       # Monthly rolling GICS + Lead-Lag PyG graphs
│   │   ├── models/
│   │   │   ├── backbone_gru.py        # CPU-pinned 1-Layer GRU + 2-Head Self-Attention (32d)
│   │   │   ├── backbone_lgbm.py       # Sample-weighted LightGBM on 105 multi-scale features
│   │   │   ├── peer_gat.py            # CPU-pinned 2-Layer PyG GAT with batch.ptr[:-1] slicing
│   │   │   ├── regime_hmm.py          # Forward-filtered 3-state Gaussian HMM (variance-sorted)
│   │   │   └── fusion_net.py          # Modality-masked Cross-Modal Attention, HMM gate & head
│   │   ├── label/
│   │   │   ├── triple_barrier.py      # Residualized triple-barrier labeling
│   │   │   └── weights.py             # Sample uniqueness & cross-ticker calendar-date purge
│   │   ├── risk/
│   │   │   ├── calibration.py         # Temperature scaling via NLL on 126d validation fold
│   │   │   └── kelly_engine.py        # Multi-outcome fractional Kelly + L1 portfolio normalizer
│   │   ├── ipc/
│   │   │   ├── socket_server.py       # UDS (/tmp/amstal.sock) JSON-RPC control plane
│   │   │   └── stream_server.py       # WebSocket real-time 100ms conflated broadcaster (:8194)
│   │   └── backtest/
│   │       ├── walk_forward.py        # 756d Train / 10d Purge / 126d Val / 10d Purge / 63d Test
│   │       └── metrics.py             # Rank IC, LONG_ONLY PnL, LONG_SHORT Sharpe & Sanity gates
│   └── tests/                         # Unit tests and Gate 1–4 falsification checks
└── amstal-terminal/                   # Institutional Desktop Terminal (Phase 4A Tier 1 -> 4B Tier 2)
    ├── src-tauri/                     # Rust desktop shell
    │   ├── Cargo.toml
    │   └── src/
    │       ├── main.rs                # Application entrypoint & engine daemon supervisor
    │       ├── ipc_client.rs          # UDS JSON-RPC & WebSocket (:8194) client
    │       └── parquet_query.rs       # Stateless :memory: duckdb-rs reader over Parquet files
    ├── src/                           # Svelte 5 Runes UI
    │   ├── App.svelte                 # Master 4-tile paneforge / CSS Grid workspace
    │   ├── components/
    │   │   ├── Omnibar.svelte         # Mnemonic REPL (<GO>) & fuzzy command palette
    │   │   ├── WorkspaceGrid.svelte   # 4-panel reactive split-pane container
    │   │   ├── ChartTile.svelte       # GP <GO>: TradingView Lightweight Charts (Tier 1)
    │   │   ├── PortfolioTile.svelte   # PORT <GO>: QARE signals & L1 Kelly sizer (Tier 1)
    │   │   ├── RegimeTile.svelte      # HMM <GO>: Forward-filtered HMM regime meter (Tier 1)
    │   │   ├── DescriptionTile.svelte # DES <GO>: Security master profile (Tier 1)
    │   │   ├── FinancialsTile.svelte  # FA <GO>: Point-in-Time XBRL statement table (Tier 1)
    │   │   ├── PeerGraphTile.svelte   # SPLC <GO>: Interactive rolling peer GAT map (Tier 2)
    │   │   ├── EdgarTile.svelte       # EDGAR <GO>: 10-Q redlines & Form 4 clusters (Tier 2)
    │   │   └── CopilotTile.svelte     # Local Ollama / LanceDB financial chat (Tier 2)
    │   └── stores/                    # Svelte 5 $state color-channel linking & tick stores
    ├── package.json
    └── tailwind.config.js
```
