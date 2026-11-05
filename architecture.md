# Technical Architecture: Amstal System
## Unified Architecture for `amstal-engine` (Quant Core) and `amstal-terminal` (Workstation)

---

## 1. Architectural Vision & Topology

**Amstal** couples an institutional quantitative research and risk engine (`amstal-engine`) with an ultra-responsive, Bloomberg-inspired desktop terminal (`amstal-terminal`).

The system is designed to provide hedge-fund caliber data pipelines, mathematical alpha isolation, and multi-asset risk management on personal workstation hardware—specifically optimized for the fanless **Apple Silicon M5 MacBook Air (24 GB Unified Memory)**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          AMSTAL COMPLETE END-TO-END TOPOLOGY                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                      [ EXTERNAL MARKET DATA & FILINGS ]
         ┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
         │ Polygon.io Stocks │ SEC EDGAR (XBRL)  │ FRED Macro        │ Tiingo / Yahoo    │
         │ (Hourly/Daily)    │ (10-Q/K, Form 4)  │ (Yields, CPI)     │ (Historical EOD)  │
         └─────────┬─────────┴─────────┬─────────┴─────────┬─────────┴─────────┬─────────┘
                   │                   │                   │                   │
                   ▼                   ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           AMSTAL ENGINE CORE (Python 3.11+)                                      │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. INGESTION & DATA LAKEHOUSE LAYER                                                                        │  │
│  │   • Provider Plugins (OpenBB Pattern): Standardized Pydantic v2 Models (`StandardCandle`, `StandardXBRL`)   │  │
│  │   • In-Process Columnar OLAP: DuckDB Embedded Engine                                                       │  │
│  │   • Storage: Partitioned Apache Parquet Files (`data/processed/{domain}/{ticker}.parquet`)                 │  │
│  │   • Point-in-Time (PIT) Validator: >16:00 EST rollover enforcement, `filed`-date fundamental alignment     │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. FEATURE ENGINEERING & ALPHA ISOLATION (QARE-MM v1.3)                                                    │  │
│  │   • Rolling 60d OLS Beta Residualization: strips SPY and Sector ETF beta to yield pure epsilon_{i,t}       │  │
│  │   • Intraday Microstructure Tensor (126, 15): Closing Power, Amihud Z-score, VWAP skew, Overnight/Day split│  │
│  │   • Alt-Data Catalyst Vector (12): Continuous exponential decay kernels (tau = 3d, 10d, 15d, 20d)          │  │
│  │   • Dynamic Rolling Peer Graph (10 nodes): Monthly GICS peers + 5d lead-lag correlation (PyG GAT)          │  │
│  │   • Macro Regime HMM: Forward-filtered 3-State Gaussian HMM (SPY + VIX), variance-sorted                   │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. MODELING, CALIBRATION & RISK SIZING                                                                     │  │
│  │   • GBDT vs Neural Net Gating: Sample-Weighted LightGBM vs. Compact GRU-Attention (hidden_size=32)         │  │
│  │   • Cross-Modal Fusion Net: Temporal series query + Alt-data attention + HMM Sigmoid Gate                  │  │
│  │   • Purged Triple-Barrier Labeling: Dynamic ATR barriers with sample uniqueness weighting                  │  │
│  │   • Temperature Scaling: Post-hoc NLL calibration on purged validation set                                 │  │
│  │   • Multi-Outcome Fractional Kelly Sizer: Maximizes expected log-wealth net of 15 bps friction             │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. LOCAL ENGINE IPC SERVER                                                                                 │  │
│  │   • Control Plane: Unix Domain Socket (UDS) / Windows Named Pipe listener                                  │  │
│  │   • Streaming Broadcaster: Local WebSocket Server (`ws://127.0.0.1:8194`) with 100ms conflation           │  │
│  │   • Zero-Copy Data Bridge: Apache Arrow Flight IPC / Shared Memory (mmap) for multi-year timeseries        │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         │ LOCAL INTER-PROCESS COMMUNICATION (IPC) BUS                   │
                         │  - Control Plane (UDS): Sub-millisecond commands & responses  │
                         │  - Stream (Port 8194): Real-time ticks, signals, regime alerts │
                         │  - Bulk (Arrow Flight): Microsecond zero-copy matrix transfers│
                         └───────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AMSTAL TERMINAL WORKSTATION (Tauri 2.0)                                    │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ TAURI 2.0 RUST BACKEND CORE                                                                                │  │
│  │   • Engine Process Supervisor: Launches, health-checks (2s ping), and gracefully terminates engine daemon  │  │
│  │   • UDS & WebSocket Client: Multiplexes incoming data streams to webview frontend                          │  │
│  │   • Hardware Interface: Listens to QMK/VIA USB keyboard events (Starboard macros, Rotary encoder)          │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │ Direct Native IPC                                       │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ SVELTE 5 / SOLIDJS USER INTERFACE LAYER                                                                    │  │
│  │   • 4-Panel Dockable Workspace (GoldenLayout / FlexLayout): Resizable, detachable multi-monitor tiles       │  │
│  │   • Mnemonic Omnibar REPL: Autocomplete parser (`AAPL US <EQUITY> DES <GO>`) + fuzzy hybrid search (`cmdk`)│  │
│  │   • Channel Linking System: Multi-color broadcast channels (Blue/Yellow/Green) syncing ticker across tiles  │  │
│  │   • TradingView Lightweight Charts: GPU WebGL candlestick rendering, volume profile, VWAP bands            │  │
│  │   • Point-in-Time Financial Viewer: Replicates Bloomberg `FA <GO>` from standardized XBRL statements       │  │
│  │   • Local Copilot Tile: Ollama (`Llama-3.1-8B`) + LanceDB vector store for automated 10-K footnote queries │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Terminal-Engine Integration Mechanics

The terminal and engine communicate locally without relying on external cloud infrastructure. This guarantees sub-millisecond execution, complete data privacy, and zero latency overhead.

### 2.1 Communication Protocols & Ports

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  LOCAL IPC CHANNEL TOPOLOGY                                      |
+──────────────────────────┬─────────────────────────────┬─────────────────────────────────────────+
| Channel Name             | Wire Transport              | Message Format / Protocol               |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **Control IPC**          | Unix Domain Socket (UDS)    | Binary Framed JSON-RPC 2.0              |
|                          | Path: `/tmp/amstal.sock`    | (Request / Synchronous Response)        |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **Market Data Feed**     | Local WebSocket             | Multiplexed UTF-8 JSON / Protobuf       |
|                          | URL: `ws://127.0.0.1:8194`  | (Pub/Sub: `ticker`, `regime`, `signals`)|
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **High-Volume Analytics**| Apache Arrow Flight IPC     | Zero-Copy Arrow RecordBatches           |
|                          | Port: `localhost:8195`      | (Direct memory-to-memory dataframes)    |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
```

### 2.2 Sequence Diagram: Mnemonic Command Execution

```
User (Keyboard)         Terminal (UI)          Tauri (Rust)          Engine (Python)         DuckDB / Storage
      │                      │                      │                       │                       │
      │ Type "NVDA US EQ FA" │                      │                       │                       │
      │ ───► <GO> (Enter)    │                      │                       │                       │
      │                      │ Parse Mnemonic       │                       │                       │
      │                      │ ───────────────────► │                       │                       │
      │                      │                      │ UDS Request           │                       │
      │                      │                      │ ────────────────────► │                       │
      │                      │                      │                       │ Resolve PIT Statements│
      │                      │                      │                       │ ────────────────────► │
      │                      │                      │                       │                       │ Parquet Read
      │                      │                      │                       │ ◄──────────────────── │ (< 5ms)
      │                      │                      │                       │ Normalize Schema      │
      │                      │                      │ UDS Response (JSON)   │                       │
      │                      │                      │ ◄──────────────────── │                       │
      │                      │ Rust Emit Event      │                       │                       │
      │                      │ ◄─────────────────── │                       │                       │
      │ Render 10-Yr Grid    │                      │                       │                       │
      │ ◄─────────────────── │                      │                       │                       │
```

### 2.3 Real-Time Tick Conflation & UI Frame Rendering
Human perception cannot digest raw ticks updating at 5,000 messages/second. Furthermore, rendering every tick locks the UI event loop.
* **Engine Ingestion**: Ingests streaming market trades and quotes asynchronously into an in-memory ring buffer.
* **100ms Conflation Loop**: Every 100 milliseconds, the engine computes the net state change (`last_price`, `bid`, `ask`, `cumulative_volume`, `vwap_deviation`) and broadcasts a single conflated frame over `ws://127.0.0.1:8194`.
* **Terminal Dirty-Rect Rendering**: The Svelte UI updates its reactive stores; WebGL canvas charts repaint at 60 FPS without DOM thrashing.

---

## 3. Reverse-Engineered Bloomberg Architecture

### 3.1 Mnemonic Command Grammar & Lexical Parser
Bloomberg's core navigation relies on an expressive Domain-Specific Language (DSL). Amstal recreates this via a deterministic lexer and state machine:

```
               ┌──────────┐
               │  TICKER  │ (e.g., "AAPL", "NVDA", "SPY")
               └────┬─────┘
                    │
                    ▼
           ┌──────────────────┐
           │   MARKET CODE    │ (Optional: "US", "LN", "GY")
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │   ASSET CLASS    │ (<EQUITY>, <GOVT>, <CORP>, <INDEX>, <CURNCY>)
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │     MNEMONIC     │ (e.g., "DES", "FA", "HP", "GP", "PORT", "HMM")
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │    PARAMETERS    │ (e.g., "QUARTERLY", "2024", "10D")
           └────────┬─────────┘
                    │
                    ▼
               ┌──────────┐
               │   <GO>   │ (Execute token / Enter)
               └──────────┘
```

#### Lexer State Machine Implementation
* When the user inputs characters into the Omnibar, the lexer tokenizes strings on whitespace.
* If a token matches a registered asset-class key (`<EQUITY>`, `<INDEX>`), it sets the execution context.
* If the terminal token is `<GO>` (or `Enter` keypress), the parser produces an AST payload dispatched via IPC:
  ```json
  {
    "action": "EXECUTE_MNEMONIC",
    "security": { "ticker": "AAPL", "market": "US", "asset_class": "EQUITY" },
    "command": "FA",
    "params": { "view": "STANDARDIZED", "horizon": "10Y" }
  }
  ```

### 3.2 Canonical Bloomberg Modules

1. **`DES <GO>` (Security Description Factsheet)**:
   * Header: Ticker, Name, Primary Exchange, CIK, GICS Sector/Sub-Industry, FIGI identifier.
   * Market Snapshot: Prior Close, Open, 52-week High/Low, Beta (60d vs SPY), Shares Outstanding, Market Cap.
   * Business Profile: Sourced Point-in-Time from SEC 10-K Item 1 Business summary.
2. **`FA <GO>` (Financial Analysis)**:
   * 10-year horizontal and vertical Point-in-Time financial statement analysis.
   * Balance Sheet, Income Statement, and Cash Flow tables standardized from SEC EDGAR XBRL facts.
   * Restatement-safe: Historical periods display values as originally filed on that date, preventing back-dated restatement bias.
3. **`GP <GO>` & `HP <GO>` (Price Charts & Historical Tables)**:
   * Candlesticks with multi-timeframe resolution (1m, 5m, 1h, 1d).
   * Overlays: Intraday VWAP, 20-day/60-day moving averages, Parkinson volatility envelopes.
   * Microstructure Pane: Overnight return bar vs. Intraday session bar decomposition.
4. **`HMM <GO>` (Macro Volatility Regime Monitor)**:
   * Displays the current forward-filtered Gaussian HMM state:
     * **State 0 (Green)**: Low-Vol Bull Market ($\sigma_{\text{market}} < 12\%$).
     * **State 1 (Yellow)**: Sideways / Normal Volatility ($\sigma_{\text{market}} \approx 12–22\%$).
     * **State 2 (Red)**: High-Vol Crash / Panic ($\sigma_{\text{market}} > 22\%$).
   * Historical regime timeline plotting market transitions without backward-looking smoothing.
5. **`PORT <GO>` & `RISK <GO>` (Portfolio Risk & Kelly Sizing)**:
   * Multi-Outcome Fractional Kelly sizing recommendations net of 15 bps friction.
   * Value-at-Risk (VaR 95% / 99%), Expected Shortfall, and single-stock 10% concentration limits.

### 3.3 Starboard Hardware Integration (QMK / VIA)
Amstal interfaces directly with USB HID devices configured via QMK firmware:
* **Asset Class Hotkeys**: `F1`–`F7` configured as raw USB HID usage codes that the Tauri Rust client maps to asset-class selector tokens (`<EQUITY>`, `<CORP>`, `<GOVT>`, etc.).
* **Action Buttons**: Distinct green Cherry MX switch for `<GO>` (`KC_ENTER`) and red switch for `<CANCEL>` (`KC_ESCAPE`).
* **Encoder Rotary Dial**: Emits custom incremental step codes captured by Tauri to scrub candlestick chart cursors with microsecond fluidity.

### 3.4 4-Panel Dockable Workspace with Color-Coded Linking
* Workspace consists of 4 independent tiles (Panel 1: Top-Left, Panel 2: Top-Right, Panel 3: Bottom-Left, Panel 4: Bottom-Right).
* **Color Channels**: Each tile displays a color-linking selector (Blue, Amber, Green, Purple).
* When Panel 1 and Panel 2 are set to Blue, changing the security in Panel 1 sends an internal event across the Svelte store bus:
  `channelStore.publish('BLUE', { ticker: 'MSFT', market: 'US', assetClass: 'EQUITY' })`.
  Panel 2 receives the event and immediately refreshes its data view for `MSFT`.

---

## 4. Reverse-Engineered OpenBB Architecture

### 4.1 Extensible Provider Architecture
`amstal-engine` structures its data ingestion around OpenBB's modular provider pattern:

```
                      ┌─────────────────────────┐
                      │   BaseProvider (ABC)    │
                      ├─────────────────────────┤
                      │ + fetch_candles()       │
                      │ + fetch_fundamentals()  │
                      │ + fetch_insiders()      │
                      │ + fetch_macro()         │
                      └────────────┬────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  PolygonProvider  │    │   EdgarProvider   │    │    FredProvider   │
├───────────────────┤    ├───────────────────┤    ├───────────────────┤
│ • Hourly/Daily    │    │ • XBRL CompanyFacts│    │ • 10Y/2Y Yields   │
│ • Real-time Quotes│    │ • Form 4 XML Insiders│  │ • CPI, M2, GDP    │
│ • Trades/VWAP     │    │ • 10-Q/K Redlines │    │ • Federal Funds   │
└───────────────────┘    └───────────────────┘    └───────────────────┘
```

### 4.2 Standardized Pydantic v2 Schemas
Every provider transforms raw vendor responses into normalized Pydantic models:

```python
from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional

class StandardCandle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

class StandardIncomeStatement(BaseModel):
    ticker: str
    period_end_date: date
    filing_date: date = Field(..., description="Point-in-Time Acceptance Date")
    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    eps_diluted: float
```

### 4.3 DuckDB Columnar Lakehouse Storage
Instead of managing heavy external database servers (PostgreSQL or ClickHouse), Amstal embeds **DuckDB**:
* Historical quotes and filings are partitioned on disk:
  ```
  data/
  ├── processed/
  │   ├── prices/
  │   │   ├── AAPL.parquet
  │   │   └── NVDA.parquet
  │   ├── fundamentals/
  │   │   └── xbrl_facts.parquet
  │   └── features/
  │       └── cross_sectional_matrix.parquet
  ```
* DuckDB runs vectorized SQL queries directly over Parquet files in memory:
  ```sql
  SELECT timestamp, close, volume,
         (close - open) / open AS intraday_return
  FROM read_parquet('data/processed/prices/NVDA.parquet')
  WHERE timestamp >= '2020-01-01'
  ORDER BY timestamp ASC;
  ```
  Execution completes in **under 15 ms** for 20 years of data.

### 4.4 Local Agentic Financial Copilot
Amstal incorporates an autonomous financial research agent running locally:
1. **Model**: `Llama-3.1-8B-Instruct` running on Apple Silicon via **Ollama** (leveraging Metal acceleration).
2. **Vector Store**: **LanceDB** embedded database indexing:
   * Item 1A (Risk Factors) from 10-K filings.
   * Item 7 (MD&A) operational commentary.
   * Quarterly earnings call transcripts.
3. **Structured Function Calling**:
   * The copilot can execute engine actions:
     * `get_peer_analysis(ticker="CRWD")`
     * `get_macro_regime()`
     * `calculate_kelly_allocation(ticker="NVDA")`

---

## 5. Machine Learning & Quantitative Modeling Pipeline (QARE-MM v1.3)

### 5.1 Idiosyncratic Alpha Isolation
Raw stock returns are approximately 65% macro and sector beta. To isolate idiosyncratic alpha:
$$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, t}\right)$$
Betas are estimated via rolling 60-day OLS on strictly historical data $[t-60, t-1]$.

### 5.2 Tensor Feature Representations

```
                                  [ MULTI-MODAL INPUT TENSORS ]
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌─────────────────────────────┐        ┌─────────────────────────────┐        ┌─────────────────────────────┐
│ 1. Time-Series Microstructure│        │ 2. SEC & Alt-Data Vector    │        │ 3. Dynamic Rolling Peer Graph│
│ Shape: (Batch, 126, 15)     │        │ Shape: (Batch, 12)          │        │ Nodes: (Total_Nodes, 10)    │
├─────────────────────────────┤        ├─────────────────────────────┤        ├─────────────────────────────┤
│ • Residual Momentum (5/20/60)│       │ • Decayed FinBERT Sentiment │        │ • Node 0: Target Stock      │
│ • Overnight/Intraday return │        │ • Decayed Form 4 Insider Buy│        │ • Nodes 1-4: GICS Peers     │
│ • Closing Power (auction)   │        │ • Decayed PEAD Shock        │        │ • Nodes 5-8: Lead-Lag Peers │
│ • Amihud Illiquidity Z-Score│        │ • Decayed 10-Q Item 1A Diff │        │ • Node 9: Sector ETF        │
│ • Close-to-VWAP Deviation   │        │ • Staleness Age Scalars     │        │ • PIT XBRL Fundamental Facts│
│ • Realized Hourly Volatility│        │ • Modality Missingness Masks│        │ • 60d Beta + Lead-Lag Coeff │
└──────────────┬──────────────┘        └──────────────┬──────────────┘        └──────────────┬──────────────┘
               │                                      │                                      │
               ▼                                      ▼                                      ▼
       ┌───────────────┐                      ┌───────────────┐                      ┌───────────────┐
       │ 1-Layer GRU   │                      │ Linear(12->32)│                      │ 2-Layer PyG   │
       │ + Self-Attn   │                      │ + LayerNorm   │                      │ GAT Layer     │
       │ (hidden = 32) │                      │ (hidden = 32) │                      │ (hidden = 32) │
       └───────┬───────┘                      └───────┬───────┘                      └───────┬───────┘
               │ (B, 126, 32)                         │ (B, 1, 32)                           │ (B, 32)
               └──────────────────────┬───────────────┘                                      │
                                      ▼                                                      │
                       ┌─────────────────────────────┐                                       │
                       │ Cross-Modal Attention       │                                       │
                       │ Query: Alt-Data (B, 1, 32)  │                                       │
                       │ Key/Val: Temporal (B,126,32)│                                       │
                       └──────────────┬──────────────┘                                       │
                                      │ (B, 32)                                              │
                                      └──────────────────────┬───────────────────────────────┘
                                                             ▼
                                                    Concat(32 + 32 + 2)
                                                        = (B, 66)
                                                             │
  [ Macro Volatility Regime HMM ]                            ▼
  Shape: (Batch, 3) ────────────────────────────────► ┌─────────────┐
                                                      │ Sigmoid Gate│
                                                      └──────┬──────┘
                                                             │ (B, 66)
                                                             ▼
                                                      ┌─────────────┐
                                                      │ Head: 66->32│
                                                      │ Linear-> 3  │
                                                      └──────┬──────┘
                                                             │ Raw Logits (B, 3)
                                                             ▼
                                                      ┌─────────────┐
                                                      │ Temperature │
                                                      │ Scaler: z/T │
                                                      └──────┬──────┘
                                                             │ Calibrated Probabilities
                                                             ▼
                                                      Multi-Outcome Kelly Sizer
```

### 5.3 Purged Triple-Barrier Labeling & Sample Uniqueness
* **Dynamic Barriers**: Upper $+2.0 \cdot \sigma_{\text{ATR}}$, Lower $-1.0 \cdot \sigma_{\text{ATR}}$, Vertical $H = 10\text{ days}$.
* **Calendar-Date Purging**: Every observation resolving after the training fold cutoff date $D_{\text{train}}$ is purged across all 250 tickers to prevent forward information leakage.
* **Sample Uniqueness Weighting**:
  $$u_{i, t} = \frac{1}{t_{\text{end}} - t + 1} \sum_{\tau = t}^{t_{\text{end}}} \frac{1}{c_{i, \tau}}$$
  where $c_{i,\tau}$ represents concurrent overlapping trade labels on day $\tau$.

### 5.4 Multi-Outcome Fractional Kelly Position Sizing
Solves the continuous log-wealth optimization net of 15 bps round-trip transaction friction ($c = 0.0015$):
$$\max_{f \in [0, f_{\max}]} \left[ \hat{p}_{\text{buy}} \ln(1 + f (b_{\text{upper}} - c)) + \hat{p}_{\text{sell}} \ln(1 - f (|b_{\text{lower}}| + c)) + \hat{p}_{\text{hold}} \ln(1 + f (\bar{r}_{\text{vert}} - c)) \right]$$
* Scaled by regime multiplier $\kappa \in [0.25, 0.50]$ (Quarter/Half Kelly).
* Hard single-stock portfolio allocation cap: $10\%$ ($15\%$ absolute maximum limit).

---

## 6. Apple Silicon M5 Air (24 GB) Thermal & Runtime Budget

The system is engineered for continuous daily operation on an uncooled, fanless M5 MacBook Air.

### 6.1 Memory Allocation Footprint
```
Total Unified Memory Budget: 24,576 MB
├── macOS & System Core:           ~4,600 MB (Constant)
├── Tauri 2.0 Webview Shell:         ~150 MB (Ultra-lightweight Rust wrapper)
├── Amstal Engine & LightGBM/PyTorch:~2,800 MB (Kept compact via hidden_size=32)
├── DuckDB In-Memory Cache:        ~2,000 MB
├── Available Headroom:           ~15,026 MB
```

### 6.2 Thermal Power Budgeting
* **Fanless Thermal Envelope**: Sustained full-core CPU/GPU utilization throttles M5 package power after ~12–15 minutes.
* **Mitigation Rules**:
  1. **Strict Offline Batch Processing**: LLM redline diff scoring (`Llama-3.1-8B`) and FinBERT embeddings never execute during active model training or live market hours. They execute as checkpointed background jobs writing directly to Parquet feature stores.
  2. **Sub-Minute Training Sweeps**: Right-sized latent vectors (`hidden_size=32`) allow full walk-forward folds across 250 universe stocks to compute in under 30 seconds on Apple Silicon MPS/CPU, finishing long before thermal throttling initiates.

---

## 7. Concrete Next Implementation Steps

To transition this architecture into executable code:
1. **Engine Setup**: Initialize `pyproject.toml`, establish the DuckDB storage layout, and implement the Point-in-Time data loader in `amstal-engine/src/data/market_loader.py`.
2. **IPC Skeleton**: Implement the UDS and WebSocket broadcaster (`localhost:8194`) in `amstal-engine/src/ipc/`.
3. **Terminal Foundation**: Initialize the Tauri 2.0 Svelte 5 application in `amstal-terminal/` with the GoldenLayout multi-tile grid and Mnemonic Omnibar component.
