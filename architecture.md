# Technical Architecture: Amstal System
## Unified Architecture for `amstal-engine` (Quant Core v1.3) and `amstal-terminal` (Workstation)

---

## 1. Architectural Vision & End-to-End Topology

**Amstal** couples an institutional quantitative research and risk engine (`amstal-engine`) with an ultra-responsive, Bloomberg-inspired desktop terminal (`amstal-terminal`).

The system is architected to provide hedge-fund caliber Point-in-Time (PIT) data pipelines, mathematical alpha isolation, and multi-asset risk management on personal workstation hardware—specifically optimized for the fanless **Apple Silicon M5 MacBook Air (24 GB Unified Memory)**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          AMSTAL COMPLETE END-TO-END TOPOLOGY                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                      [ EXTERNAL MARKET DATA & FILINGS ]
         ┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
         │ Polygon.io /      │ SEC EDGAR (XBRL)  │ FRED Macro        │ Historical Index  │
         │ yfinance (Hybrid) │ (`edgartools`)    │ (Yields, VIX)     │ Joiners / Leavers │
         └─────────┬─────────┴─────────┬─────────┴─────────┬─────────┴─────────┬─────────┘
                   │                   │                   │                   │
                   ▼                   ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           AMSTAL ENGINE CORE (Python 3.11+)                                      │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. INGESTION & STATELESS LAKEHOUSE LAYER                                                                   │  │
│  │   • Provider Plugins (OpenBB Pattern): Standardized Pydantic v2 Models (`StandardCandle`, `StandardXBRL`)   │  │
│  │   • Stateless OLAP: In-memory `duckdb.connect(":memory:")` querying partitioned Apache Parquet files       │  │
│  │   • Lock-Free Concurrency: Writes to `{ticker}.parquet.tmp` followed by POSIX atomic `os.replace()`        │  │
│  │   • Point-in-Time (PIT) Validator: >16:00 America/New_York (market close) rollover enforcement, `filed` date fundamental alignment │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. FEATURE ENGINEERING & ALPHA ISOLATION (QARE-MM v1.3)                                                    │  │
│  │   • 2-Stage Sequential OLS Orthogonalization: SPY -> Sector ETF_s -> Stock epsilon_{i,t} (No Collinearity) │  │
│  │   • Dual-Mode Microstructure Tensor (126, 15): Full Hourly Bars + Automatic Daily-OHLCV Proxy Fallback     │  │
│  │   • LightGBM Multi-Scale Summary (Batch, 105): 7 statistics per channel [t0, EWMA 5/21/63/126, Std, Slope] │  │
│  │   • Alt-Data Catalyst Vector (12): Continuous exponential decay kernels (tau = 3d, 10d, 15d, 20d)          │  │
│  │   • Dynamic Rolling Peer Graph (10 nodes): Monthly GICS peers + 5d lead-lag correlation (PyG GAT on CPU)   │  │
│  │   • Macro Regime HMM: Forward-filtered 3-State Gaussian HMM (SPY + VIX), variance-sorted                   │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. MODELING, CALIBRATION & RISK SIZING                                                                     │  │
│  │   • GBDT vs Neural Net Gating: Sample-Weighted LightGBM (105d) vs. CPU-Pinned GRU-Attention (32d)          │  │
│  │   • Modality-Masked Fusion Net: Locks `(B, 66)` across Phase 2 (Temporal Fallback) & Phase 3 (Cross-Modal) │  │
│  │   • Purged Walk-Forward: 756d Train -> 10d Purge -> 126d Val -> 10d Purge -> 63d OOS Test (Quarterly)      │  │
│  │   • Temperature Scaling: Post-hoc NLL calibration fit strictly on the 126-day purged validation fold       │  │
│  │   • Configurable Kelly Sizer: Default `LONG_ONLY` Cash + Portfolio-Level L1 Gross Exposure Cap (<= 1.0)    │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │                                                         │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. LOCAL ENGINE IPC SERVER                                                                                 │  │
│  │   • Control Plane: Unix Domain Socket (UDS `/tmp/amstal.sock`) / Windows Named Pipe JSON-RPC 2.0 listener  │  │
│  │   • Streaming Broadcaster: Local WebSocket Server (`ws://127.0.0.1:8194`) with 100ms conflation scheduler  │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         │ STREAMLINED LOCAL IPC & STORAGE BUS                           │
                         │  - Control Plane (UDS): Sub-millisecond commands & responses  │
                         │  - Stream (Port 8194): Real-time ticks, signals, regime alerts │
                         │  - Lock-Free Parquet: Stateless `:memory:` duckdb-rs in Tauri │
                         └───────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AMSTAL TERMINAL WORKSTATION (Tauri 2.0)                                    │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ TAURI 2.0 RUST BACKEND CORE                                                                                │  │
│  │   • Engine Process Supervisor: Spawns, health-checks (2s ping), and gracefully terminates engine daemon    │  │
│  │   • UDS & WebSocket Client: Multiplexes incoming data streams to webview frontend                          │  │
│  │   • Stateless DuckDB Reader: Queries immutable `.parquet` files directly in Rust (`:memory:`, <15ms)       │  │
│  └─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                        │ Direct Native IPC                                       │
│                                                        ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ SVELTE 5 RUNES USER INTERFACE LAYER (Two-Tier Phased Rollout)                                              │  │
│  │   • 4-Panel Reactive Split-Pane Grid (`paneforge` + CSS Grid with `$state` / `$derived` Runes)             │  │
│  │   • Mnemonic Omnibar REPL: Autocomplete parser (`AAPL US <EQUITY> DES <GO>`) + fuzzy hybrid search (`cmdk`)│  │
│  │   • Channel Linking System: Multi-color broadcast channels (Blue/Amber/Green/Red) syncing active security  │  │
│  │   • Tier 1 Core (Phase 4A): `GP <GO>` (Charts), `PORT <GO>` (Kelly/Signals), `HMM <GO>`, `DES`, `FA <GO>`  │  │
│  │   • Tier 2 Suite (Phase 4B): `SPLC <GO>` (Peer GAT), `EDGAR <GO>`, `NEWS`, `HP`, `ANR`, Local Copilot      │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Terminal-Engine Integration Mechanics

The terminal and engine communicate locally without relying on external cloud infrastructure or redundant serialization layers.

### 2.1 Streamlined Communication Protocols & Storage Concurrency

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  LOCAL IPC & STORAGE TOPOLOGY                                    |
+──────────────────────────┬─────────────────────────────┬─────────────────────────────────────────+
| Channel Name             | Wire / Storage Transport    | Message Format / Concurrency Rule       |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **Control & RPC**        | Local Domain Socket / Pipe  | Binary Framed JSON-RPC 2.0 (Ping <1ms)  |
|                          | Unix: `$XDG_RUNTIME_DIR/`   | Mode 0700 private user boundary.        |
|                          | Windows: `\\.\pipe\amstal`  | Windows Named Pipe security descriptor. |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **Market Data Feed**     | Local WebSocket             | Multiplexed UTF-8 JSON                  |
|                          | URL: `ws://127.0.0.1:8194`  | (100ms Conflated Pub/Sub with Origin    |
|                          | Token Authenticated         | check & ephemeral session token)        |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
| **Lock-Free Analytical** | Stateless `:memory:`        | Zero-Lock Direct Parquet Queries        |
| **Storage Bridge**       | `duckdb-rs` & Python DuckDB | Python writes `.parquet.tmp` -> POSIX   |
|                          | over `data/processed/`      | atomic `os.replace()` for safe reads    |
+──────────────────────────┼─────────────────────────────┼─────────────────────────────────────────+
```

### 2.2 Why Stateless `:memory:` DuckDB + Atomic Rename?
A persistent `.duckdb` database file enforces a single-writer OS file lock: if `amstal-engine` (Python) holds write access to `amstal.duckdb`, `amstal-terminal` (Tauri Rust) cannot open it concurrently without throwing `IO Error: Could not set lock on file`. Furthermore, overwriting `.parquet` files in-place during a concurrent read corrupts the Parquet footer.

Amstal solves both failure modes deterministically:
1. **Zero `.duckdb` Lock Files**: Both Python and Tauri Rust open ephemeral `duckdb.connect(":memory:")` sessions and query `.parquet` files directly via `SELECT * FROM read_parquet(...)`.
2. **Atomic POSIX Replacement**:
```python
import os
from pathlib import Path

def atomic_write_parquet(df, target_path: Path) -> None:
    tmp_path = target_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    os.replace(tmp_path, target_path)  # Atomic POSIX inode swap
```
   Any concurrent `duckdb-rs` query in Tauri either reads the complete previous inode or the complete new inode—never a partial file.

### 2.3 Sequence Diagram: Mnemonic Command & Analytical Query Execution

```
User (Keyboard)         Terminal (Svelte 5)    Tauri (Rust Core)      Engine (Python UDS)    Parquet Lakehouse
      │                         │                      │                       │                     │
      │ Type "NVDA US EQ FA"    │                      │                       │                     │
      │ ───► <GO> (Enter)       │                      │                       │                     │
      │                         │ Parse Mnemonic AST   │                       │                     │
      │                         │ ───────────────────► │                       │                     │
      │                         │                      │ [If Cached in Lakehouse: Direct Read]       │
      │                         │                      │ ──────────────────────────────────────────► │
      │                         │                      │                       │     :memory: DuckDB │
      │                         │                      │ ◄────────────────────────────────────────── │ (< 5ms)
      │                         │                      │                       │                     │
      │                         │                      │ [If Uncached / Live Compute Required]       │
      │                         │                      │ UDS JSON-RPC Request  │                     │
      │                         │                      │ ────────────────────► │                     │
      │                         │                      │                       │ Fetch & Compute PIT │
      │                         │                      │                       │ Atomic os.replace() │
      │                         │                      │                       │ ──────────────────► │
      │                         │                      │ UDS JSON Response     │                     │
      │                         │                      │ ◄──────────────────── │                     │
      │                         │ Update $state Store  │                       │                     │
      │                         │ ◄─────────────────── │                       │                     │
      │ Render 10-Yr PIT Grid   │                      │                       │                     │
      │ ◄────────────────────── │                      │                       │                     │
```

### 2.4 Real-Time Tick Conflation & 60 FPS UI Rendering
* **Engine Ring Buffer**: Ingests streaming market trades and quotes asynchronously.
* **100ms Conflation Scheduler**: Every 100 milliseconds, the engine aggregates net state changes (`last_price`, `bid`, `ask`, `cumulative_volume`, `vwap_deviation`) and broadcasts a single conflated frame over `ws://127.0.0.1:8194`.
* **Svelte 5 Runes Reactivity**: Fine-grained `$state` signals update only modified DOM text nodes and WebGL canvas buffers at 60 FPS without virtual-DOM diffing overhead.

---

## 3. Reverse-Engineered Bloomberg Architecture

### 3.1 Mnemonic Command Grammar & Lexical Parser
Amstal recreates Bloomberg's Reverse Polish Notation (RPN) command workflow via a deterministic lexer and state machine:

```
               ┌──────────┐
               │  TICKER  │ (e.g., "AAPL", "NVDA", "SPY")
               └────┬─────┘
                    │
                    ▼
           ┌──────────────────┐
           │   MARKET CODE    │ (Optional: "US", "LN", "GY" — defaults to "US")
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │   ASSET CLASS    │ (<EQUITY>, <GOVT>, <CORP>, <INDEX>, <CURNCY>)
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │     MNEMONIC     │ ("GP", "PORT", "HMM", "DES", "FA", "SPLC", "EDGAR", "NEWS", "HP", "ANR")
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │    PARAMETERS    │ (Optional: "QUARTERLY", "10Y", "LONG_ONLY")
           └────────┬─────────┘
                    │
                    ▼
               ┌──────────┐
               │   <GO>   │ (Execute token / Enter)
               └──────────┘
```

### 3.2 Two-Tier Module Architecture

#### Tier 1: Core Quantitative Workstation (Phase 4A)
1. **`GP <GO>` (Graph Price & Microstructure)**:
   * TradingView Lightweight Charts (WebGL) with multi-timeframe candlesticks, intraday VWAP, dynamic Triple-Barrier levels ($+2\sigma_{\text{ATR}} / -1\sigma_{\text{ATR}}$), and an overnight-vs-intraday return decomposition sub-pane.
2. **`PORT <GO>` / `QARE <GO>` (Portfolio & Kelly Allocation Sizer)**:
   * Displays real-time QARE-MM v1.3 signal breakdown across all 11 factor channels, Temperature-calibrated probabilities ($\hat{p}_{\text{buy}}, \hat{p}_{\text{hold}}, \hat{p}_{\text{sell}}$), unconstrained single-stock Kelly fraction, and $L_1$-normalized portfolio weights ($w_i$).
3. **`HMM <GO>` (Macro Volatility Regime Monitor)**:
   * Visualizes forward-filtered 3-state Gaussian HMM probabilities:
     * **State 0 (Green)**: Low-Volatility Regime ($\Sigma_{0, rr}$ lowest return variance).
     * **State 1 (Amber)**: Normal / Medium-Volatility Regime ($\Sigma_{1, rr}$ medium return variance).
     * **State 2 (Red)**: High-Volatility / Turbulent Regime ($\Sigma_{2, rr}$ highest return variance).
4. **`DES <GO>` (Security Master Profile)**:
   * Header with Ticker, Name, Primary Exchange, CIK, FIGI, GICS Sector/Sub-Industry, 60-day orthogonalized market beta ($\hat{\beta}_{\text{mkt}}$) and sector beta ($\hat{\beta}_{\text{sec}}$), ADV, and Point-in-Time Item 1 business summary.
5. **`FA <GO>` (Point-in-Time Financial Analysis)**:
   * 10-year quarterly and annual Income Statement, Balance Sheet, and Cash Flow tables stamped strictly by SEC XBRL `filed` acceptance date (preventing back-dated restatement look-ahead).

#### Tier 2: Alt-Data & Graph Suite (Phase 4B)
6. **`SPLC <GO>` (Dynamic Rolling Peer Graph)**:
   * Force-directed interactive graph rendering the 10-node subgraph ($v_0$ target, $v_1 \dots v_4$ GICS peers, $v_5 \dots v_8$ 5-day lead-lag cross-correlation peers, $v_9$ Sector ETF) and GAT attention weights.
7. **`EDGAR <GO>` (SEC 10-Q/10-K Redlines & Form 4 Clusters)**:
   * Side-by-side `difflib` redline viewer for Item 1A Risk Factors and Item 7 MD&A, scored by `Llama-3.1-8B`, alongside Form 4 open-market cluster buying/selling tables (filtering out Rule 10b5-1 scheduled trades).
8. **`NEWS <GO>` (FinBERT Sentiment Stream)**:
   * Real-time headline feed with FinBERT sentiment scores and exponential decay curves ($\tau = 3.0\text{d}$).
9. **`HP <GO>` & `ANR <GO>`**:
   * Tabular historical prices with overnight/intraday splits (`HP`) and analyst consensus target distributions (`ANR`).
10. **Local Copilot (`HELP <GO>`)**:
    * Embedded Ollama + LanceDB RAG assistant for natural language equity and quantitative research queries.

### 3.3 4-Panel Reactive Split-Pane Grid (`paneforge` + Svelte 5 Runes)
Instead of legacy imperative DOM layout managers (such as GoldenLayout) that conflict with Svelte 5's reactive lifecycle, `WorkspaceGrid.svelte` uses `paneforge` and CSS Grid backed by Svelte 5 runes (`$state`, `$derived`):
* **Color-Channel Synchronization**:
  Each of the 4 panels binds to a reactive channel store (`BLUE | AMBER | GREEN | RED | UNLINKED`). Executing `NVDA US <EQUITY> GP <GO>` in a Blue-linked panel mutates `channelState.BLUE.ticker = "NVDA"`, automatically triggering reactive re-fetches in all other Blue-linked panels.

---

## 4. Reverse-Engineered OpenBB Architecture

### 4.1 Extensible Provider Pattern
`amstal-engine` isolates external vendor APIs behind an abstract `BaseProvider` interface:

```
                      ┌─────────────────────────────────┐
                      │        BaseProvider (ABC)       │
                      ├─────────────────────────────────┤
                      │ + fetch_candles()               │
                      │ + fetch_fundamentals_pit()      │
                      │ + fetch_form4_insiders()        │
                      │ + fetch_macro_series()          │
                      └────────────────┬────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│ PolygonYFinanceHybrid │    │     EdgarProvider     │    │      FredProvider     │
├───────────────────────┤    ├───────────────────────┤    ├───────────────────────┤
│ • Polygon Hourly/Daily│    │ • `edgartools` Engine │    │ • 10Y/2Y Yields       │
│ • Automatic yfinance  │    │ • US-GAAP Tag Hierarchy│   │ • VIX, CPI, M2, GDP   │
│   Daily Proxy Fallback│    │ • Form 4 XML & 10-Q/K │    │ • Fed Funds Rate      │
└───────────────────────┘    └───────────────────────┘    └───────────────────────┘
```

### 4.2 Deterministic US-GAAP XBRL Tag Fallback Hierarchy (`sec_edgar.py`)
Companies frequently switch XBRL taxonomy tags across reporting years. To construct unbroken 10-year Point-in-Time fundamental series for `FA <GO>` and Peer GAT nodes (`Earnings Yield`, `YoY Revenue Growth`, `Operating Margin`), `sec_edgar.py` integrates `edgartools` with a strict tag resolution hierarchy:
* **Revenue**: `RevenueFromContractWithCustomerExcludingAssessedTax` $\to$ `Revenues` $\to$ `SalesRevenueNet` $\to$ `SalesRevenueGoodsNet`
* **Operating Income**: `OperatingIncomeLoss` $\to$ `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest`
* **Diluted EPS**: `EarningsPerShareDiluted` $\to$ `IncomeLossFromContinuingOperationsPerDilutedShare`
* **Point-in-Time Rule**: Every fact is indexed strictly by its SEC `filed` acceptance timestamp (never back-dated to fiscal period `end` date). Filings accepted after `16:00 America/New_York` (local market close) on day $t$ are shifted strictly to $t+1$.

### 4.3 Hybrid LLM Backfill vs. Local Ongoing Inference (`sec_diff.py`)
* **Historical Seed (One-Time Batch Backfill)**: Scoring `difflib` redlines for ~8,000 historical 10-Q/10-K filings (`250 stocks * 4 quarters * 8 years`) locally on a fanless M5 Air would require ~40 hours of sustained compute. `sec_diff.py` includes an async batch adapter for OpenAI-compatible inference endpoints (Groq / DeepInfra / Together running `Llama-3.1-8B-Instruct` at `temperature=0.0`) that materializes the historical Parquet cache in <45 minutes.
* **Live & Ongoing Inference (100% Local Ollama)**: Once seeded, all new daily/quarterly SEC filings and interactive terminal Copilot prompts execute 100% locally via **Ollama** (`llama3.1:8b` Q4_K_M / Q8_0) with zero external API dependency.

---

## 5. Quantitative & Machine Learning Pipeline (QARE-MM v1.3)

### 5.1 Idiosyncratic Alpha Isolation (2-Stage Sequential OLS Orthogonalization)
To eliminate collinearity between `SPY` and GICS Sector ETFs ($\rho \ge 0.80$), rolling 60-day betas are estimated over strictly historical windows $[t-60, t-1]$ in two sequential stages:
1. **Stage 1 (Sector Orthogonalization against SPY)**:
   $$r_{\text{ETF}_s, t} = \hat{\alpha}_{s, t-1} + \hat{\beta}_{s, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \nu_{s, t} \implies r_{\text{ETF}_s, \perp, t} = \nu_{s, t}$$
2. **Stage 2 (Stock Residualization against SPY and Orthogonal Sector)**:
   $$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, \perp, t}\right)$$
By construction, $\text{Cov}(r_{\text{SPY}}, r_{\text{ETF}_s, \perp}) = 0$, guaranteeing stable regression betas and pure idiosyncratic residuals $\epsilon_{i, t}$.

### 5.2 Multi-Modal Tensor Topology & Phase 2/3 Modality Masking

```
                                  [ MULTI-MODAL INPUT TENSORS ]
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌─────────────────────────────┐        ┌─────────────────────────────┐        ┌─────────────────────────────┐
│ 1. Dual-Mode Microstructure │        │ 2. SEC & Alt-Data Vector    │        │ 3. Dynamic Rolling Peer Graph│
│ Shape: (Batch, 126, 15)     │        │ Shape: (Batch, 12)          │        │ Nodes: (Total_Nodes, 10)    │
├─────────────────────────────┤        ├─────────────────────────────┤        ├─────────────────────────────┤
│ • 15 Rank-Normalized Chans  │        │ • FinBERT, Form 4, PEAD,    │        │ • Node 0: Target Stock      │
│ • Hourly Full OR Daily Proxy│        │   10-Q Redline Diffs        │        │ • Nodes 1-4: GICS Peers     │
│ • Also summarized into      │        │ • Staleness Age Scalars     │        │ • Nodes 5-8: Lead-Lag Peers │
│   (Batch, 105) for LightGBM │        │ • Missingness Masks (9..11) │        │ • Node 9: Sector ETF        │
└──────────────┬──────────────┘        └──────────────┬──────────────┘        └──────────────┬──────────────┘
               │                                      │                                      │
               ▼                                      ▼                                      ▼
       ┌───────────────┐                      ┌───────────────┐                      ┌───────────────┐
       │ 1-Layer GRU   │                      │ Linear(12->32)│                      │ 2-Layer PyG   │
       │ + 2H Self-Attn│                      │ + LayerNorm   │                      │ GAT Layer     │
       │ (CPU, d = 32) │                      │ (CPU, d = 32) │                      │ (CPU, d = 32) │
       └───────┬───────┘                      └───────┬───────┘                      └───────┬───────┘
               │ (B, 126, 32)                         │ (B, 1, 32)                           │ (B, 32)
               └──────────────────────┬───────────────┘                                      │
                                      ▼                                                      │
                       ┌─────────────────────────────┐                                       │
                       │ Modality-Masked Fusion      │                                       │
                       │ • Phase 2 (Masks=0):        │                                       │
                       │   Temporal Pooled (B, 32)   │                                       │
                       │ • Phase 3 (Masks=1):        │                                       │
                       │   Cross-Modal Attn (B, 32)  │                                       │
                       └──────────────┬──────────────┘                                       │
                                      │ (B, 32)                                              │
                                      └──────────────────────┬───────────────────────────────┘
                                                             ▼
                                      Concat: [ (B, 32) || (B, 32) || [D+, D-] (B, 2) ]
                                                         = (B, 66)
                                                             │
  [ Macro Volatility Regime HMM ]                            ▼
  Shape: (Batch, 3) ────────────────────────────────► ┌─────────────┐
                                                      │ Sigmoid Gate│
                                                      │ Linear(3,66)│
                                                      └──────┬──────┘
                                                             │ (B, 66)
                                                             ▼
                                               ┌───────────────────┐
                                               │ Classification    │
                                               │ Linear(66 -> 32)  │
                                               │ + LayerNorm + SiLU│
                                               │ + Dropout(0.3)    │
                                               │ + Linear(32 -> 3) │
                                               └─────────┬─────────┘
                                                             │ Raw Logits (B, 3)
                                                             ▼
                                                      ┌─────────────┐
                                                      │ Temperature │
                                                      │ Scaler: z/T │
                                                      └──────┬──────┘
                                                             │ Calibrated Probabilities (B, 3)
                                                             ▼
                                               L1-Normalized Multi-Outcome Kelly Sizer
```

### 5.3 Purged Walk-Forward Cross-Validation Geometry (`walk_forward.py`)
Every walk-forward fold strictly enforces two 10-trading-day purge buffers ($t_{\text{end}} \le D_{\text{cutoff}}$ across all 250 tickers) to prevent overlapping triple-barrier labels from leaking across train, validation, and out-of-sample test windows:

```
┌───────────────────────────┬──────────────┬───────────────────────────┬──────────────┬───────────────────────────┐
│    ROLLING TRAIN FOLD     │ PURGE BUFFER │   VALIDATION FOLD (NLL)   │ PURGE BUFFER │    OOS TEST FOLD (STEP)   │
│   3 Years (756 Trading D) │  10 Trad. D  │  6 Months (126 Trading D) │  10 Trad. D  │   3 Months (63 Trading D) │
├───────────────────────────┼──────────────┼───────────────────────────┼──────────────┼───────────────────────────┤
│ • Fit LightGBM (105ch)    │ • Drop all   │ • Early Stopping          │ • Drop all   │ • Strictly Out-of-Sample  │
│ • Fit GRU/GAT (32d)       │   samples w/ │ • Fit Ensemble Weights    │   samples w/ │   Evaluation (Rank IC,    │
│ • Fit 3-State Gaussian HMM│   t_end > D1 │ • Fit Temp Scaler (T > 0) │   t_end > D2 │   Brier, Net-of-15bps PnL)│
└───────────────────────────┴──────────────┴───────────────────────────┴──────────────┴───────────────────────────┘
```

### 5.4 Configurable Execution Mode & Portfolio-Level $L_1$ Kelly Normalization (`kelly_engine.py`)
1. **Single-Stock Fractional Kelly**:
   Solves the expected log-wealth objective using centered validation payoffs ($\tilde{b}_k = R_k - \mu_{\bar{p}}$, guaranteeing $\sum \bar{p}_k \tilde{b}_k = 0$) net of 15 bps round-trip friction ($c = 0.0015$), scaled by regime multiplier $\kappa \in [0.25, 0.50]$ and capped at $f_{\max} = 0.10$, subject to strict positive wealth multiplier domain bounds:
   $$\max_{f \in [0, f_{\max}]} \left[ \hat{p}_{\text{buy}} \ln(1 + f (\tilde{b}_{\text{upper}} - c)) + \hat{p}_{\text{sell}} \ln(1 - f (|\tilde{b}_{\text{lower}}| + c)) + \hat{p}_{\text{hold}} \ln(1 + f (\tilde{b}_{\text{vert}} - c)) \right]$$
   $$\text{subject to } 1 + f(\tilde{b}_{\text{upper}} - c) > 0, \quad 1 - f(|\tilde{b}_{\text{lower}}| + c) > 0, \quad 1 + f(\tilde{b}_{\text{vert}} - c) > 0$$
2. **Execution Mode Switch**:
   * `LONG_ONLY` (Default): Constrains $f_i \in [0, 0.10]$. `Underperform/Sell` signals exit long positions to cash ($0.0\%$).
   * `LONG_SHORT` (Diagnostic Stat-Arb): Evaluates cross-sectional Rank IC and dollar-neutral long/short performance in `metrics.py`.
3. **Portfolio-Level $L_1$ Gross Exposure Normalization**:
   $$w_i = \frac{f_i^* \cdot \kappa}{\max\left(1.0, \sum_{j=1}^N |f_j^* \cdot \kappa|\right)} \implies \sum_{i=1}^N |w_i| \le 1.0$$

---

## 6. Apple Silicon M5 Air (24 GB) Hardware & Runtime Constraints

### 6.1 Unified Memory Budget (24,576 MB)
```
Total Unified Memory: 24,576 MB
├── macOS & System Core:                 ~4,600 MB
├── Tauri 2.0 Webview Shell:               ~150 MB
├── Amstal Engine (LightGBM + CPU PyTorch):~2,800 MB (hidden_size=32)
├── Stateless :memory: DuckDB Buffer:    ~2,000 MB
└── Available Headroom:                 ~15,026 MB
```

### 6.2 CPU Thread Pinning & Thermal Envelope
* **Explicit CPU Pinning (`torch.set_num_threads(4)`)**: PyG `GATConv` scatter/gather operations lack native Apple Metal (MPS) kernels, and for 10-node subgraphs and 32-unit GRUs, CPU execution avoids Metal kernel dispatch latency. All neural modules run deterministically on CPU threads.
* **Burst Thermal Budgeting**: With pre-computed Parquet caches and `hidden_size = 32`, each quarterly walk-forward fold trains and evaluates in <30 seconds, finishing well below the 12-minute fanless M5 thermal throttling threshold.

---

## 7. Development Safeguards & Numerical Stability Rules

To prevent subtle runtime bugs, look-ahead leakage, or numerical blowups during Phase 1–4 implementation, all modules must enforce these 8 engineering invariants:

1. **Out-of-Sample OLS Projection & Beta Clipping (`residualizer.py`)**:
   * Stage 1 and Stage 2 OLS parameters $(\hat{\alpha}_{t-1}, \hat{\beta}_{t-1})$ are fit strictly within the historical window $\tau \in [t-60, t-1]$ (where $\nu_{s, \tau}$ is orthogonal to $r_{\text{SPY}, \tau}$, decomposing Stage 2 into two independent univariate projections) and applied out-of-sample to day $t$.
   * Variance denominators include a ridge floor ($\text{Var}(x) + 10^{-8}$) and betas are clipped to $\hat{\beta}_{\text{mkt}} \in [-1.0, +4.0]$, $\hat{\beta}_{\text{sec}} \in [-3.0, +3.0]$ to prevent trading-halt spikes.
2. **NaN-Safe Cross-Sectional Ranking (`microstructure.py`)**:
   * Uses `scipy.stats.rankdata(x, method='average', nan_policy='omit')` divided by $M_{t, \text{valid}} - 1$ (count of valid non-NaN universe stocks on day $t$), with post-rank `NaN` imputation to `0.0` so a single missing bar never corrupts the 250-stock cross-section.
3. **1D Residual Volatility Scale $\sigma_{\text{ATR}(\epsilon), i, t}$ (`triple_barrier.py`)**:
   * Because $\epsilon_{i, t}$ is a 1D residual series rather than an OHLC bar, the horizon-scaled volatility barrier unit is defined causally over $[t-19, t]$ as $\sigma_{\text{ATR}(\epsilon), i, t} = \sqrt{10} \cdot \sqrt{\frac{1}{20}\sum_{m=0}^{19} \epsilon_{i, t-m}^2}$.
4. **Intra-Ticker Concurrency & Weight Normalization (`weights.py`)**:
   * Concurrency $c_{i, \tau} \in [1, 10]$ is computed **intra-ticker** (counting overlapping active barrier windows for stock $i$ on day $\tau$), and uniqueness weights $u_{i, t}$ are mean-normalized across each training fold ($\frac{1}{N_{\text{train}}}\sum u_{i, t} = 1.0$) to preserve effective learning rate in LightGBM and PyTorch.
5. **Causal Feature Warm-Up vs. Fold-Scoped Parametric Fitting (`walk_forward.py`)**:
   * Purely causal rolling features (`60d OLS` + `60d Momentum/Amihud` + `126d Sequence Window` = $246\text{d}$ warm-up burn-in) are computed continuously along the timeline so Validation ($126\text{d}$) and OOS Test ($63\text{d}$) folds never lose their initial rows to warm-up `NaN`s.
   * All parametric estimators (`LightGBM`, `GRU`, `GAT`, `HMM`, `TemperatureScaler`, and `Kelly` base-rate payoff centering) are fit strictly inside their purged fold boundaries.
6. **Custom Log-Space Forward Filter (`regime_hmm.py`)**:
   * Calling `hmmlearn.GaussianHMM.predict_proba(X)` is strictly forbidden because it executes the forward-backward Baum-Welch smoother (leaking future states $x_{t+1:T}$). `regime_hmm.py` implements a custom causal log-space forward recursion using `scipy.special.logsumexp`, and fits `StandardScaler` strictly on the 756-day training fold.
7. **Bidirectional + Self-Loop Subgraphs & Pure PyG (`graph_builder.py` / `peer_gat.py`)**:
   * Subgraph edges are constructed bidirectionally between peers/ETF ($v_1 \dots v_9$) and target ($v_0$, always local index `0`) with `add_self_loops=True` so $v_0$ receives incoming peer messages in `GATConv`.
   * Uses pure `torch-geometric>=2.5.0` native PyTorch scatter operations (no legacy C++ `torch-scatter`/`torch-sparse` compilation required on Python 3.11–3.13).
8. **Base-Rate Payoff Centering for Gate 4 Zero-Skill Guarantee (`kelly_engine.py`)**:
   * Realized class payoffs $b_k$ on the validation fold are centered against the empirical validation prior $\bar{\mathbf{p}}$ so $\sum_k \bar{p}_k b_k = 0$. Any zero-skill model outputting prior probabilities ($\hat{\mathbf{p}} = \bar{\mathbf{p}}$) has expected net return $0 - c = -0.0015 < 0$, mathematically guaranteeing $f_i^* = 0.0\%$ allocation at Gate 4.
