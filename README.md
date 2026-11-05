# Amstal (QARE-MM v1.3)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-0288d1.svg)](https://lightgbm.readthedocs.io/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3f51b5.svg)](https://pyg.org/)
[![Platform](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20x86_64-lightgrey.svg)]()

> **Multi-Modal Quantitative Alpha & Risk Engine (`amstal-engine`) + Institutional Workstation (`amstal-terminal`)**  
> An institutional-grade cross-sectional swing trading and portfolio risk-management system designed for sample efficiency, strict Point-in-Time (PIT) integrity, and market/sector-residualized idiosyncratic alpha.

---

## Table of Contents

- [Overview](#overview)
- [Key Architectural Tenets](#key-architectural-tenets)
- [System Architecture](#system-architecture)
- [Quantitative & Mathematical Framework](#quantitative--mathematical-framework)
  - [1. Market- & Sector-Residualized Returns (2-Stage Sequential OLS)](#1-market---sector-residualized-returns-2-stage-sequential-ols)
  - [2. Cross-Sectional Rank Normalization](#2-cross-sectional-rank-normalization)
  - [3. Hybrid Dual-Mode Microstructure & Overnight Decomposition](#3-hybrid-dual-mode-microstructure--overnight-decomposition)
  - [4. Multi-Scale Tabular Summary for LightGBM `(Batch, 105)`](#4-multi-scale-tabular-summary-for-lightgbm-batch-105)
  - [5. Continuous Event-Decay Kernel](#5-continuous-event-decay-kernel)
  - [6. Variance-Sorted Forward-Filtered HMM](#6-variance-sorted-forward-filtered-hmm)
  - [7. Purged Triple-Barrier Labeling, Uniqueness & Walk-Forward Geometry](#7-purged-triple-barrier-labeling-uniqueness--walk-forward-geometry)
  - [8. Signed Smart-Money Divergence & Modality-Masked `(B, 66)` Fusion](#8-signed-smart-money-divergence--modality-masked-b-66-fusion)
  - [9. Temperature Calibration, Execution Modes & L1-Normalized Kelly Sizing](#9-temperature-calibration-execution-modes--l1-normalized-kelly-sizing)
- [Multi-Modal Feature Dictionary](#multi-modal-feature-dictionary)
  - [Price & Intraday Microstructure Matrix `(Batch, 126, 15)`](#price--intraday-microstructure-matrix-batch-126-15)
  - [SEC EDGAR & NLP Alt-Data Vector `(Batch, 12)`](#sec-edgar--nlp-alt-data-vector-batch-12)
  - [Dynamic Rolling Peer Graph `(Total_Nodes, 10)`](#dynamic-rolling-peer-graph-total_nodes-10)
- [Phase Gating & Falsification Pipeline](#phase-gating--falsification-pipeline)
- [Target Output API Contract](#target-output-api-contract)
- [Hardware, Storage & Thermal Ergonomics](#hardware-storage--thermal-ergonomics)
- [Legal, Regulatory & Licensing Disclaimers](#legal-regulatory--licensing-disclaimers)
- [Directory Layout](#directory-layout)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

Most academic multi-modal deep learning models applied to finance suffer from severe sample starvation, retail indicator decay, look-ahead leakage, and overparameterization. **Amstal (QARE-MM v1.3)** is built around institutional reality:

- **Universe:** 250 liquid US mid-to-large-cap equities ($>\$25\text{M}$ ADV, S&P 500 ranks 100–500 and S&P MidCap 400, ex-Financials & Utilities), snapshotted annually from Point-in-Time historical index joiners/leavers tables (explicitly retaining delisted/acquired tickers to eliminate survivorship bias).
- **Target Horizon:** 5 to 20 trading days (10-day vertical swing horizon).
- **Walk-Forward Cadence:** Rolling 3-Year Train ($756\text{d}$) $\to$ $10\text{d}$ Purge $\to$ 6-Month Validation ($126\text{d}$) $\to$ $10\text{d}$ Purge $\to$ 3-Month Out-of-Sample Test ($63\text{d}$, quarterly re-fit).
- **Frictional Hurdle:** 15 bps (0.15%) round-trip penalty per trade (slippage, spread, and brokerage).
- **Lookback:** Rolling 126 trading days ($T = 126$, ~6 calendar months).
- **Core Signal Channels:**
  1. High-frequency price, flow, and hybrid dual-mode intraday/daily microstructure (`(Batch, 126, 15)` for GRU and `(Batch, 105)` multi-scale summary for LightGBM).
  2. SEC EDGAR catalysts (`edgartools` XBRL hierarchy, Form 4 insider buying/selling, 10-Q/10-K Item 1A/MD&A redline diffs, and PEAD).
  3. Dynamic rolling GICS and 5-day lead-lag cross-correlation peer graphs.
  4. Macro volatility regime filtering via forward-filtered 3-state Gaussian HMM.

---

## Key Architectural Tenets

1. **Idiosyncratic Alpha Isolation (2-Stage Sequential OLS):** Raw equity returns are roughly 65% macro and sector beta. Amstal orthogonalizes the GICS Sector ETF against `SPY` first, then strips both `SPY` and orthogonal sector beta from each stock before labeling or modeling.
2. **Strict Point-in-Time (PIT) Integrity:** Any SEC filing, corporate announcement, or news headline published after 16:00 EST on day $t$ is strictly stamped to day $t+1$. Fundamentals are keyed strictly by SEC XBRL `filed` date; back-dated restatements and static look-ahead graphs are forbidden.
3. **Model-Agnostic Gating:** Sample-Weighted Gradient Boosted Decision Trees (`(Batch, 105)` multi-scale tabular features) and a compact CPU-pinned temporal neural network (`(Batch, 126, 15)`, `hidden_size = 32`) compete as co-equals. If trees outperform sequence nets out-of-sample, the neural sequence branch is pruned.
4. **Lock-Free Lakehouse & Thermal Efficiency:** Designed to run full walk-forward sweeps on Apple Silicon Unified Memory (M5 Air 24 GB) without thermal throttling, using stateless `:memory:` DuckDB sessions over atomic POSIX-renamed Apache Parquet partitions (`os.replace()`).
5. **Falsification-First Discipline:** Mandatory sanity gates (shuffled label tests, 1-day forward-leak checks, and strict Sharpe ceilings) ensure signals represent genuine predictive alpha rather than look-ahead artifacts.

---

## System Architecture

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

---

## Quantitative & Mathematical Framework

### 1. Market- & Sector-Residualized Returns (2-Stage Sequential OLS)
Because `SPY` and GICS Sector ETFs ($\text{ETF}_s$) frequently exhibit correlations $\rho \ge 0.80$, rolling 60-day betas over $[t-60, t-1]$ are estimated via 2-stage sequential orthogonalization:
1. **Stage 1 (Sector Orthogonalization):**
   $$r_{\text{ETF}_s, t} = \hat{\alpha}_{s, t-1} + \hat{\beta}_{s, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + r_{\text{ETF}_s, \perp, t}$$
2. **Stage 2 (Stock Residualization):**
   $$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$
   $$\epsilon_{i, t} = r_{i, t} - \left(\hat{\alpha}_{i, t-1} + \hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, \perp, t}\right)$$
   Guarantees $\text{Cov}(r_{\text{SPY}}, r_{\text{ETF}_s, \perp}) = 0$, eliminating collinear beta sign swings and isolating pure idiosyncratic return $\epsilon_{i, t}$. Both regressions are fit strictly within $[t-60, t-1]$ and applied out-of-sample to day $t$ with a variance ridge floor ($\text{Var}(x) + 10^{-8}$) and beta clipping ($\hat{\beta}_{\text{mkt}} \in [-1, 4]$, $\hat{\beta}_{\text{sec}} \in [-3, 3]$).

### 2. Cross-Sectional Rank Normalization
To prevent distributional shifts across market cycles, every feature $x_{i, t}$ is cross-sectionally ranked across all valid non-NaN universe assets ($M_{t, \text{valid}} \approx 250$) into $[-1.0, +1.0]$ using `scipy.stats.rankdata(x, method='average', nan_policy='omit')` with post-rank `NaN` imputation to `0.0`:

$$\tilde{x}_{i, t} = 2 \cdot \left(\frac{\text{rank}(x_{i, t}) - 1}{M_{t, \text{valid}} - 1}\right) - 1.0$$

### 3. Hybrid Dual-Mode Microstructure & Overnight Decomposition
Informed institutional flow concentrates overnight and during the closing cross; retail liquidity shocks dominate intraday. Channels `12–14` automatically switch between full intraday hourly bars (when Polygon/Alpaca credentials are present) and deterministic Daily-OHLCV proxies (for zero-cost 10-year daily backtests):
* **Overnight Log-Return:** $r_{\text{overnight}, t} = \ln(P_{\text{open}, t} / P_{\text{close}, t-1})$
* **Intraday Session Log-Return:** $r_{\text{intraday}, t} = \ln(P_{\text{close}, t} / P_{\text{open}, t})$
* **Realized Intraday Volatility (#12):** $\sigma_{\text{intraday}, t} = \sqrt{\sum_{h=1}^7 r_{t, h}^2}$  
  *(Daily Fallback: Rogers-Satchell Volatility $\sqrt{\ln(H_t/C_t)\ln(H_t/O_t) + \ln(L_t/C_t)\ln(L_t/O_t)}$)*
* **Closing Power (#13):** $\text{CP}_t = \ln\left(\frac{V_{t, \text{hour 7}} + 10^{-5}}{V_{t, \text{hour 1}} + 10^{-5}}\right)$  
  *(Daily Fallback: Volume-Scaled CLV Skew $\left(\frac{2C_t - H_t - L_t}{H_t - L_t + 10^{-8}}\right) \cdot \ln\left(\frac{V_t}{\bar{V}_{20}} + 10^{-5}\right)$)*
* **Close-to-VWAP Deviation (#14):** $\Delta \text{VWAP}_t = \frac{P_{t, \text{close}} - \text{VWAP}_t}{\text{VWAP}_t}$  
  *(Daily Fallback: Close-to-Typical-Price Deviation $(C_t - \text{TP}_t)/\text{TP}_t$, where $\text{TP}_t = \frac{H_t+L_t+C_t}{3}$)*

### 4. Multi-Scale Tabular Summary for LightGBM `(Batch, 105)`
To give Sample-Weighted LightGBM a compact, non-overfitting tabular representation of the `(Batch, 126, 15)` sequence tensor at Gate 1, each of the 15 rank-normalized channels is summarized into 7 statistics ($15 \times 7 = 105$ features):
$$[\tilde{x}_{i, t}, \, \text{EWMA}_5(\tilde{x}), \, \text{EWMA}_{21}(\tilde{x}), \, \text{EWMA}_{63}(\tilde{x}), \, \text{EWMA}_{126}(\tilde{x}), \, \text{Std}_{21}(\tilde{x}), \, \text{Slope}_{21}(\tilde{x})]$$

### 5. Continuous Event-Decay Kernel
Catalyst scores $X_0 \in [-1.0, 1.0]$ from discrete filings, transactions, and earnings decay exponentially over elapsed trading days ($\Delta d$):

$$X_{\text{event}}(t) = X_0 \cdot \exp\left(-\frac{\ln(2)}{\tau} \cdot \Delta d\right)$$

* News / Sentiment: $\tau_{\text{news}} = 3.0\text{ days}$
* PEAD Shocks: $\tau_{\text{pead}} = 10.0\text{ days}$
* SEC 10-Q/10-K Redline Diffs: $\tau_{\text{legal}} = 15.0\text{ days}$
* SEC Form 4 Insider Clusters: $\tau_{\text{form4}} = 20.0\text{ days}$

### 6. Variance-Sorted Forward-Filtered HMM
A 3-state Gaussian HMM is fit on standardized `SPY` returns and `VIX` levels (with 10 random EM restarts, `StandardScaler` fit strictly on the 756-day training fold):
* **Custom Log-Space Forward Filter Only:** Explicitly implements causal log-space forward recursion ($\alpha_t(k) = P(S_t = k \mid x_{1:t})$ via `scipy.special.logsumexp`). Calling `hmmlearn.GaussianHMM.predict_proba(X)` (which runs the forward-backward Baum-Welch smoother) is strictly banned to prevent forward-looking bias.
* **Return-Variance State Alignment:** States are sorted by variance component $\Sigma_{k, rr}$ to ensure permutation invariance across walk-forward folds:
  $$\Sigma_{0, rr} < \Sigma_{1, rr} < \Sigma_{2, rr} \implies [\text{State 0: Low-Vol Bull, State 1: Sideways, State 2: High-Vol Crash}]$$

### 7. Purged Triple-Barrier Labeling, Uniqueness & Walk-Forward Geometry
Ternary labels $y_{i, t} \in \{0: \text{Outperform}, 1: \text{Neutral}, 2: \text{Underperform}\}$ are defined on cumulative idiosyncratic return paths $C_{\epsilon, i}(t, k) = \sum_{\Delta=1}^k \epsilon_{i, t+\Delta}$ using the 1D horizon-scaled residual volatility $\sigma_{\text{ATR}(\epsilon), i, t} = \sqrt{10} \cdot \sqrt{\frac{1}{20}\sum_{m=0}^{19}\epsilon_{i, t-m}^2}$:
* **Upper Barrier:** $+2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Lower Barrier:** $-1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Vertical Barrier:** $H = 10\text{ trading days}$
* **Intra-Ticker Sample Uniqueness Weighting:** Observations are weighted by average inverse intra-ticker concurrency $u_{i, t} = \frac{1}{t_{\text{end}} - t + 1} \sum_{\tau = t}^{t_{\text{end}}} \frac{1}{c_{i, \tau}}$ ($c_{i, \tau} \in [1, 10]$) and mean-normalized across each training fold ($\frac{1}{N_{\text{train}}}\sum u_{i, t} = 1.0$).
* **Locked Walk-Forward Cadence:** $246\text{d}$ Causal Feature Warm-Up $\to$ Rolling 3-Year Train ($756\text{d}$) $\to$ $10\text{d}$ Purge $\to$ 6-Month Validation ($126\text{d}$, for early stopping and Temperature Scaling $T > 0$) $\to$ $10\text{d}$ Purge $\to$ 3-Month Out-of-Sample Test Step ($63\text{d}$, quarterly re-fit).

### 8. Signed Smart-Money Divergence & Modality-Masked `(B, 66)` Fusion
The $(B, 66)$ fused vector entering the HMM Sigmoid Gate concatenates:
1. Temporal / Cross-Modal representation $(B, 32)$ (uses pooled Temporal Self-Attention in Phase 2 when Alt-Data masks = `0.0`, and Cross-Modal Attention in Phase 3).
2. Dynamic Rolling Peer GAT root node embedding $(B, 32)$ (bidirectional peer-to-target edges with `add_self_loops=True`, root $v_0$ at local index `0` sliced via `batch.ptr[:-1]`, using pure `torch-geometric>=2.5.0` native PyTorch scatter ops).
3. Signed Smart-Money Divergence $[D_t^+, D_t^-] \in \mathbb{R}^{(B, 2)}$ (zero-masked `[0.0, 0.0]` in Phase 2; populated in Phase 3 via $D_t = S_{\text{FinBERT}, t} \times (-\epsilon_t V_t / \bar{V}_{20})$, where $D_t^+ = \max(D_t, 0)$ and $D_t^- = \max(-D_t, 0)$).

### 9. Temperature Calibration, Execution Modes & L1-Normalized Kelly Sizing
1. **Temperature Scaling:** Raw logits $z_k$ are calibrated on the purged 126-day validation split by finding scalar $T > 0$ minimizing Negative Log-Likelihood:
   $$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_{j=0}^2 \exp(z_j / T)}$$
2. **Base-Rate Payoff Centering & Configurable Kelly Sizing:**
   - Validation class payoffs $b_k$ are centered against the empirical validation prior $\bar{\mathbf{p}}$ ($\sum_k \bar{p}_k b_k = 0$), mathematically guaranteeing that any zero-skill model ($\hat{\mathbf{p}} = \bar{\mathbf{p}}$) yields net return $-c = -0.0015 < 0$ and allocates $f_i^* = 0.0\%$ at Gate 4.
   - **Default (`LONG_ONLY` Cash):** Solves for $f_i^* \in [0, 0.10]$ maximizing expected log-wealth net of 15 bps round-trip friction ($c = 0.0015$), scaled by regime multiplier $\kappa \in [0.25, 0.50]$. `Underperform/Sell` signals exit long positions to cash ($0.0\%$).
   - **Diagnostic (`LONG_SHORT` Stat-Arb):** Reports cross-sectional Rank IC and dollar-neutral Long-Short Sharpe in backtest diagnostics.
3. **Portfolio-Level $L_1$ Gross Exposure Normalization:**
   $$w_i = \frac{f_i^* \cdot \kappa}{\max\left(1.0, \sum_{j=1}^N |f_j^* \cdot \kappa|\right)} \implies \sum_{i=1}^N |w_i| \le 1.0$$

---

## Multi-Modal Feature Dictionary

### Price & Intraday Microstructure Matrix `(Batch, 126, 15)`
| Channel | Feature Name | Description (Hourly Primary / Daily Proxy Fallback) |
| :---: | :--- | :--- |
| `0` | Daily Log-Return | Total closing return $r_t = \ln(C_t / C_{t-1})$ |
| `1` | Idiosyncratic Residual Return | Return residualized via 2-Stage Sequential OLS against SPY & Sector ETF $\epsilon_t$ |
| `2` | Overnight Return | Overnight gap $\ln(O_t / C_{t-1})$ |
| `3` | Intraday Session Return | Regular session drift $\ln(C_t / O_t)$ |
| `4-6`| Residual Momentum (5d, 20d, 60d) | Multi-scale cumulative idiosyncratic momentum |
| `7` | Relative Volume Ratio | Participation volume $V_t / \bar{V}_{20}$ |
| `8` | Amihud Illiquidity Z-Score | 60-day Z-score of price impact $\|r_t\| / (C_t \cdot V_t)$ |
| `9` | ATR Volatility Ratio | Normalized range $\text{ATR}_{14, t} / C_t$ |
| `10`| Idiosyncratic Volatility (20d) | Annualized 20-day residual variance |
| `11`| Parkinson Volatility | Intraday extreme variance $(\ln(H_t / L_t))^2 / (4 \ln 2)$ |
| `12`| Realized Intraday Volatility | Hourly sum of squared returns (Fallback: Rogers-Satchell Daily Volatility) |
| `13`| Closing Power | Ratio of Hour 7 to Hour 1 volume (Fallback: Volume-Scaled CLV Skew) |
| `14`| Close-to-VWAP Deviation | EOD accumulation vs session VWAP (Fallback: Close-to-Typical-Price Deviation) |

### SEC EDGAR & NLP Alt-Data Vector `(Batch, 12)`
| Index | Feature Name | Source / Kernel |
| :---: | :--- | :--- |
| `0-1` | Decayed FinBERT Sentiment & Age | News headlines passed through local FinBERT ($\tau = 3\text{d}$) |
| `2-3` | Decayed Form 4 Insider Score & Age | SEC EDGAR Form 4 open-market cluster purchases/sales ex-10b5-1 ($\tau = 20\text{d}$) |
| `4-5` | Decayed PEAD Earnings Shock & Age | Post-earnings announcement volume-scaled residual shock ($\tau = 10\text{d}$) |
| `6` | Decayed 10-Q Item 1A Risk Delta | `difflib` Item 1A redline delta scored via `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `7` | Decayed MD&A Catalyst Score | `difflib` Item 7/Item 2 redline delta scored via `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `8` | SEC Filing Staleness Age | Time elapsed since SEC acceptance timestamp ($>16:00\text{ EST} \to t+1$) |
| `9-11`| Modality Missingness Masks | Explicit binary indicators (news, insider, 10-Q) |

### Dynamic Rolling Peer Graph `(Total_Nodes, 10)`
- **10 Nodes per Subgraph:** Target ($v_0$), 4 GICS sub-industry peers ($v_1 \dots v_4$), 4 rolling 5-day lead-lag cross-correlation peers ($v_5 \dots v_8$), and the GICS Sector ETF ($v_9$).
- **Features:** 5d/20d residual return, 60d market beta, Point-in-Time XBRL fundamentals via `edgartools` + deterministic US-GAAP tag hierarchy (Earnings Yield, Revenue Growth, Operating Margin stamped strictly by `filed` date), realized idiosyncratic volatility, and lead-lag cross-correlation.

---

## Phase Gating & Falsification Pipeline

Development is bound by sequential, non-negotiable gates:

```
Phase 0: Lock Spec v2.1        ──► [GATE 0: All Tensors, PIT Rules & Decisions Locked] (PASS)
          │
          ▼
Phase 1: Residualized Backbone ──► [GATE 1: Passes Shuffled/Lagged Sanity Checks;
                                            Beats Baseline Net of 15 bps OOS; IC > 0.025]
          │
          ▼
Phase 2: Regime & Peer Graph   ──► [GATE 2: Ablation Shows Filtered HMM & CPU-Pinned GAT
                                            Improve Out-of-Sample Sharpe / Drawdown >= 15%]
          │
          ▼
Phase 3: SEC EDGAR & Alt-Data  ──► [GATE 3: Form 4, PEAD, FinBERT & 10-Q Diffs
                                            Add Incremental IC > 0.015 Post-Cutoff]
          │
          ▼
Phase 4: Calibration, Kelly &  ──► [GATE 4: Brier Score Drops > 10% OOS; Zero-Skill -> 0.0%;
         Two-Tier Terminal                  Tier 1 4-Panel Svelte 5 Terminal Verified]
```

---

## Target Output API Contract

The engine emits structured, machine-verifiable trade decisions adhering to this JSON schema:

```json
{
  "ticker": "CRWD",
  "timestamp_utc": "2026-10-19T20:00:00Z",
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

## Hardware, Storage & Thermal Ergonomics

Amstal is tuned for execution on an **Apple Silicon M5 MacBook Air (24 GB Unified Memory)** without active cooling:

- **Memory Footprint:** Core workspace, stateless `:memory:` DuckDB, and CPU-pinned PyTorch/LightGBM use $\approx 4.8\text{ GB}$, leaving $\approx 15.0\text{ GB}$ headroom.
- **Lock-Free Lakehouse Concurrency:** Neither Python nor Tauri Rust ever creates a persistent `.duckdb` lock file. Python writes to `{ticker}.parquet.tmp` and executes an atomic POSIX `os.replace()`, allowing concurrent in-memory `duckdb-rs` queries in the Terminal with zero file-lock contention.
- **CPU Thread Pinning:** `peer_gat.py` and `backbone_gru.py` are explicitly pinned to CPU (`torch.set_num_threads(4)`) to avoid PyG Apple Metal (MPS) scatter/gather fallback overhead.
- **Hybrid LLM Backfill:** Supports an optional batch API adapter (Groq/DeepInfra/Together) for the one-time 8,000-filing historical 10-Q diff backfill, while all ongoing daily/quarterly filings and Copilot queries run 100% locally via **Ollama** (`llama3.1:8b`).

---

## Legal, Regulatory & Licensing Disclaimers

1. **Non-Investment Advice:** Amstal is strictly a quantitative research, backtesting, and educational simulation platform provided "AS IS" under the MIT License. Nothing generated by this software constitutes investment, legal, or tax advice.
2. **Market Data Vendor TOS:** All raw and processed market data files (`data/*.parquet`) are `.gitignored`. Users must supply their own API keys and comply with vendor redistribution terms.
3. **SEC EDGAR Fair Access:** All requests to `sec.gov` enforce a $\le 10\text{ req/s}$ rate limit and include a compliant `User-Agent` header.
4. **Trademark Notice:** Amstal is an independent open-source project and is not affiliated with, sponsored by, or endorsed by Bloomberg Finance L.P.
5. **FEMA / LRS Compliance (India):** Default portfolio sizing (`LONG_ONLY`) restricts allocations to unleveraged, cash-settled equity delivery trades in accordance with RBI Liberalised Remittance Scheme (LRS) guidelines.

---

## Directory Layout

```
amstal/
├── PRD.md                         # Master Product Requirements Document (Spec v2.1)
├── architecture.md                # Master End-to-End Technical Architecture
├── amstal-engine/
│   ├── PRD.md                     # Engine-synced PRD
│   ├── architecture.md            # Engine-synced Technical Architecture
│   ├── README.md                  # Architecture, Quant Specs & Usage Guide
│   ├── LICENSE                    # MIT License
│   ├── pyproject.toml             # Build & dependency specification
│   ├── data/                      # Data store (git-ignored)
│   │   ├── raw/                   # Daily/hourly OHLCV, SEC raw filings
│   │   ├── processed/             # Atomic-renamed Parquet feature lakehouse
│   │   └── universe/              # Survivor-bias-free 250-stock annual snapshots
│   ├── src/
│   │   ├── data/
│   │   │   ├── market_loader.py   # Hybrid hourly/daily OHLCV loader (Polygon/yfinance)
│   │   │   ├── sec_edgar.py       # edgartools XBRL hierarchy & Form 4 parser
│   │   │   └── universe.py        # Liquid universe selector & PIT delisting filters
│   │   ├── features/
│   │   │   ├── residualizer.py    # 2-Stage Sequential OLS beta orthogonalization
│   │   │   ├── microstructure.py  # 15ch Dual-Mode sequence + 105ch LightGBM summary
│   │   │   ├── sec_diff.py        # Item 1A/MD&A redlines (Hybrid Batch / Local Ollama)
│   │   │   ├── news_sentiment.py  # FinBERT scoring & continuous event decay
│   │   │   └── graph_builder.py   # Monthly rolling GICS + Lead-Lag PyG graphs
│   │   ├── models/
│   │   │   ├── backbone_gru.py    # CPU-pinned 1-Layer GRU + 2-Head Self-Attention
│   │   │   ├── backbone_lgbm.py   # Sample-weighted LightGBM on 105 summary features
│   │   │   ├── peer_gat.py        # CPU-pinned 2-Layer PyG GAT with root extraction
│   │   │   ├── regime_hmm.py      # Forward-filtered 3-state Gaussian HMM
│   │   │   └── fusion_net.py      # Modality-masked Cross-Modal Attention, HMM gate & head
│   │   ├── label/
│   │   │   ├── triple_barrier.py  # Residualized triple-barrier labeling
│   │   │   └── weights.py         # Sample uniqueness & cross-ticker calendar-date purge
│   │   ├── risk/
│   │   │   ├── calibration.py     # Temperature scaling via NLL on 126d validation fold
│   │   │   └── kelly_engine.py    # Multi-outcome fractional Kelly + L1 normalizer
│   │   ├── ipc/
│   │   │   ├── socket_server.py   # UDS (/tmp/amstal.sock) JSON-RPC listener
│   │   │   └── stream_server.py   # WebSocket 100ms conflated broadcaster (:8194)
│   │   └── backtest/
│   │       ├── walk_forward.py    # 756d Train / 10d Purge / 126d Val / 10d Purge / 63d Test
│   │       └── metrics.py         # Rank IC, LONG_ONLY PnL, LONG_SHORT Sharpe & Sanity tests
│   └── tests/                     # Unit tests and Gate 1-4 falsification checks
└── amstal-terminal/               # Tauri 2.0 + Svelte 5 Workstation (Tier 1 -> Tier 2)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) (for Phase 3 SEC 10-Q/10-K local LLM extraction with `llama3.1:8b`)
- Recommended: macOS with Apple Silicon (M-series) or Linux

### Installation

```bash
git clone git@github.com:wwwrryyy/amstal.git
cd amstal/amstal-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## License

This project is licensed under the [MIT License](LICENSE).
