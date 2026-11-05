# Amstal (QARE-MM v1.3)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-0288d1.svg)](https://lightgbm.readthedocs.io/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3f51b5.svg)](https://pyg.org/)
[![Platform](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20x86_64-lightgrey.svg)]()

> **Multi-Modal Quantitative Alpha & Risk Engine**  
> An institutional-grade cross-sectional swing trading and portfolio risk-management system designed for sample efficiency, strict Point-in-Time (PIT) integrity, and market/sector-residualized idiosyncratic alpha.

---

## Table of Contents

- [Overview](#overview)
- [Key Architectural Tenets](#key-architectural-tenets)
- [System Architecture](#system-architecture)
- [Quantitative & Mathematical Framework](#quantitative--mathematical-framework)
  - [1. Market- & Sector-Residualized Returns](#1-market---sector-residualized-returns)
  - [2. Cross-Sectional Rank Normalization](#2-cross-sectional-rank-normalization)
  - [3. Microstructure & Overnight Decomposition](#3-microstructure--overnight-decomposition)
  - [4. Continuous Event-Decay Kernel](#4-continuous-event-decay-kernel)
  - [5. Variance-Sorted Forward-Filtered HMM](#5-variance-sorted-forward-filtered-hmm)
  - [6. Purged Triple-Barrier Labeling & Sample Uniqueness](#6-purged-triple-barrier-labeling--sample-uniqueness)
  - [7. Temperature Calibration & Multi-Outcome Kelly Sizing](#7-temperature-calibration--multi-outcome-kelly-sizing)
- [Multi-Modal Feature Dictionary](#multi-modal-feature-dictionary)
  - [Price & Intraday Microstructure Matrix `(Batch, 126, 15)`](#price--intraday-microstructure-matrix-batch-126-15)
  - [SEC EDGAR & NLP Alt-Data Vector `(Batch, 12)`](#sec-edgar--nlp-alt-data-vector-batch-12)
  - [Dynamic Rolling Peer Graph `(Total_Nodes, 10)`](#dynamic-rolling-peer-graph-total_nodes-10)
- [Phase Gating & Falsification Pipeline](#phase-gating--falsification-pipeline)
- [Target Output API Contract](#target-output-api-contract)
- [Hardware & Thermal Ergonomics](#hardware--thermal-ergonomics)
- [Directory Layout](#directory-layout)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

Most academic multi-modal deep learning models applied to finance suffer from severe sample starvation, retail indicator decay, look-ahead leakage, and overparameterization. **Amstal (QARE-MM v1.3)** is built around institutional reality:

- **Universe:** 250 liquid US mid-to-large-cap equities ($>\$25\text{M}$ ADV, S&P 500 ranks 100–500 and S&P MidCap 400, ex-Financials & Utilities).
- **Target Horizon:** 5 to 20 trading days (10-day vertical swing horizon).
- **Frictional Hurdle:** 15 bps (0.15%) round-trip penalty per trade (slippage, spread, and brokerage).
- **Lookback:** Rolling 126 trading days ($T = 126$, ~6 calendar months).
- **Core Signal Channels:**
  1. High-frequency price, flow, and intraday microstructure.
  2. SEC EDGAR catalysts (Form 4 insider buying/selling, 10-Q/10-K Item 1A/MD&A redline diffs, and PEAD).
  3. Dynamic rolling GICS and lead-lag cross-correlation peer graphs.
  4. Macro volatility regime filtering via forward-filtered Gaussian HMM.

---

## Key Architectural Tenets

1. **Idiosyncratic Alpha Isolation:** Raw equity returns are roughly 65% macro and sector beta. Amstal explicitly strips rolling market (`SPY`) and GICS sector ETF betas before labeling or modeling.
2. **Strict Point-in-Time (PIT) Integrity:** Any SEC filing, corporate announcement, or news headline published after 16:00 EST on day $t$ is strictly stamped to day $t+1$. Restated fundamentals and look-ahead graph structures are strictly forbidden.
3. **Model-Agnostic Gating:** Sample-Weighted Gradient Boosted Decision Trees (LightGBM/CatBoost) and a compact temporal neural network (`hidden_size = 32`) compete as co-equals. If trees outperform sequence nets out-of-sample, the neural sequence branch is pruned.
4. **Thermal & Memory Efficiency:** Designed to run full walk-forward sweeps and training on Apple Silicon Unified Memory (M5 Air 24 GB) without thermal throttling, leveraging right-sized 32-channel latent representations and offline Apache Parquet caching.
5. **Falsification-First Discipline:** Mandatory sanity gates (shuffled label tests, forward-leak sanity tests, and strict Sharpe ceilings) ensure signals represent genuine predictive alpha rather than look-ahead artifacts.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AMSTAL NEURAL TOPOLOGY (v1.3)                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Cross-Sectional Price & Flow]      [SEC EDGAR & NLP Alt-Data]       [Rolling Peer & GICS Graph]
     Shape: (Batch, 126, 15)              Shape: (Batch, 12)             Nodes: (Total_Nodes, 10)
                │                                  │                     Edges: (2, Total_Edges)
                ▼                                  ▼                                │
    ┌───────────────────────┐          ┌───────────────────────┐                    ▼
    │ 1-Layer GRU / LSTM    │          │ Linear(12 -> 32)      │          ┌───────────────────┐
    │ + 2-Head Self-Attn    │          │ + LayerNorm           │          │ 2-Layer PyG GAT   │
    │ hidden_size = 32      │          │ + GELU + Dropout(0.2) │          │ Root extracted    │
    └───────────┬───────────┘          └───────────┬───────────┘          │ via batch.ptr[:-1]│
                │ (B, 126, 32)                     │ (B, 1, 32)           └─────────┬─────────┘
                └────────────────┬─────────────────┘                                │
                                 ▼                                                  │
                  ┌─────────────────────────────┐                                   │
                  │ Cross-Modal Attention       │                                   │
                  │ Q: Alt-Data (B, 1, 32)      │                                   │
                  │ K, V: Temporal (B, 126, 32) │                                   │
                  │ + Residual (t_126)          │                                   │
                  └──────────────┬──────────────┘                                   │
                                 │ (B, 32)                                          │ (B, 32)
                                 └──────────────────────┬───────────────────────────┘
                                                        ▼
                                              Concat(32 + 32 + 2)
                                                  = (B, 66)
                                                        │
                                                        ▼
   [Filtered HMM Probabilities]               ┌───────────────────┐
       Shape: (Batch, 3) ───────────────────► │ Sigmoid Gate      │
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
                                              Multi-Outcome Kelly Sizer
```

---

## Quantitative & Mathematical Framework

### 1. Market- & Sector-Residualized Returns
Rolling 60-day OLS betas against the market (`SPY`) and the corresponding GICS Sector ETF ($\text{ETF}_s$) isolate purely idiosyncratic stock price movement:

$$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$

$$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, t}\right)$$

### 2. Cross-Sectional Rank Normalization
To prevent distributional shifts across market cycles, every feature $x_{i, t}$ is cross-sectionally ranked across all active universe assets ($M_t \approx 250$) into $[-1.0, +1.0]$:

$$\tilde{x}_{i, t} = 2 \cdot \left(\frac{\text{rank}(x_{i, t}) - 1}{M_t - 1}\right) - 1.0$$

### 3. Microstructure & Overnight Decomposition
Informed institutional flow concentrates overnight and during the closing cross; retail liquidity shocks dominate intraday:
* **Overnight Log-Return:** $r_{\text{overnight}, t} = \ln(P_{\text{open}, t} / P_{\text{close}, t-1})$
* **Intraday Session Log-Return:** $r_{\text{intraday}, t} = \ln(P_{\text{close}, t} / P_{\text{open}, t})$
* **Realized Hourly Volatility:** $\sigma_{\text{intraday}, t} = \sqrt{\sum_{h=1}^7 r_{t, h}^2}$
* **Closing Power:** $\text{CP}_t = \ln\left(\frac{V_{t, \text{hour 7}} + 10^{-5}}{V_{t, \text{hour 1}} + 10^{-5}}\right)$
* **Close-to-VWAP Deviation:** $\Delta \text{VWAP}_t = \frac{P_{t, \text{close}} - \text{VWAP}_t}{\text{VWAP}_t}$

### 4. Continuous Event-Decay Kernel
Catalyst scores $X_0 \in [-1.0, 1.0]$ from discrete filings, transactions, and earnings decay exponentially over elapsed trading days ($\Delta d$):

$$X_{\text{event}}(t) = X_0 \cdot \exp\left(-\frac{\ln(2)}{\tau} \cdot \Delta d\right)$$

* News / Sentiment: $\tau_{\text{news}} = 3.0\text{ days}$
* PEAD Shocks: $\tau_{\text{pead}} = 10.0\text{ days}$
* SEC 10-Q/10-K Redline Diffs: $\tau_{\text{legal}} = 15.0\text{ days}$
* SEC Form 4 Insider Clusters: $\tau_{\text{form4}} = 20.0\text{ days}$

### 5. Variance-Sorted Forward-Filtered HMM
A 3-state Gaussian HMM is fit on standardized `SPY` returns and `VIX` levels (with 10 random EM restarts):
* **Forward-Filtered Probabilities Only:** Computes $\alpha_t(k) = P(x_{1:t}, S_t = k)$ and normalizes forward. Backward-pass smoothing is strictly banned to prevent forward-looking bias.
* **Return-Variance State Alignment:** States are sorted by variance component $\Sigma_{k, rr}$ to ensure permutation invariance across walk-forward folds:
  $$\Sigma_{0, rr} < \Sigma_{1, rr} < \Sigma_{2, rr} \implies [\text{State 0: Low-Vol Bull, State 1: Sideways, State 2: High-Vol Crash}]$$

### 6. Purged Triple-Barrier Labeling & Sample Uniqueness
Ternary labels $y_{i, t} \in \{0: \text{Outperform}, 1: \text{Neutral}, 2: \text{Underperform}\}$ are defined on cumulative idiosyncratic return paths:
* **Upper Barrier:** $+2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Lower Barrier:** $-1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Vertical Barrier:** $H = 10\text{ trading days}$
* **Calendar-Date Purging:** Any sample resolving after training cutoff date $D_{\text{train}}$ is purged across all tickers.
* **Sample Uniqueness Weighting:** Observations are weighted by average inverse concurrency $u_{i, t}$ over their active barrier resolution window:
  $$u_{i, t} = \frac{1}{t_{\text{end}} - t + 1} \sum_{\tau = t}^{t_{\text{end}}} \frac{1}{c_{i, \tau}}$$

### 7. Temperature Calibration & Multi-Outcome Kelly Sizing
1. **Temperature Scaling:** Raw logits $z_k$ are calibrated on an independent purged validation set by finding scalar $T > 0$ minimizing Negative Log-Likelihood:
   $$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_{j=0}^2 \exp(z_j / T)}$$
2. **Multi-Outcome Kelly Sizing:** Portfolio weights maximize expected log-wealth net of 15 bps round-trip friction ($c = 0.0015$):
   $$\max_{f \in [0, f_{\max}]} \mathbb{E}[\ln(1 + f \cdot R_{\text{net}})] = \sum_{k \in \{\text{buy}, \text{sell}, \text{hold}\}} \hat{p}_k \ln(1 + f (b_k - c))$$
   - Multiplied by regime scalar $\kappa \in [0.25, 0.50]$ (Quarter-to-Half Kelly).
   - Single-name position size capped at 10% (15% hard absolute maximum).

---

## Multi-Modal Feature Dictionary

### Price & Intraday Microstructure Matrix `(Batch, 126, 15)`
| Channel | Feature Name | Description |
| :---: | :--- | :--- |
| `0` | Daily Log-Return | Total closing return $r_t = \ln(C_t / C_{t-1})$ |
| `1` | Idiosyncratic Residual Return | Return residualized against SPY and Sector ETF $\epsilon_t$ |
| `2` | Overnight Return | Overnight gap $\ln(O_t / C_{t-1})$ |
| `3` | Intraday Session Return | Regular session drift $\ln(C_t / O_t)$ |
| `4-6`| Residual Momentum (5d, 20d, 60d) | Multi-scale cumulative idiosyncratic momentum |
| `7` | Relative Volume Ratio | Participation volume $V_t / \bar{V}_{20}$ |
| `8` | Amihud Illiquidity Z-Score | 60-day Z-score of price impact $\|r_t\| / (C_t \cdot V_t)$ |
| `9` | ATR Volatility Ratio | Normalized range $\text{ATR}_{14, t} / C_t$ |
| `10`| Idiosyncratic Volatility (20d) | Annualized 20-day residual variance |
| `11`| Parkinson Volatility | Intraday extreme variance $(\ln(H_t / L_t))^2 / (4 \ln 2)$ |
| `12`| Realized Intraday Volatility | High-frequency hourly sum of squared returns |
| `13`| Closing Power | Ratio of auction hour volume to market open volume |
| `14`| Close-to-VWAP Deviation | Institutional accumulation above or below session VWAP |

### SEC EDGAR & NLP Alt-Data Vector `(Batch, 12)`
| Index | Feature Name | Source / Kernel |
| :---: | :--- | :--- |
| `0-1` | Decayed FinBERT Sentiment & Age | News headlines passed through local FinBERT ($\tau = 3\text{d}$) |
| `2-3` | Decayed Form 4 Insider Score & Age | SEC EDGAR Form 4 open-market cluster purchases/sales ($\tau = 20\text{d}$) |
| `4-5` | Decayed PEAD Earnings Shock & Age | Post-earnings announcement volume-scaled residual shock ($\tau = 10\text{d}$) |
| `6` | Decayed 10-Q Item 1A Risk Delta | `difflib` Item 1A redline delta scored via local `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `7` | Decayed MD&A Catalyst Score | `difflib` Item 7/Item 2 redline delta scored via local `Llama-3.1-8B` ($\tau = 15\text{d}$) |
| `8` | SEC Filing Staleness Age | Time elapsed since SEC acceptance timestamp |
| `9-11`| Modality Missingness Masks | Explicit binary indicators (news, insider, 10-Q) |

### Dynamic Rolling Peer Graph `(Total_Nodes, 10)`
- **10 Nodes per Subgraph:** Target ($v_0$), 4 GICS sub-industry peers ($v_1 \dots v_4$), 4 rolling 5-day lead-lag cross-correlation peers ($v_5 \dots v_8$), and the GICS Sector ETF ($v_9$).
- **Features:** 5d/20d residual return, 60d beta, Point-in-Time XBRL fundamentals (Earnings Yield, Revenue Growth, Operating Margin stamped strictly by `filed` date), realized idiosyncratic volatility, and lead-lag cross-correlation.

---

## Phase Gating & Falsification Pipeline

Development is bound by sequential, non-negotiable gates:

```
Phase 0: Lock Spec v1.3       ──► [GATE 0: All Tensors, PIT Rules & Universe Locked] (PASS)
          │
          ▼
Phase 1: Residualized Backbone ──► [GATE 1: Passes Shuffled/Lagged Sanity Checks;
                                            Beats Baseline Net of 15 bps OOS]
          │
          ▼
Phase 2: Regime & Peer Graph   ──► [GATE 2: Ablation Shows Filtered HMM & GAT
                                            Improve Out-of-Sample Sharpe / DD]
          │
          ▼
Phase 3: SEC EDGAR & Alt-Data  ──► [GATE 3: Form 4, PEAD, FinBERT & 10-Q Diffs
                                            Add Incremental IC > 0.02 Post-Cutoff]
          │
          ▼
Phase 4: Calibration & Kelly   ──► [GATE 4: Brier Score Drops > 10% OOS;
                                            Zero-Skill Model Sizes to ~0.0%]
```

### Stop Rules
- **No Modality Stacking Before Verification:** GAT and LLM pipelines are prohibited until Phase 1 proves predictive edge on pure market/microstructure data.
- **Model Agnosticism:** If Sample-Weighted LightGBM/CatBoost beats the Attention-GRU out-of-sample, the sequence net is dropped or weighted purely by validation log-likelihood.
- **Sharpe Ratio Cap:** Any backtest yielding annualized Sharpe $> 2.5$ or win rate $> 65\%$ on 10-day swing horizons is flagged as an automatic timestamp leak until disproven.

---

## Target Output API Contract

The engine emits structured, machine-verifiable trade decisions adhering to this JSON schema:

```json
{
  "ticker": "CRWD",
  "timestamp_utc": "2026-10-19T20:00:00Z",
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

## Hardware & Thermal Ergonomics

Amstal is tuned for execution on an **Apple Silicon M5 MacBook Air (24 GB Unified Memory)** without active cooling:

- **Memory Footprint:** Core workspace and PyTorch/LightGBM use $\approx 2.8\text{ GB}$, leaving ample headroom for OS and background tasks.
- **Process Isolation:** Heavy NLP models (`Llama-3.1-8B` Q8_0/Q4_K_M via Ollama and `ProsusAI/finbert`) never execute concurrently with model training. They run as offline checkpointed batch jobs pre-materializing scores into local Apache Parquet caches (`data/features/*.parquet`).
- **Burst Thermal Envelope:** Right-sized 32-channel latent representations allow full walk-forward epochs over 250 stocks to complete in seconds on CPU/MPS, avoiding thermal power throttling.

---

## Directory Layout

```
amstal-engine/
├── PRD.md                         # Product Requirements Document (Spec v1.3)
├── README.md                      # Architecture, Quant Specs & Usage Guide
├── LICENSE                        # MIT License
├── data/                          # Data store (git-ignored)
│   ├── raw/                       # Daily/hourly OHLCV, SEC raw filings
│   ├── processed/                 # Parquet feature stores
│   └── universe/                  # 250-stock annual constituent snapshots
├── src/
│   ├── data/
│   │   ├── market_loader.py       # Ingests OHLCV, SPY, and sector ETFs
│   │   ├── sec_edgar.py           # Form 4 parser & XBRL fundamentals
│   │   └── universe.py            # Liquid universe selector & PIT filters
│   ├── features/
│   │   ├── residualizer.py        # Rolling 60d OLS beta residualization
│   │   ├── microstructure.py      # Overnight/intraday, VWAP skew, closing power
│   │   ├── sec_diff.py            # Item 1A/MD&A redlines & Llama-3.1 scoring
│   │   ├── news_sentiment.py      # FinBERT scoring & continuous event decay
│   │   └── graph_builder.py       # Monthly rolling GICS + Lead-Lag PyG graphs
│   ├── models/
│   │   ├── backbone_gru.py        # 1-Layer GRU/LSTM + 2-Head Self-Attention
│   │   ├── backbone_lgbm.py       # Sample-weighted LightGBM/CatBoost
│   │   ├── peer_gat.py            # 2-Layer PyG GAT with root extraction
│   │   ├── regime_hmm.py          # Forward-filtered 3-state Gaussian HMM
│   │   └── fusion_net.py          # Cross-modal attention, HMM gate & head
│   ├── label/
│   │   ├── triple_barrier.py      # Residualized triple-barrier labeling
│   │   └── weights.py             # Sample uniqueness & calendar-date purge
│   ├── risk/
│   │   ├── calibration.py         # Temperature scaling via NLL optimization
│   │   └── kelly_engine.py        # Multi-outcome fractional Kelly optimizer
│   └── backtest/
│       ├── walk_forward.py        # Purged cross-validation engine
│       └── metrics.py             # IC, Sharpe, Drawdown, and Sanity tests
└── tests/                         # Unit tests and Gate 1-4 falsification checks
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) (for Phase 3 SEC 10-Q/10-K local LLM extraction with `llama3.1:8b`)
- Recommended: macOS with Apple Silicon (MPS acceleration) or Linux with CUDA

### Installation

```bash
git clone git@github.com:wwwrryyy/amstal.git
cd amstal/amstal-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).
