# P2-ETF-REGIME-PREDICTOR

A regime-aware ETF rotation model combining Wasserstein distance-based market clustering with composite momentum ranking to generate daily trading signals.

Two independent options:
- **Option A** — FI / Commodities ETFs: TLT · VNQ · SLV · GLD · LQD · HYG
- **Option B** — Equity ETFs: SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME

---

## Overview

The model answers one practical question each trading day: **which ETF is most likely to beat the risk-free rate over the next 5 trading days?**

It does this in two layers:

**Layer 1 — Wasserstein k-means Regime Detection**
Each rolling 20-day window of ETF returns is treated as an empirical probability distribution. Wasserstein distance (Earth Mover's Distance) measures how far apart two distributions are, capturing their full shape rather than just mean and variance. The optimal number of regimes k is auto-selected using Maximum Mean Discrepancy (MMD) scoring. Based on Horvath, Issa & Muguruza (2021/2024).

**Layer 2 — Momentum Ranking**
A composite momentum score ranks all ETFs each day using Rate-of-Change (5d/10d/21d/63d), OBV accumulation (21d), and Breakout score (20d range position). Scores are Z-normalised cross-sectionally. The top-ranked ETF with Z ≥ 0.7σ enters; otherwise the strategy holds CASH earning the 3M T-Bill rate.

---

## Strategy Rules

| Rule | Detail |
|---|---|
| Conviction gate | Z ≥ 0.7σ to enter; below threshold earns T-Bill |
| Stop-loss | 2-day cumulative loss ≤ −12% → CASH until Z ≥ 1.0σ |
| Transaction cost | 5bps per one-way trade |

---

## Validation

Walk-forward validated using expanding 3-year training / 1-year test windows (2011–present). The model never sees test-period data during any training fold. Walk-forward OOS predictions are updated incrementally each day — only new/missing folds are computed, keeping daily runs fast.

---

## Infrastructure

```
GitHub            — code + Actions workflows
Hugging Face      — dataset, models, predictions, signals (namespaced by option)
Streamlit Cloud   — read-only UI
```

### HF Dataset Structure

```
P2SAMAPA/p2-etf-regime-predictor
├── option_a/
│   ├── etf_data.parquet               — Option A feature dataset
│   ├── mom_pred_history.parquet       — Option A in-sample predictions
│   ├── wf_mom_pred_history.parquet    — Option A walk-forward OOS predictions
│   ├── signals.parquet                — Option A daily signal log
│   ├── models/
│   │   ├── regime_detector.pkl
│   │   └── momentum_ranker.pkl
│   ├── meta/feature_list.json
│   └── sweep/sweep_{year}_{date}.json
└── option_b/
    └── (same structure as option_a)
```

---

## Repository Structure

```
P2-ETF-REGIME-PREDICTOR/
├── app.py                    # Streamlit UI
├── train_hf.py               # Training pipeline (--option a/b)
├── daily_data_update.py      # Incremental data update (both options)
├── data_manager_hf.py        # Data fetching, features, HF I/O
├── config.py                 # Central config (ETF universes, paths, params)
├── regime_detection.py       # Wasserstein k-means regime detector
├── models.py                 # MomentumRanker
├── strategy.py               # Signal execution, backtesting, metrics
├── utils.py                  # Shared utilities
├── requirements.txt
└── .github/workflows/
    ├── seed_hf_dataset.yml   # One-time full seed (manual only)
    ├── daily_data_update.yml # Incremental data update (Mon–Fri after close)
    ├── daily_pipeline.yml    # Full train + WF + sweep, Options A & B in parallel
    └── manual_retrain.yml    # Full retrain, Options A & B in parallel (daily + on-demand)
```

---

## Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `seed_hf_dataset.yml` | Manual only | One-time full dataset seed from scratch |
| `daily_data_update.yml` | Mon–Fri 21:30 UTC | Incremental OHLCV + macro data update |
| `daily_pipeline.yml` | Mon–Fri 22:30 UTC | Full train + incremental WF CV + sweep (A & B parallel) |
| `manual_retrain.yml` | Mon–Fri 03:00 UTC + on-demand | Full retrain all folds (A & B parallel) |

Options A and B run as parallel jobs within the same workflow, making efficient use of GitHub Actions free-tier concurrency.

---

## Setup

### Required Secrets

**GitHub Secrets** (Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write token |
| `FRED_API_KEY` | FRED API key (fred.stlouisfed.org) |

**Streamlit Community Cloud Secrets**:

```toml
HF_TOKEN        = "your_hf_token"
HF_DATASET_REPO = "P2SAMAPA/p2-etf-regime-predictor"
FRED_API_KEY    = "your_fred_key"
GITHUB_REPO     = "P2SAMAPA/P2-ETF-REGIME-PREDICTOR"
GH_PAT          = "your_github_pat"
```

### First Run

1. Push all files to GitHub
2. Actions → **Seed HF Dataset** → Run workflow (seeds both options)
3. Actions → **Manual Retrain** → Run workflow (trains both options, all folds)
4. Verify HF dataset populates under `option_a/` and `option_b/`
5. Deploy to Streamlit Community Cloud

### Local Testing

```bash
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=...
export FRED_API_KEY=...

# Full train Option A
python train_hf.py --option a

# Full train Option B  
python train_hf.py --option b

# Walk-forward CV (incremental)
python train_hf.py --option a --wfcv

# Consensus sweep
python train_hf.py --option a --sweep

# Incremental data update (both options)
python daily_data_update.py
```

---

## train_hf.py Usage

```
python train_hf.py --option a                    # Full train Option A
python train_hf.py --option b                    # Full train Option B
python train_hf.py --option a --force-refresh    # Force data rebuild
python train_hf.py --option a --wfcv             # Incremental walk-forward CV
python train_hf.py --option a --sweep            # Consensus sweep (all years)
python train_hf.py --option a --sweep-year 2015  # Sweep for one year only
python train_hf.py --option a --single-year 2015 # Single-year WF test
```

---

## Disclaimer

This project is for research and educational purposes only. It is not investment advice. Past backtest performance does not guarantee future results.

---

## References

Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein Distance*. SSRN Working Paper 3947905. Published in Journal of Computational Finance (2024). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905
