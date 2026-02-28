# P2-ETF-REGIME-PREDICTOR

A regime-aware ETF momentum rotation model combining Wasserstein distance-based market clustering with a composite momentum ranking score to generate daily trading signals across 6 fixed income and commodity ETFs.

---

## Overview

Most quantitative models treat all market environments the same. This model explicitly detects the current market regime before ranking ETFs — a regime of rising rates calls for different momentum signals than a crisis or risk-on environment.

The model answers one practical question each trading day: **which ETF is most likely to beat the risk-free rate over the next 5 trading days?**

---

## Methodology

### Layer 1 — Wasserstein k-means Regime Detection

Based on *"Clustering Market Regimes Using the Wasserstein Distance"* (Horvath, Issa, Muguruza, 2021 — [SSRN 3947905](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905)), published in the Journal of Computational Finance (2024).

Each rolling 20-day window of ETF returns is treated as an empirical probability distribution. The **Wasserstein distance** (Earth Mover's Distance) measures how far apart two such distributions are, accounting for their full shape rather than just mean and variance. This is mathematically more rigorous than standard k-means on moments and outperforms Hidden Markov Models on non-Gaussian financial returns.

The optimal number of regimes k is auto-selected using Maximum Mean Discrepancy (MMD) scoring. Typical regimes detected:

| Regime | Characteristics |
|--------|----------------|
| Risk-On | Low VIX, positive returns, normal yield curve |
| Risk-Off | Elevated VIX, negative returns, widening spreads |
| Rate-Rising | Inverted/flattening yield curve, rising DGS10 |
| Crisis | Very high VIX, extreme HY spread widening |

### Layer 2 — Momentum Ranking

A pure rules-based composite momentum score ranks all 6 ETFs each day. The top-ranked ETF with sufficient conviction enters; below threshold the strategy holds CASH earning the 3M T-Bill rate.

**Composite score weights:**
- 40% × Rate-of-Change 5d
- 30% × Rate-of-Change 10d
- 20% × Rate-of-Change 21d
- 10% × Rate-of-Change 63d
- 15% × OBV accumulation (21d volume-weighted momentum)
- 15% × 20d Breakout score (position within rolling high/low range)

Scores are Z-normalised cross-sectionally across all 6 ETFs each day. The top-ranked ETF with Z ≥ threshold enters; otherwise CASH.

### Strategy Execution

| Rule | Detail |
|------|--------|
| Conviction gate | Z-score ≥ 0.7σ to enter — below this earns 3M T-Bill |
| Stop-loss | 2-day cumulative loss ≤ −12% → CASH until Z ≥ 1.0σ |
| Transaction cost | 5bps per one-way trade |

### Walk-Forward Validation

Performance is validated using rolling 3-year training / 1-year test windows across the full history (2011–2026), producing 3,812 genuinely out-of-sample days. The model never sees test-period data during training for any fold. Walk-forward OOS predictions refresh monthly.

---

## ETF Universe

| ETF | Description | Regime Affinity |
|-----|-------------|-----------------|
| TLT | iShares 20+ Year Treasury Bond | Risk-Off, Rate-Falling |
| VNQ | Vanguard Real Estate ETF | Risk-On, low inflation |
| SLV | iShares Silver Trust | Crisis, high inflation |
| GLD | SPDR Gold Shares | Crisis, Stagflation |
| LQD | iShares Investment Grade Corporate Bond | Risk-On, stable rates |
| HYG | iShares High Yield Corporate Bond | Risk-On, credit expansion |

Benchmarks: SPY (S&P 500) · AGG (US Aggregate Bond)

---

## Infrastructure

```
GitHub (this repo)               — code + GitHub Actions workflows
GitLab (p2-etf-regime-predictor) — dataset, models, signals storage
GitHub Actions                   — daily 6:30am EST training pipeline
Streamlit Community Cloud        — read-only UI
```

### Daily Pipeline (GitHub Actions)

```
Weekday 6:30am EST:
1. Load dataset from GitLab
2. Fetch yesterday's new data (yfinance + FRED)
3. Wasserstein k-means regime detection
4. Fit MomentumRanker on full history
5. Generate in-sample prediction history
6. Produce next-day signal
7. Push dataset, model, predictions, signal to GitLab

1st of each month (automatic):
8. Run walk-forward OOS validation (3y train / 1y test, rolling)
9. Save fresh OOS predictions to GitLab
```

### GitLab Storage Structure

```
data/
  etf_data.csv              — full feature dataset, daily updated
  mom_pred_history.csv      — in-sample momentum predictions
  wf_mom_pred_history.csv   — walk-forward OOS predictions (monthly refresh)
models/
  regime_detector.pkl       — fitted WassersteinKMeans model
  momentum_ranker.pkl       — fitted MomentumRanker
signals/
  signals.csv               — daily signal log
```

---

## Repository Structure

```
P2-ETF-REGIME-PREDICTOR/
├── app.py                  # Streamlit UI (read-only, loads from GitLab)
├── train.py                # Pipeline orchestrator (called by Actions)
├── data_manager.py         # FRED + yfinance fetching, feature engineering
├── models.py               # MomentumRanker + walk-forward CV
├── strategy.py             # Signal execution, backtesting, metrics
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        ├── daily_pipeline.yml    # Weekday 6:30am EST + monthly WF refresh
        └── manual_retrain.yml    # On-demand full rebuild or WF run
```

---

## Setup

### Required Secrets

**GitHub Secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `GITLAB_API_TOKEN` | GitLab Personal Access Token |
| `GITLAB_REPO_URL` | `https://gitlab.com/P2SAMAPA/p2-etf-regime-predictor` |
| `FRED_API_KEY` | FRED API key from fred.stlouisfed.org |

**Streamlit Community Cloud Secrets** (App Settings → Secrets):

```toml
GITLAB_API_TOKEN = "your_token"
GITLAB_REPO_URL  = "https://gitlab.com/P2SAMAPA/p2-etf-regime-predictor"
FRED_API_KEY     = "your_fred_key"
```

### First Run

1. Push all files to GitHub
2. Go to Actions → **Manual Retrain** → Run workflow (`force_refresh=true`, `run_wfcv=true`)
3. Wait ~35 min for retrain, then ~3 hours for walk-forward job to complete
4. Verify GitLab populates with `etf_data.csv`, `momentum_ranker.pkl`, `mom_pred_history.csv`, `wf_mom_pred_history.csv`, `signals.csv`
5. Connect repo to Streamlit Community Cloud and deploy

### Local Testing

```bash
pip install -r requirements.txt
# Create .env with your secrets (never commit this file)
python train.py --local --force-refresh   # skips GitLab write
python train.py --wfcv-only               # run WF only (writes to GitLab)
```

---

## Disclaimer

This project is for research and educational purposes only. It is not investment advice. Past backtest performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## References

Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein Distance*. SSRN Working Paper 3947905. Published in Journal of Computational Finance (2024). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905
