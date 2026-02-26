# P2-ETF-REGIME-PREDICTOR

A regime-aware ETF rotation model combining Wasserstein distance-based market clustering with a LightGBM + Logistic Regression ensemble to generate daily trading signals across 5 fixed income and commodity ETFs.

---

## Overview

Most quantitative models treat all market environments the same. This model explicitly detects the current market regime before making predictions — a regime of rising rates calls for different signals than a crisis or risk-on environment. The key insight is borrowed from academic research: instead of assuming returns follow a known distribution, we treat each rolling window of returns as an empirical probability distribution and measure how far apart those distributions are using the Wasserstein distance.

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

### Layer 2 — LightGBM + Logistic Regression Ensemble

For each regime, **5 independent binary classifiers** are trained — one per ETF. Each model answers: *"Will this ETF beat the 3M T-Bill rate over the next 5 trading days?"*

- **LightGBM** captures non-linear interactions between macro features (60% ensemble weight)
- **Logistic Regression (L1)** provides a sparse interpretable baseline (40% weight)
- When the two models **disagree** by more than 15%, conviction is halved — a natural uncertainty filter
- Falls back to a global model when a regime has fewer than 150 training rows

### Input Features (~60-70 signals)

**FRED Macro:**
DGS10 · T10Y2Y · T10Y3M · DTB3 · MORTGAGE30US · VIXCLS · DTWEXBGS · DCOILWTICO · BAMLC0A0CM · BAMLH0A0HYM2 · UMCSENT · T10YIE

**Derived Macro:**
Real Yield (DGS10 − T10YIE) · Rate momentum (20d/60d) · Rising/falling regime flags · Yield curve shape flags · Inflation regime · VIX regime · Credit stress · Rolling Z-scores (60d window)

**ETF Signals (per ETF):**
Daily return · Realised volatility (10d/21d) · Momentum (5d/21d/63d) · Volume ratio · ATR14 · Relative strength vs SPY

### Strategy Execution

| Rule | Detail |
|------|--------|
| Conviction gate | Z-score ≥ 0.5σ to enter — below this earns 3M T-Bill |
| Stop-loss | 2-day cumulative loss ≤ −12% → CASH until Z ≥ 1.0σ |
| Rotation | Top pick 5-day cumulative loss → rotate to #2 ETF |
| Disagreement filter | LightGBM vs LogReg gap > 15% → halve conviction |
| Transaction cost | 5bps per one-way trade |

---

## ETF Universe

| ETF | Description | Regime Affinity |
|-----|-------------|-----------------|
| TLT | iShares 20+ Year Treasury Bond | Risk-Off, Rate-Falling |
| TBT | ProShares UltraShort 20+ Year Treasury | Rate-Rising |
| VNQ | Vanguard Real Estate ETF | Risk-On, low inflation |
| SLV | iShares Silver Trust | Crisis, high inflation |
| GLD | SPDR Gold Shares | Crisis, Stagflation |

Benchmarks: SPY (S&P 500) · AGG (US Aggregate Bond)

---

## Infrastructure

```
GitHub (this repo)          — code + GitHub Actions workflows
GitLab (p2-etf-regime-predictor) — dataset, models, signals storage
GitHub Actions              — daily 6:30am EST training pipeline
Streamlit Community Cloud   — read-only UI
```

### Daily Pipeline (GitHub Actions, 6:30am EST weekdays)

```
1. Load dataset from GitLab
2. Fetch yesterday's new data (yfinance + FRED)
3. Wasserstein k-means regime detection
4. Build binary forward return targets (5-day horizon)
5. Train LightGBM + LogReg per (ETF, regime)
6. Generate prediction history for backtest
7. Produce next-day signal
8. Push dataset, models, signals to GitLab
```

### GitLab Storage Structure

```
data/
  etf_data.csv          — full feature dataset, daily updated
models/
  regime_detector.pkl   — fitted WassersteinKMeans model
  model_bank.pkl        — all regime-specific classifiers
signals/
  signals.csv           — daily signal log
meta/
  feature_list.json     — saved feature names for inference consistency
```

---

## Repository Structure

```
P2-ETF-REGIME-PREDICTOR/
├── app.py                  # Streamlit UI (read-only, loads from GitLab)
├── train.py                # Pipeline orchestrator (called by Actions)
├── data_manager.py         # FRED + yfinance fetching, feature engineering
├── regime_detection.py     # Wasserstein k-means implementation
├── models.py               # LightGBM + LogReg ensemble, RegimeModelBank
├── strategy.py             # Signal execution, backtesting, metrics
├── utils.py                # Shared helpers
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        ├── daily_pipeline.yml    # Weekday 6:30am EST run
        └── manual_retrain.yml    # On-demand full rebuild
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
2. Go to Actions → **Manual Retrain** → Run workflow (force refresh: true)
3. Wait ~10 minutes for first full pipeline run
4. Verify GitLab folders populate with `etf_data.csv`, `model_bank.pkl`, `signals.csv`
5. Connect repo to Streamlit Community Cloud and deploy

### Local Testing (optional)

```bash
pip install -r requirements.txt
# Create .env with your secrets (never commit this file)
python train.py --local --force-refresh   # skips GitLab write
```

---

## Disclaimer

This project is for research and educational purposes only. It is not investment advice. Past backtest performance does not guarantee future results. TBT is a leveraged inverse ETF with structural decay and is unsuitable for long holding periods. Always consult a qualified financial advisor before making investment decisions.

---

## References

Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein Distance*. SSRN Working Paper 3947905. Published in Journal of Computational Finance (2024). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905
