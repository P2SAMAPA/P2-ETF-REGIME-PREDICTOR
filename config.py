# config.py — P2-ETF-REGIME-PREDICTOR v2
# =========================================
# Central configuration for Option A (FI/Commodities) and Option B (Equity ETFs).

import os

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-regime-predictor")

# ── GitHub (for Streamlit workflow triggers) ──────────────────────────────────
GITHUB_REPO = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
GH_PAT      = os.environ.get("GH_PAT", "")

# ── Option A — FI / Commodities ETFs ─────────────────────────────────────────
OPTION_A_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
OPTION_A_BENCHMARKS = ["SPY", "AGG"]
OPTION_A_ALL_TICKERS = OPTION_A_ETFS + OPTION_A_BENCHMARKS

# HF namespaced paths for Option A
OPTION_A_HF = {
    "dataset":      "option_a/etf_data.parquet",
    "predictions":  "option_a/mom_pred_history.parquet",
    "wf_preds":     "option_a/wf_mom_pred_history.parquet",
    "signals":      "option_a/signals.parquet",
    "detector":     "option_a/models/regime_detector.pkl",
    "ranker":       "option_a/models/momentum_ranker.pkl",
    "feature_list": "option_a/meta/feature_list.json",
    "sweep_prefix": "option_a/sweep/sweep_",
}

# ── Option B — Equity ETFs ────────────────────────────────────────────────────
OPTION_B_ETFS = [
    "SPY",  # S&P 500
    "QQQ",  # NASDAQ 100
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "GDX",  # Gold Miners
    "XME",  # Metals & Mining
]
OPTION_B_BENCHMARKS = ["SPY", "QQQ"]  # SPY already in universe, used as benchmark ref
OPTION_B_ALL_TICKERS = list(dict.fromkeys(OPTION_B_ETFS + OPTION_B_BENCHMARKS))  # deduplicated

# HF namespaced paths for Option B
OPTION_B_HF = {
    "dataset":      "option_b/etf_data.parquet",
    "predictions":  "option_b/mom_pred_history.parquet",
    "wf_preds":     "option_b/wf_mom_pred_history.parquet",
    "signals":      "option_b/signals.parquet",
    "detector":     "option_b/models/regime_detector.pkl",
    "ranker":       "option_b/models/momentum_ranker.pkl",
    "feature_list": "option_b/meta/feature_list.json",
    "sweep_prefix": "option_b/sweep/sweep_",
}

# ── Shared data settings ──────────────────────────────────────────────────────
START_YEAR_DEFAULT = 2008

FRED_SERIES = {
    "DGS10":        "10Y Treasury Yield",
    "T10Y2Y":       "10Y-2Y Yield Spread",
    "T10Y3M":       "10Y-3M Yield Spread",
    "DTB3":         "3M T-Bill Rate",
    "MORTGAGE30US": "30Y Mortgage Rate",
    "VIXCLS":       "VIX",
    "DTWEXBGS":     "USD Broad Index",
    "DCOILWTICO":   "WTI Crude Oil",
    "BAMLC0A0CM":   "IG Corporate Spread",
    "BAMLH0A0HYM2": "HY Credit Spread",
    "UMCSENT":      "UMich Consumer Sentiment",
    "T10YIE":       "10Y Breakeven Inflation",
}

# ── Strategy parameters ───────────────────────────────────────────────────────
CONVICTION_Z_MIN = 0.7
STOP_LOSS_PCT    = -0.12
Z_REENTRY        = 1.0
TRANSACTION_BPS  = 5
RISK_FREE_RATE   = 0.045   # 4.5% annual fallback

# ── Walk-forward windows (expanding window from 2008) ────────────────────────
WINDOWS = [
    {"id": 1,  "train_start": "2008-01-01", "train_end": "2010-12-31", "test_year": "2011"},
    {"id": 2,  "train_start": "2008-01-01", "train_end": "2011-12-31", "test_year": "2012"},
    {"id": 3,  "train_start": "2008-01-01", "train_end": "2012-12-31", "test_year": "2013"},
    {"id": 4,  "train_start": "2008-01-01", "train_end": "2013-12-31", "test_year": "2014"},
    {"id": 5,  "train_start": "2008-01-01", "train_end": "2014-12-31", "test_year": "2015"},
    {"id": 6,  "train_start": "2008-01-01", "train_end": "2015-12-31", "test_year": "2016"},
    {"id": 7,  "train_start": "2008-01-01", "train_end": "2016-12-31", "test_year": "2017"},
    {"id": 8,  "train_start": "2008-01-01", "train_end": "2017-12-31", "test_year": "2018"},
    {"id": 9,  "train_start": "2008-01-01", "train_end": "2018-12-31", "test_year": "2019"},
    {"id": 10, "train_start": "2008-01-01", "train_end": "2019-12-31", "test_year": "2020"},
    {"id": 11, "train_start": "2008-01-01", "train_end": "2020-12-31", "test_year": "2021"},
    {"id": 12, "train_start": "2008-01-01", "train_end": "2021-12-31", "test_year": "2022"},
    {"id": 13, "train_start": "2008-01-01", "train_end": "2022-12-31", "test_year": "2023"},
    {"id": 14, "train_start": "2008-01-01", "train_end": "2023-12-31", "test_year": "2024"},
]

# ── Sweep years for consensus (used in app.py) ────────────────────────────────
SWEEP_YEARS = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

# ── ETF display colours ───────────────────────────────────────────────────────
ETF_COLORS = {
    # Option A
    "TLT": "#4e79a7", "VNQ": "#76b7b2", "SLV": "#edc948",
    "GLD": "#b07aa1", "LQD": "#59a14f", "HYG": "#e15759",
    # Option B
    "QQQ": "#f28e2b", "XLK": "#ff9da7", "XLF": "#9c755f",
    "XLE": "#bab0ac", "XLV": "#499894", "XLI": "#86bcb6",
    "XLY": "#e15759", "XLP": "#79706e", "XLU": "#d37295",
    "GDX": "#edc948", "XME": "#b07aa1",
    # Shared
    "SPY": "#4e79a7", "AGG": "#76b7b2", "CASH": "#aaaaaa",
}

# ── Regime colours ────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Risk-On":           "#16a34a",
    "Risk-Off":          "#dc2626",
    "Rate-Rising":       "#d97706",
    "Rate-Falling":      "#3b82f6",
    "Crisis":            "#8b5cf6",
    "Risk-On-Commodity": "#f97316",
}

# ── Conviction colours ────────────────────────────────────────────────────────
CONVICTION_COLORS = {
    "Very High": "#16a34a",
    "High":      "#22c55e",
    "Moderate":  "#f97316",
    "Low":       "#dc2626",
}
