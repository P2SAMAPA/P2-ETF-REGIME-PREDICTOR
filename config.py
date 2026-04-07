# config.py — P2-ETF-REGIME-PREDICTOR v4
# =========================================
# Central configuration for Option A (FI/Commodities) and Option B (Equity ETFs).
#
# KEY ARCHITECTURAL FIX (v4):
#   The original WINDOWS definition gave every window the same test period
#   (2025-01-01 to present), so the "Single Year" dropdown and consensus sweep
#   always produced the same signal regardless of which start year was selected.
#
#   v4 uses EXPANDING-WINDOW walk-forward validation:
#     - Train always starts at 2008-01-01 (fixed anchor)
#     - Each window's TEST YEAR advances by one calendar year
#     - 14 historical windows (test years 2012–2025)
#     - 1 live window (test 2025-01-01 → present)
#
#   This produces genuinely different OOS predictions for each test year because:
#     (a) the model is fitted on more data each year, and
#     (b) the test period covers a completely different market regime
#         (2020 = COVID crash, 2022 = rate hike bear, 2024 = AI bull, etc.)
#
#   The dropdown label now reads "Test year" not "Train start year", which is
#   more intuitive: the user is asking "what would the model have picked in 2020?"

import os

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-regime-predictor")

# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_REPO = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
GH_PAT      = os.environ.get("GH_PAT", "")

# ── Option A — FI / Commodities ETFs ─────────────────────────────────────────
OPTION_A_ETFS       = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
OPTION_A_BENCHMARKS = ["SPY", "AGG"]
OPTION_A_ALL_TICKERS = OPTION_A_ETFS + OPTION_A_BENCHMARKS

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
OPTION_B_BENCHMARKS  = ["SPY", "QQQ"]
OPTION_B_ALL_TICKERS = list(dict.fromkeys(OPTION_B_ETFS + OPTION_B_BENCHMARKS))

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
TRAIN_ANCHOR       = "2008-01-01"   # fixed train start for all WF windows

# ── Strategy parameters ───────────────────────────────────────────────────────
CONVICTION_Z_MIN = 0.7
STOP_LOSS_PCT    = -0.12
Z_REENTRY        = 1.0
TRANSACTION_BPS  = 5
RISK_FREE_RATE   = 0.045   # 4.5% annual fallback

# ── Fixed live test period (dynamic end) ─────────────────────────────────────
TEST_START = "2025-01-01"
TEST_END   = None   # interpreted as "latest available date" at runtime

# ── FRED series ───────────────────────────────────────────────────────────────
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

# ── Expanding-window WF definition ───────────────────────────────────────────
#
# Each historical window:  train [2008-01-01 .. test_year-1-12-31]
#                          test  [test_year-01-01 .. test_year-12-31]
#
# Live window:             train [2008-01-01 .. 2024-12-31]
#                          test  [2025-01-01 .. latest]
#
# The key "test_year" is an integer used as the dropdown selection and
# as the primary key in the wf_predictions parquet (replaces "train_start").
# It is also the key saved in sweep JSON files.
#
# Minimum training requirement: 4 years (first test year = 2012).

WINDOWS = []

# Historical windows: test years 2012..2025
for test_year in range(2012, 2026):
    WINDOWS.append({
        "test_year":   test_year,
        "train_start": TRAIN_ANCHOR,
        "train_end":   f"{test_year - 1}-12-31",
        "test_start":  f"{test_year}-01-01",
        "test_end":    f"{test_year}-12-31",
        "is_live":     False,
        "label":       f"Test {test_year}  (train 2008–{test_year-1})",
    })

# Live window: train 2008-2024, test 2025-present
# Uses test_year=9999 as a sentinel so it never collides with a real year
# and is excluded from historical sweep (is_live=True).
WINDOWS.append({
    "test_year":   9999,
    "train_start": TRAIN_ANCHOR,
    "train_end":   "2024-12-31",
    "test_start":  TEST_START,
    "test_end":    None,          # replaced by df.index.max() at runtime
    "is_live":     True,
    "label":       "Live  (train 2008–2024, test 2025-present)",
})

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
    # Benchmarks
    "SPY": "#4e79a7", "AGG": "#76b7b2", "CASH": "#aaaaaa",
}

# ── Regime colours ────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Low Volatility":  "#16a34a",
    "High Volatility": "#dc2626",
    "Trending":        "#3b82f6",
    "Mean Reverting":  "#d97706",
    "Crisis":          "#8b5cf6",
    # Legacy labels (kept for backward compat)
    "Risk-On":           "#16a34a",
    "Risk-Off":          "#dc2626",
    "Rate-Rising":       "#d97706",
    "Rate-Falling":      "#3b82f6",
    "Risk-On-Commodity": "#f97316",
}

# ── Conviction colours ────────────────────────────────────────────────────────
CONVICTION_COLORS = {
    "Very High": "#16a34a",
    "High":      "#22c55e",
    "Moderate":  "#f97316",
    "Low":       "#dc2626",
    "Very Low":  "#991b1b",
}
