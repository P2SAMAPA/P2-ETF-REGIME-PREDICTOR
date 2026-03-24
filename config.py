# config.py — P2-ETF-REGIME-PREDICTOR (HF Dataset Version)
# ==========================================================
# Central configuration for the project.

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-regime-predictor")
HF_SOURCE_REPO  = os.environ.get("HF_SOURCE_REPO", "P2SAMAPA/p2-etf-deepwave-dl")

# ── GitHub Actions trigger (for Streamlit) ─────────────────────────────────────
GITHUB_REPO = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
GH_PAT      = os.environ.get("GH_PAT", "")   # Personal Access Token (set in Streamlit secrets)

# ── Data ───────────────────────────────────────────────────────────────────────
START_YEAR_DEFAULT = 2008
TARGET_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
BENCHMARK_ETFS = ["SPY", "AGG"]
ALL_TICKERS = TARGET_ETFS + BENCHMARK_ETFS

# ── Strategy parameters (used in app and strategy.py) ──────────────────────────
CONVICTION_Z_MIN = 0.7
STOP_LOSS_PCT    = -0.12
Z_REENTRY        = 1.0
TRANSACTION_BPS  = 5
RISK_FREE_RATE   = 0.045   # 4.5% annual

# ── Walk‑forward windows (for single‑year runs and CV) ─────────────────────────
WINDOWS = [
    {'id': 1,  'train_start': '2008-01-01', 'train_end': '2010-12-31', 'test_year': '2011'},
    {'id': 2,  'train_start': '2008-01-01', 'train_end': '2011-12-31', 'test_year': '2012'},
    {'id': 3,  'train_start': '2008-01-01', 'train_end': '2012-12-31', 'test_year': '2013'},
    {'id': 4,  'train_start': '2008-01-01', 'train_end': '2013-12-31', 'test_year': '2014'},
    {'id': 5,  'train_start': '2008-01-01', 'train_end': '2014-12-31', 'test_year': '2015'},
    {'id': 6,  'train_start': '2008-01-01', 'train_end': '2015-12-31', 'test_year': '2016'},
    {'id': 7,  'train_start': '2008-01-01', 'train_end': '2016-12-31', 'test_year': '2017'},
    {'id': 8,  'train_start': '2008-01-01', 'train_end': '2017-12-31', 'test_year': '2018'},
    {'id': 9,  'train_start': '2008-01-01', 'train_end': '2018-12-31', 'test_year': '2019'},
    {'id': 10, 'train_start': '2008-01-01', 'train_end': '2019-12-31', 'test_year': '2020'},
    {'id': 11, 'train_start': '2008-01-01', 'train_end': '2020-12-31', 'test_year': '2021'},
    {'id': 12, 'train_start': '2008-01-01', 'train_end': '2021-12-31', 'test_year': '2022'},
    {'id': 13, 'train_start': '2008-01-01', 'train_end': '2022-12-31', 'test_year': '2023'},
    {'id': 14, 'train_start': '2008-01-01', 'train_end': '2023-12-31', 'test_year': '2024'},
]
