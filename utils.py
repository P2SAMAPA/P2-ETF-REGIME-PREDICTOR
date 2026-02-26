"""
utils.py — P2-ETF-REGIME-PREDICTOR
=====================================
Shared utility functions used across modules.

Author: P2SAMAPA
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
import pytz

log = logging.getLogger(__name__)

EST = pytz.timezone("US/Eastern")


def get_est_time() -> datetime:
    """Return current time in US/Eastern timezone."""
    return datetime.now(tz=EST)


def is_market_open() -> bool:
    """
    Returns True if US equity market is currently open.
    Simple check: weekday + 9:30-16:00 EST.
    Does not account for holidays.
    """
    now = get_est_time()
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def is_training_window() -> bool:
    """
    Returns True if we are in the daily training window.
    GitHub Actions runs at 6:30am EST — this checks if it is
    between 6:00am and 9:00am EST (pre-market, post-data-availability).
    """
    now = get_est_time()
    if now.weekday() >= 5:
        return False
    window_open  = now.replace(hour=6, minute=0,  second=0, microsecond=0)
    window_close = now.replace(hour=9, minute=0,  second=0, microsecond=0)
    return window_open <= now <= window_close


def next_trading_day(from_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Return the next trading day (weekday) after from_date.
    Skips weekends only — does not account for holidays.
    """
    if from_date is None:
        from_date = pd.Timestamp.now()
    nxt = from_date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def prev_trading_day(from_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the previous trading day before from_date."""
    if from_date is None:
        from_date = pd.Timestamp.now()
    prev = from_date - pd.Timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= pd.Timedelta(days=1)
    return prev


def format_pct(value: float, decimals: int = 2) -> str:
    """Format float as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_zscore(z: float) -> str:
    """Format Z-score with sign."""
    return f"{z:+.2f}σ"


def conviction_colour(label: str) -> str:
    """Return Streamlit-compatible colour string for conviction label."""
    return {
        "Very High": "green",
        "High":      "green",
        "Moderate":  "orange",
        "Low":       "red",
    }.get(label, "grey")


def regime_colour(regime_name: str) -> str:
    """Return colour for regime badge."""
    return {
        "Risk-On":     "#00d1b2",
        "Risk-Off":    "#ff6b6b",
        "Rate-Rising": "#ffa500",
        "Crisis":      "#cc0000",
        "Stagflation": "#9b59b6",
        "Recovery":    "#3498db",
    }.get(regime_name, "#888888")
