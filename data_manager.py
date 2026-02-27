"""
data_manager.py — P3-ETF-REGIME-PREDICTOR
==========================================
Handles all data fetching, feature engineering, and GitLab storage.

Data sources:
  - FRED API: macro signals (DGS10, T10Y2Y, T10Y3M, DTB3, MORTGAGE30US,
               VIXCLS, DTWEXBGS, DCOILWTICO, BAMLC0A0CM, BAMLH0A0HYM2,
               UMCSENT, T10YIE)
  - yfinance (fallback: Stooq): ETF OHLCV for TLT, TBT, VNQ, SLV, GLD,
               SPY, AGG

GitLab storage:
  - data/etf_data.csv         — full feature dataset
  - signals/signals.csv       — daily predictions
  - models/                   — serialised model artefacts
  - meta/feature_list.json    — saved feature names

Author: P2SAMAPA
"""

import os
import io
import json
import time
import base64
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GITLAB_API_BASE  = "https://gitlab.com/api/v4"
GITLAB_REPO_URL  = os.getenv("GITLAB_REPO_URL",
                              "https://gitlab.com/P2SAMAPA/p2-etf-regime-predictor")
GITLAB_API_TOKEN = os.getenv("GITLAB_API_TOKEN", "")
FRED_API_KEY     = os.getenv("FRED_API_KEY", "")

# GitLab project ID — URL-encoded namespace/project
GITLAB_PROJECT   = "P2SAMAPA%2Fp2-etf-regime-predictor"

TARGET_ETFS      = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
BENCHMARK_ETFS   = ["SPY", "AGG"]
ALL_TICKERS      = TARGET_ETFS + BENCHMARK_ETFS

FRED_SERIES = {
    "DGS10":        "10Y Treasury Yield",
    "T10Y2Y":      "10Y-2Y Yield Spread",
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

# ── GitLab helpers ───────────────────────────────────────────────────────────

def _gitlab_headers() -> dict:
    return {"PRIVATE-TOKEN": GITLAB_API_TOKEN}


def gitlab_read_file(path: str) -> Optional[str]:
    """Read a file from GitLab repo. Returns raw content string or None."""
    encoded_path = requests.utils.quote(path, safe="")
    url = f"{GITLAB_API_BASE}/projects/{GITLAB_PROJECT}/repository/files/{encoded_path}/raw"
    try:
        r = requests.get(url, headers=_gitlab_headers(),
                         params={"ref": "main"}, timeout=30)
        if r.status_code == 200:
            return r.text
        log.warning(f"GitLab read {path} → {r.status_code}")
        return None
    except Exception as e:
        log.error(f"GitLab read error {path}: {e}")
        return None


def gitlab_write_file(path: str, content: str, commit_msg: str) -> bool:
    """Write/update a text file in GitLab repo. Returns True on success."""
    encoded_path = requests.utils.quote(path, safe="")
    url     = f"{GITLAB_API_BASE}/projects/{GITLAB_PROJECT}/repository/files/{encoded_path}"
    payload = {
        "branch":         "main",
        "content":        content,
        "commit_message": commit_msg,
        "encoding":       "text",
    }
    # Try update (PUT) first, then create (POST)
    r = requests.put(url, headers=_gitlab_headers(), json=payload, timeout=30)
    if r.status_code in (200, 201):
        log.info(f"GitLab updated: {path}")
        return True
    r = requests.post(url, headers=_gitlab_headers(), json=payload, timeout=30)
    if r.status_code in (200, 201):
        log.info(f"GitLab created: {path}")
        return True
    log.error(f"GitLab write {path} → {r.status_code}: {r.text[:200]}")
    return False


def gitlab_write_binary(path: str, data: bytes, commit_msg: str) -> bool:
    """Write binary file (e.g. pickle) to GitLab using base64 encoding."""
    encoded_path = requests.utils.quote(path, safe="")
    url     = f"{GITLAB_API_BASE}/projects/{GITLAB_PROJECT}/repository/files/{encoded_path}"
    payload = {
        "branch":         "main",
        "content":        base64.b64encode(data).decode(),
        "commit_message": commit_msg,
        "encoding":       "base64",
    }
    r = requests.put(url, headers=_gitlab_headers(), json=payload, timeout=30)
    if r.status_code in (200, 201):
        return True
    r = requests.post(url, headers=_gitlab_headers(), json=payload, timeout=30)
    return r.status_code in (200, 201)


def gitlab_read_binary(path: str) -> Optional[bytes]:
    """Read a binary file from GitLab. Returns raw bytes or None."""
    encoded_path = requests.utils.quote(path, safe="")
    url = f"{GITLAB_API_BASE}/projects/{GITLAB_PROJECT}/repository/files/{encoded_path}"
    try:
        r = requests.get(url, headers=_gitlab_headers(),
                         params={"ref": "main"}, timeout=30)
        if r.status_code == 200:
            return base64.b64decode(r.json().get("content", ""))
        return None
    except Exception as e:
        log.error(f"GitLab binary read error {path}: {e}")
        return None


# ── FRED data fetching ───────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str = "2005-01-01") -> pd.Series:
    """Fetch a single FRED series. Returns a named Series indexed by date."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":          series_id,
        "api_key":            FRED_API_KEY,
        "file_type":          "json",
        "observation_start":  start,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        obs  = r.json().get("observations", [])
        data = {
            pd.Timestamp(o["date"]): float(o["value"])
            for o in obs if o["value"] != "."
        }
        s = pd.Series(data, name=series_id)
        log.info(f"FRED {series_id}: {len(s)} observations")
        return s
    except Exception as e:
        log.error(f"FRED fetch error {series_id}: {e}")
        return pd.Series(name=series_id, dtype=float)


def fetch_all_fred(start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch all FRED series and return as aligned DataFrame."""
    frames = []
    for sid in FRED_SERIES:
        s = fetch_fred_series(sid, start=start)
        if not s.empty:
            frames.append(s)
        time.sleep(0.3)   # respectful pacing
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill()
    return df


# ── yfinance / Stooq ETF fetching ────────────────────────────────────────────

def fetch_yfinance(tickers: list, start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch OHLCV from yfinance for multiple tickers."""
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            raise ValueError("yfinance returned empty DataFrame")
        return raw
    except Exception as e:
        log.error(f"yfinance error: {e}")
        return pd.DataFrame()


def fetch_stooq_fallback(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:
    """Fallback single-ticker fetch from Stooq."""
    stooq_map = {t: f"{t}.US" for t in ALL_TICKERS}
    stooq_ticker = stooq_map.get(ticker, f"{ticker}.US")
    url = (f"https://stooq.com/q/d/l/?s={stooq_ticker}"
           f"&d1={start.replace('-','')}&i=d")
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True).sort_index()
        log.info(f"Stooq {ticker}: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Stooq error {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_etfs(start: str = "2005-01-01") -> pd.DataFrame:
    """
    Fetch OHLCV for all ETFs. Returns flat DataFrame with columns like
    TLT_Open, TLT_High, TLT_Low, TLT_Close, TLT_Volume, etc.
    """
    raw    = fetch_yfinance(ALL_TICKERS, start=start)
    result = pd.DataFrame()

    for ticker in ALL_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df_t = raw[[c for c in raw.columns
                             if c in ["Open","High","Low","Close","Volume"]]].copy()
            if df_t.empty:
                raise ValueError(f"Empty for {ticker}")
            for col in df_t.columns:
                result[f"{ticker}_{col}"] = df_t[col]
        except Exception as e:
            log.warning(f"yfinance failed {ticker}, trying Stooq: {e}")
            df_fb = fetch_stooq_fallback(ticker, start=start)
            if not df_fb.empty:
                for col in df_fb.columns:
                    result[f"{ticker}_{col}"] = df_fb[col]

    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Feature engineering ──────────────────────────────────────────────────────

def compute_etf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived ETF features from OHLCV:
    daily return, realised vol (10d/21d), momentum (5d/21d/63d),
    volume ratio, ATR, relative strength vs SPY.
    """
    new_cols = {}

    for ticker in ALL_TICKERS:
        close_col = f"{ticker}_Close"
        high_col  = f"{ticker}_High"
        low_col   = f"{ticker}_Low"
        vol_col   = f"{ticker}_Volume"

        if close_col not in df.columns:
            continue

        close = df[close_col]
        ret   = close.pct_change()

        new_cols[f"{ticker}_Ret"]      = ret
        new_cols[f"{ticker}_RVol10d"]  = ret.rolling(10).std() * np.sqrt(252)
        new_cols[f"{ticker}_RVol21d"]  = ret.rolling(21).std() * np.sqrt(252)
        new_cols[f"{ticker}_Mom5d"]    = close.pct_change(5)
        new_cols[f"{ticker}_Mom21d"]   = close.pct_change(21)
        new_cols[f"{ticker}_Mom63d"]   = close.pct_change(63)

        if vol_col in df.columns:
            vol_ma = df[vol_col].rolling(20).mean()
            new_cols[f"{ticker}_VolRatio"] = df[vol_col] / (vol_ma + 1e-9)

        if high_col in df.columns and low_col in df.columns:
            prev_close = close.shift(1)
            tr = pd.concat([
                df[high_col] - df[low_col],
                (df[high_col] - prev_close).abs(),
                (df[low_col]  - prev_close).abs(),
            ], axis=1).max(axis=1)
            new_cols[f"{ticker}_ATR14"] = tr.rolling(14).mean() / (close + 1e-9)

    # Relative strength vs SPY (21d rolling return differential)
    if "SPY_Ret" in new_cols:
        spy_ret = new_cols["SPY_Ret"]
        for ticker in TARGET_ETFS:
            rk = f"{ticker}_Ret"
            if rk in new_cols:
                new_cols[f"{ticker}_RelSPY21d"] = (
                    new_cols[rk].rolling(21).mean() -
                    spy_ret.rolling(21).mean()
                )

    # Rate-of-Change momentum features (for Layer 2B momentum ranker)
    # Raw RoC — no volatility scaling per user preference
    for ticker in TARGET_ETFS:
        ret_key = f"{ticker}_Ret"
        if ret_key not in new_cols:
            continue
        price_proxy = (1 + new_cols[ret_key]).cumprod()
        # Rate of change: today vs N days ago
        for n in [5, 10, 21, 63]:
            new_cols[f"{ticker}_RoC{n}d"] = price_proxy.pct_change(n)
        # Volume accumulation: OBV-like — sum of signed volume
        vol_key = f"{ticker}_Volume"
        if vol_key in df.columns:
            signed_vol = df[vol_key] * np.sign(new_cols[ret_key].fillna(0))
            new_cols[f"{ticker}_OBV10d"] = signed_vol.rolling(10).sum()
            new_cols[f"{ticker}_OBV21d"] = signed_vol.rolling(21).sum()
        # Breakout: price vs 20d high (1 = at high, 0 = at low)
        rolling_high = price_proxy.rolling(20).max()
        rolling_low  = price_proxy.rolling(20).min()
        rng = rolling_high - rolling_low
        new_cols[f"{ticker}_Breakout20d"] = (
            (price_proxy - rolling_low) / (rng + 1e-9)
        )

    # Breakout features — 1d and 3d return vs 21d realised vol
    # Critical for catching explosive precious metals / commodity moves
    for ticker in TARGET_ETFS:
        ret_key  = f"{ticker}_Ret"
        rvol_key = f"{ticker}_RVol21d"
        if ret_key in new_cols and rvol_key in new_cols:
            daily_vol = new_cols[rvol_key] / np.sqrt(252)
            ret_1d    = new_cols[ret_key]
            ret_3d    = new_cols[ret_key].rolling(3).sum()
            # Z-score of 1d and 3d return vs recent vol — spikes signal breakouts
            new_cols[f"{ticker}_RetZ1d"] = ret_1d  / (daily_vol + 1e-9)
            new_cols[f"{ticker}_RetZ3d"] = ret_3d  / (daily_vol * np.sqrt(3) + 1e-9)
            # Acceleration: 3d momentum vs 21d momentum
            mom3  = new_cols[ret_key].rolling(3).mean()
            mom21 = new_cols[ret_key].rolling(21).mean()
            new_cols[f"{ticker}_MomAccel"] = mom3 - mom21

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived macro features:
    real yield, rate momentum/flags, yield curve shape,
    inflation regime, VIX regime, credit stress, Z-scores.
    """
    new_cols = {}

    # Real yield
    if "DGS10" in df.columns and "T10YIE" in df.columns:
        new_cols["Real_Yield_10Y"] = df["DGS10"] - df["T10YIE"]

    # Rate momentum + rising/falling flags
    for series in ["DGS10", "T10Y2Y", "T10Y3M", "T10YIE"]:
        if series in df.columns:
            mom20 = df[series].diff(20)
            mom60 = df[series].diff(60)
            new_cols[f"{series}_Mom20d"]    = mom20
            new_cols[f"{series}_Mom60d"]    = mom60
            new_cols[f"{series}_Rising20d"] = (mom20 > 0).astype(int)
            new_cols[f"{series}_Rising60d"] = (mom60 > 0).astype(int)

    # Rate acceleration
    if "DGS10" in df.columns:
        new_cols["DGS10_Accel"] = df["DGS10"].diff(20).diff(20)

    # Yield curve shape
    if "T10Y2Y" in df.columns:
        new_cols["YC_Inverted"] = (df["T10Y2Y"] < 0).astype(int)
        new_cols["YC_Flat"]     = ((df["T10Y2Y"] >= 0) &
                                   (df["T10Y2Y"] < 0.5)).astype(int)
        new_cols["YC_Steep"]    = (df["T10Y2Y"] >= 0.5).astype(int)

    # Inflation regime
    if "T10YIE" in df.columns:
        new_cols["Inflation_High"] = (df["T10YIE"] > 2.5).astype(int)
        new_cols["Inflation_Low"]  = (df["T10YIE"] < 1.5).astype(int)

    # VIX regime
    if "VIXCLS" in df.columns:
        new_cols["VIX_Low"]  = (df["VIXCLS"] < 15).astype(int)
        new_cols["VIX_Med"]  = ((df["VIXCLS"] >= 15) &
                                (df["VIXCLS"] < 25)).astype(int)
        new_cols["VIX_High"] = (df["VIXCLS"] >= 25).astype(int)

    # Credit stress
    if "BAMLH0A0HYM2" in df.columns:
        new_cols["HY_Stress_High"] = (df["BAMLH0A0HYM2"] > 600).astype(int)
        new_cols["HY_Stress_Med"]  = ((df["BAMLH0A0HYM2"] >= 400) &
                                      (df["BAMLH0A0HYM2"] < 600)).astype(int)

    # Combine
    combined = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Credit spread features — LQD/HYG relative to TLT (regime signals)
    if "LQD_Ret" in new_cols and "HYG_Ret" in new_cols:
        # HY-IG spread proxy: HYG return minus LQD return
        new_cols["HYG_vs_LQD_5d"]  = (new_cols["HYG_Ret"].rolling(5).mean() -
                                       new_cols["LQD_Ret"].rolling(5).mean())
        new_cols["HYG_vs_LQD_21d"] = (new_cols["HYG_Ret"].rolling(21).mean() -
                                       new_cols["LQD_Ret"].rolling(21).mean())
    if "LQD_Ret" in new_cols and "TLT_Ret" in new_cols:
        # IG spread vs rates: LQD minus TLT (credit risk premium signal)
        new_cols["LQD_vs_TLT_5d"]  = (new_cols["LQD_Ret"].rolling(5).mean() -
                                       new_cols["TLT_Ret"].rolling(5).mean())
        new_cols["LQD_vs_TLT_21d"] = (new_cols["LQD_Ret"].rolling(21).mean() -
                                       new_cols["TLT_Ret"].rolling(21).mean())
    if "HYG_Ret" in new_cols and "TLT_Ret" in new_cols:
        # HY spread vs rates: risk-on/off signal
        new_cols["HYG_vs_TLT_5d"]  = (new_cols["HYG_Ret"].rolling(5).mean() -
                                       new_cols["TLT_Ret"].rolling(5).mean())
        new_cols["HYG_vs_TLT_21d"] = (new_cols["HYG_Ret"].rolling(21).mean() -
                                       new_cols["TLT_Ret"].rolling(21).mean())

    # USD daily change — key leading indicator for SLV/GLD moves
    if "DTWEXBGS" in df.columns:
        new_cols["USD_Ret1d"]  = df["DTWEXBGS"].pct_change(1)
        new_cols["USD_Ret5d"]  = df["DTWEXBGS"].pct_change(5)
        new_cols["USD_Ret21d"] = df["DTWEXBGS"].pct_change(21)

    # Cross-ETF relative momentum features
    # Gives LambdaRank direct signals about which ETF is currently winning
    etf_pairs = [
        ("SLV", "VNQ"), ("SLV", "TLT"), ("SLV", "LQD"), ("SLV", "HYG"),
        ("GLD", "VNQ"), ("GLD", "LQD"), ("GLD", "HYG"),
        ("LQD", "HYG"), ("LQD", "TLT"), ("HYG", "TLT"),
    ]
    for e1, e2 in etf_pairs:
        r1_5  = new_cols.get(f"{e1}_Ret", pd.Series(dtype=float)).rolling(5).mean()
        r2_5  = new_cols.get(f"{e2}_Ret", pd.Series(dtype=float)).rolling(5).mean()
        r1_21 = new_cols.get(f"{e1}_Ret", pd.Series(dtype=float)).rolling(21).mean()
        r2_21 = new_cols.get(f"{e2}_Ret", pd.Series(dtype=float)).rolling(21).mean()
        if not r1_5.empty and not r2_5.empty:
            new_cols[f"{e1}_vs_{e2}_5d"]  = r1_5  - r2_5
            new_cols[f"{e1}_vs_{e2}_21d"] = r1_21 - r2_21

    # Rolling Z-scores for all macro series + derived
    zscore_targets = list(FRED_SERIES.keys()) + list(new_cols.keys())
    z_new = {}
    for col in zscore_targets:
        if col in combined.columns:
            mu  = combined[col].rolling(60, min_periods=20).mean()
            std = combined[col].rolling(60, min_periods=20).std()
            z_new[f"{col}_Z"] = (combined[col] - mu) / (std + 1e-9)

    return pd.concat([combined,
                      pd.DataFrame(z_new, index=combined.index)], axis=1)


def build_full_dataset(start_year: int = 2008) -> pd.DataFrame:
    """
    Full pipeline: fetch FRED + ETF data, engineer features, return clean df.
    Called by train.py on first run or force refresh.
    """
    start = f"{start_year}-01-01"
    log.info(f"Building full dataset from {start}...")

    macro_df = fetch_all_fred(start=start)
    etf_df   = fetch_all_etfs(start=start)

    if macro_df.empty or etf_df.empty:
        raise ValueError("Failed to fetch data — check API keys and network")

    # Align macro to ETF trading days
    macro_df = macro_df.reindex(etf_df.index, method="ffill")
    df = pd.concat([etf_df, macro_df], axis=1).sort_index()

    df = compute_etf_features(df)
    df = compute_macro_features(df)

    # Filter to start year (allow warmup period before)
    df = df[df.index >= pd.Timestamp(start)]

    # Drop rows where core ETF closes are missing
    core_cols = [f"{t}_Close" for t in TARGET_ETFS]
    df = df.dropna(subset=core_cols)
    df = df.ffill().dropna(how="all", axis=1)

    log.info(f"Dataset: {len(df)} rows × {df.shape[1]} cols "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── Forward return targets ───────────────────────────────────────────────────

def build_forward_targets(df: pd.DataFrame,
                           forward_days: int = 5,
                           rf_rate_col: str = "DTB3") -> pd.DataFrame:
    # Build forward return targets at multiple horizons (5, 10, 15, 20 days).
    # Stores raw forward returns at each horizon per ETF.
    # The ranking model selects optimal horizon per (ETF, regime).
    fwd = pd.DataFrame(index=df.index)
    daily_rf = (df[rf_rate_col] / 100 / 252
                if rf_rate_col in df.columns
                else pd.Series(0.045 / 252, index=df.index))

    for ticker in TARGET_ETFS:
        ret_col = f"{ticker}_Ret"
        if ret_col not in df.columns:
            continue
        for h in [5, 10, 15, 20]:
            fwd_ret = df[ret_col].rolling(h).sum().shift(-h)
            rf_thr  = daily_rf * h
            fwd[f"{ticker}_FwdRet{h}d"]   = fwd_ret
            fwd[f"{ticker}_BeatCash{h}d"] = (fwd_ret > rf_thr).astype(int)
        # Default 5d for backward compatibility
        fwd[f"{ticker}_FwdRet"]   = fwd[f"{ticker}_FwdRet5d"]
        fwd[f"{ticker}_BeatCash"] = fwd[f"{ticker}_BeatCash5d"]

    return fwd
# ── Incremental update ───────────────────────────────────────────────────────

def incremental_update(existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch only data since last date in existing_df,
    append and re-engineer features for the new tail.
    """
    last_date = existing_df.index[-1]
    start     = (last_date - timedelta(days=90)).strftime("%Y-%m-%d")
    log.info(f"Incremental update from {start}...")

    macro_df = fetch_all_fred(start=start)
    etf_df   = fetch_all_etfs(start=start)

    if macro_df.empty or etf_df.empty:
        log.warning("Incremental fetch empty — returning existing dataset")
        return existing_df

    macro_df = macro_df.reindex(etf_df.index, method="ffill")
    new_df   = pd.concat([etf_df, macro_df], axis=1).sort_index()
    new_df   = compute_etf_features(new_df)
    new_df   = compute_macro_features(new_df)

    new_rows = new_df[new_df.index > last_date]
    if new_rows.empty:
        log.info("No new rows — dataset is current")
        return existing_df

    combined = (pd.concat([existing_df, new_rows])
                .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                .sort_index())
    log.info(f"Appended {len(new_rows)} rows. Total: {len(combined)}")
    return combined


# ── GitLab storage ───────────────────────────────────────────────────────────

def load_dataset_from_gitlab() -> Optional[pd.DataFrame]:
    """Load etf_data.csv from GitLab."""
    content = gitlab_read_file("data/etf_data.csv")
    if content is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(content), index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        log.info(f"Loaded from GitLab: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Parse error: {e}")
        return None


def save_dataset_to_gitlab(df: pd.DataFrame) -> bool:
    return gitlab_write_file(
        "data/etf_data.csv", df.to_csv(),
        f"Update dataset {df.index[-1].date()} ({len(df)} rows)"
    )


def load_signals_from_gitlab() -> Optional[pd.DataFrame]:
    content = gitlab_read_file("signals/signals.csv")
    if content is None:
        return None
    try:
        return pd.read_csv(io.StringIO(content), index_col=0, parse_dates=True)
    except Exception as e:
        log.error(f"Signals parse error: {e}")
        return None


def save_signals_to_gitlab(signals_df: pd.DataFrame) -> bool:
    existing = load_signals_from_gitlab()
    if existing is not None:
        combined = (pd.concat([existing, signals_df])
                    .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                    .sort_index())
    else:
        combined = signals_df
    return gitlab_write_file(
        "signals/signals.csv", combined.to_csv(),
        f"Update signals {combined.index[-1].date()}"
    )


def save_model_to_gitlab(model_bytes: bytes, filename: str) -> bool:
    return gitlab_write_binary(
        f"models/{filename}", model_bytes, f"Update model: {filename}"
    )


def load_model_from_gitlab(filename: str) -> Optional[bytes]:
    return gitlab_read_binary(f"models/{filename}")


def save_predictions_to_gitlab(pred_df: pd.DataFrame,
                                path: str = "data/pred_history.csv") -> bool:
    return gitlab_write_file(
        path, pred_df.to_csv(),
        f"Update predictions {pred_df.index[-1].date()} ({len(pred_df)} rows)"
    )


def load_predictions_from_gitlab(path: str = "data/pred_history.csv") -> Optional[pd.DataFrame]:
    content = gitlab_read_file(path)
    if content is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(content), index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        log.info(f"Loaded predictions from GitLab: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Predictions parse error: {e}")
        return None


def save_feature_list_to_gitlab(feature_names: list) -> bool:
    content = json.dumps({"features":  feature_names,
                          "updated":   datetime.utcnow().isoformat()})
    return gitlab_write_file("meta/feature_list.json", content,
                              "Update feature list")


def load_momentum_ranker_from_gitlab() -> Optional[bytes]:
    return load_model_from_gitlab("momentum_ranker.pkl")


def load_momentum_predictions_from_gitlab() -> Optional[pd.DataFrame]:
    return load_predictions_from_gitlab("data/mom_pred_history.csv")


def load_feature_list_from_gitlab() -> Optional[list]:
    content = gitlab_read_file("meta/feature_list.json")
    if content is None:
        return None
    try:
        return json.loads(content)["features"]
    except Exception:
        return None


# ── Main entry point for Streamlit ───────────────────────────────────────────

def get_data(start_year: int = 2008,
             force_refresh: bool = False) -> pd.DataFrame:
    """
    Streamlit entry point.
    Loads from GitLab if available; full rebuild on force_refresh or first run.
    """
    if not force_refresh:
        df = load_dataset_from_gitlab()
        if df is not None:
            df = df[df.index >= pd.Timestamp(f"{start_year}-01-01")]
            log.info(f"Using GitLab dataset: {len(df)} rows from {start_year}")
            return df

    log.info("Full dataset rebuild...")
    df = build_full_dataset(start_year=start_year)
    save_dataset_to_gitlab(df)
    return df
