"""
data_manager_hf.py — P2-ETF-REGIME-PREDICTOR (HF Dataset Version)
==========================================================
Handles all data fetching, feature engineering, and Hugging Face Dataset storage.

Data sources:
  - FRED API: macro signals (DGS10, T10Y2Y, T10Y3M, DTB3, MORTGAGE30US,
               VIXCLS, DTWEXBGS, DCOILWTICO, BAMLC0A0CM, BAMLH0A0HYM2,
               UMCSENT, T10YIE)
  - yfinance: ETF OHLCV for TLT, VNQ, SLV, GLD, LQD, HYG, SPY, AGG

Hugging Face Dataset storage (replaces GitLab):
  - data/etf_data.parquet         — full feature dataset
  - data/mom_pred_history.parquet — in-sample momentum predictions
  - data/wf_mom_pred_history.parquet — walk-forward OOS predictions
  - signals/signals.parquet       — daily predictions
  - models/                       — serialised model artefacts
  - meta/feature_list.json        — saved feature names
  - sweep/                        — multi-year consensus results

Author: P2SAMAPA (HF Migration)
"""

import os
import io
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO

import yfinance as yf
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO_ID = os.getenv("HF_DATASET_REPO", os.getenv("HF_REPO_ID", "P2SAMAPA/p2-etf-regime-predictor"))
HF_TOKEN         = os.getenv("HF_TOKEN", "")
FRED_API_KEY     = os.getenv("FRED_API_KEY", "")

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

# ── HF Dataset helpers ───────────────────────────────────────────────────────

def _get_hf_api() -> HfApi:
    """Get authenticated HF API client."""
    token = HF_TOKEN or os.getenv("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set — public read-only access")
    return HfApi(token=token)


def hf_read_file(path: str, repo_type: str = "dataset") -> Optional[str]:
    """Read a text file from HF Dataset. Returns content string or None."""
    try:
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=path,
            repo_type=repo_type,
            token=HF_TOKEN or None,  # None for public repos
        )
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        log.warning(f"HF read {path} failed: {e}")
        return None


def hf_write_file(path: str, content: str, commit_msg: str) -> bool:
    """Write/update a text file in HF Dataset."""
    try:
        api = _get_hf_api()
        api.upload_file(
            path_or_fileobj=content.encode('utf-8'),
            path_in_repo=path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        log.info(f"HF updated: {path}")
        return True
    except Exception as e:
        log.error(f"HF write {path} failed: {e}")
        return False


def hf_write_binary(path: str, data: bytes, commit_msg: str) -> bool:
    """Write binary file (e.g. pickle) to HF Dataset."""
    try:
        api = _get_hf_api()
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        log.info(f"HF binary updated: {path}")
        return True
    except Exception as e:
        log.error(f"HF binary write {path} failed: {e}")
        return False


def hf_read_binary(path: str) -> Optional[bytes]:
    """Read a binary file from HF Dataset. Returns raw bytes or None."""
    try:
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=path,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        log.warning(f"HF binary read {path} failed: {e}")
        return None


def hf_read_parquet(path: str) -> Optional[pd.DataFrame]:
    """Read a parquet file from HF Dataset."""
    try:
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=path,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        df = pd.read_parquet(file_path)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date':
            df.index = pd.to_datetime(df.index)
        log.info(f"Loaded from HF: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"HF parquet read {path} failed: {e}")
        return None


def hf_write_parquet(path: str, df: pd.DataFrame, commit_msg: str) -> bool:
    """Write DataFrame as parquet to HF Dataset."""
    try:
        buffer = BytesIO()
        # Ensure index is preserved
        df_out = df.copy()
        if df_out.index.name is None:
            df_out.index.name = 'Date'
        df_out.to_parquet(buffer, index=True)
        buffer.seek(0)

        api = _get_hf_api()
        api.upload_file(
            path_or_fileobj=buffer.getvalue(),
            path_in_repo=path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        log.info(f"HF parquet updated: {path} ({len(df)} rows)")
        return True
    except Exception as e:
        log.error(f"HF parquet write {path} failed: {e}")
        return False


def hf_list_files(path: str = "", recursive: bool = False) -> list:
    """List files in HF Dataset repository."""
    try:
        files = list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        if path:
            files = [f for f in files if f.startswith(path)]
        return files
    except Exception as e:
        log.warning(f"HF list files failed: {e}")
        return []


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


# ── yfinance ETF fetching ────────────────────────────────────────────────────

def fetch_yfinance(tickers: list, start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch OHLCV from yfinance for multiple tickers."""
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True,
                          progress=False, threads=False)
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

    # Rate-of-Change momentum features
    for ticker in TARGET_ETFS:
        ret_key = f"{ticker}_Ret"
        if ret_key not in new_cols:
            continue
        price_proxy = (1 + new_cols[ret_key]).cumprod()
        for n in [5, 10, 21, 63]:
            new_cols[f"{ticker}_RoC{n}d"] = price_proxy.pct_change(n)
        vol_key = f"{ticker}_Volume"
        if vol_key in df.columns:
            signed_vol = df[vol_key] * np.sign(new_cols[ret_key].fillna(0))
            new_cols[f"{ticker}_OBV10d"] = signed_vol.rolling(10).sum()
            new_cols[f"{ticker}_OBV21d"] = signed_vol.rolling(21).sum()
        rolling_high = price_proxy.rolling(20).max()
        rolling_low  = price_proxy.rolling(20).min()
        rng = rolling_high - rolling_low
        new_cols[f"{ticker}_Breakout20d"] = (
            (price_proxy - rolling_low) / (rng + 1e-9)
        )

    # Breakout features — 1d and 3d return vs 21d realised vol
    for ticker in TARGET_ETFS:
        ret_key  = f"{ticker}_Ret"
        rvol_key = f"{ticker}_RVol21d"
        if ret_key in new_cols and rvol_key in new_cols:
            daily_vol = new_cols[rvol_key] / np.sqrt(252)
            ret_1d    = new_cols[ret_key]
            ret_3d    = new_cols[ret_key].rolling(3).sum()
            new_cols[f"{ticker}_RetZ1d"] = ret_1d  / (daily_vol + 1e-9)
            new_cols[f"{ticker}_RetZ3d"] = ret_3d  / (daily_vol * np.sqrt(3) + 1e-9)
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

    # Credit spread features
    if "LQD_Ret" in new_cols and "HYG_Ret" in new_cols:
        new_cols["HYG_vs_LQD_5d"]  = (new_cols["HYG_Ret"].rolling(5).mean() -
                                       new_cols["LQD_Ret"].rolling(5).mean())
        new_cols["HYG_vs_LQD_21d"] = (new_cols["HYG_Ret"].rolling(21).mean() -
                                       new_cols["LQD_Ret"].rolling(21).mean())
    if "LQD_Ret" in new_cols and "TLT_Ret" in new_cols:
        new_cols["LQD_vs_TLT_5d"]  = (new_cols["LQD_Ret"].rolling(5).mean() -
                                       new_cols["TLT_Ret"].rolling(5).mean())
        new_cols["LQD_vs_TLT_21d"] = (new_cols["LQD_Ret"].rolling(21).mean() -
                                       new_cols["TLT_Ret"].rolling(21).mean())
    if "HYG_Ret" in new_cols and "TLT_Ret" in new_cols:
        new_cols["HYG_vs_TLT_5d"]  = (new_cols["HYG_Ret"].rolling(5).mean() -
                                       new_cols["TLT_Ret"].rolling(5).mean())
        new_cols["HYG_vs_TLT_21d"] = (new_cols["HYG_Ret"].rolling(21).mean() -
                                       new_cols["TLT_Ret"].rolling(21).mean())

    # USD daily change
    if "DTWEXBGS" in df.columns:
        new_cols["USD_Ret1d"]  = df["DTWEXBGS"].pct_change(1)
        new_cols["USD_Ret5d"]  = df["DTWEXBGS"].pct_change(5)
        new_cols["USD_Ret21d"] = df["DTWEXBGS"].pct_change(21)

    # Cross-ETF relative momentum features
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


# ── HF Dataset storage ───────────────────────────────────────────────────────────

def load_dataset_from_hf() -> Optional[pd.DataFrame]:
    """Load etf_data.parquet from HF Dataset."""
    return hf_read_parquet("data/etf_data.parquet")


def save_dataset_to_hf(df: pd.DataFrame) -> bool:
    return hf_write_parquet(
        "data/etf_data.parquet", df,
        f"Update dataset {df.index[-1].date()} ({len(df)} rows)"
    )


def load_signals_from_hf() -> Optional[pd.DataFrame]:
    return hf_read_parquet("signals/signals.parquet")


def save_signals_to_hf(signals_df: pd.DataFrame) -> bool:
    existing = load_signals_from_hf()
    if existing is not None:
        combined = (pd.concat([existing, signals_df])
                    .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                    .sort_index())
    else:
        combined = signals_df
    return hf_write_parquet(
        "signals/signals.parquet", combined,
        f"Update signals {combined.index[-1].date()}"
    )


def save_model_to_hf(model_bytes: bytes, filename: str) -> bool:
    return hf_write_binary(
        f"models/{filename}", model_bytes, f"Update model: {filename}"
    )


def load_model_from_hf(filename: str) -> Optional[bytes]:
    return hf_read_binary(f"models/{filename}")


def save_predictions_to_hf(pred_df: pd.DataFrame,
                              path: str = "data/mom_pred_history.parquet") -> bool:
    return hf_write_parquet(
        path, pred_df,
        f"Update predictions {pred_df.index[-1].date()} ({len(pred_df)} rows)"
    )


def save_sweep_to_hf(results: dict, start_year: int) -> bool:
    """
    Save sweep result for one start_year as a date-stamped JSON file.
    Filename: sweep/sweep_{start_year}_{YYYYMMDD}.json
    app.py reads these files to build the consensus view.
    """
    from datetime import datetime, timezone, timedelta
    today_est = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")
    path      = f"sweep/sweep_{start_year}_{today_est}.json"
    content_str = json.dumps(results, indent=2, default=str)
    ok = hf_write_file(
        path, content_str,
        f"Sweep result {start_year} — {today_est}"
    )
    log.info(f"  Sweep result saved ({start_year}): {ok} → {path}")
    return ok


def load_predictions_from_hf(path: str = "data/mom_pred_history.parquet") -> Optional[pd.DataFrame]:
    return hf_read_parquet(path)


def load_wf_predictions_from_hf() -> Optional[pd.DataFrame]:
    return hf_read_parquet("data/wf_mom_pred_history.parquet")


def save_feature_list_to_hf(feature_names: list) -> bool:
    content = json.dumps({"features":  feature_names,
                          "updated":   datetime.utcnow().isoformat()})
    return hf_write_file("meta/feature_list.json", content,
                              "Update feature list")


def load_feature_list_from_hf() -> Optional[list]:
    content = hf_read_file("meta/feature_list.json")
    if content is None:
        return None
    try:
        return json.loads(content)["features"]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# NEW HF-SPECIFIC FUNCTIONS (for app.py imports)
# ═══════════════════════════════════════════════════════════════════════════════

def load_momentum_ranker_from_hf() -> Optional[bytes]:
    """Load momentum ranker model from HF Dataset."""
    return load_model_from_hf("momentum_ranker.pkl")


def load_momentum_predictions_from_hf() -> Optional[pd.DataFrame]:
    """Load in-sample momentum predictions from HF Dataset."""
    return load_predictions_from_hf("data/mom_pred_history.parquet")


def load_wf_momentum_predictions_from_hf() -> Optional[pd.DataFrame]:
    """Load walk-forward momentum predictions from HF Dataset."""
    return load_wf_predictions_from_hf()


def load_wf_ensemble_predictions_from_hf() -> Optional[pd.DataFrame]:
    """Load walk-forward ensemble predictions from HF Dataset."""
    return load_predictions_from_hf("data/wf_pred_history.parquet")


# ── Main entry point for Streamlit ───────────────────────────────────────────

def get_data(start_year: int = 2008,
             force_refresh: bool = False) -> pd.DataFrame:
    """
    Streamlit entry point.
    Loads from HF Dataset if available; full rebuild on force_refresh or first run.
    """
    if not force_refresh:
        df = load_dataset_from_hf()
        if df is not None:
            df = df[df.index >= pd.Timestamp(f"{start_year}-01-01")]
            log.info(f"Using HF dataset: {len(df)} rows from {start_year}")
            return df

    log.info("Full dataset rebuild...")
    df = build_full_dataset(start_year=start_year)
    save_dataset_to_hf(df)
    return df


# ── Backward compatibility aliases ───────────────────────────────────────────
# These maintain API compatibility with old GitLab-based code

load_dataset_from_gitlab = load_dataset_from_hf
save_dataset_to_gitlab = save_dataset_to_hf
load_signals_from_gitlab = load_signals_from_hf
save_signals_to_gitlab = save_signals_to_hf
load_model_from_gitlab = load_model_from_hf
save_model_to_gitlab = save_model_to_hf
load_predictions_from_gitlab = load_predictions_from_hf
save_predictions_to_gitlab = save_predictions_to_hf
save_sweep_to_gitlab = save_sweep_to_hf
load_feature_list_from_gitlab = load_feature_list_from_hf
save_feature_list_to_gitlab = save_feature_list_to_hf
load_momentum_ranker_from_gitlab = load_momentum_ranker_from_hf
load_momentum_predictions_from_gitlab = load_momentum_predictions_from_hf
load_wf_momentum_predictions_from_gitlab = load_wf_momentum_predictions_from_hf
load_wf_ensemble_predictions_from_gitlab = load_wf_ensemble_predictions_from_hf
