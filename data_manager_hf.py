data_manager_hf.py — P2-ETF-REGIME-PREDICTOR v2 (CORRECTED)
=================================================
Handles all data fetching, feature engineering, and Hugging Face Dataset
storage for both Option A (FI/Commodities) and Option B (Equity ETFs).

Fixes in this version:
- Added custom JSON encoder to handle NaN/Inf values
- Fixed JSON serialization issues with numpy types

Data sources:
 - FRED API: macro signals (shared across both options)
 - yfinance: ETF OHLCV (fetched once for all tickers combined)

HF Dataset storage (namespaced by option):
 option_a/etf_data.parquet — Option A feature dataset
 option_a/mom_pred_history.parquet — Option A in-sample predictions
 option_a/wf_mom_pred_history.parquet — Option A walk-forward predictions
 option_a/signals.parquet — Option A daily signals
 option_a/models/ — Option A model artefacts
 option_a/meta/feature_list.json — Option A feature names
 option_a/sweep/ — Option A consensus sweep results

 option_b/... — Same structure for Option B
"""

import os
import io
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO

import requests
import yfinance as yf
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Runtime constants ─────────────────────────────────────────────────────────

HF_REPO_ID = os.getenv("HF_DATASET_REPO", cfg.HF_DATASET_REPO)
HF_TOKEN = os.getenv("HF_TOKEN", cfg.HF_TOKEN)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Convenience aliases (Option A defaults — used by legacy callers)
TARGET_ETFS = cfg.OPTION_A_ETFS
BENCHMARK_ETFS = cfg.OPTION_A_BENCHMARKS
ALL_TICKERS = cfg.OPTION_A_ALL_TICKERS


# CORRECTED: Custom JSON encoder to handle numpy types and NaN
def _clean_for_json(obj):
    """Recursively clean object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── HF low-level helpers ──────────────────────────────────────────────────────

def _get_hf_api() -> HfApi:
    token = HF_TOKEN or os.getenv("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set — public read-only access")
    return HfApi(token=token)

def hf_read_file(path: str) -> Optional[str]:
    try:
        fp = hf_hub_download(repo_id=HF_REPO_ID, filename=path,
                             repo_type="dataset", token=HF_TOKEN or None)
        with open(fp) as f:
            return f.read()
    except Exception as e:
        log.warning(f"HF read {path} failed: {e}")
        return None

def hf_write_file(path: str, content: str, commit_msg: str) -> bool:
    try:
        _get_hf_api().upload_file(
            path_or_fileobj=content.encode("utf-8"),
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
    try:
        _get_hf_api().upload_file(
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
    try:
        fp = hf_hub_download(repo_id=HF_REPO_ID, filename=path,
                             repo_type="dataset", token=HF_TOKEN or None)
        with open(fp, "rb") as f:
            return f.read()
    except Exception as e:
        log.warning(f"HF binary read {path} failed: {e}")
        return None

def hf_read_parquet(path: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    try:
        fp = hf_hub_download(repo_id=HF_REPO_ID, filename=path,
                             repo_type="dataset", token=HF_TOKEN or None,
                             force_download=force_download)
        df = pd.read_parquet(fp)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        elif df.index.name != "Date":
            df.index = pd.to_datetime(df.index)
        log.info(f"HF loaded: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"HF parquet read {path} failed: {e}")
        return None

def hf_write_parquet(path: str, df: pd.DataFrame, commit_msg: str) -> bool:
    try:
        buf = BytesIO()
        df_out = df.copy()
        if df_out.index.name is None:
            df_out.index.name = "Date"
        df_out.to_parquet(buf, index=True)
        _get_hf_api().upload_file(
            path_or_fileobj=buf.getvalue(),
            path_in_repo=path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        log.info(f"HF parquet saved: {path} ({len(df)} rows)")
        return True
    except Exception as e:
        log.error(f"HF parquet write {path} failed: {e}")
        return False

def hf_list_files(prefix: str = "") -> list:
    try:
        files = list(list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset",
                                     token=HF_TOKEN or None))
        return [f for f in files if f.startswith(prefix)] if prefix else files
    except Exception as e:
        log.warning(f"HF list files failed: {e}")
        return []

# ── FRED fetching ─────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str = "2005-01-01") -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "file_type": "json", "observation_start": start}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = {pd.Timestamp(o["date"]): float(o["value"])
                for o in r.json().get("observations", []) if o["value"] != "."}
        s = pd.Series(data, name=series_id)
        log.info(f"FRED {series_id}: {len(s)} obs")
        return s
    except Exception as e:
        log.error(f"FRED {series_id} error: {e}")
        return pd.Series(name=series_id, dtype=float)

def fetch_all_fred(start: str = "2005-01-01") -> pd.DataFrame:
    frames = []
    for sid in cfg.FRED_SERIES:
        s = fetch_fred_series(sid, start=start)
        if not s.empty:
            frames.append(s)
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill()

# ── yfinance fetching ─────────────────────────────────────────────────────────

def fetch_yfinance(tickers: list, start: str = "2005-01-01") -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True,
                          progress=False, threads=False)
        return raw
    except Exception as e:
        log.error(f"yfinance fetch error: {e}")
        return pd.DataFrame()

def fetch_stooq_fallback(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:
    url = (f"https://stooq.com/q/d/l/?s={ticker}.US"
           f"&d1={start.replace('-', '')}&i=d")
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True).sort_index()
        log.info(f"Stooq {ticker}: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Stooq {ticker} error: {e}")
        return pd.DataFrame()

def fetch_etfs_for_option(tickers: list, start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch OHLCV for a given ticker list, returns flat DataFrame."""
    raw = fetch_yfinance(tickers, start=start)
    result = pd.DataFrame()

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df_t = raw[[c for c in raw.columns
                            if c in ["Open", "High", "Low", "Close", "Volume"]]].copy()
            if df_t.empty:
                raise ValueError(f"Empty for {ticker}")
            for col in df_t.columns:
                result[f"{ticker}_{col}"] = df_t[col]
        except Exception:
            log.warning(f"yfinance failed for {ticker}, trying Stooq fallback")
            df_fb = fetch_stooq_fallback(ticker, start=start)
            if not df_fb.empty:
                for col in df_fb.columns:
                    result[f"{ticker}_{col}"] = df_fb[col]

    result.index = pd.to_datetime(result.index)
    return result.sort_index()

# ── Feature engineering ───────────────────────────────────────────────────────

def compute_etf_features(df: pd.DataFrame, target_etfs: list) -> pd.DataFrame:
    """Compute derived ETF features for the given target ETF list."""
    new_cols = {}
    all_tickers_in_df = [t for t in target_etfs if f"{t}_Close" in df.columns]

    for ticker in all_tickers_in_df:
        close = df[f"{ticker}_Close"]
        ret = close.pct_change()

        new_cols[f"{ticker}_Ret"] = ret
        new_cols[f"{ticker}_RVol10d"] = ret.rolling(10).std() * np.sqrt(252)
        new_cols[f"{ticker}_RVol21d"] = ret.rolling(21).std() * np.sqrt(252)
        new_cols[f"{ticker}_Mom5d"] = close.pct_change(5)
        new_cols[f"{ticker}_Mom21d"] = close.pct_change(21)
        new_cols[f"{ticker}_Mom63d"] = close.pct_change(63)

        vol_col = f"{ticker}_Volume"
        if vol_col in df.columns:
            vol_ma = df[vol_col].rolling(20).mean()
            new_cols[f"{ticker}_VolRatio"] = df[vol_col] / (vol_ma + 1e-9)

        high_col = f"{ticker}_High"
        low_col = f"{ticker}_Low"
        if high_col in df.columns and low_col in df.columns:
            prev_close = close.shift(1)
            tr = pd.concat([
                df[high_col] - df[low_col],
                (df[high_col] - prev_close).abs(),
                (df[low_col] - prev_close).abs(),
            ], axis=1).max(axis=1)
            new_cols[f"{ticker}_ATR14"] = tr.rolling(14).mean() / (close + 1e-9)

        # Relative strength vs SPY
        if "SPY_Ret" in new_cols:
            spy_ret = new_cols["SPY_Ret"]
            for ticker in target_etfs:
                rk = f"{ticker}_Ret"
                if rk in new_cols and ticker != "SPY":
                    new_cols[f"{ticker}_RelSPY21d"] = (
                        new_cols[rk].rolling(21).mean() -
                        spy_ret.rolling(21).mean()
                    )

        # RoC, OBV, Breakout for each target ETF
        for ticker in target_etfs:
            ret_key = f"{ticker}_Ret"
            if ret_key not in new_cols:
                continue

            ret = new_cols[ret_key]
            cum_ret = (1 + ret).cumprod()
            price_proxy = cum_ret

            # OBV proxy
            vol_col = f"{ticker}_Volume"
            if vol_col in df.columns:
                obv = (np.sign(ret) * df[vol_col]).cumsum()
                new_cols[f"{ticker}_OBV_Z"] = (
                    (obv - obv.rolling(21).mean()) /
                    (obv.rolling(21).std() + 1e-9)
                )

            # Breakout
            rolling_high = price_proxy.rolling(20).max()
            rolling_low = price_proxy.rolling(20).min()
            rng = rolling_high - rolling_low
            new_cols[f"{ticker}_Breakout20d"] = (price_proxy - rolling_low) / (rng + 1e-9)

            # RetZ and momentum acceleration
            rvol_key = f"{ticker}_RVol21d"
            if rvol_key in new_cols:
                daily_vol = new_cols[rvol_key] / np.sqrt(252)
                ret_1d = new_cols[ret_key]
                ret_3d = new_cols[ret_key].rolling(3).sum()
                new_cols[f"{ticker}_RetZ1d"] = ret_1d / (daily_vol + 1e-9)
                new_cols[f"{ticker}_RetZ3d"] = ret_3d / (daily_vol * np.sqrt(3) + 1e-9)
                mom3 = new_cols[ret_key].rolling(3).mean()
                mom21 = new_cols[ret_key].rolling(21).mean()
                new_cols[f"{ticker}_MomAccel"] = mom3 - mom21

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features (shared across both options)."""
    new_cols = {}

    if "DGS10" in df.columns and "T10YIE" in df.columns:
        new_cols["Real_Yield_10Y"] = df["DGS10"] - df["T10YIE"]

    for series in ["DGS10", "T10Y2Y", "T10Y3M", "T10YIE"]:
        if series in df.columns:
            mom20 = df[series].diff(20)
            mom60 = df[series].diff(60)
            new_cols[f"{series}_Mom20d"] = mom20
            new_cols[f"{series}_Mom60d"] = mom60
            new_cols[f"{series}_Rising20d"] = (mom20 > 0).astype(int)
            new_cols[f"{series}_Rising60d"] = (mom60 > 0).astype(int)

    if "DGS10" in df.columns:
        new_cols["DGS10_Accel"] = df["DGS10"].diff(20).diff(20)

    if "T10Y2Y" in df.columns:
        new_cols["YC_Inverted"] = (df["T10Y2Y"] < 0).astype(int)
        new_cols["YC_Flat"] = ((df["T10Y2Y"] >= 0) & (df["T10Y2Y"] < 0.5)).astype(int)
        new_cols["YC_Steep"] = (df["T10Y2Y"] >= 0.5).astype(int)

    if "T10YIE" in df.columns:
        new_cols["Inflation_High"] = (df["T10YIE"] > 2.5).astype(int)
        new_cols["Inflation_Low"] = (df["T10YIE"] < 1.5).astype(int)

    if "VIXCLS" in df.columns:
        new_cols["VIX_Low"] = (df["VIXCLS"] < 15).astype(int)
        new_cols["VIX_Med"] = ((df["VIXCLS"] >= 15) & (df["VIXCLS"] < 25)).astype(int)
        new_cols["VIX_High"] = (df["VIXCLS"] >= 25).astype(int)

    if "BAMLH0A0HYM2" in df.columns:
        new_cols["HY_Stress_High"] = (df["BAMLH0A0HYM2"] > 600).astype(int)
        new_cols["HY_Stress_Med"] = ((df["BAMLH0A0HYM2"] >= 400) &
                                     (df["BAMLH0A0HYM2"] < 600)).astype(int)

    if "DTWEXBGS" in df.columns:
        new_cols["USD_Ret1d"] = df["DTWEXBGS"].pct_change(1)
        new_cols["USD_Ret5d"] = df["DTWEXBGS"].pct_change(5)
        new_cols["USD_Ret21d"] = df["DTWEXBGS"].pct_change(21)

    combined = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Rolling Z-scores for all macro series + derived
    z_new = {}
    for col in list(cfg.FRED_SERIES.keys()) + list(new_cols.keys()):
        if col in combined.columns:
            mu = combined[col].rolling(60, min_periods=20).mean()
            std = combined[col].rolling(60, min_periods=20).std()
            z_new[f"{col}_Z"] = (combined[col] - mu) / (std + 1e-9)

    return pd.concat([combined, pd.DataFrame(z_new, index=combined.index)], axis=1)

# ── Full dataset build ────────────────────────────────────────────────────────

def build_full_dataset(option: str, start_year: int = 2008) -> pd.DataFrame:
    """
    Full pipeline for a given option ('a' or 'b').
    Fetches FRED + ETF data, engineers features, returns clean df.
    """
    option = option.lower()
    if option == "a":
        target_etfs = cfg.OPTION_A_ETFS
        all_tickers = cfg.OPTION_A_ALL_TICKERS
    elif option == "b":
        target_etfs = cfg.OPTION_B_ETFS
        all_tickers = cfg.OPTION_B_ALL_TICKERS
    else:
        raise ValueError(f"Unknown option: {option}. Use 'a' or 'b'.")

    start = f"{start_year}-01-01"
    log.info(f"Building Option {option.upper()} dataset from {start} "
             f"({len(target_etfs)} ETFs)...")

    macro_df = fetch_all_fred(start=start)
    etf_df = fetch_etfs_for_option(all_tickers, start=start)

    if macro_df.empty or etf_df.empty:
        raise ValueError(f"Failed to fetch data for Option {option.upper()}")

    macro_df = macro_df.reindex(etf_df.index, method="ffill")
    df = pd.concat([etf_df, macro_df], axis=1).sort_index()

    df = compute_etf_features(df, target_etfs)
    df = compute_macro_features(df)

    df = df[df.index >= pd.Timestamp(start)]

    core_cols = [f"{t}_Close" for t in target_etfs if f"{t}_Close" in df.columns]
    df = df.dropna(subset=core_cols)
    df = df.ffill().dropna(how="all", axis=1)

    log.info(f"Option {option.upper()} dataset: {len(df)} rows × {df.shape[1]} cols "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df

# ── Forward return targets ────────────────────────────────────────────────────

def build_forward_targets(df: pd.DataFrame, target_etfs: list,
                          rf_rate_col: str = "DTB3") -> pd.DataFrame:
    """
    Build forward return targets at multiple horizons (5, 10, 15, 20 days).
    Stores raw forward returns and BeatCash binary labels at each horizon per ETF.
    """
    fwd = pd.DataFrame(index=df.index)
    daily_rf = (df[rf_rate_col] / 100 / 252
                if rf_rate_col in df.columns
                else pd.Series(cfg.RISK_FREE_RATE / 252, index=df.index))

    for ticker in target_etfs:
        ret_col = f"{ticker}_Ret"
        if ret_col not in df.columns:
            continue
        for h in [5, 10, 15, 20]:
            fwd_ret = df[ret_col].rolling(h).sum().shift(-h)
            rf_thr = daily_rf * h
            fwd[f"{ticker}_FwdRet{h}d"] = fwd_ret
            fwd[f"{ticker}_BeatCash{h}d"] = (fwd_ret > rf_thr).astype(int)
        # Default 5d for backward compatibility
        fwd[f"{ticker}_FwdRet"] = fwd[f"{ticker}_FwdRet5d"]
        fwd[f"{ticker}_BeatCash"] = fwd[f"{ticker}_BeatCash5d"]

    return fwd

# ── Incremental update ────────────────────────────────────────────────────────

def incremental_update(existing_df: pd.DataFrame, option: str) -> pd.DataFrame:
    """
    Fetch only data since last date in existing_df for the given option.
    Re-engineers features for the new tail and appends.
    """
    option = option.lower()
    target_etfs = cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS
    all_tickers = cfg.OPTION_A_ALL_TICKERS if option == "a" else cfg.OPTION_B_ALL_TICKERS

    last_date = existing_df.index[-1]
    start = (last_date - timedelta(days=90)).strftime("%Y-%m-%d")
    log.info(f"Option {option.upper()} incremental update from {start}...")

    macro_df = fetch_all_fred(start=start)
    etf_df = fetch_etfs_for_option(all_tickers, start=start)

    if macro_df.empty or etf_df.empty:
        log.warning(f"Option {option.upper()} incremental fetch empty — returning existing")
        return existing_df

    macro_df = macro_df.reindex(etf_df.index, method="ffill")
    new_df = pd.concat([etf_df, macro_df], axis=1).sort_index()
    new_df = compute_etf_features(new_df, target_etfs)
    new_df = compute_macro_features(new_df)

    new_rows = new_df[new_df.index > last_date]
    if new_rows.empty:
        log.info(f"Option {option.upper()}: no new rows — dataset is current")
        return existing_df

    combined = (pd.concat([existing_df, new_rows])
                .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                .sort_index())
    log.info(f"Option {option.upper()}: appended {len(new_rows)} rows. "
             f"Total: {len(combined)}")
    return combined

# ── Option-aware HF storage ───────────────────────────────────────────────────

def _paths(option: str) -> dict:
    """Return HF path dict for the given option."""
    option = option.lower()
    if option == "a":
        return cfg.OPTION_A_HF
    elif option == "b":
        return cfg.OPTION_B_HF
    raise ValueError(f"Unknown option: {option}")

def load_dataset(option: str) -> Optional[pd.DataFrame]:
    return hf_read_parquet(_paths(option)["dataset"])

def save_dataset(df: pd.DataFrame, option: str) -> bool:
    return hf_write_parquet(
        _paths(option)["dataset"], df,
        f"Option {option.upper()} dataset update {df.index[-1].date()} ({len(df)} rows)"
    )

def load_predictions(option: str) -> Optional[pd.DataFrame]:
    return hf_read_parquet(_paths(option)["predictions"])

def save_predictions(df: pd.DataFrame, option: str) -> bool:
    return hf_write_parquet(
        _paths(option)["predictions"], df,
        f"Option {option.upper()} predictions update {df.index[-1].date()}"
    )

def load_wf_predictions(option: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    return hf_read_parquet(_paths(option)["wf_preds"], force_download=force_download)

def save_wf_predictions(df: pd.DataFrame, option: str) -> bool:
    return hf_write_parquet(
        _paths(option)["wf_preds"], df,
        f"Option {option.upper()} WF predictions update {df.index[-1].date()}"
    )

def load_signals(option: str) -> Optional[pd.DataFrame]:
    return hf_read_parquet(_paths(option)["signals"])

def save_signals(signals_df: pd.DataFrame, option: str) -> bool:
    existing = load_signals(option)
    if existing is not None:
        combined = (pd.concat([existing, signals_df])
                    .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                    .sort_index())
    else:
        combined = signals_df
    return hf_write_parquet(
        _paths(option)["signals"], combined,
        f"Option {option.upper()} signals update {combined.index[-1].date()}"
    )

def load_detector(option: str) -> Optional[bytes]:
    return hf_read_binary(_paths(option)["detector"])

def save_detector(model_bytes: bytes, option: str) -> bool:
    return hf_write_binary(
        _paths(option)["detector"], model_bytes,
        f"Option {option.upper()} regime detector update"
    )

def load_ranker(option: str) -> Optional[bytes]:
    return hf_read_binary(_paths(option)["ranker"])

def save_ranker(model_bytes: bytes, option: str) -> bool:
    return hf_write_binary(
        _paths(option)["ranker"], model_bytes,
        f"Option {option.upper()} momentum ranker update"
    )

def load_model(option: str, filename: str) -> Optional[bytes]:
    base = _paths(option)["detector"].rsplit("/", 1)[0]
    return hf_read_binary(f"{base}/{filename}")

def save_model(model_bytes: bytes, option: str, filename: str) -> bool:
    base = _paths(option)["detector"].rsplit("/", 1)[0]
    return hf_write_binary(
        f"{base}/{filename}", model_bytes,
        f"Option {option.upper()} model update: {filename}"
    )

def save_feature_list(feature_names: list, option: str) -> bool:
    content = json.dumps({"features": feature_names,
                          "updated": datetime.utcnow().isoformat()})
    return hf_write_file(_paths(option)["feature_list"], content,
                         f"Option {option.upper()} feature list update")

def load_feature_list(option: str) -> Optional[list]:
    content = hf_read_file(_paths(option)["feature_list"])
    if content is None:
        return None
    try:
        return json.loads(content).get("features")
    except Exception:
        return None

def save_sweep_result(results: dict, start_year: int, option: str) -> bool:
    today_est = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y%m%d")
    path = f"{_paths(option)['sweep_prefix']}{start_year}_{today_est}.json"

    # CORRECTED: Clean results for JSON serialization
    clean_results = _clean_for_json(results)

    return hf_write_file(
        path,
        json.dumps(clean_results, indent=2, cls=NumpyEncoder),
        f"Option {option.upper()} sweep {start_year} — {today_est}"
    )

def load_sweep_results(option: str) -> tuple:
    """
    Load the most recent sweep result per year for the given option.
    Returns (dict of {year: result}, best_date_str).
    """
    prefix = _paths(option)["sweep_prefix"]
    folder = prefix.rsplit("/", 1)[0]
    files = hf_list_files(folder)

    year_best = {}
    for name in files:
        if not name.endswith(".json"):
            continue
        base = name.replace(prefix, "")
        parts = base.replace(".json", "").split("_")
        if len(parts) < 2:
            continue
        try:
            yr = int(parts[0])
            dt = parts[1]
            if yr not in year_best or dt > year_best[yr]:
                year_best[yr] = dt
        except Exception:
            continue

    found, best_date = {}, None
    for yr, dt in year_best.items():
        fname = f"{prefix}{yr}_{dt}.json"
        content = hf_read_file(fname)
        if content:
            try:
                found[yr] = json.loads(content)
                best_date = dt if best_date is None or dt > best_date else best_date
            except Exception:
                pass

    return found, best_date

# ── Streamlit entry point ─────────────────────────────────────────────────────

def get_data(option: str, start_year: int = 2008,
             force_refresh: bool = False) -> pd.DataFrame:
    """
    Load dataset for the given option from HF.
    Falls back to full rebuild if not found or force_refresh=True.
    """
    if not force_refresh:
        df = load_dataset(option)
        if df is not None:
            df = df[df.index >= pd.Timestamp(f"{start_year}-01-01")]
            log.info(f"Option {option.upper()} HF dataset: {len(df)} rows from {start_year}")
            return df

    log.info(f"Option {option.upper()} full dataset rebuild...")
    df = build_full_dataset(option=option, start_year=start_year)
    save_dataset(df, option)
    return df

# ── Backward-compatible aliases (Option A defaults) ───────────────────────────

def load_dataset_from_hf() -> Optional[pd.DataFrame]: return load_dataset("a")
def save_dataset_to_hf(df) -> bool: return save_dataset(df, "a")
def load_signals_from_hf() -> Optional[pd.DataFrame]: return load_signals("a")
def save_signals_to_hf(df) -> bool: return save_signals(df, "a")
def load_model_from_hf(fn) -> Optional[bytes]: return hf_read_binary(f"option_a/models/{fn}")
def save_model_to_hf(b, fn) -> bool: return save_model(b, "a", fn)
def load_predictions_from_hf(p=None) -> Optional[pd.DataFrame]: return load_predictions("a")
def save_predictions_to_hf(df, p=None) -> bool: return save_predictions(df, "a")
def load_wf_predictions_from_hf() -> Optional[pd.DataFrame]: return load_wf_predictions("a")
def load_momentum_ranker_from_hf() -> Optional[bytes]: return load_ranker("a")
def load_momentum_predictions_from_hf() -> Optional[pd.DataFrame]: return load_predictions("a")
def load_wf_momentum_predictions_from_hf() -> Optional[pd.DataFrame]: return load_wf_predictions("a")
def save_sweep_to_hf(r, y) -> bool: return save_sweep_result(r, y, "a")
def save_feature_list_to_hf(fl) -> bool: return save_feature_list(fl, "a")
def load_feature_list_from_hf() -> Optional[list]: return load_feature_list("a")
