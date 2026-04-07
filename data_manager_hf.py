"""
data_manager_hf.py - P2-ETF-REGIME-PREDICTOR v4 (CORRECTED)
=========================================

FIXES vs v3:
  1. save_dataframe() reverts to simple index=True for ALL file types.
     The v3 reset_index() approach for wf_predictions caused a double-reset
     bug when _merge_wf() in train_hf.py also reset the index, producing
     an integer-indexed parquet that load_dataframe() could not restore.
     The correct approach (v4): _merge_wf() handles composite dedup in
     memory and the result already has a clean DatetimeIndex before save.

  2. load_dataframe() no longer tries to detect/repair the reset-index
     format — that format no longer exists in v4.

  3. All other fixes from v2 (repo_type="dataset", dated sweep files,
     hf_write_file, build_full_dataset, incremental_update) preserved.
"""

import os
import io
import pickle
import json
import logging
import re
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── huggingface_hub ────────────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file, login, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Local mode only.")

# ── Configuration ──────────────────────────────────────────────────────────────
try:
    import config as cfg
    REPO_ID = cfg.HF_DATASET_REPO
except Exception:
    REPO_ID = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-regime-predictor")

HF_REPO_ID = REPO_ID
REPO_TYPE  = "dataset"

try:
    import config as _cfg
    TARGET_ETFS = _cfg.OPTION_A_ETFS
    ALL_TICKERS = list(dict.fromkeys(
        _cfg.OPTION_A_ETFS + _cfg.OPTION_A_BENCHMARKS +
        _cfg.OPTION_B_ETFS + _cfg.OPTION_B_BENCHMARKS
    ))
except Exception:
    TARGET_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
    ALL_TICKERS = TARGET_ETFS + ["SPY", "AGG"]

LOCAL_CACHE = "./hf_cache"

# ── Authentication ─────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN and HF_AVAILABLE:
    try:
        login(token=HF_TOKEN)
        print("✅ Authenticated with Hugging Face Hub")
        api = HfApi()
        try:
            files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
            print(f"\n📁 Files in {REPO_ID}:")
            for f in files:
                print(f"  - {f}")
            print()
        except Exception as e:
            print(f"⚠️ Could not list repo files: {e}")
    except Exception as e:
        print(f"❌ HF authentication failed: {e}")
else:
    print("⚠️ HF_TOKEN not set or huggingface_hub not available")


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _ensure_cache_dir():
    os.makedirs(LOCAL_CACHE, exist_ok=True)

def _get_cache_path(filename: str, option: str) -> str:
    _ensure_cache_dir()
    return os.path.join(LOCAL_CACHE, f"{option}_{filename}")


# ── Low-level HF I/O ───────────────────────────────────────────────────────────
def upload_to_hub(local_path: str, remote_path: str, repo_id: str = REPO_ID):
    if not HF_AVAILABLE:
        print(f"HF not available, skipping upload: {local_path}")
        return
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print(f"✅ Uploaded {remote_path} to {repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload {remote_path}: {e}")


def hf_write_file(content: bytes, path_in_repo: str,
                  repo_id: str = REPO_ID, commit_message: str = "update"):
    if not HF_AVAILABLE:
        return
    try:
        upload_file(
            path_or_fileobj=io.BytesIO(content),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=commit_message,
        )
    except Exception as e:
        log.error(f"hf_write_file failed for {path_in_repo}: {e}")


def hf_write_parquet(df: pd.DataFrame, path_in_repo: str,
                     repo_id: str = REPO_ID, commit_message: str = "update"):
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    hf_write_file(buf.getvalue(), path_in_repo,
                  repo_id=repo_id, commit_message=commit_message)


def download_from_hub(remote_path: str, local_path: str,
                      repo_id: str = REPO_ID) -> bool:
    if not HF_AVAILABLE:
        return False
    try:
        print(f"  Attempting: {remote_path}")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type=REPO_TYPE,
            local_dir=os.path.dirname(local_path) or ".",
            local_dir_use_symlinks=False,
            force_download=True,
        )
        if downloaded_path != local_path and os.path.exists(downloaded_path):
            import shutil
            shutil.copy2(downloaded_path, local_path)
        return os.path.exists(local_path) or os.path.exists(downloaded_path)
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:120]}")
        return False


# ── DataFrame I/O ──────────────────────────────────────────────────────────────
_FILENAME_MAP = {
    "data":           "etf_data.parquet",
    "predictions":    "mom_pred_history.parquet",
    "wf_predictions": "wf_mom_pred_history.parquet",
    "signals":        "signals.parquet",
}


def save_dataframe(df: pd.DataFrame, name: str, option: str, upload: bool = True):
    """
    Save a DataFrame to local cache then upload to HF.

    Always writes with index=True.  For wf_predictions, the caller
    (_merge_wf in train_hf.py) is responsible for ensuring the DataFrame
    has a clean DatetimeIndex and a 'test_year' column before calling here.
    """
    actual_name = _FILENAME_MAP.get(name, f"{name}.parquet")
    cache_path  = _get_cache_path(actual_name, option)

    df.to_parquet(cache_path, index=True)

    if name == "wf_predictions" and "test_year" in df.columns:
        n_years = df["test_year"].nunique()
        print(f"✅ Saved wf_predictions for option {option}: "
              f"{len(df)} rows across {n_years} test years "
              f"{sorted(df['test_year'].unique())}")
    else:
        print(f"✅ Saved {name} for option {option}: {len(df)} rows")

    if upload:
        upload_to_hub(cache_path, f"option_{option}/{actual_name}")


def load_dataframe(name: str, option: str,
                   force_download: bool = False) -> Optional[pd.DataFrame]:
    """Load a DataFrame from local cache or HF."""
    actual_name = _FILENAME_MAP.get(name, f"{name}.parquet")
    cache_path  = _get_cache_path(actual_name, option)

    if not force_download and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df = df.set_index("Date")
                    df.index = pd.DatetimeIndex(df.index)
                    df.index.name = "Date"
            return df
        except Exception as e:
            print(f"⚠️ Error loading local {cache_path}: {e}")

    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        print(f"\n🔍 Searching HF for {name} (option {option})...")
        for remote_path in [
            f"option_{option}/{actual_name}",
            f"{option}/{actual_name}",
            f"{option}_{actual_name}",
            actual_name,
        ]:
            if download_from_hub(remote_path, temp_path):
                if os.path.exists(temp_path):
                    os.replace(temp_path, cache_path)
                    try:
                        df = pd.read_parquet(cache_path)
                        if not isinstance(df.index, pd.DatetimeIndex):
                            if "Date" in df.columns:
                                df = df.set_index("Date")
                                df.index = pd.DatetimeIndex(df.index)
                                df.index.name = "Date"
                        print(f"✅ Loaded {name} for option {option} from HF")
                        return df
                    except Exception as e:
                        print(f"⚠️ Downloaded file unreadable: {e}")

    print(f"❌ Could not find {name} for option {option}")
    return None


# ── Pickle I/O ─────────────────────────────────────────────────────────────────
def save_pickle(obj: Any, name: str, option: str, upload: bool = True):
    cache_path = _get_cache_path(f"{name}.pkl", option)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved {name} for option {option}")
    if upload:
        upload_to_hub(cache_path, f"option_{option}/models/{name}.pkl")


def load_pickle(name: str, option: str, force_download: bool = False) -> Optional[Any]:
    cache_path = _get_cache_path(f"{name}.pkl", option)
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {cache_path}: {e}")

    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        for remote_path in [
            f"option_{option}/models/{name}.pkl",
            f"option_{option}/{name}.pkl",
            f"models/{name}.pkl",
        ]:
            if download_from_hub(remote_path, temp_path):
                if os.path.exists(temp_path):
                    os.replace(temp_path, cache_path)
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
    return None


# ── JSON I/O ───────────────────────────────────────────────────────────────────
def save_json(obj: Dict, name: str, option: str, upload: bool = True):
    cache_path = _get_cache_path(f"{name}.json", option)
    with open(cache_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"✅ Saved {name} for option {option}")
    if upload:
        upload_to_hub(cache_path, f"option_{option}/meta/{name}.json")


def load_json(name: str, option: str, force_download: bool = False) -> Optional[Dict]:
    cache_path = _get_cache_path(f"{name}.json", option)
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {cache_path}: {e}")

    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        if download_from_hub(f"option_{option}/meta/{name}.json", temp_path):
            if os.path.exists(temp_path):
                os.replace(temp_path, cache_path)
                with open(cache_path) as f:
                    return json.load(f)
    return None


# ── Named accessors ────────────────────────────────────────────────────────────
def get_data(option: str, start_year: int = 2000, force_refresh: bool = False):
    df = load_dataframe("data", option, force_download=force_refresh)
    if df is not None:
        print(f"✅ Loaded data for option {option}: {len(df)} rows")
        return df
    raise ValueError(
        f"Data not found for option {option}. "
        f"Run seed_hf_dataset workflow or check HF_TOKEN."
    )

def load_predictions(option, force_download=False):
    return load_dataframe("predictions", option, force_download=force_download)

def save_predictions(df, option, upload=True):
    save_dataframe(df, "predictions", option, upload=upload)

def load_wf_predictions(option, force_download=False):
    return load_dataframe("wf_predictions", option, force_download=force_download)

def save_wf_predictions(df, option, upload=True):
    save_dataframe(df, "wf_predictions", option, upload=upload)

def load_signals(option, force_download=False):
    return load_dataframe("signals", option, force_download=force_download)

def save_signals(df, option, upload=True):
    save_dataframe(df, "signals", option, upload=upload)

def load_detector(option, force_download=False):
    return load_pickle("regime_detector", option, force_download=force_download)

def save_detector(detector, option, upload=True):
    save_pickle(detector, "regime_detector", option, upload=upload)

def save_ranker(ranker, option, upload=True):
    save_pickle(ranker, "momentum_ranker", option, upload=upload)

def load_ranker(option, force_download=False):
    return load_pickle("momentum_ranker", option, force_download=force_download)

def save_feature_list(features, option, upload=True):
    save_json({"features": features, "timestamp": str(datetime.now())},
              "features", option, upload=upload)

def load_feature_list(option, force_download=False):
    data = load_json("features", option, force_download=force_download)
    return data.get("features") if data else None


# ── Sweep I/O ─────────────────────────────────────────────────────────────────
def save_sweep_result(result: Dict, year: int, option: str, upload: bool = True):
    """Save sweep result keyed by test_year to dated file on HF."""
    date_str   = datetime.now().strftime("%Y%m%d")
    name       = f"sweep_{year}"
    cache_path = _get_cache_path(f"{name}.json", option)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if upload:
        remote = f"option_{option}/sweep/sweep_{year}_{date_str}.json"
        upload_to_hub(cache_path, remote)


def load_sweep_result(year: int, option: str, force_download: bool = False) -> Optional[Dict]:
    """Load latest sweep result for a given test_year."""
    cache_path = _get_cache_path(f"sweep_{year}.json", option)
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    if not HF_AVAILABLE:
        return None

    temp_path = cache_path + ".tmp"

    # Try legacy un-dated paths first
    for remote in [
        f"option_{option}/meta/sweep_{year}.json",
        f"option_{option}/sweep/sweep_{year}.json",
    ]:
        if download_from_hub(remote, temp_path):
            if os.path.exists(temp_path):
                os.replace(temp_path, cache_path)
                with open(cache_path) as f:
                    return json.load(f)

    # Scan for latest dated file
    try:
        files     = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        pattern   = re.compile(rf"option_{option}/sweep/sweep_{year}_(\d{{8}})\.json")
        candidates = {}
        for fpath in files:
            m = pattern.match(fpath)
            if m:
                candidates[m.group(1)] = fpath
        if candidates:
            latest = candidates[max(candidates.keys())]
            if download_from_hub(latest, temp_path):
                if os.path.exists(temp_path):
                    os.replace(temp_path, cache_path)
                    with open(cache_path) as f:
                        return json.load(f)
    except Exception as e:
        log.warning(f"load_sweep_result({year}): scan failed: {e}")

    return None


def load_sweep_results(option: str) -> tuple:
    """Load all sweep results by scanning HF for dated sweep files."""
    results   = {}
    best_date = None

    if not HF_AVAILABLE:
        return results, best_date

    try:
        files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    except Exception as e:
        log.warning(f"load_sweep_results: could not list repo files: {e}")
        return results, best_date

    pattern = re.compile(rf"option_{option}/sweep/sweep_(\d{{4}})_(\d{{8}})\.json")
    best_per_year: Dict[int, str] = {}
    for fpath in files:
        m = pattern.match(fpath)
        if m:
            yr, ds = int(m.group(1)), m.group(2)
            if yr not in best_per_year or ds > best_per_year[yr]:
                best_per_year[yr] = ds

    if not best_per_year:
        log.warning(f"load_sweep_results: no sweep files for option {option}")
        return results, best_date

    best_date = max(best_per_year.values())

    for year, date_stamp in best_per_year.items():
        remote     = f"option_{option}/sweep/sweep_{year}_{date_stamp}.json"
        cache_key  = f"sweep_{year}_{date_stamp}"
        cache_path = _get_cache_path(f"{cache_key}.json", option)
        if not os.path.exists(cache_path):
            temp = cache_path + ".tmp"
            if download_from_hub(remote, temp):
                if os.path.exists(temp):
                    os.replace(temp, cache_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    results[year] = json.load(f)
            except Exception as e:
                log.warning(f"Could not parse {cache_path}: {e}")

    log.info(f"load_sweep_results: {len(results)} years for option {option}")
    return results, best_date


def list_available_data(option: str) -> Dict[str, bool]:
    return {
        "data":           load_dataframe("data", option) is not None,
        "predictions":    load_predictions(option) is not None,
        "wf_predictions": load_wf_predictions(option) is not None,
        "signals":        load_signals(option) is not None,
        "detector":       load_detector(option) is not None,
        "ranker":         load_ranker(option) is not None,
        "features":       load_feature_list(option) is not None,
    }


# ── High-level functions (seed + daily_data_update) ───────────────────────────
def build_full_dataset(option: str, start_year: int = 2008) -> pd.DataFrame:
    return _build_dataset_inline(option=option, start_year=start_year)


def _build_dataset_inline(option: str, start_year: int = 2008) -> pd.DataFrame:
    import yfinance as yf
    import pandas_datareader as pdr

    try:
        import config as cfg
        etfs       = cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS
        benchmarks = cfg.OPTION_A_BENCHMARKS if option == "a" else cfg.OPTION_B_BENCHMARKS
        fred_map   = cfg.FRED_SERIES
    except Exception:
        etfs       = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
        benchmarks = ["SPY", "AGG"]
        fred_map   = {"DGS10": "10Y", "DTB3": "3M", "VIXCLS": "VIX"}

    all_tickers = list(dict.fromkeys(etfs + benchmarks))
    start_str   = f"{start_year}-01-01"
    end_str     = datetime.today().strftime("%Y-%m-%d")
    fred_key    = os.environ.get("FRED_API_KEY", "")

    log.info(f"Fetching {all_tickers} ({start_str}→{end_str})…")
    raw   = yf.download(all_tickers, start=start_str, end=end_str,
                        auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw

    df = pd.DataFrame(index=close.index)
    df.index.name = "Date"

    for t in all_tickers:
        if t in close.columns:
            df[f"{t}_Close"] = close[t]
            df[f"{t}_Ret"]   = close[t].pct_change()

    if "Volume" in raw.columns:
        for t in all_tickers:
            if t in raw["Volume"].columns:
                df[f"{t}_Vol"] = raw["Volume"][t]

    if fred_key:
        for sid in fred_map:
            try:
                s = pdr.DataReader(sid, "fred", start=start_str, end=end_str)
                df[sid] = s[sid].reindex(df.index, method="ffill")
                log.info(f"  ✅ FRED {sid}")
            except Exception as e:
                log.warning(f"  ⚠️ FRED {sid}: {e}")
    else:
        log.warning("FRED_API_KEY not set — macro features omitted")

    for t in etfs:
        if f"{t}_Close" not in df.columns:
            continue
        for w in [5, 10, 21, 63]:
            df[f"{t}_RoC_{w}d"] = df[f"{t}_Close"].pct_change(w)
        if f"{t}_Vol" in df.columns:
            df[f"{t}_OBV_21d"] = (
                np.sign(df[f"{t}_Ret"]) * df[f"{t}_Vol"]
            ).rolling(21).sum()
        roll_hi = df[f"{t}_Close"].rolling(20).max()
        roll_lo = df[f"{t}_Close"].rolling(20).min()
        rng     = (roll_hi - roll_lo).replace(0, np.nan)
        df[f"{t}_Breakout_20d"] = (df[f"{t}_Close"] - roll_lo) / rng

    df.dropna(how="all", inplace=True)
    log.info(f"Dataset built: {len(df)} rows × {df.shape[1]} cols")
    return df


def save_dataset(df: pd.DataFrame, option: str) -> bool:
    try:
        save_dataframe(df, "data", option, upload=True)
        return True
    except Exception as e:
        log.error(f"save_dataset option {option}: {e}")
        return False


def load_dataset(option: str) -> Optional[pd.DataFrame]:
    return load_dataframe("data", option, force_download=False)


def incremental_update(df_existing: pd.DataFrame, option: str) -> pd.DataFrame:
    import yfinance as yf
    import pandas_datareader as pdr

    last_date = df_existing.index[-1]
    start_str = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = datetime.today().strftime("%Y-%m-%d")

    if start_str >= end_str:
        log.info("incremental_update: already up to date.")
        return df_existing

    try:
        import config as cfg
        etfs       = cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS
        benchmarks = cfg.OPTION_A_BENCHMARKS if option == "a" else cfg.OPTION_B_BENCHMARKS
        fred_map   = cfg.FRED_SERIES
    except Exception:
        etfs       = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
        benchmarks = ["SPY", "AGG"]
        fred_map   = {"DGS10": "10Y", "DTB3": "3M", "VIXCLS": "VIX"}

    all_tickers  = list(dict.fromkeys(etfs + benchmarks))
    fred_key     = os.environ.get("FRED_API_KEY", "")
    lookback_str = (last_date - pd.Timedelta(days=100)).strftime("%Y-%m-%d")

    log.info(f"incremental_update: {start_str} → {end_str}…")
    raw   = yf.download(all_tickers, start=lookback_str, end=end_str,
                        auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw
    new_idx = close.index[close.index > last_date]

    if len(new_idx) == 0:
        log.info("incremental_update: no new trading days.")
        return df_existing

    existing_close_cols = {
        t: f"{t}_Close" for t in all_tickers if f"{t}_Close" in df_existing.columns
    }
    existing_close = df_existing[list(existing_close_cols.values())].rename(
        columns={v: k for k, v in existing_close_cols.items()}
    )
    combined_close = pd.concat([existing_close, close]).groupby(level=0).last()
    delta = pd.DataFrame(index=new_idx)

    for t in all_tickers:
        if t not in close.columns:
            continue
        delta[f"{t}_Close"] = close[t].reindex(new_idx)
        delta[f"{t}_Ret"]   = combined_close[t].pct_change().reindex(new_idx)
        if "Volume" in raw.columns and t in raw["Volume"].columns:
            delta[f"{t}_Vol"] = raw["Volume"][t].reindex(new_idx)

    for t in etfs:
        if t not in combined_close.columns:
            continue
        ser_c = combined_close[t]
        for w in [5, 10, 21, 63]:
            delta[f"{t}_RoC_{w}d"] = ser_c.pct_change(w).reindex(new_idx)
        if f"{t}_Vol" in delta.columns:
            sign_ret = np.sign(combined_close[t].pct_change())
            vol_ser  = (raw["Volume"][t] if "Volume" in raw.columns
                        and t in raw["Volume"].columns
                        else pd.Series(np.nan, index=combined_close.index))
            delta[f"{t}_OBV_21d"] = (sign_ret * vol_ser).rolling(21).sum().reindex(new_idx)
        hi  = ser_c.rolling(20).max().reindex(new_idx)
        lo  = ser_c.rolling(20).min().reindex(new_idx)
        rng = (hi - lo).replace(0, np.nan)
        delta[f"{t}_Breakout_20d"] = (ser_c.reindex(new_idx) - lo) / rng

    if fred_key:
        for sid in fred_map:
            try:
                s = pdr.DataReader(sid, "fred", start=lookback_str, end=end_str)
                delta[sid] = s[sid].reindex(new_idx, method="ffill")
            except Exception as e:
                log.warning(f"FRED {sid}: {e}")
                if sid in df_existing.columns:
                    delta[sid] = df_existing[sid].iloc[-1]

    for col in df_existing.columns:
        if col not in delta.columns:
            delta[col] = np.nan

    result = (pd.concat([df_existing, delta[df_existing.columns]])
              .pipe(lambda d: d[~d.index.duplicated(keep="last")])
              .sort_index())
    log.info(f"incremental_update: +{len(new_idx)} rows → {len(result)} total")
    return result
