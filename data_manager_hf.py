"""
data_manager_hf.py - P2-ETF-REGIME-PREDICTOR v2 (CORRECTED)
=========================================
Data management for Hugging Face Hub storage.
Handles caching, downloading, and uploading of data artifacts.
"""

import os
import pickle
import json
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Local mode only.")


# Configuration - YOUR ACTUAL REPO
REPO_ID = "P2SAMAPA/p2-etf-regime-predictor"
LOCAL_CACHE = "./hf_cache"

# Try to authenticate with token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN and HF_AVAILABLE:
    try:
        login(token=HF_TOKEN)
        print("Authenticated with Hugging Face Hub")
    except Exception as e:
        print(f"Authentication failed: {e}")


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(LOCAL_CACHE):
        os.makedirs(LOCAL_CACHE)


def _get_cache_path(filename: str, option: str) -> str:
    """Get local cache path for a file."""
    _ensure_cache_dir()
    return os.path.join(LOCAL_CACHE, f"{option}_{filename}")


def upload_to_hub(local_path: str, remote_path: str, repo_id: str = REPO_ID):
    """Upload a file to Hugging Face Hub."""
    if not HF_AVAILABLE:
        print(f"HF not available, keeping local: {local_path}")
        return
    
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
        )
        print(f"Uploaded {remote_path} to {repo_id}")
    except Exception as e:
        print(f"Failed to upload {remote_path}: {e}")


def download_from_hub(remote_path: str, local_path: str, repo_id: str = REPO_ID) -> bool:
    """Download a file from Hugging Face Hub."""
    if not HF_AVAILABLE:
        return False
    
    try:
        # Try downloading without symlinks
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False,
            force_download=True,
        )
        
        # If the downloaded path is different, copy/move it
        if downloaded_path != local_path and os.path.exists(downloaded_path):
            import shutil
            shutil.copy2(downloaded_path, local_path)
        
        return os.path.exists(local_path)
    except Exception as e:
        print(f"Failed to download {remote_path}: {e}")
        return False


def save_dataframe(df: pd.DataFrame, name: str, option: str, upload: bool = True):
    """Save a dataframe to cache and optionally upload to HF."""
    # Map logical names to actual filenames in repo
    filename_map = {
        "data": "etf_data.parquet",
        "predictions": "mom_pred_history.parquet",
        "wf_predictions": "wf_mom_pred_history.parquet",
        "signals": "signals.parquet",
    }
    
    actual_name = filename_map.get(name, f"{name}.parquet")
    cache_path = _get_cache_path(actual_name, option)
    df.to_parquet(cache_path, index=True)
    print(f"Saved {name} for option {option} to {cache_path}")
    
    if upload:
        # Upload to option-specific subfolder
        remote_path = f"option_{option}/{actual_name}"
        upload_to_hub(cache_path, remote_path)


def load_dataframe(name: str, option: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    """Load a dataframe from cache or HF."""
    # Map logical names to actual filenames in repo
    filename_map = {
        "data": "etf_data.parquet",
        "predictions": "mom_pred_history.parquet",
        "wf_predictions": "wf_mom_pred_history.parquet",
        "signals": "signals.parquet",
    }
    
    actual_name = filename_map.get(name, f"{name}.parquet")
    cache_path = _get_cache_path(actual_name, option)
    
    # Try local cache first
    if not force_download and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {name} for option {option} from local cache")
            return df
        except Exception as e:
            print(f"Error loading local {cache_path}: {e}")
    
    # Try downloading from HF
    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        
        # Try different possible remote paths
        remote_paths = [
            f"option_{option}/{actual_name}",  # option_a/etf_data.parquet
            f"{actual_name}",                   # etf_data.parquet
            f"{option}_{actual_name}",          # a_etf_data.parquet
        ]
        
        for remote_path in remote_paths:
            print(f"Trying to download: {remote_path}")
            if download_from_hub(remote_path, temp_path):
                if os.path.exists(temp_path):
                    os.rename(temp_path, cache_path)
                    df = pd.read_parquet(cache_path)
                    print(f"Successfully loaded {name} for option {option} from HF")
                    return df
    
    return None


def save_pickle(obj: Any, name: str, option: str, upload: bool = True):
    """Save a pickle object to cache and optionally upload to HF."""
    cache_path = _get_cache_path(f"{name}.pkl", option)
    with open(cache_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved {name} for option {option} to {cache_path}")
    
    if upload:
        remote_path = f"option_{option}/models/{name}.pkl"
        upload_to_hub(cache_path, remote_path)


def load_pickle(name: str, option: str, force_download: bool = False) -> Optional[Any]:
    """Load a pickle object from cache or HF."""
    cache_path = _get_cache_path(f"{name}.pkl", option)
    
    # Try local cache first
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                print(f"Loaded {name} for option {option} from local cache")
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading local {cache_path}: {e}")
    
    # Try downloading from HF
    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        
        # Try different possible remote paths
        remote_paths = [
            f"option_{option}/models/{name}.pkl",
            f"option_{option}/{name}.pkl",
            f"models/{name}.pkl",
        ]
        
        for remote_path in remote_paths:
            print(f"Trying to download: {remote_path}")
            if download_from_hub(remote_path, temp_path):
                if os.path.exists(temp_path):
                    os.rename(temp_path, cache_path)
                    with open(cache_path, 'rb') as f:
                        print(f"Successfully loaded {name} for option {option} from HF")
                        return pickle.load(f)
    
    return None


def save_json(obj: Dict, name: str, option: str, upload: bool = True):
    """Save a JSON object to cache and optionally upload to HF."""
    cache_path = _get_cache_path(f"{name}.json", option)
    with open(cache_path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved {name} for option {option} to {cache_path}")
    
    if upload:
        remote_path = f"option_{option}/meta/{name}.json"
        upload_to_hub(cache_path, remote_path)


def load_json(name: str, option: str, force_download: bool = False) -> Optional[Dict]:
    """Load a JSON object from cache or HF."""
    cache_path = _get_cache_path(f"{name}.json", option)
    
    # Try local cache first
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading local {cache_path}: {e}")
    
    # Try downloading from HF
    if HF_AVAILABLE:
        temp_path = cache_path + ".tmp"
        remote_path = f"option_{option}/meta/{name}.json"
        if download_from_hub(remote_path, temp_path):
            if os.path.exists(temp_path):
                os.rename(temp_path, cache_path)
                with open(cache_path, 'r') as f:
                    return json.load(f)
    
    return None


# Specific functions for the project
def get_data(option: str, start_year: int = 2000, force_refresh: bool = False) -> pd.DataFrame:
    """Get the main dataset for an option."""
    # First try to load the main data file
    df = load_dataframe("data", option, force_download=force_refresh)
    
    if df is not None:
        print(f"Loaded data for option {option}: {len(df)} rows")
        return df
    
    # If not found, raise error with helpful message
    raise ValueError(
        f"Data not found for option {option}. "
        f"Make sure you have:\n"
        f"1. Set HF_TOKEN environment variable for authentication\n"
        f"2. The repository {REPO_ID} exists and contains option_{option}/etf_data.parquet\n"
        f"3. Run data collection first if data doesn't exist"
    )


def load_predictions(option: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    """Load predictions for an option."""
    return load_dataframe("predictions", option, force_download=force_download)


def save_predictions(df: pd.DataFrame, option: str, upload: bool = True):
    """Save predictions for an option."""
    save_dataframe(df, "predictions", option, upload=upload)


def load_wf_predictions(option: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    """Load walk-forward predictions for an option."""
    return load_dataframe("wf_predictions", option, force_download=force_download)


def save_wf_predictions(df: pd.DataFrame, option: str, upload: bool = True):
    """Save walk-forward predictions for an option."""
    save_dataframe(df, "wf_predictions", option, upload=upload)


def load_signals(option: str, force_download: bool = False) -> Optional[pd.DataFrame]:
    """Load signals for an option."""
    return load_dataframe("signals", option, force_download=force_download)


def save_signals(df: pd.DataFrame, option: str, upload: bool = True):
    """Save signals for an option."""
    save_dataframe(df, "signals", option, upload=upload)


def load_detector(option: str, force_download: bool = False) -> Optional[bytes]:
    """Load the regime detector for an option."""
    return load_pickle("regime_detector", option, force_download=force_download)


def save_detector(detector, option: str, upload: bool = True):
    """Save the regime detector for an option."""
    save_pickle(detector, "regime_detector", option, upload=upload)


def save_ranker(ranker, option: str, upload: bool = True):
    """Save the momentum ranker for an option."""
    save_pickle(ranker, "momentum_ranker", option, upload=upload)


def load_ranker(option: str, force_download: bool = False):
    """Load the momentum ranker for an option."""
    return load_pickle("momentum_ranker", option, force_download=force_download)


def save_feature_list(features: list, option: str, upload: bool = True):
    """Save the feature list for an option."""
    save_json({"features": features, "timestamp": str(datetime.now())}, "features", option, upload=upload)


def load_feature_list(option: str, force_download: bool = False) -> Optional[list]:
    """Load the feature list for an option."""
    data = load_json("features", option, force_download=force_download)
    return data.get("features") if data else None


def save_sweep_result(result: Dict, year: int, option: str, upload: bool = True):
    """Save a sweep result for a specific year."""
    save_json(result, f"sweep_{year}", option, upload=upload)


def load_sweep_result(year: int, option: str, force_download: bool = False) -> Optional[Dict]:
    """Load a sweep result for a specific year."""
    return load_json(f"sweep_{year}", option, force_download=force_download)


def list_available_data(option: str) -> Dict[str, bool]:
    """Check what data is available for an option."""
    return {
        "data": load_dataframe("data", option) is not None,
        "predictions": load_predictions(option) is not None,
        "wf_predictions": load_wf_predictions(option) is not None,
        "signals": load_signals(option) is not None,
        "detector": load_detector(option) is not None,
        "ranker": load_ranker(option) is not None,
        "features": load_feature_list(option) is not None,
    }
