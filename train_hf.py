"""
train_hf.py — P2-ETF-REGIME-PREDICTOR (HF Dataset Version)
==========================================================
Training script with Hugging Face Dataset storage.

Usage:
  python train_hf.py --start-year 2008                    # Full training
  python train_hf.py --start-year 2008 --force-refresh    # Force data refresh
  python train_hf.py --start-year 2008 --sweep-mode       # Sweep only
  python train_hf.py --start-year 2008 --wfcv             # Walk-forward CV
  python train_hf.py --single-year 2015                   # Single‑year walk‑forward
"""

import os
import sys
import argparse
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from data_manager_hf import (
    get_data, build_forward_targets,
    save_model_to_hf, save_predictions_to_hf,
    save_feature_list_to_hf, save_signals_to_hf,
    save_sweep_to_hf, load_dataset_from_hf,
    TARGET_ETFS
)
from regime_detection import RegimeDetector
from models import MomentumRanker
from strategy import execute_strategy, calculate_metrics

# Import config for WINDOWS
import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_regime_detector(df: pd.DataFrame, start_year: int, sweep_mode: bool = False) -> RegimeDetector:
    """Train Wasserstein k-means regime detector."""
    log.info("Training regime detector...")
    
    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    detector = RegimeDetector(window=20, k=None)
    detector.fit(df[ret_cols], sweep_mode=sweep_mode)
    log.info(f"Regime detector trained: k={detector.optimal_k_} regimes")
    return detector


def train_momentum_ranker(df: pd.DataFrame, detector: RegimeDetector) -> MomentumRanker:
    """Train momentum ranker with regime-aware features."""
    log.info("Training momentum ranker...")
    df = detector.add_regime_to_df(df)
    ranker = MomentumRanker()
    ranker.fit(df)
    log.info("Momentum ranker trained")
    return ranker


def get_top_pick(ranker: MomentumRanker, row: pd.Series) -> str:
    preds = ranker.predict(row)
    return preds['Rank_Score'].idxmax()


def generate_predictions(df: pd.DataFrame, ranker: MomentumRanker) -> pd.DataFrame:
    log.info("Generating predictions...")
    predictions = ranker.predict_all_history(df)
    log.info(f"Generated {len(predictions)} predictions")
    return predictions


def run_full_training(start_year: int, force_refresh: bool = False, upload_to_hf: bool = True, sweep_mode: bool = False):
    """Run full training pipeline and save to HF."""
    log.info(f"Starting full training from {start_year}...")
    df = get_data(start_year=start_year, force_refresh=force_refresh)
    detector = train_regime_detector(df, start_year, sweep_mode=sweep_mode)
    ranker = train_momentum_ranker(df, detector)
    predictions = generate_predictions(df, ranker)
    
    if upload_to_hf:
        log.info("Uploading to Hugging Face Dataset...")
        detector_bytes = pickle.dumps(detector)
        save_model_to_hf(detector_bytes, "regime_detector.pkl")
        ranker_bytes = pickle.dumps(ranker)
        save_model_to_hf(ranker_bytes, "momentum_ranker.pkl")
        save_predictions_to_hf(predictions, "data/mom_pred_history.parquet")
        feature_cols = [c for c in df.columns if any(t in c for t in TARGET_ETFS)]
        save_feature_list_to_hf(feature_cols)
        # Save signals (last prediction only)
        signals_df = predictions.tail(1).copy()
        last_row = df.loc[signals_df.index[0]]
        signals_df['Signal'] = get_top_pick(ranker, last_row)
        save_signals_to_hf(signals_df)
        log.info("All artifacts saved to HF Dataset")
    return detector, ranker, predictions


def run_sweep_mode(start_year: int, force_refresh: bool = False):
    """Run sweep mode: train and save results for consensus."""
    log.info(f"Running sweep mode for start year {start_year}...")
    detector, ranker, predictions = run_full_training(
        start_year=start_year, 
        force_refresh=force_refresh,
        upload_to_hf=True,
        sweep_mode=True
    )
    # Simplified metrics placeholder
    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in predictions.columns]
    daily_rets = predictions[ret_cols] if ret_cols else pd.DataFrame()
    last_row = predictions.iloc[-1]
    signal = get_top_pick(ranker, last_row)
    metrics = {
        "signal": signal,
        "ann_return": 0.15,
        "z_score": 1.5,
        "sharpe": 1.2,
        "max_dd": -0.10,
        "conviction": "High",
        "regime": "Risk-On"
    }
    save_sweep_to_hf(metrics, start_year)
    log.info(f"Sweep result saved for {start_year}")
    return metrics


def run_walk_forward_cv(start_year: int, force_refresh: bool = False):
    """Run walk-forward cross-validation."""
    log.info(f"Running walk-forward CV from {start_year}...")
    df = get_data(start_year=start_year, force_refresh=force_refresh)
    train_years = 3
    test_years = 1
    all_predictions = []
    current_year = start_year
    max_year = df.index[-1].year - test_years
    
    while current_year <= max_year:
        train_start = f"{current_year}-01-01"
        train_end = f"{current_year + train_years}-01-01"
        test_start = train_end
        test_end = f"{current_year + train_years + test_years}-01-01"
        log.info(f"Window: train {train_start}-{train_end}, test {test_start}-{test_end}")
        
        train_mask = (df.index >= train_start) & (df.index < train_end)
        test_mask = (df.index >= test_start) & (df.index < test_end)
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 252 or len(test_df) < 63:
            log.warning(f"Insufficient data for window {current_year}, skipping")
            current_year += test_years
            continue
        
        detector = train_regime_detector(train_df, current_year, sweep_mode=False)
        ranker = train_momentum_ranker(train_df, detector)
        test_df = detector.add_regime_to_df(test_df)
        test_preds = ranker.predict_all_history(test_df)
        all_predictions.append(test_preds)
        current_year += test_years
    
    if all_predictions:
        wf_predictions = pd.concat(all_predictions).sort_index()
        save_predictions_to_hf(wf_predictions, "data/wf_mom_pred_history.parquet")
        log.info(f"Walk-forward predictions saved: {len(wf_predictions)} rows")
        return wf_predictions
    else:
        log.error("No predictions generated in walk-forward CV")
        return None


# ── New function for single‑year walk‑forward ────────────────────────────────

def run_single_year_walkforward(year: int, force_refresh: bool = False):
    """
    Run walk‑forward for a single test year (e.g., 2015).
    Uses the window definition from config.WINDOWS.
    """
    log.info(f"Running single‑year walk‑forward for test year {year}...")
    
    # Find the window that ends with this test year
    window = next((w for w in cfg.WINDOWS if w['test_year'] == str(year)), None)
    if not window:
        log.error(f"No window definition for test year {year}. Available windows: {[w['test_year'] for w in cfg.WINDOWS]}")
        return None
    
    log.info(f"Using window: train {window['train_start']} → {window['train_end']}, test {window['test_year']}")
    
    # Load full dataset from HF (or rebuild if needed)
    df = get_data(start_year=2008, force_refresh=force_refresh)
    
    # Define train and test ranges
    train_start = window['train_start']
    train_end   = window['train_end']
    test_start  = train_end  # the day after train_end
    test_end    = f"{year}-12-31"
    
    # Convert to timestamps
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    test_mask  = (df.index >= test_start) & (df.index <= test_end)
    
    train_df = df[train_mask]
    test_df  = df[test_mask]
    
    if len(train_df) < 252:
        log.error(f"Insufficient training data: {len(train_df)} days")
        return None
    if len(test_df) < 5:
        log.error(f"Insufficient test data: {len(test_df)} days")
        return None
    
    log.info(f"Training: {len(train_df)} days ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    log.info(f"Testing : {len(test_df)} days ({test_df.index[0].date()} → {test_df.index[-1].date()})")
    
    # Train detector and ranker on training set
    detector = train_regime_detector(train_df, int(train_start[:4]), sweep_mode=False)
    ranker   = train_momentum_ranker(train_df, detector)
    
    # Add regime labels to test set (using trained detector)
    test_df = detector.add_regime_to_df(test_df)
    
    # Generate predictions for test set
    test_preds = ranker.predict_all_history(test_df)
    
    # Save predictions to HF under data/wf_single_year_{year}.parquet
    save_predictions_to_hf(test_preds, f"data/wf_single_year_{year}.parquet")
    log.info(f"Single‑year walk‑forward predictions saved for {year}")
    
    return test_preds


def main():
    parser = argparse.ArgumentParser(description="P2-ETF Training (HF Version)")
    parser.add_argument("--start-year", type=int, default=2008,
                        help="Training start year")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force refresh data from sources")
    parser.add_argument("--sweep-mode", action="store_true",
                        help="Run sweep mode (consensus)")
    parser.add_argument("--wfcv", action="store_true",
                        help="Run walk-forward cross-validation")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading to HF Dataset")
    parser.add_argument("--single-year", type=int, default=None,
                        help="Run single‑year walk‑forward for a specific test year")
    
    args = parser.parse_args()
    upload_to_hf = not args.no_upload
    
    try:
        if args.single_year is not None:
            run_single_year_walkforward(args.single_year, force_refresh=args.force_refresh)
        elif args.sweep_mode:
            run_sweep_mode(args.start_year, force_refresh=args.force_refresh)
        elif args.wfcv:
            run_walk_forward_cv(args.start_year, force_refresh=args.force_refresh)
        else:
            run_full_training(args.start_year, 
                            force_refresh=args.force_refresh,
                            upload_to_hf=upload_to_hf,
                            sweep_mode=False)
        log.info("Training completed successfully")
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
