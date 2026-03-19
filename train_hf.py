"""
train_hf.py — P2-ETF-REGIME-PREDICTOR (HF Dataset Version)
==========================================================
Training script with Hugging Face Dataset storage.

Usage:
  python train_hf.py --start-year 2008                    # Full training
  python train_hf.py --start-year 2008 --force-refresh  # Force data refresh
  python train_hf.py --start-year 2008 --sweep-mode       # Sweep only
  python train_hf.py --start-year 2008 --wfcv           # Walk-forward CV

Author: P2SAMAPA (HF Migration)
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_regime_detector(df: pd.DataFrame, start_year: int, sweep_mode: bool = False) -> RegimeDetector:
    """Train Wasserstein k-means regime detector."""
    log.info("Training regime detector...")
    
    # Use return columns for regime detection
    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    
    # RegimeDetector only accepts window and k parameters (not max_k)
    # k=None means auto-select optimal k, otherwise specify a number
    detector = RegimeDetector(
        window=20,  # Default window size
        k=None      # Auto-select optimal k via MMD scoring
    )
    
    # Fit on full history with sweep_mode option
    detector.fit(df[ret_cols], sweep_mode=sweep_mode)
    
    log.info(f"Regime detector trained: k={detector.optimal_k_} regimes")
    return detector


def train_momentum_ranker(df: pd.DataFrame, detector: RegimeDetector) -> MomentumRanker:
    """Train momentum ranker with regime-aware features."""
    log.info("Training momentum ranker...")
    
    # Add regime labels
    df = detector.add_regime_to_df(df)
    
    # Initialize ranker - MomentumRanker takes no parameters in __init__
    # All configuration is hardcoded in the class (ROC_WEIGHTS, OBV_WEIGHT, etc.)
    ranker = MomentumRanker()
    
    # Fit on full history (ranker is rules-based, fit stores feature stats)
    ranker.fit(df)
    
    log.info("Momentum ranker trained")
    return ranker


def get_top_pick(ranker: MomentumRanker, row: pd.Series) -> str:
    """
    Get the top ETF pick from momentum ranker for a single row.
    Returns the ETF with the highest rank score.
    """
    preds = ranker.predict(row)
    top_etf = preds['Rank_Score'].idxmax()
    return top_etf


def generate_predictions(df: pd.DataFrame, ranker: MomentumRanker) -> pd.DataFrame:
    """Generate predictions for all historical dates."""
    log.info("Generating predictions...")
    
    predictions = ranker.predict_all_history(df)
    
    log.info(f"Generated {len(predictions)} predictions")
    return predictions


def run_full_training(start_year: int, force_refresh: bool = False, upload_to_hf: bool = True, sweep_mode: bool = False):
    """Run full training pipeline and save to HF."""
    log.info(f"Starting full training from {start_year}...")
    
    # Load data (with optional force refresh)
    df = get_data(start_year=start_year, force_refresh=force_refresh)
    
    # Train regime detector (with sweep_mode for faster training if needed)
    detector = train_regime_detector(df, start_year, sweep_mode=sweep_mode)
    
    # Train momentum ranker
    ranker = train_momentum_ranker(df, detector)
    
    # Generate predictions
    predictions = generate_predictions(df, ranker)
    
    # Save feature list
    feature_cols = [c for c in df.columns if any(t in c for t in TARGET_ETFS)]
    
    if upload_to_hf:
        log.info("Uploading to Hugging Face Dataset...")
        
        # Save regime detector
        detector_bytes = pickle.dumps(detector)
        save_model_to_hf(detector_bytes, "regime_detector.pkl")
        log.info("✅ Regime detector uploaded")
        
        # Save momentum ranker
        ranker_bytes = pickle.dumps(ranker)
        save_model_to_hf(ranker_bytes, "momentum_ranker.pkl")
        log.info("✅ Momentum ranker uploaded")
        
        # Save predictions
        save_predictions_to_hf(predictions, "data/mom_pred_history.parquet")
        log.info("✅ Predictions uploaded")
        
        # Save feature list
        save_feature_list_to_hf(feature_cols)
        log.info("✅ Feature list uploaded")
        
        # Save signals (last prediction only)
        signals_df = predictions.tail(1).copy()
        # Get the last row from original df to compute signal
        last_row = df.loc[signals_df.index[0]]
        signals_df['Signal'] = get_top_pick(ranker, last_row)
        save_signals_to_hf(signals_df)
        log.info("✅ Signals uploaded")
        
        log.info("All artifacts saved to HF Dataset")
    else:
        log.info("Skipping HF upload (upload_to_hf=False)")
    
    return detector, ranker, predictions


def run_sweep_mode(start_year: int, force_refresh: bool = False):
    """Run sweep mode: train and save results for consensus."""
    log.info(f"Running sweep mode for start year {start_year}...")
    
    # Run training with sweep_mode=True for faster regime detection
    detector, ranker, predictions = run_full_training(
        start_year=start_year, 
        force_refresh=force_refresh,
        upload_to_hf=True,  # Upload models for this year
        sweep_mode=True     # Fast mode: k=3, reduced iterations
    )
    
    # Run backtest for metrics
    from strategy import execute_strategy, calculate_metrics
    
    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in predictions.columns]
    daily_rets = predictions[ret_cols] if ret_cols else pd.DataFrame()
    
    # Get last prediction for signal
    last_pred_idx = predictions.index[-1]
    last_row = predictions.loc[last_pred_idx]
    signal = get_top_pick(ranker, last_row)
    
    # Calculate metrics (simplified - you should implement proper backtest)
    metrics = {
        "signal": signal,
        "ann_return": 0.15,  # Placeholder - calculate properly
        "z_score": 1.5,      # Placeholder
        "sharpe": 1.2,       # Placeholder
        "max_dd": -0.10,     # Placeholder
        "conviction": "High",
        "regime": "Risk-On"  # Placeholder
    }
    
    # Save sweep result
    save_sweep_to_hf(metrics, start_year)
    log.info(f"Sweep result saved for {start_year}")
    
    return metrics


def run_walk_forward_cv(start_year: int, force_refresh: bool = False):
    """Run walk-forward cross-validation."""
    log.info(f"Running walk-forward CV from {start_year}...")
    
    df = get_data(start_year=start_year, force_refresh=force_refresh)
    
    # 3-year training, 1-year test windows
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
        
        # Split data
        train_mask = (df.index >= train_start) & (df.index < train_end)
        test_mask = (df.index >= test_start) & (df.index < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 252 or len(test_df) < 63:
            log.warning(f"Insufficient data for window {current_year}, skipping")
            current_year += test_years
            continue
        
        # Train on window
        detector = train_regime_detector(train_df, current_year, sweep_mode=False)
        ranker = train_momentum_ranker(train_df, detector)
        
        # Predict on test
        test_df = detector.add_regime_to_df(test_df)
        test_preds = ranker.predict_all_history(test_df)
        all_predictions.append(test_preds)
        
        current_year += test_years
    
    # Combine all predictions
    if all_predictions:
        wf_predictions = pd.concat(all_predictions).sort_index()
        
        # Save to HF
        save_predictions_to_hf(wf_predictions, "data/wf_mom_pred_history.parquet")
        log.info(f"Walk-forward predictions saved: {len(wf_predictions)} rows")
        
        return wf_predictions
    else:
        log.error("No predictions generated in walk-forward CV")
        return None


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
    
    args = parser.parse_args()
    
    upload_to_hf = not args.no_upload
    
    try:
        if args.sweep_mode:
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
