"""
train_hf.py — P2-ETF-REGIME-PREDICTOR v2
=========================================
Training pipeline for Option A (FI/Commodities) and Option B (Equity ETFs).

Usage:
  python train_hf.py --option a                        # Full train Option A
  python train_hf.py --option b                        # Full train Option B
  python train_hf.py --option a --force-refresh        # Force data rebuild
  python train_hf.py --option a --wfcv                 # Walk-forward CV (incremental)
  python train_hf.py --option a --sweep                # Consensus sweep
  python train_hf.py --option a --sweep-year 2015      # Sweep for one year
  python train_hf.py --option a --single-year 2015     # Single-year WF test
"""

import os
import sys
import argparse
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

import config as cfg
from data_manager_hf import (
    get_data, build_forward_targets,
    load_dataset, save_dataset,
    load_predictions, save_predictions,
    load_wf_predictions, save_wf_predictions,
    load_signals, save_signals,
    save_detector, save_ranker,
    save_feature_list, save_sweep_result,
)
from regime_detection import RegimeDetector
from models import MomentumRanker
from strategy import execute_strategy, calculate_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _target_etfs(option: str) -> list:
    return cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS


def _label(option: str) -> str:
    return f"Option {option.upper()}"


# ── Core training steps ───────────────────────────────────────────────────────

def train_regime_detector(df: pd.DataFrame, option: str,
                           sweep_mode: bool = False) -> RegimeDetector:
    log.info(f"{_label(option)}: training regime detector...")
    ret_cols = [f"{t}_Ret" for t in _target_etfs(option) if f"{t}_Ret" in df.columns]
    detector = RegimeDetector(window=20, k=None)
    detector.fit(df[ret_cols], sweep_mode=sweep_mode)
    log.info(f"{_label(option)}: regime detector trained — k={detector.optimal_k_} regimes")
    return detector


def train_momentum_ranker(df: pd.DataFrame, detector: RegimeDetector,
                           option: str) -> MomentumRanker:
    log.info(f"{_label(option)}: training momentum ranker...")
    df = detector.add_regime_to_df(df)
    ranker = MomentumRanker()
    ranker.fit(df)
    log.info(f"{_label(option)}: momentum ranker trained")
    return ranker


def generate_predictions(df: pd.DataFrame, ranker: MomentumRanker,
                          option: str) -> pd.DataFrame:
    log.info(f"{_label(option)}: generating predictions...")
    predictions = ranker.predict_all_history(df)
    log.info(f"{_label(option)}: {len(predictions)} predictions generated")
    return predictions


def get_top_pick(ranker: MomentumRanker, row: pd.Series) -> str:
    preds = ranker.predict(row)
    return preds["Rank_Score"].idxmax()


# ── Full training pipeline ────────────────────────────────────────────────────

def run_full_training(option: str, force_refresh: bool = False,
                      sweep_mode: bool = False) -> tuple:
    """
    Full train for a single option. Uploads all artefacts to HF.
    Returns (detector, ranker, predictions).
    """
    log.info(f"{'='*60}")
    log.info(f"{_label(option)}: full training pipeline")
    log.info(f"{'='*60}")

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)
    detector    = train_regime_detector(df, option, sweep_mode=sweep_mode)
    ranker      = train_momentum_ranker(df, detector, option)
    predictions = generate_predictions(df, ranker, option)

    log.info(f"{_label(option)}: uploading artefacts to HF...")
    save_detector(pickle.dumps(detector), option)
    save_ranker(pickle.dumps(ranker), option)
    save_predictions(predictions, option)

    feature_cols = [c for c in df.columns
                    if any(t in c for t in _target_etfs(option))]
    save_feature_list(feature_cols, option)

    # Daily signal — last row
    last_row    = df.loc[predictions.index[-1]]
    top_pick    = get_top_pick(ranker, last_row)
    signals_row = predictions.tail(1).copy()
    signals_row["Signal"] = top_pick
    save_signals(signals_row, option)

    log.info(f"{_label(option)}: ✅ full training complete. Today's signal: {top_pick}")
    return detector, ranker, predictions


# ── Walk-forward CV (incremental) ─────────────────────────────────────────────

def run_walk_forward_cv(option: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Incremental walk-forward CV. Checks which folds already exist in HF
    and only computes missing/new folds. Saves incrementally.
    """
    log.info(f"{_label(option)}: incremental walk-forward CV...")
    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    # Load existing WF predictions to determine already-computed folds
    existing_wf = load_wf_predictions(option, force_download=True)
    completed_years = set()
    if existing_wf is not None and not existing_wf.empty:
        completed_years = set(existing_wf.index.year.unique().tolist())
        log.info(f"{_label(option)}: WF folds already computed: {sorted(completed_years)}")

    new_folds = []
    for window in cfg.WINDOWS:
        test_year = int(window["test_year"])

        # Skip folds already computed
        if test_year in completed_years:
            log.info(f"{_label(option)}: WF fold {test_year} already exists — skipping")
            continue

        train_mask = ((df.index >= window["train_start"]) &
                      (df.index <= window["train_end"]))
        test_mask  = ((df.index >= window["train_end"]) &
                      (df.index <= f"{test_year}-12-31"))
        train_df = df[train_mask]
        test_df  = df[test_mask]

        if len(train_df) < 252 or len(test_df) < 5:
            log.warning(f"{_label(option)}: insufficient data for WF fold {test_year} — skipping")
            continue

        log.info(f"{_label(option)}: WF fold {test_year} — "
                 f"train {train_df.index[0].date()}→{train_df.index[-1].date()}, "
                 f"test {test_df.index[0].date()}→{test_df.index[-1].date()}")

        detector = train_regime_detector(train_df, option)
        ranker   = train_momentum_ranker(train_df, detector, option)
        test_df  = detector.add_regime_to_df(test_df)
        fold_preds = ranker.predict_all_history(test_df)
        new_folds.append(fold_preds)

    if not new_folds:
        log.info(f"{_label(option)}: no new WF folds to compute.")
        return existing_wf

    new_preds = pd.concat(new_folds).sort_index()

    # Merge with existing
    if existing_wf is not None and not existing_wf.empty:
        merged = (pd.concat([existing_wf, new_preds])
                  .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                  .sort_index())
    else:
        merged = new_preds

    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: ✅ WF CV complete — {len(merged)} total OOS rows")
    return merged


# ── Single-year walk-forward ──────────────────────────────────────────────────

def run_single_year(year: int, option: str,
                    force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Run walk-forward for a single test year and save to HF."""
    window = next((w for w in cfg.WINDOWS if w["test_year"] == str(year)), None)
    if not window:
        log.error(f"{_label(option)}: no window for test year {year}")
        return None

    log.info(f"{_label(option)}: single-year WF for {year}...")
    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    train_mask = ((df.index >= window["train_start"]) &
                  (df.index <= window["train_end"]))
    test_mask  = ((df.index >= window["train_end"]) &
                  (df.index <= f"{year}-12-31"))
    train_df, test_df = df[train_mask], df[test_mask]

    if len(train_df) < 252 or len(test_df) < 5:
        log.error(f"{_label(option)}: insufficient data for year {year}")
        return None

    detector   = train_regime_detector(train_df, option)
    ranker     = train_momentum_ranker(train_df, detector, option)
    test_df    = detector.add_regime_to_df(test_df)
    test_preds = ranker.predict_all_history(test_df)

    # Merge into existing WF predictions
    existing_wf = load_wf_predictions(option, force_download=True)
    if existing_wf is not None and not existing_wf.empty:
        merged = (pd.concat([existing_wf, test_preds])
                  .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                  .sort_index())
    else:
        merged = test_preds

    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: ✅ single-year WF {year} saved ({len(test_preds)} rows)")
    return test_preds


# ── Consensus sweep ───────────────────────────────────────────────────────────

def run_sweep(option: str, years: Optional[list] = None,
              force_refresh: bool = False) -> None:
    """Run full training for each sweep year and save consensus metrics."""
    years = years or cfg.SWEEP_YEARS
    log.info(f"{_label(option)}: running consensus sweep for years {years}...")
    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    for year in years:
        window = next((w for w in cfg.WINDOWS if w["test_year"] == str(year)), None)
        if not window:
            log.warning(f"{_label(option)}: no window for sweep year {year} — skipping")
            continue

        log.info(f"{_label(option)}: sweep year {year}...")
        train_mask = ((df.index >= window["train_start"]) &
                      (df.index <= window["train_end"]))
        train_df = df[train_mask]

        if len(train_df) < 252:
            log.warning(f"{_label(option)}: insufficient training data for {year} — skipping")
            continue

        detector   = train_regime_detector(train_df, option, sweep_mode=True)
        ranker     = train_momentum_ranker(train_df, detector, option)
        predictions = generate_predictions(train_df, ranker, option)

        # Strategy metrics on the training set (sweep uses in-sample for signal ranking)
        ret_cols = [f"{t}_Ret" for t in _target_etfs(option) if f"{t}_Ret" in train_df.columns]
        rf_rate  = (float(train_df["DTB3"].iloc[-1] / 100)
                    if "DTB3" in train_df.columns else cfg.RISK_FREE_RATE)

        try:
            (strat_rets, _, _, next_signal, conviction_z,
             conviction_label, last_p) = execute_strategy(
                predictions_df=predictions,
                daily_ret_df=train_df[ret_cols],
                rf_rate=rf_rate,
                z_reentry=cfg.Z_REENTRY,
                stop_loss_pct=cfg.STOP_LOSS_PCT,
                fee_bps=cfg.TRANSACTION_BPS,
                regime_series=train_df.get("Regime_Name",
                                           pd.Series("Unknown", index=train_df.index)),
            )
            metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
        except Exception as e:
            log.warning(f"{_label(option)}: strategy failed for sweep year {year}: {e}")
            metrics = {}
            next_signal = "CASH"
            conviction_z = 0.0
            conviction_label = "Low"

        regime_name = (train_df["Regime_Name"].iloc[-1]
                       if "Regime_Name" in train_df.columns else "Unknown")

        result = {
            "signal":     next_signal,
            "ann_return": round(metrics.get("ann_return", 0.0), 4),
            "z_score":    round(float(conviction_z), 3),
            "sharpe":     round(metrics.get("sharpe", 0.0), 3),
            "max_dd":     round(metrics.get("max_dd", 0.0), 4),
            "conviction": conviction_label,
            "regime":     str(regime_name),
        }
        save_sweep_result(result, year, option)
        log.info(f"{_label(option)}: sweep {year} — signal={next_signal}, "
                 f"return={result['ann_return']:.1%}, sharpe={result['sharpe']:.2f}")

    log.info(f"{_label(option)}: ✅ sweep complete")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P2-ETF-REGIME-PREDICTOR v2 training pipeline"
    )
    parser.add_argument(
        "--option", required=True, choices=["a", "b"],
        help="Which option to train: a (FI/Commodities) or b (Equity ETFs)"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Force full dataset rebuild from source APIs"
    )
    parser.add_argument(
        "--wfcv", action="store_true",
        help="Run incremental walk-forward CV"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run consensus sweep across all sweep years"
    )
    parser.add_argument(
        "--sweep-year", type=int, default=None,
        help="Run sweep for a single year only"
    )
    parser.add_argument(
        "--single-year", type=int, default=None,
        help="Run single-year walk-forward for a specific test year"
    )

    args   = parser.parse_args()
    option = args.option.lower()

    try:
        if args.single_year is not None:
            run_single_year(args.single_year, option,
                            force_refresh=args.force_refresh)
        elif args.sweep_year is not None:
            run_sweep(option, years=[args.sweep_year],
                      force_refresh=args.force_refresh)
        elif args.sweep:
            run_sweep(option, force_refresh=args.force_refresh)
        elif args.wfcv:
            run_walk_forward_cv(option, force_refresh=args.force_refresh)
        else:
            run_full_training(option, force_refresh=args.force_refresh)

        log.info(f"Option {option.upper()}: pipeline completed successfully")

    except Exception as e:
        log.error(f"Option {option.upper()}: pipeline failed — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Fix missing Optional import
from typing import Optional

if __name__ == "__main__":
    main()
