"""
train.py — P2-ETF-REGIME-PREDICTOR
=====================================
Daily training pipeline orchestrator.
Called by GitHub Actions at 6:30am EST on weekdays.

Pipeline steps:
  1. Load existing dataset from GitLab
  2. Incrementally fetch new data (yesterday's close)
  3. Run Wasserstein k-means regime detection
  4. Build forward return targets
  5. Train LightGBM + LogReg model bank (per regime)
  6. Run full backtest to generate prediction history
  7. Generate next-day signal
  8. Save dataset, models, signals to GitLab

Author: P2SAMAPA
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Imports ───────────────────────────────────────────────────────────────────
from data_manager import (
    save_predictions_to_gitlab,
    get_data, build_full_dataset, incremental_update,
    build_forward_targets, load_dataset_from_gitlab,
    save_dataset_to_gitlab, save_model_to_gitlab,
    save_signals_to_gitlab, save_feature_list_to_gitlab,
    load_model_from_gitlab, TARGET_ETFS,
)
from regime_detection import RegimeDetector
from models import RegimeModelBank, MomentumRanker, get_feature_columns
from strategy import (
    execute_strategy, calculate_metrics,
    build_signal_row, TARGET_ETFS as STRAT_ETFS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
START_YEAR    = 2008
FORWARD_DAYS  = 5
VAL_PCT       = 0.15
TRAIN_PCT     = 0.70


def run_pipeline(force_refresh: bool = False,
                 skip_gitlab_write: bool = False,
                 run_wfcv: bool = False) -> dict:
    """
    Full daily training pipeline.

    Parameters
    ----------
    force_refresh     : rebuild dataset from scratch (ignore GitLab cache)
    skip_gitlab_write : run locally without pushing to GitLab
    run_wfcv          : run walk-forward CV (slower, for diagnostics)

    Returns dict with pipeline results summary.
    """
    results = {}
    log.info("=" * 60)
    log.info("P2-ETF-REGIME-PREDICTOR — Daily Training Pipeline")
    log.info(f"Started: {datetime.utcnow().isoformat()}Z")
    log.info("=" * 60)

    # ── Step 1: Load / build dataset ──────────────────────────────────────────
    log.info("Step 1: Loading dataset...")

    if force_refresh:
        log.info("Force refresh — rebuilding from scratch")
        df = build_full_dataset(start_year=START_YEAR)
    else:
        existing = load_dataset_from_gitlab()
        if existing is not None:
            log.info(f"Loaded from GitLab: {len(existing)} rows")
            df = incremental_update(existing)
        else:
            log.info("No GitLab dataset found — building from scratch")
            df = build_full_dataset(start_year=START_YEAR)

    log.info(f"Dataset: {len(df)} rows × {df.shape[1]} cols "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    results["dataset_rows"] = len(df)
    results["dataset_cols"] = df.shape[1]
    results["date_range"]   = f"{df.index[0].date()} → {df.index[-1].date()}"

    # ── Step 2: Regime detection ───────────────────────────────────────────────
    log.info("Step 2: Wasserstein k-means regime detection...")

    detector = RegimeDetector()
    try:
        detector.fit(df)
        df = detector.add_regime_to_df(df)
        log.info(detector.summary())
        results["optimal_k"]    = detector.optimal_k_
        results["regime_names"] = detector.regime_names_
    except Exception as e:
        log.error(f"Regime detection failed: {e}")
        log.info("Falling back to single global regime (0)")
        df["Regime"]      = 0
        df["Regime_Name"] = "Global"
        results["optimal_k"] = 1

    # Fill any NaN regime values (first ~20 rows before first window)
    df["Regime"]      = df["Regime"].fillna(0).astype(int)
    df["Regime_Name"] = df["Regime_Name"].fillna("Global")

    # ── Step 3: Build forward targets ─────────────────────────────────────────
    log.info("Step 3: Building forward return targets...")

    # Get current risk-free rate
    rf_rate = 0.045
    if "DTB3" in df.columns:
        rf_rate = float(df["DTB3"].dropna().iloc[-1]) / 100
    log.info(f"Risk-free rate: {rf_rate*100:.2f}%")

    fwd_df = build_forward_targets(df, forward_days=FORWARD_DAYS)
    log.info(f"Forward targets: {fwd_df.shape[1]} columns, "
             f"{fwd_df.dropna().shape[0]} valid rows")
    results["rf_rate"] = rf_rate

    # ── Step 4: Train model bank ───────────────────────────────────────────────
    log.info("Step 4: Training LightGBM + LogReg model bank...")

    # Align df and fwd_df — drop rows where targets are all NaN (last 5 rows)
    common_idx = df.index.intersection(fwd_df.dropna(how="all").index)
    df_train   = df.loc[common_idx]
    fwd_train  = fwd_df.loc[common_idx]

    feature_cols = get_feature_columns(df_train)
    log.info(f"Features: {len(feature_cols)}")

    # Clean NaNs from feature matrix — fill with column median
    df_train[feature_cols] = df_train[feature_cols].fillna(
        df_train[feature_cols].median()
    ).fillna(0.0)
    log.info("NaN cleaning applied to feature matrix")

    bank = RegimeModelBank()
    try:
        bank.fit(df_train, fwd_train,
                 feature_cols=feature_cols,
                 val_pct=VAL_PCT)
        results["n_models"]  = len(bank.models_)
        log.info(f"Trained {len(bank.models_)} regime-specific ranking models "
                 f"+ 1 global fallback")
    except Exception as e:
        log.error(f"Model training failed: {e}")
        raise

    # ── Step 5: Generate prediction history (for backtest) ────────────────────
    log.info("Step 5: Generating prediction history...")

    # Ensure Regime column is clean integer before prediction
    df["Regime"] = df["Regime"].fillna(0).astype(int)
    df[feature_cols] = df[feature_cols].fillna(
        df[feature_cols].median()
    ).fillna(0.0)

    try:
        pred_history = bank.predict_all_history(df)
        log.info(f"Prediction history: {len(pred_history)} rows")
        results["pred_history_rows"] = len(pred_history)
    except Exception as e:
        log.error(f"Prediction history failed: {e}")
        raise

    # ── Step 6: Run backtest ───────────────────────────────────────────────────
    log.info("Step 6: Running backtest...")

    # Daily return columns for strategy execution
    ret_cols   = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    # Also include benchmarks for context
    for b in ["SPY_Ret", "AGG_Ret"]:
        if b in df.columns:
            ret_cols.append(b)
    daily_rets = df[ret_cols]

    regime_series = df["Regime_Name"] if "Regime_Name" in df.columns else None

    try:
        (strat_rets, audit_trail, next_date, next_signal,
         conviction_z, conviction_label, last_p) = execute_strategy(
            predictions_df = pred_history,
            daily_ret_df   = daily_rets,
            rf_rate        = rf_rate,
            regime_series  = df["Regime_Name"] if "Regime_Name" in df.columns else None,
        )

        metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
        log.info(f"Backtest: Ann Return={metrics.get('ann_return',0)*100:.1f}% "
                 f"Sharpe={metrics.get('sharpe',0):.2f} "
                 f"MaxDD={metrics.get('max_dd',0)*100:.1f}%")
        results["ann_return"] = round(metrics.get("ann_return", 0) * 100, 2)
        results["sharpe"]     = round(metrics.get("sharpe", 0), 3)
        results["max_dd"]     = round(metrics.get("max_dd", 0) * 100, 2)
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        raise

    # ── Step 7: Generate next-day signal ──────────────────────────────────────
    log.info(f"Step 7: Next trading day signal...")

    # Get current regime
    regime_int, regime_name = 0, "Unknown"
    try:
        regime_int, regime_name = detector.get_current_regime(df)
    except Exception:
        pass

    signal_row = build_signal_row(
        next_date       = next_date,
        next_signal     = next_signal,
        conviction_z    = conviction_z,
        conviction_label= conviction_label,
        p_array         = last_p,
        regime_int      = regime_int,
        regime_name     = regime_name,
        metrics         = metrics,
    )

    log.info(f"Signal: {next_signal} on {next_date.date()} "
             f"(Z={conviction_z:.2f}, {conviction_label}) "
             f"Regime: {regime_name}")
    results["next_signal"]     = next_signal
    results["next_date"]       = str(next_date.date())
    results["conviction_z"]    = round(conviction_z, 3)
    results["conviction_label"]= conviction_label
    results["regime_name"]     = regime_name

    # ── Step 8: Save to GitLab ─────────────────────────────────────────────────
    if not skip_gitlab_write:
        log.info("Step 8: Saving to GitLab...")

        # Prediction history (saves heavy compute from Streamlit)
        ok = save_predictions_to_gitlab(pred_history)
        log.info(f"  Prediction history saved: {ok}")

        # Dataset
        ok = save_dataset_to_gitlab(df)
        log.info(f"  Dataset saved: {ok}")

        # Feature list
        ok = save_feature_list_to_gitlab(feature_cols)
        log.info(f"  Feature list saved: {ok}")

        # Regime detector
        ok = save_model_to_gitlab(detector.to_bytes(), "regime_detector.pkl")
        log.info(f"  Regime detector saved: {ok}")

        # Model bank
        ok = save_model_to_gitlab(bank.to_bytes(), "model_bank.pkl")
        log.info(f"  Model bank saved: {ok}")

        # ── Layer 2B: Momentum ranker ────────────────────────────────────
        log.info("Step 5b: Training Layer 2B momentum ranker...")
        try:
            momentum_ranker = MomentumRanker()
            momentum_ranker.fit(df)
            ok = save_model_to_gitlab(momentum_ranker.to_bytes(), "momentum_ranker.pkl")
            log.info(f"  Momentum ranker saved: {ok}")
            mom_pred = momentum_ranker.predict_all_history(df)
            ok = save_predictions_to_gitlab(mom_pred, "data/mom_pred_history.csv")
            log.info(f"  Momentum predictions saved: {ok}")
        except Exception as e:
            log.error(f"  Momentum ranker failed: {e}")


        # Signals
        ok = save_signals_to_gitlab(signal_row)
        log.info(f"  Signal saved: {ok}")

        results["gitlab_saved"] = True
    else:
        log.info("Step 8: Skipping GitLab write (--local flag set)")
        results["gitlab_saved"] = False

    # ── Optional: Walk-forward CV ──────────────────────────────────────────────
    if run_wfcv:
        log.info("Running walk-forward cross-validation (diagnostic)...")
        from models import walk_forward_cv
        try:
            wfcv_results = walk_forward_cv(df_train, fwd_train, n_splits=5)
            log.info(f"\nWalk-forward CV results:\n{wfcv_results.to_string()}")
            results["wfcv"] = wfcv_results.to_dict()
        except Exception as e:
            log.error(f"Walk-forward CV failed: {e}")

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Next signal: {next_signal} ({conviction_label}, Z={conviction_z:.2f})")
    log.info(f"  Regime: {regime_name}")
    log.info(f"  Ann Return: {results.get('ann_return','?')}%")
    log.info(f"  Sharpe: {results.get('sharpe','?')}")
    log.info("=" * 60)

    return results


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P2-ETF-REGIME-PREDICTOR Training Pipeline"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Rebuild dataset from scratch (ignore GitLab cache)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run without writing to GitLab (for local testing)"
    )
    parser.add_argument(
        "--wfcv", action="store_true",
        help="Run walk-forward cross-validation (diagnostic, slower)"
    )
    args = parser.parse_args()

    results = run_pipeline(
        force_refresh     = args.force_refresh,
        skip_gitlab_write = args.local,
        run_wfcv          = args.wfcv,
    )
    sys.exit(0)
