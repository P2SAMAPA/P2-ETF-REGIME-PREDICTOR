"""
train_hf.py — P2-ETF-REGIME-PREDICTOR (Hugging Face Version)
=============================================================
Daily training pipeline orchestrator.
Called by GitHub Actions at 6:30am EST on weekdays.

HF Dataset Integration:
  - Loads/saves to P2SAMAPA/p2-etf-regime-predictor
  - Uses Parquet format for data (faster, smaller)
  - Binary pickle files for models

Author: P2SAMAPA (HF Migration)
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# HF VERSION: Import from data_manager_hf
from data_manager_hf import (
    save_predictions_to_hf, save_predictions_to_gitlab,  # backward compat
    get_data, build_full_dataset, incremental_update,
    build_forward_targets, load_dataset_from_hf,
    save_dataset_to_hf, save_model_to_hf,
    save_signals_to_hf, save_feature_list_to_hf,
    load_model_from_hf, save_sweep_to_hf, TARGET_ETFS,
    HF_REPO_ID,
)
from regime_detection import RegimeDetector
from models import MomentumRanker, get_feature_columns
from strategy import (
    execute_strategy, calculate_metrics,
    build_signal_row, compute_sweep_z,
    TARGET_ETFS as STRAT_ETFS,
)

DEFAULT_START_YEAR = 2008
FORWARD_DAYS       = 5
VAL_PCT            = 0.15
TRAIN_PCT          = 0.70


def run_pipeline(force_refresh: bool = False,
                 skip_hf_write: bool = False,
                 run_wfcv: bool = False,
                 sweep_mode: bool = False,
                 start_year: int = None) -> dict:
    results = {}
    effective_start_year = start_year if start_year else DEFAULT_START_YEAR

    log.info("=" * 60)
    log.info("P2-ETF-REGIME-PREDICTOR — Daily Training Pipeline (HF Dataset)")
    log.info(f"Started:    {datetime.utcnow().isoformat()}Z")
    log.info(f"Start year: {effective_start_year}")
    log.info(f"Sweep mode: {sweep_mode}")
    log.info(f"HF Repo:    {HF_REPO_ID}")
    log.info("=" * 60)

    # ── Step 1: Load / build dataset ──────────────────────────────────────────
    log.info("Step 1: Loading dataset from HF...")
    if sweep_mode:
        existing = load_dataset_from_hf()
        df = existing if existing is not None else build_full_dataset(
            start_year=DEFAULT_START_YEAR)
    elif force_refresh:
        df = build_full_dataset(start_year=DEFAULT_START_YEAR)
    else:
        existing = load_dataset_from_hf()
        df = (incremental_update(existing) if existing is not None
              else build_full_dataset(start_year=DEFAULT_START_YEAR))

    log.info(f"Dataset: {len(df)} rows × {df.shape[1]} cols "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    results.update({"dataset_rows": len(df), "dataset_cols": df.shape[1],
                    "date_range": f"{df.index[0].date()} → {df.index[-1].date()}"})

    # ── Step 2: Regime detection ───────────────────────────────────────────────
    log.info("Step 2: Wasserstein k-means regime detection...")
    detector = None

    if sweep_mode:
        # SWEEP OPTIMISATION: reuse saved detector
        log.info("  Sweep mode: attempting to load saved detector from HF...")
        try:
            det_bytes = load_model_from_hf("regime_detector.pkl")
            if det_bytes:
                detector = RegimeDetector.from_bytes(det_bytes)
                log.info(f"  ✅ Loaded saved detector "
                         f"(k={detector.optimal_k_}, "
                         f"regimes={list(detector.regime_names_.values())})")
            else:
                log.warning("  No saved detector found — will refit with fast path")
        except Exception as e:
            log.warning(f"  Could not load saved detector: {e} — will refit")

    if detector is None:
        detector = RegimeDetector()
        try:
            detector.fit(df, sweep_mode=sweep_mode)
            log.info(detector.summary())
            results["optimal_k"]    = detector.optimal_k_
            results["regime_names"] = detector.regime_names_
        except Exception as e:
            log.error(f"Regime detection failed: {e}")
            df["Regime"]      = 0
            df["Regime_Name"] = "Global"
            results["optimal_k"] = 1
    else:
        results["optimal_k"]    = detector.optimal_k_
        results["regime_names"] = detector.regime_names_

    # Drop existing Regime columns before adding new ones
    if "Regime" in df.columns:
        df = df.drop(columns=["Regime", "Regime_Name"])
        log.info("  Dropped existing Regime columns before re-detection")

    try:
        df = detector.add_regime_to_df(df)
    except Exception as e:
        log.error(f"add_regime_to_df failed: {e}")
        df["Regime"]      = 0
        df["Regime_Name"] = "Global"

    df["Regime"]      = df["Regime"].fillna(0).astype(int)
    df["Regime_Name"] = df["Regime_Name"].fillna("Global")

    # ── Step 3: Build forward targets ─────────────────────────────────────────
    log.info("Step 3: Building forward return targets...")
    rf_rate = 0.045
    if "DTB3" in df.columns:
        rf_rate = float(df["DTB3"].dropna().iloc[-1]) / 100
    log.info(f"Risk-free rate: {rf_rate*100:.2f}%")
    fwd_df = build_forward_targets(df, forward_days=FORWARD_DAYS)
    results["rf_rate"] = rf_rate

    # ── Step 4: Train model bank ───────────────────────────────────────────────
    log.info("Step 4: Training LightGBM + LogReg model bank...")
    from models import MomentumRanker, RegimeModelBank
    bank         = RegimeModelBank()
    common_idx   = df.index.intersection(fwd_df.dropna(how="all").index)
    df_train     = df.loc[common_idx]
    fwd_train    = fwd_df.loc[common_idx]
    feature_cols = get_feature_columns(df_train)
    df_train[feature_cols] = (df_train[feature_cols]
                              .fillna(df_train[feature_cols].median())
                              .fillna(0.0))
    try:
        bank.fit(df_train, fwd_train, feature_cols=feature_cols,
                 val_pct=VAL_PCT)
        results["n_models"] = len(bank.models_)
    except Exception as e:
        log.error(f"Model training failed: {e}")
        raise

    # ── Step 5: Generate prediction history ───────────────────────────────────
    log.info("Step 5: Generating prediction history...")
    df["Regime"] = df["Regime"].fillna(0).astype(int)
    df[feature_cols] = (df[feature_cols]
                        .fillna(df[feature_cols].median())
                        .fillna(0.0))
    try:
        pred_history = bank.predict_all_history(df)
        results["pred_history_rows"] = len(pred_history)
    except Exception as e:
        log.error(f"Prediction history failed: {e}")
        raise

    # ── Step 5b: Momentum ranker ───────────────────────────────────────────────
    log.info("Step 5b: Training Layer 2B momentum ranker...")
    momentum_ranker = None
    mom_pred        = None

    if sweep_mode:
        # SWEEP OPTIMISATION: reuse saved momentum ranker
        log.info("  Sweep mode: attempting to load saved momentum ranker from HF...")
        try:
            from data_manager_hf import load_model_from_hf
            mom_bytes = load_model_from_hf("momentum_ranker.pkl")
            if mom_bytes:
                momentum_ranker = MomentumRanker.from_bytes(mom_bytes)
                mom_pred = momentum_ranker.predict_all_history(df)
                log.info(f"  ✅ Loaded saved momentum ranker, "
                         f"generated {len(mom_pred)} prediction rows")
            else:
                log.warning("  No saved ranker found — will refit")
        except Exception as e:
            log.warning(f"  Could not load saved ranker: {e} — will refit")

    if momentum_ranker is None:
        try:
            momentum_ranker = MomentumRanker()
            momentum_ranker.fit(df)
            mom_pred = momentum_ranker.predict_all_history(df)
        except Exception as e:
            log.error(f"Momentum ranker failed: {e}")

    # ── Step 6: Run backtest ───────────────────────────────────────────────────
    log.info("Step 6: Running backtest...")
    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    for b in ["SPY_Ret", "AGG_Ret"]:
        if b in df.columns:
            ret_cols.append(b)
    daily_rets = df[ret_cols]

    cutoff  = pd.Timestamp(f"{effective_start_year}-01-01")

    # SWEEP FIX: use momentum ranker predictions for backtest
    if sweep_mode and mom_pred is not None:
        log.info("  Sweep mode: using momentum ranker predictions for backtest")
        bt_predictions = mom_pred
    else:
        bt_predictions = pred_history

    pred_bt = bt_predictions[bt_predictions.index >= cutoff]
    rets_bt = daily_rets[daily_rets.index >= cutoff]
    reg_bt  = (df["Regime_Name"][df.index >= cutoff]
               if "Regime_Name" in df.columns else None)

    log.info(f"Backtest window: {pred_bt.index[0].date()} → "
             f"{pred_bt.index[-1].date()} ({len(pred_bt)} days)")

    try:
        (strat_rets, audit_trail, next_date, next_signal,
         conviction_z, conviction_label, last_p) = execute_strategy(
            predictions_df = pred_bt,
            daily_ret_df   = rets_bt,
            rf_rate        = rf_rate,
            regime_series  = reg_bt,
        )
        metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
        log.info(f"Backtest: Ann Return={metrics.get('ann_return',0)*100:.1f}%  "
                 f"Sharpe={metrics.get('sharpe',0):.2f}  "
                 f"MaxDD={metrics.get('max_dd',0)*100:.1f}%")
        results.update({
            "ann_return": round(metrics.get("ann_return", 0) * 100, 2),
            "sharpe":     round(metrics.get("sharpe", 0), 3),
            "max_dd":     round(metrics.get("max_dd", 0) * 100, 2),
        })
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        raise

    # ── Step 7: Generate next-day signal ──────────────────────────────────────
    log.info("Step 7: Next trading day signal...")

    regime_int, regime_name = 0, "Unknown"
    try:
        regime_int, regime_name = detector.get_current_regime(df)
    except Exception as e:
        log.warning(f"get_current_regime failed: {e} — using last labelled regime")
    if regime_name == "Unknown" and "Regime_Name" in df.columns:
        last_regime_name = str(df["Regime_Name"].dropna().iloc[-1])
        if last_regime_name not in ("Unknown", "Global", "nan", ""):
            regime_name = last_regime_name
            log.info(f"  Regime from df column: {regime_name}")
    if regime_int == 0 and "Regime" in df.columns:
        regime_int = int(df["Regime"].dropna().iloc[-1])

    sweep_regime_name = regime_name
    if reg_bt is not None and len(reg_bt) > 0:
        sweep_regime_name = str(reg_bt.iloc[-1])

    signal_row = build_signal_row(
        next_date=next_date, next_signal=next_signal,
        conviction_z=conviction_z, conviction_label=conviction_label,
        p_array=last_p, regime_int=regime_int, regime_name=regime_name,
        metrics=metrics,
    )

    log.info(f"Signal: {next_signal} on {next_date.date()} "
             f"(Z={conviction_z:.2f}, {conviction_label}) "
             f"Regime: {regime_name}")
    results.update({
        "next_signal":      next_signal,
        "next_date":        str(next_date.date()),
        "conviction_z":     round(conviction_z, 3),
        "conviction_label": conviction_label,
        "regime_name":      regime_name,
    })

    # ── Step 8: Save to Hugging Face ───────────────────────────────────────────
    if not skip_hf_write:
        log.info("Step 8: Saving to Hugging Face Dataset...")

        # Save predictions (Parquet format)
        save_predictions_to_hf(pred_history, "data/pred_history.parquet")

        # Save dataset
        save_dataset_to_hf(df)

        # Save feature list
        save_feature_list_to_hf(feature_cols)

        # Only save detector + bank in full pipeline mode (not sweep)
        if not sweep_mode:
            save_model_to_hf(detector.to_bytes(), "regime_detector.pkl")
            save_model_to_hf(bank.to_bytes(), "model_bank.pkl")
            if momentum_ranker is not None:
                save_model_to_hf(momentum_ranker.to_bytes(),
                                 "momentum_ranker.pkl")
            if mom_pred is not None:
                save_predictions_to_hf(mom_pred, "data/mom_pred_history.parquet")

        # Walk-forward CV
        if run_wfcv:
            log.info("  Running walk-forward CV...")
            try:
                from models import walk_forward_cv
                wf_mom = walk_forward_cv(df=df, fwd_df=fwd_df,
                                          mode="momentum",
                                          train_years=3, test_years=1)
                if not wf_mom.empty:
                    save_predictions_to_hf(wf_mom, "data/wf_mom_pred_history.parquet")
            except Exception as e:
                log.error(f"  Walk-forward failed: {e}")

        # Sweep results
        if sweep_mode and effective_start_year:
            sweep_z, sweep_conv = compute_sweep_z(strat_rets, rf_rate=rf_rate)

            sweep_payload = {
                "signal":      next_signal,
                "ann_return":  round(metrics.get("ann_return", 0), 4),
                "z_score":     sweep_z,
                "sharpe":      round(metrics.get("sharpe", 0), 3),
                "max_dd":      round(metrics.get("max_dd", 0), 4),
                "conviction":  sweep_conv,
                "regime":      sweep_regime_name,
                "start_year":  effective_start_year,
            }
            ok = save_sweep_to_hf(sweep_payload, effective_start_year)
            log.info(f"  Sweep saved (year={effective_start_year} "
                     f"signal={next_signal} "
                     f"ann_ret={sweep_payload['ann_return']*100:.2f}% "
                     f"sweep_z={sweep_z:.2f} "
                     f"regime={sweep_regime_name}): {ok}")

        # Save signals
        save_signals_to_hf(signal_row)
        results["hf_saved"] = True
    else:
        log.info("Step 8: Skipping HF write (--local flag set)")
        results["hf_saved"] = False

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Start year:  {effective_start_year}")
    log.info(f"  Next signal: {next_signal} ({conviction_label}, Z={conviction_z:.2f})")
    log.info(f"  Regime:      {regime_name}")
    log.info(f"  Ann Return:  {results.get('ann_return','?')}%")
    log.info(f"  Sharpe:      {results.get('sharpe','?')}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P2-ETF-REGIME-PREDICTOR Training Pipeline (HF Version)")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--local",         action="store_true")
    parser.add_argument("--wfcv-only",     action="store_true")
    parser.add_argument("--wfcv",          action="store_true")
    parser.add_argument("--sweep-mode",    action="store_true")
    parser.add_argument("--start-year",    type=int, default=None)
    args = parser.parse_args()

    if args.wfcv_only:
        log.info("=== WALK-FORWARD ONLY MODE ===")
        from data_manager_hf import get_data, build_forward_targets, save_predictions_to_hf
        from models import walk_forward_cv
        df     = get_data(start_year=DEFAULT_START_YEAR)
        fwd_df = build_forward_targets(df)
        wf_mom = walk_forward_cv(df=df, fwd_df=fwd_df, mode="momentum",
                                  train_years=3, test_years=1)
        if not wf_mom.empty:
            save_predictions_to_hf(wf_mom, "data/wf_mom_pred_history.parquet")
            log.info(f"WF momentum saved: {len(wf_mom)} OOS days")
        sys.exit(0)

    results = run_pipeline(
        force_refresh     = args.force_refresh,
        skip_hf_write     = args.local,
        run_wfcv          = args.wfcv,
        sweep_mode        = args.sweep_mode,
        start_year        = args.start_year,
    )
    sys.exit(0)
