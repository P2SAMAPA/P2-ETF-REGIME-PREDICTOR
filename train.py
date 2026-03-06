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
    load_model_from_gitlab, save_sweep_to_gitlab, TARGET_ETFS,
)
from regime_detection import RegimeDetector
from models import MomentumRanker, get_feature_columns
from strategy import (
    execute_strategy, calculate_metrics,
    build_signal_row, TARGET_ETFS as STRAT_ETFS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_START_YEAR = 2008
FORWARD_DAYS       = 5
VAL_PCT            = 0.15
TRAIN_PCT          = 0.70


def run_pipeline(force_refresh: bool = False,
                 skip_gitlab_write: bool = False,
                 run_wfcv: bool = False,
                 sweep_mode: bool = False,
                 start_year: int = None) -> dict:
    """
    Full daily training pipeline.

    Parameters
    ----------
    force_refresh     : rebuild dataset from scratch (ignore GitLab cache)
    skip_gitlab_write : run locally without pushing to GitLab
    run_wfcv          : run walk-forward CV (slower, for diagnostics)
    sweep_mode        : load shared dataset from GitLab, skip yfinance calls
    start_year        : override training start year (backtest cutoff)
    """
    results = {}

    # ── Resolve effective start year ──────────────────────────────────────────
    # Fix: pass start_year explicitly into run_pipeline() instead of
    # trying to mutate the module-level START_YEAR from __main__, which
    # only modifies a freshly imported copy and has no effect on this run.
    effective_start_year = start_year if start_year else DEFAULT_START_YEAR

    log.info("=" * 60)
    log.info("P2-ETF-REGIME-PREDICTOR — Daily Training Pipeline")
    log.info(f"Started:    {datetime.utcnow().isoformat()}Z")
    log.info(f"Start year: {effective_start_year}")
    log.info(f"Sweep mode: {sweep_mode}")
    log.info("=" * 60)

    # ── Step 1: Load / build dataset ──────────────────────────────────────────
    log.info("Step 1: Loading dataset...")

    if sweep_mode:
        log.info("Sweep mode — loading shared dataset from GitLab (no yfinance calls)")
        existing = load_dataset_from_gitlab()
        if existing is not None:
            log.info(f"Loaded from GitLab: {len(existing)} rows")
            df = existing
        else:
            log.warning("No GitLab dataset found in sweep mode — building from scratch")
            df = build_full_dataset(start_year=DEFAULT_START_YEAR)
    elif force_refresh:
        log.info("Force refresh — rebuilding from scratch")
        df = build_full_dataset(start_year=DEFAULT_START_YEAR)
    else:
        existing = load_dataset_from_gitlab()
        if existing is not None:
            log.info(f"Loaded from GitLab: {len(existing)} rows")
            df = incremental_update(existing)
        else:
            log.info("No GitLab dataset found — building from scratch")
            df = build_full_dataset(start_year=DEFAULT_START_YEAR)

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

    df["Regime"]      = df["Regime"].fillna(0).astype(int)
    df["Regime_Name"] = df["Regime_Name"].fillna("Global")

    # ── Step 3: Build forward targets ─────────────────────────────────────────
    log.info("Step 3: Building forward return targets...")

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
    from models import MomentumRanker, RegimeModelBank
    bank = RegimeModelBank()

    common_idx = df.index.intersection(fwd_df.dropna(how="all").index)
    df_train   = df.loc[common_idx]
    fwd_train  = fwd_df.loc[common_idx]

    feature_cols = get_feature_columns(df_train)
    log.info(f"Features: {len(feature_cols)}")

    df_train[feature_cols] = df_train[feature_cols].fillna(
        df_train[feature_cols].median()
    ).fillna(0.0)
    log.info("NaN cleaning applied to feature matrix")

    try:
        bank.fit(df_train, fwd_train,
                 feature_cols=feature_cols,
                 val_pct=VAL_PCT)
        results["n_models"] = len(bank.models_)
        log.info(f"Trained {len(bank.models_)} regime-specific ranking models "
                 f"+ 1 global fallback")
    except Exception as e:
        log.error(f"Model training failed: {e}")
        raise

    # ── Step 5: Generate prediction history ───────────────────────────────────
    log.info("Step 5: Generating prediction history...")

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

    # ── Step 5b: Momentum ranker ───────────────────────────────────────────────
    log.info("Step 5b: Training Layer 2B momentum ranker...")
    momentum_ranker = None
    mom_pred        = None
    try:
        momentum_ranker = MomentumRanker()
        momentum_ranker.fit(df)
        mom_pred = momentum_ranker.predict_all_history(df)
        log.info(f"Momentum predictions: {len(mom_pred)} rows")
    except Exception as e:
        log.error(f"Momentum ranker failed: {e}")

    # ── Step 6: Run backtest ───────────────────────────────────────────────────
    log.info("Step 6: Running backtest...")

    ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    for b in ["SPY_Ret", "AGG_Ret"]:
        if b in df.columns:
            ret_cols.append(b)
    daily_rets = df[ret_cols]

    # Fix: apply effective_start_year cutoff BEFORE running the backtest so
    # each sweep year genuinely backtests only from its own start year.
    # Previously START_YEAR was never overridden, so all 6 jobs used 2008
    # and produced identical metrics.
    cutoff     = pd.Timestamp(f"{effective_start_year}-01-01")
    pred_bt    = pred_history[pred_history.index >= cutoff]
    rets_bt    = daily_rets[daily_rets.index >= cutoff]
    reg_bt     = (df["Regime_Name"][df.index >= cutoff]
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
        results["ann_return"] = round(metrics.get("ann_return", 0) * 100, 2)
        results["sharpe"]     = round(metrics.get("sharpe", 0), 3)
        results["max_dd"]     = round(metrics.get("max_dd", 0) * 100, 2)
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        raise

    # ── Step 7: Generate next-day signal ──────────────────────────────────────
    log.info("Step 7: Next trading day signal...")

    regime_int, regime_name = 0, "Unknown"
    try:
        regime_int, regime_name = detector.get_current_regime(df)
    except Exception:
        pass

    signal_row = build_signal_row(
        next_date        = next_date,
        next_signal      = next_signal,
        conviction_z     = conviction_z,
        conviction_label = conviction_label,
        p_array          = last_p,
        regime_int       = regime_int,
        regime_name      = regime_name,
        metrics          = metrics,
    )

    log.info(f"Signal: {next_signal} on {next_date.date()} "
             f"(Z={conviction_z:.2f}, {conviction_label}) "
             f"Regime: {regime_name}")
    results["next_signal"]      = next_signal
    results["next_date"]        = str(next_date.date())
    results["conviction_z"]     = round(conviction_z, 3)
    results["conviction_label"] = conviction_label
    results["regime_name"]      = regime_name

    # ── Step 8: Save to GitLab ─────────────────────────────────────────────────
    if not skip_gitlab_write:
        log.info("Step 8: Saving to GitLab...")

        ok = save_predictions_to_gitlab(pred_history)
        log.info(f"  Prediction history saved: {ok}")

        ok = save_dataset_to_gitlab(df)
        log.info(f"  Dataset saved: {ok}")

        ok = save_feature_list_to_gitlab(feature_cols)
        log.info(f"  Feature list saved: {ok}")

        ok = save_model_to_gitlab(detector.to_bytes(), "regime_detector.pkl")
        log.info(f"  Regime detector saved: {ok}")

        ok = save_model_to_gitlab(bank.to_bytes(), "model_bank.pkl")
        log.info(f"  Model bank saved: {ok}")

        if momentum_ranker is not None:
            ok = save_model_to_gitlab(momentum_ranker.to_bytes(), "momentum_ranker.pkl")
            log.info(f"  Momentum ranker saved: {ok}")
        if mom_pred is not None:
            ok = save_predictions_to_gitlab(mom_pred, "data/mom_pred_history.csv")
            log.info(f"  Momentum predictions saved: {ok}")

        # ── Walk-forward OOS validation ────────────────────────────────────
        # Fix: run_wfcv is now passed in as a parameter, so it's always
        # in scope here. Previously the code referenced args.wfcv which
        # caused a NameError inside run_pipeline(), silently skipping the
        # walk-forward and sweep saves.
        if run_wfcv:
            log.info("  Running Option B (Momentum) walk-forward...")
            try:
                from models import walk_forward_cv
                wf_mom = walk_forward_cv(
                    df=df, fwd_df=fwd_df,
                    mode="momentum",
                    train_years=3, test_years=1,
                )
                if not wf_mom.empty:
                    ok = save_predictions_to_gitlab(
                        wf_mom, "data/wf_mom_pred_history.csv"
                    )
                    log.info(f"  WF momentum saved ({len(wf_mom)} OOS days): {ok}")
            except Exception as e:
                log.error(f"  Walk-forward failed: {e}")
        else:
            log.info("  Skipping walk-forward (run_wfcv=False)")

        # ── Sweep result — save per-year JSON ─────────────────────────────
        # Fix: sweep_mode and effective_start_year are now both guaranteed
        # to be in scope here. Previously args.start_year was referenced
        # inside run_pipeline() but args only existed in __main__, causing
        # a NameError and silently skipping the sweep JSON save entirely.
        if sweep_mode and effective_start_year:
            sweep_payload = {
                "signal":      next_signal,
                "ann_return":  round(metrics.get("ann_return", 0), 4),
                "z_score":     round(conviction_z, 3),
                "sharpe":      round(metrics.get("sharpe", 0), 3),
                "max_dd":      round(metrics.get("max_dd", 0), 4),
                "conviction":  conviction_label,
                "regime":      regime_name,
                "start_year":  effective_start_year,
            }
            ok = save_sweep_to_gitlab(sweep_payload, effective_start_year)
            log.info(f"  Sweep result saved (start_year={effective_start_year}  "
                     f"signal={next_signal}  ann_ret={sweep_payload['ann_return']*100:.2f}%  "
                     f"sharpe={sweep_payload['sharpe']:.2f}): {ok}")

        ok = save_signals_to_gitlab(signal_row)
        log.info(f"  Signal saved: {ok}")

        results["gitlab_saved"] = True
    else:
        log.info("Step 8: Skipping GitLab write (--local flag set)")
        results["gitlab_saved"] = False

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Start year:   {effective_start_year}")
    log.info(f"  Next signal:  {next_signal} ({conviction_label}, Z={conviction_z:.2f})")
    log.info(f"  Regime:       {regime_name}")
    log.info(f"  Ann Return:   {results.get('ann_return','?')}%")
    log.info(f"  Sharpe:       {results.get('sharpe','?')}")
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
        "--wfcv-only", action="store_true",
        help="Run walk-forward CV only — skip data fetch and model retrain"
    )
    parser.add_argument(
        "--wfcv", action="store_true",
        help="Run walk-forward cross-validation (diagnostic, slower)"
    )
    parser.add_argument(
        "--sweep-mode", action="store_true",
        help="Sweep mode: load dataset from GitLab only, no yfinance calls"
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Override training start year for sweep backtest cutoff"
    )
    args = parser.parse_args()

    # Walk-forward only mode — load existing data, skip retrain
    if args.wfcv_only:
        log.info("=== WALK-FORWARD ONLY MODE ===")
        from data_manager import get_data, build_forward_targets
        from models import (get_feature_columns, walk_forward_cv, MomentumRanker)
        from data_manager import save_predictions_to_gitlab
        df     = get_data(start_year=DEFAULT_START_YEAR)
        fwd_df = build_forward_targets(df)
        log.info("Running WF Option B (Momentum)...")
        wf_mom = walk_forward_cv(df=df, fwd_df=fwd_df, mode="momentum",
                                  train_years=3, test_years=1)
        if not wf_mom.empty:
            save_predictions_to_gitlab(wf_mom, "data/wf_mom_pred_history.csv")
            log.info(f"WF momentum saved: {len(wf_mom)} OOS days")
        log.info("Walk-forward complete")
        sys.exit(0)

    # Fix: pass all args explicitly into run_pipeline() so they are
    # always in scope inside the function. Previously --start-year was
    # applied by mutating _self.START_YEAR on a freshly imported module
    # copy, which had no effect on the already-running instance.
    results = run_pipeline(
        force_refresh     = args.force_refresh,
        skip_gitlab_write = args.local,
        run_wfcv          = args.wfcv,
        sweep_mode        = args.sweep_mode,
        start_year        = args.start_year,   # ✅ passed in cleanly
    )
    sys.exit(0)
