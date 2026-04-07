"""
train_hf.py - P2-ETF-REGIME-PREDICTOR v3 (FULLY CORRECTED)
=========================================
Training pipeline for Option A (FI/Commodities) and Option B (Equity ETFs).

FIXES vs v2:
  1. run_full_training(): deduplication on the wf_predictions DataFrame now
     uses subset=["Date", "train_start"] instead of the DatetimeIndex alone.
     Previously, pd.concat followed by index-dedup kept only the LAST
     train_start window for every shared date — meaning 17 windows were
     silently collapsed into 1. Now every (Date, train_start) pair is unique.

  2. run_single_year(): same composite-key dedup applied when merging a
     single new window into existing predictions.

  3. NaN / numpy-type handling (kept from v2): NumpyEncoder + _clean_numpy_values
     ensure sweep JSON is always serialisable.

  4. run_sweep(): uses TEST data for predictions, not training data (kept
     from v2 correction).

Usage:
  python train_hf.py --option a              # Full train all windows (Option A)
  python train_hf.py --option b              # Full train all windows (Option B)
  python train_hf.py --option a --force-refresh   # Force dataset rebuild
  python train_hf.py --option a --force-retrain   # Discard cached windows, retrain all
  python train_hf.py --option a --wfcv       # Incremental WF CV (fast path)
  python train_hf.py --option a --sweep      # Consensus sweep across all windows
  python train_hf.py --option a --sweep-year 2010  # Sweep one window only
  python train_hf.py --option a --single-year 2010 # Single-window WF
"""

import os
import sys
import argparse
import pickle
import logging
import json
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from data_manager_hf import (
    get_data,
    load_predictions, save_predictions,
    load_wf_predictions, save_wf_predictions,
    load_signals, save_signals,
    load_detector as hf_load_detector,
    save_detector, save_ranker,
    save_feature_list, save_sweep_result,
)
from regime_detection import RegimeDetector
from models import MomentumRanker
from strategy import execute_strategy, calculate_metrics, compute_sweep_z

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── JSON helpers ───────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and NaN values."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _clean_numpy_values(obj):
    """Recursively clean numpy values and NaN from nested structures."""
    if isinstance(obj, dict):
        return {k: _clean_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_numpy_values(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_clean_numpy_values(v) for v in obj.tolist()]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


# ── Helpers ────────────────────────────────────────────────────────────────────
def _target_etfs(option: str) -> list:
    return cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS


def _label(option: str) -> str:
    return f"Option {option.upper()}"


def _load_fixed_k(option: str) -> Optional[int]:
    """
    Load saved HF detector and return its optimal_k_.
    Avoids re-running k-selection in WF CV jobs.
    Falls back to None if detector not yet saved.
    """
    try:
        detector = hf_load_detector(option)
        if detector is None:
            log.warning(f"{_label(option)}: no saved detector on HF — will run k-selection")
            return None
        k = detector.optimal_k_
        log.info(f"{_label(option)}: loaded fixed_k={k} from saved HF detector")
        return k
    except Exception as e:
        log.warning(f"{_label(option)}: could not load detector ({e}) — will run k-selection")
        return None


def _dedup_wf_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    KEY FIX: deduplicate on (Date, train_start) not on the DatetimeIndex.

    All 17 training windows share the same test-period dates (2025-01-01
    onward).  Deduplicating on the index alone keeps only the last row per
    date — collapsing all windows into one.  This function instead treats
    (Date, train_start) as the composite primary key.
    """
    if df.empty:
        return df
    # If Date is the index, move it to a column temporarily
    has_date_index = isinstance(df.index, pd.DatetimeIndex)
    if has_date_index:
        df = df.reset_index()  # "Date" becomes a column
        index_name = df.columns[0]  # usually "Date"
    else:
        index_name = None

    if "train_start" in df.columns and "Date" in df.columns:
        df = df.drop_duplicates(subset=["Date", "train_start"], keep="last")
    elif "Date" in df.columns:
        df = df.drop_duplicates(subset=["Date"], keep="last")

    if has_date_index and "Date" in df.columns:
        df = df.set_index("Date")
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"

    return df.sort_index()


# ── Training functions ─────────────────────────────────────────────────────────
def train_regime_detector(
    df: pd.DataFrame,
    option: str,
    sweep_mode: bool = False,
    wf_mode: bool = False,
    fixed_k: Optional[int] = None,
) -> RegimeDetector:
    mode_str = ("wf_mode" if wf_mode else
                "sweep_mode" if sweep_mode else "full")
    log.info(f"{_label(option)}: training regime detector [{mode_str}]...")
    ret_cols = [f"{t}_Ret" for t in _target_etfs(option)
                if f"{t}_Ret" in df.columns]
    detector = RegimeDetector(window=20, k=None)
    detector.fit(df[ret_cols], sweep_mode=sweep_mode,
                 wf_mode=wf_mode, fixed_k=fixed_k)
    log.info(f"{_label(option)}: regime detector trained — "
             f"k={detector.optimal_k_} regimes [{mode_str}]")
    return detector


def train_momentum_ranker(df: pd.DataFrame, detector: RegimeDetector,
                          option: str) -> MomentumRanker:
    """Train MomentumRanker with the correct ETF universe for this option."""
    log.info(f"{_label(option)}: training momentum ranker...")
    df = detector.add_regime_to_df(df)
    ranker = MomentumRanker(target_etfs=_target_etfs(option))
    ranker.fit(df)
    log.info(f"{_label(option)}: momentum ranker trained — "
             f"universe: {_target_etfs(option)}")
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


# ── Full training (all windows, incremental) ───────────────────────────────────
def run_full_training(option: str, force_refresh: bool = False,
                      force_retrain: bool = False) -> None:
    """
    Train all windows sequentially. Saves detector (with optimal_k_) once.
    Predictions are stored in a single table with a train_start column.

    force_retrain=True clears existing_start_years so all windows are
    recomputed. Use whenever models.py column schema changes.

    KEY FIX: uses _dedup_wf_predictions() which deduplicates on
    (Date, train_start) — not on DatetimeIndex alone — so all 17 windows
    are preserved in the HF parquet file.
    """
    log.info(f"{'='*60}")
    log.info(f"{_label(option)}: full training pipeline (all windows)")
    if force_retrain:
        log.info(f"{_label(option)}: --force-retrain set — all windows will be recomputed")
    log.info(f"{'='*60}")

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    # Determine which windows already have predictions (incremental).
    existing_start_years = set()
    existing = None
    if not force_retrain:
        existing = load_wf_predictions(option, force_download=True)
        if existing is not None and not existing.empty and "train_start" in existing.columns:
            existing_start_years = set(existing["train_start"].unique())
            log.info(f"{_label(option)}: already computed windows: {sorted(existing_start_years)}")
    else:
        log.info(f"{_label(option)}: skipping existing predictions check (force_retrain)")

    # Load or compute fixed_k once
    fixed_k = _load_fixed_k(option)
    if fixed_k is None:
        log.info(f"{_label(option)}: no saved detector — running k-selection on full data...")
        ret_cols = [f"{t}_Ret" for t in _target_etfs(option)
                    if f"{t}_Ret" in df.columns]
        ref_detector = RegimeDetector(window=20, k=None)
        ref_detector.fit(df[ret_cols], wf_mode=False)
        fixed_k = ref_detector.optimal_k_
        log.info(f"{_label(option)}: selected k={fixed_k}")
        # Save the detector so future WF jobs reuse fixed_k
        ref_ranker = train_momentum_ranker(df, ref_detector, option)
        save_detector(ref_detector, option)
        save_ranker(ref_ranker, option)
        log.info(f"{_label(option)}: saved detector and ranker to HF")

    all_preds = []
    for window in cfg.WINDOWS:
        start_year = int(window["train_start"].split("-")[0])
        if start_year in existing_start_years:
            log.info(f"{_label(option)}: window {start_year} already exists — skipping")
            continue

        train_mask = (df.index >= window["train_start"]) & (df.index <= window["train_end"])
        test_end   = window["test_end"] if window["test_end"] is not None else df.index.max()
        test_mask  = (df.index >= window["test_start"]) & (df.index <= test_end)
        train_df   = df[train_mask]
        test_df    = df[test_mask]

        if len(train_df) < 252 or len(test_df) < 5:
            log.warning(f"{_label(option)}: insufficient data for train start {start_year} — skipping")
            continue

        log.info(f"{_label(option)}: window {start_year} — "
                 f"train {train_df.index[0].date()}→{train_df.index[-1].date()}, "
                 f"test {test_df.index[0].date()}→{test_df.index[-1].date()}")

        detector  = train_regime_detector(train_df, option, wf_mode=True, fixed_k=fixed_k)
        ranker    = train_momentum_ranker(train_df, detector, option)
        test_df_r = detector.add_regime_to_df(test_df)
        fold_preds = ranker.predict_all_history(test_df_r)

        # Tag with training start year — this column must survive into the parquet
        fold_preds["train_start"] = start_year

        all_preds.append(fold_preds)
        log.info(f"{_label(option)}: window {start_year} complete — "
                 f"{len(fold_preds)} OOS predictions")

    if not all_preds:
        log.info(f"{_label(option)}: no new windows to compute.")
        return

    new_preds = pd.concat(all_preds).sort_index()

    # KEY FIX: use composite (Date, train_start) dedup
    existing_safe = existing if not force_retrain else None
    if existing_safe is not None and not existing_safe.empty:
        if "train_start" not in existing_safe.columns:
            log.warning("Existing predictions lack train_start column; resetting.")
            merged = new_preds
        else:
            merged = _dedup_wf_predictions(pd.concat([existing_safe, new_preds]))
    else:
        merged = new_preds

    log.info(f"{_label(option)}: saving {len(merged)} rows across "
             f"{merged['train_start'].nunique()} training windows")
    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: WF CV complete — {len(merged)} total OOS rows")


# ── Single-window WF ───────────────────────────────────────────────────────────
def run_single_year(start_year: int, option: str,
                    force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Train a single window for a given train start year and merge it into
    the existing wf_predictions parquet.

    KEY FIX: deduplicates on (Date, train_start) when merging.
    """
    window = next((w for w in cfg.WINDOWS
                   if w["train_start"].startswith(str(start_year))), None)
    if not window:
        log.error(f"{_label(option)}: no window for train start year {start_year}")
        return None

    log.info(f"{_label(option)}: single‑window for train start {start_year}...")
    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    train_mask = (df.index >= window["train_start"]) & (df.index <= window["train_end"])
    test_end   = window["test_end"] if window["test_end"] is not None else df.index.max()
    test_mask  = (df.index >= window["test_start"]) & (df.index <= test_end)
    train_df, test_df = df[train_mask], df[test_mask]

    if len(train_df) < 252 or len(test_df) < 5:
        log.error(f"{_label(option)}: insufficient data for start year {start_year}")
        return None

    fixed_k = _load_fixed_k(option)
    if fixed_k is None:
        ret_cols = [f"{t}_Ret" for t in _target_etfs(option)
                    if f"{t}_Ret" in df.columns]
        ref_detector = RegimeDetector(window=20, k=None)
        ref_detector.fit(df[ret_cols], wf_mode=False)
        fixed_k = ref_detector.optimal_k_

    detector   = train_regime_detector(train_df, option, wf_mode=True, fixed_k=fixed_k)
    ranker     = train_momentum_ranker(train_df, detector, option)
    test_df_r  = detector.add_regime_to_df(test_df)
    fold_preds = ranker.predict_all_history(test_df_r)
    fold_preds["train_start"] = start_year

    existing = load_wf_predictions(option, force_download=True)
    if existing is not None and not existing.empty:
        if "train_start" not in existing.columns:
            log.warning("Existing predictions lack train_start column; resetting.")
            merged = fold_preds
        else:
            # KEY FIX: composite (Date, train_start) dedup
            merged = _dedup_wf_predictions(pd.concat([existing, fold_preds]))
    else:
        merged = fold_preds

    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: single‑window {start_year} saved "
             f"({len(fold_preds)} rows, {merged['train_start'].nunique()} total windows)")
    return fold_preds


# ── Consensus sweep ────────────────────────────────────────────────────────────
def run_sweep(option: str, years: Optional[list] = None,
              force_refresh: bool = False) -> None:
    """
    Consensus sweep over all windows (or a subset of train start years).
    For each window, runs the full strategy on the test period and records metrics.
    Uses TEST data for predictions (not training data).
    """
    if years:
        windows = [w for w in cfg.WINDOWS if int(w["train_start"].split("-")[0]) in years]
    else:
        windows = cfg.WINDOWS

    etfs = _target_etfs(option)
    log.info(f"{_label(option)}: consensus sweep for windows: "
             f"{[w['train_start'] for w in windows]}...")

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    fixed_k = _load_fixed_k(option)
    if fixed_k is None:
        ret_cols = [f"{t}_Ret" for t in _target_etfs(option)
                    if f"{t}_Ret" in df.columns]
        ref_detector = RegimeDetector(window=20, k=None)
        ref_detector.fit(df[ret_cols], wf_mode=False)
        fixed_k = ref_detector.optimal_k_

    for window in windows:
        train_mask = (df.index >= window["train_start"]) & (df.index <= window["train_end"])
        test_end   = window["test_end"] if window["test_end"] is not None else df.index.max()
        test_mask  = (df.index >= window["test_start"]) & (df.index <= test_end)
        train_df   = df[train_mask]
        test_df    = df[test_mask]

        if len(train_df) < 252 or len(test_df) < 5:
            log.warning(f"{_label(option)}: insufficient data for window "
                        f"{window['train_start']} — skipping")
            continue

        start_year = int(window["train_start"].split("-")[0])
        log.info(f"{_label(option)}: sweep window train start {start_year} "
                 f"({len(train_df)} training days, {len(test_df)} test days)")

        detector        = train_regime_detector(train_df, option, sweep_mode=True,
                                                fixed_k=fixed_k)
        train_r         = detector.add_regime_to_df(train_df)
        ranker          = MomentumRanker(target_etfs=etfs)
        ranker.fit(train_r)

        # Use TEST data for predictions (not training data)
        test_df_regime  = detector.add_regime_to_df(test_df)
        predictions     = generate_predictions(test_df_regime, ranker, option)

        common = predictions.index.intersection(test_df.index)
        if len(common) == 0:
            log.warning(f"{_label(option)}: no common dates for test period in "
                        f"window {start_year}")
            continue

        ret_cols     = [f"{t}_Ret" for t in etfs if f"{t}_Ret" in test_df.columns]
        ret_df       = test_df.loc[common, ret_cols]
        pred_aligned = predictions.loc[common]

        rf_rate = (float(test_df["DTB3"].iloc[-1] / 100)
                   if "DTB3" in test_df.columns else cfg.RISK_FREE_RATE)

        regime_series = (test_df["Regime_Name"].reindex(common)
                         if "Regime_Name" in test_df.columns
                         else pd.Series("Unknown", index=common))

        try:
            (strat_rets, _, _, next_signal, conviction_z,
             conviction_label, last_p) = execute_strategy(
                predictions_df=pred_aligned,
                daily_ret_df=ret_df,
                rf_rate=rf_rate,
                z_reentry=cfg.Z_REENTRY,
                stop_loss_pct=cfg.STOP_LOSS_PCT,
                fee_bps=cfg.TRANSACTION_BPS,
                regime_series=regime_series,
                target_etfs=etfs,
            )
            strat_rets_clean = strat_rets[~np.isnan(strat_rets)]
            metrics          = calculate_metrics(strat_rets_clean, rf_rate=rf_rate)
            sweep_z, sweep_label = compute_sweep_z(strat_rets_clean, rf_rate=rf_rate)
        except Exception as e:
            log.warning(f"{_label(option)}: strategy failed for window {start_year}: {e}")
            import traceback
            traceback.print_exc()
            metrics      = {}
            next_signal  = "CASH"
            sweep_z      = 0.0
            sweep_label  = "Low"
            last_p       = np.full(len(etfs), 1.0 / len(etfs))

        regime_name = (test_df["Regime_Name"].iloc[-1]
                       if "Regime_Name" in test_df.columns else "Unknown")

        etf_probs = {}
        for i, etf in enumerate(etfs):
            p_val = float(last_p[i]) if i < len(last_p) else (1.0 / len(etfs))
            if np.isnan(p_val) or np.isinf(p_val):
                p_val = 1.0 / len(etfs)
            etf_probs[etf] = round(p_val, 4)

        result = {
            "signal":     next_signal,
            "ann_return": round(metrics.get("ann_return", 0.0), 4),
            "z_score":    sweep_z,
            "sharpe":     round(metrics.get("sharpe", 0.0), 3),
            "max_dd":     round(metrics.get("max_dd", 0.0), 4),
            "conviction": sweep_label,
            "regime":     str(regime_name),
            "etf_probs":  etf_probs,
        }
        result = _clean_numpy_values(result)

        save_sweep_result(result, start_year, option)
        log.info(f"{_label(option)}: sweep window {start_year} — "
                 f"signal={next_signal}, "
                 f"return={result['ann_return']:.1%}, "
                 f"sharpe={result['sharpe']:.2f}, "
                 f"z={sweep_z:.2f}σ")

    log.info(f"{_label(option)}: sweep complete")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="P2-ETF-REGIME-PREDICTOR v3 training pipeline"
    )
    parser.add_argument("--option", required=True, choices=["a", "b"],
                        help="Which option: a (FI/Commodities) or b (Equity ETFs)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force full dataset rebuild from source APIs")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Ignore existing wf predictions and retrain all windows from scratch. "
                             "Use this when models.py column schema has changed.")
    parser.add_argument("--wfcv", action="store_true",
                        help="Run incremental walk-forward CV (fast path)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run consensus sweep across all windows")
    parser.add_argument("--sweep-year", type=int, default=None,
                        help="Run sweep for a single train start year only")
    parser.add_argument("--single-year", type=int, default=None,
                        help="Run single‑window walk‑forward for a specific train start year")

    args = parser.parse_args()
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
            run_full_training(option, force_refresh=args.force_refresh,
                              force_retrain=args.force_retrain)
        else:
            run_full_training(option, force_refresh=args.force_refresh,
                              force_retrain=args.force_retrain)

        log.info(f"Option {option.upper()}: pipeline completed successfully")

    except Exception as e:
        log.error(f"Option {option.upper()}: pipeline failed — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
