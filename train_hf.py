"""
train_hf.py - P2-ETF-REGIME-PREDICTOR v4 (ARCHITECTURAL REWRITE)
=========================================

ROOT CAUSE FIXED (v4):
  The original config used ALL 17 windows with the SAME test period
  (2025-01-01 → present).  Every window therefore produced identical
  predictions on the same 250 trading days, so the consensus sweep
  always returned the same ETF (GDX / GLD) for every "year" in the
  dropdown, and the 813% return was the in-sample fit on 2025 data
  where momentum happened to concentrate on one ETF.

  v4 switches to EXPANDING-WINDOW walk-forward (see config.py):
    - 14 historical windows, each testing a different calendar year
      (2012, 2013, … 2025).  Each window sees a unique market regime:
      2020 = COVID crash, 2022 = rate hike bear, 2024 = AI bull, etc.
    - 1 live window: train 2008-2024, test 2025-present.
    - The WF parquet is keyed on "test_year" (not "train_start") and
      deduplication uses (Date, test_year) as the composite key.
    - Sweep JSON files are keyed on test_year so the per-year consensus
      table shows genuinely different best-ETF signals across regimes.

  The double-reset bug (save_dataframe reset_index + dedup reset again)
  is fixed: save_dataframe always writes index=True; the composite-key
  dedup works entirely in memory before saving.

Usage:
  python train_hf.py --option a            # expand WF for all windows
  python train_hf.py --option b
  python train_hf.py --option a --force-retrain  # wipe + rebuild all
  python train_hf.py --option a --sweep          # consensus sweep
  python train_hf.py --option a --sweep-year 2022  # single sweep year
  python train_hf.py --option a --single-year 2022  # single WF window
"""

import os
import sys
import argparse
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
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _clean(obj):
    """Recursively sanitise numpy scalars, NaN, Inf for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ── Helpers ────────────────────────────────────────────────────────────────────
def _target_etfs(option: str) -> list:
    return cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS


def _label(option: str) -> str:
    return f"Option {option.upper()}"


def _load_fixed_k(option: str) -> Optional[int]:
    try:
        det = hf_load_detector(option)
        if det is None:
            return None
        k = det.optimal_k_
        log.info(f"{_label(option)}: loaded fixed_k={k} from saved detector")
        return k
    except Exception as e:
        log.warning(f"{_label(option)}: could not load detector ({e})")
        return None


def _merge_wf(existing: Optional[pd.DataFrame],
              new_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new fold predictions into existing WF table.
    Deduplication key: (Date index, test_year column).
    No reset_index stored — the DatetimeIndex is preserved throughout.
    """
    if existing is None or existing.empty:
        return new_preds.sort_index()

    if "test_year" not in existing.columns:
        log.warning("Existing wf_predictions lack 'test_year' column — resetting.")
        return new_preds.sort_index()

    combined = pd.concat([existing, new_preds])
    # Deduplicate on composite key (Date, test_year)
    combined = combined.reset_index()           # Date -> column temporarily
    combined = combined.drop_duplicates(
        subset=["Date", "test_year"], keep="last"
    )
    combined = combined.set_index("Date")
    combined.index = pd.DatetimeIndex(combined.index)
    combined.index.name = "Date"
    return combined.sort_index()


# ── Core training helpers ──────────────────────────────────────────────────────
def _fit_detector(df: pd.DataFrame, option: str,
                  wf_mode: bool = False,
                  fixed_k: Optional[int] = None) -> RegimeDetector:
    ret_cols = [f"{t}_Ret" for t in _target_etfs(option) if f"{t}_Ret" in df.columns]
    det = RegimeDetector(window=20, k=None)
    det.fit(df[ret_cols], wf_mode=wf_mode, fixed_k=fixed_k)
    return det


def _fit_ranker(train_df: pd.DataFrame, detector: RegimeDetector,
                option: str) -> MomentumRanker:
    train_r = detector.add_regime_to_df(train_df)
    ranker  = MomentumRanker(target_etfs=_target_etfs(option))
    ranker.fit(train_r)
    return ranker


def _predict_on_test(test_df: pd.DataFrame,
                     detector: RegimeDetector,
                     ranker: MomentumRanker) -> pd.DataFrame:
    """
    Apply detector + ranker to the TEST slice only.
    detector.predict() is called on test_df — no training-period data is
    exposed to the model at prediction time.
    """
    test_r = detector.add_regime_to_df(test_df)
    preds  = ranker.predict_all_history(test_r)
    return preds


# ── Full expanding-window WF training ─────────────────────────────────────────
def run_full_training(option: str, force_refresh: bool = False,
                      force_retrain: bool = False) -> None:
    """
    Run all expanding-window WF folds and store predictions keyed by test_year.
    """
    log.info("=" * 60)
    log.info(f"{_label(option)}: expanding-window WF training")
    log.info("=" * 60)

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    # Determine which test_years already have predictions
    existing_test_years: set = set()
    existing: Optional[pd.DataFrame] = None
    if not force_retrain:
        existing = load_wf_predictions(option, force_download=True)
        if (existing is not None and not existing.empty
                and "test_year" in existing.columns):
            existing_test_years = set(existing["test_year"].unique())
            log.info(f"{_label(option)}: already computed test years: "
                     f"{sorted(existing_test_years)}")
    else:
        log.info(f"{_label(option)}: force_retrain — rebuilding all folds")

    # Select or compute fixed_k from the full dataset once
    fixed_k = _load_fixed_k(option)
    if fixed_k is None:
        log.info(f"{_label(option)}: running k-selection on full dataset...")
        full_det = _fit_detector(df, option, wf_mode=False)
        fixed_k  = full_det.optimal_k_
        log.info(f"{_label(option)}: k={fixed_k} selected")
        full_ranker = _fit_ranker(df, full_det, option)
        save_detector(full_det, option)
        save_ranker(full_ranker, option)
        log.info(f"{_label(option)}: saved full detector + ranker to HF")

    all_new_preds = []
    for window in cfg.WINDOWS:
        test_year = window["test_year"]
        is_live   = window["is_live"]

        # Skip historical folds already computed (unless force_retrain)
        if test_year in existing_test_years and not is_live:
            log.info(f"{_label(option)}: test_year={test_year} already exists — skipping")
            continue
        # Always recompute live window to stay fresh
        if test_year in existing_test_years and is_live:
            log.info(f"{_label(option)}: refreshing live window (test_year={test_year})")

        train_mask = ((df.index >= window["train_start"]) &
                      (df.index <= window["train_end"]))
        test_end   = (window["test_end"] if window["test_end"] is not None
                      else df.index.max().strftime("%Y-%m-%d"))
        test_mask  = ((df.index >= window["test_start"]) &
                      (df.index <= test_end))
        train_df   = df[train_mask]
        test_df    = df[test_mask]

        if len(train_df) < 252:
            log.warning(f"{_label(option)}: test_year={test_year}: "
                        f"only {len(train_df)} train days — need 252, skipping")
            continue
        if len(test_df) < 5:
            log.warning(f"{_label(option)}: test_year={test_year}: "
                        f"only {len(test_df)} test days — skipping")
            continue

        log.info(f"{_label(option)}: test_year={test_year}"
                 f"{'(LIVE)' if is_live else ''} "
                 f"train {train_df.index[0].date()}→{train_df.index[-1].date()} "
                 f"({len(train_df)}d) "
                 f"test {test_df.index[0].date()}→{test_df.index[-1].date()} "
                 f"({len(test_df)}d)")

        det        = _fit_detector(train_df, option, wf_mode=True, fixed_k=fixed_k)
        ranker     = _fit_ranker(train_df, det, option)
        fold_preds = _predict_on_test(test_df, det, ranker)

        if fold_preds.empty:
            log.warning(f"{_label(option)}: test_year={test_year}: "
                        f"no predictions — test slice shorter than lookback?")
            continue

        fold_preds["test_year"] = test_year
        all_new_preds.append(fold_preds)
        log.info(f"{_label(option)}: test_year={test_year}: "
                 f"{len(fold_preds)} predictions generated")

    if not all_new_preds:
        log.info(f"{_label(option)}: no new folds to compute.")
        return

    new_df = pd.concat(all_new_preds).sort_index()

    if force_retrain:
        merged = new_df
    else:
        merged = _merge_wf(existing, new_df)

    log.info(f"{_label(option)}: saving {len(merged)} rows across "
             f"{merged['test_year'].nunique()} test years")
    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: WF training complete")


# ── Single-fold WF ─────────────────────────────────────────────────────────────
def run_single_year(test_year: int, option: str,
                    force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Train and save predictions for a single test_year fold."""
    window = next((w for w in cfg.WINDOWS if w["test_year"] == test_year), None)
    if not window:
        log.error(f"{_label(option)}: no window for test_year={test_year}")
        return None

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    train_mask = ((df.index >= window["train_start"]) &
                  (df.index <= window["train_end"]))
    test_end   = (window["test_end"] if window["test_end"] is not None
                  else df.index.max().strftime("%Y-%m-%d"))
    test_mask  = ((df.index >= window["test_start"]) &
                  (df.index <= test_end))
    train_df, test_df = df[train_mask], df[test_mask]

    if len(train_df) < 252 or len(test_df) < 5:
        log.error(f"{_label(option)}: insufficient data for test_year={test_year}")
        return None

    fixed_k = _load_fixed_k(option)
    if fixed_k is None:
        full_det = _fit_detector(df, option, wf_mode=False)
        fixed_k  = full_det.optimal_k_

    det        = _fit_detector(train_df, option, wf_mode=True, fixed_k=fixed_k)
    ranker     = _fit_ranker(train_df, det, option)
    fold_preds = _predict_on_test(test_df, det, ranker)

    if fold_preds.empty:
        log.error(f"{_label(option)}: test_year={test_year}: no predictions generated")
        return None

    fold_preds["test_year"] = test_year
    existing = load_wf_predictions(option, force_download=True)
    merged   = _merge_wf(existing, fold_preds)
    save_wf_predictions(merged, option)
    log.info(f"{_label(option)}: test_year={test_year} saved "
             f"({len(fold_preds)} rows, {merged['test_year'].nunique()} total)")
    return fold_preds


# ── Consensus sweep ────────────────────────────────────────────────────────────
def run_sweep(option: str, years: Optional[list] = None,
              force_refresh: bool = False) -> None:
    """
    For each historical test year, run strategy on that year's OOS predictions
    and save a sweep JSON keyed by test_year.
    """
    target_windows = [w for w in cfg.WINDOWS if not w["is_live"]]
    if years:
        target_windows = [w for w in target_windows if w["test_year"] in years]

    etfs = _target_etfs(option)
    log.info(f"{_label(option)}: sweep for test years: "
             f"{[w['test_year'] for w in target_windows]}")

    df = get_data(option=option, start_year=cfg.START_YEAR_DEFAULT,
                  force_refresh=force_refresh)

    # Load pre-saved WF predictions
    wf_preds = load_wf_predictions(option, force_download=True)
    if wf_preds is None or wf_preds.empty or "test_year" not in wf_preds.columns:
        log.warning(f"{_label(option)}: no WF predictions found — "
                    f"run `python train_hf.py --option {option}` first")
        return

    for window in target_windows:
        test_year = window["test_year"]

        fold_preds = wf_preds[wf_preds["test_year"] == test_year].copy()
        if fold_preds.empty:
            log.warning(f"{_label(option)}: no WF preds for test_year={test_year} — "
                        f"run: --single-year {test_year}")
            continue

        test_end  = (window["test_end"] if window["test_end"] is not None
                     else df.index.max().strftime("%Y-%m-%d"))
        test_mask = ((df.index >= window["test_start"]) &
                     (df.index <= test_end))
        test_df   = df[test_mask]

        if len(test_df) < 10:
            log.warning(f"{_label(option)}: test_year={test_year}: "
                        f"insufficient test data — skipping")
            continue

        common = fold_preds.index.intersection(test_df.index)
        if len(common) < 5:
            log.warning(f"{_label(option)}: test_year={test_year}: "
                        f"only {len(common)} overlapping dates — skipping")
            continue

        ret_cols     = [f"{t}_Ret" for t in etfs if f"{t}_Ret" in test_df.columns]
        ret_df       = test_df.loc[common, ret_cols]
        pred_aligned = fold_preds.loc[common]

        rf_rate = (float(test_df["DTB3"].iloc[-1] / 100)
                   if "DTB3" in test_df.columns else cfg.RISK_FREE_RATE)

        if "Regime_Name" in test_df.columns:
            regime_series = test_df["Regime_Name"].reindex(common).ffill().fillna("Unknown")
        elif "Regime" in pred_aligned.columns:
            regime_series = pred_aligned["Regime"].reindex(common).fillna("Unknown").astype(str)
        else:
            regime_series = pd.Series("Unknown", index=common)

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
            clean_rets           = strat_rets[np.isfinite(strat_rets)]
            metrics              = calculate_metrics(clean_rets, rf_rate=rf_rate)
            sweep_z, sweep_label = compute_sweep_z(clean_rets, rf_rate=rf_rate)
        except Exception as e:
            log.warning(f"{_label(option)}: strategy failed for test_year={test_year}: {e}")
            import traceback; traceback.print_exc()
            metrics      = {}
            next_signal  = "CASH"
            sweep_z      = 0.0
            sweep_label  = "Low"
            last_p       = np.full(len(etfs), 1.0 / len(etfs))

        regime_name = str(regime_series.iloc[-1]) if len(regime_series) else "Unknown"

        etf_probs = {}
        for i, etf in enumerate(etfs):
            p = float(last_p[i]) if i < len(last_p) else 1.0 / len(etfs)
            etf_probs[etf] = round(p if np.isfinite(p) else 1.0 / len(etfs), 4)

        result = _clean({
            "signal":     next_signal,
            "ann_return": round(metrics.get("ann_return", 0.0), 4),
            "z_score":    sweep_z,
            "sharpe":     round(metrics.get("sharpe", 0.0), 3),
            "max_dd":     round(metrics.get("max_dd", 0.0), 4),
            "conviction": sweep_label,
            "regime":     regime_name,
            "etf_probs":  etf_probs,
            "n_days":     int(len(common)),
            "test_year":  test_year,
        })

        save_sweep_result(result, test_year, option)
        log.info(f"{_label(option)}: sweep test_year={test_year} — "
                 f"signal={next_signal}, "
                 f"return={result['ann_return']:.1%}, "
                 f"sharpe={result['sharpe']:.2f}, "
                 f"z={sweep_z:.2f}σ, "
                 f"n={len(common)}d")

    log.info(f"{_label(option)}: sweep complete")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="P2-ETF-REGIME-PREDICTOR v4 — expanding-window WF pipeline"
    )
    parser.add_argument("--option", required=True, choices=["a", "b"])
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force full dataset rebuild from yfinance + FRED")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Wipe existing predictions and recompute all folds")
    parser.add_argument("--wfcv", action="store_true",
                        help="Run WF training (alias for default invocation)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run consensus sweep across all historical test years")
    parser.add_argument("--sweep-year", type=int, default=None,
                        help="Run sweep for one test year only")
    parser.add_argument("--single-year", type=int, default=None,
                        help="Run WF for one test year only")

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
        else:
            run_full_training(option,
                              force_refresh=args.force_refresh,
                              force_retrain=args.force_retrain)

        log.info(f"Option {option.upper()}: pipeline completed successfully")

    except Exception as e:
        log.error(f"Option {option.upper()}: pipeline failed — {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
