"""
daily_data_update.py — P2-ETF-REGIME-PREDICTOR v2
--------------------------------------------------
Incremental data update for both Option A (FI/Commodities) and Option B
(Equity ETFs). Runs after US market close via GitHub Actions.

For each option:
  1. Load existing dataset from HF
  2. Fetch only new trading days
  3. Re-engineer features for the new tail
  4. Push updated dataset back to HF

Both options run sequentially in a single script invocation.
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def update_option(option: str) -> bool:
    """Run incremental update for a single option ('a' or 'b'). Returns True on success."""
    from data_manager_hf import (
        load_dataset, save_dataset, incremental_update,
    )
    label = f"Option {option.upper()}"
    log.info(f"{'='*60}")
    log.info(f"{label} — starting incremental data update")
    log.info(f"{'='*60}")

    df_existing = load_dataset(option)
    if df_existing is None or df_existing.empty:
        log.error(
            f"{label}: no existing dataset found on HF. "
            "Run full seed first via manual_retrain workflow."
        )
        return False

    log.info(f"{label}: existing dataset — {len(df_existing)} rows, "
             f"last date: {df_existing.index[-1].date()}")

    df_updated = incremental_update(df_existing, option)
    new_rows   = len(df_updated) - len(df_existing)

    if new_rows == 0:
        log.info(f"{label}: already up to date — nothing to push.")
        return True

    log.info(f"{label}: pushing +{new_rows} rows to HF...")
    ok = save_dataset(df_updated, option)
    if ok:
        log.info(
            f"{label}: ✅ done — dataset now covers "
            f"{df_updated.index[0].date()} → {df_updated.index[-1].date()} "
            f"({len(df_updated)} rows)"
        )
    else:
        log.error(f"{label}: ❌ failed to push updated dataset to HF")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Daily incremental data update for P2-ETF-REGIME-PREDICTOR"
    )
    parser.add_argument(
        "--option", choices=["a", "b", "both"], default="both",
        help="Which option to update (default: both)"
    )
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        log.error("HF_TOKEN not set — cannot read/write HF Dataset")
        sys.exit(1)
    if not os.getenv("FRED_API_KEY"):
        log.warning("FRED_API_KEY not set — macro features may be stale")

    options = ["a", "b"] if args.option == "both" else [args.option]
    failures = []

    for opt in options:
        ok = update_option(opt)
        if not ok:
            failures.append(opt.upper())

    if failures:
        log.error(f"Update failed for: Option(s) {', '.join(failures)}")
        sys.exit(1)

    log.info("✅ Daily data update complete for all options.")


if __name__ == "__main__":
    main()
