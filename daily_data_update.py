#!/usr/bin/env python3
"""
daily_data_update.py — P2-ETF-REGIME-PREDICTOR
------------------------------------------------
Standalone incremental data update script.
Loads existing dataset from HF, fetches only new trading days,
re-engineers features for the new tail, and pushes back to HF.

Uses incremental_update() and storage functions already in
data_manager_hf.py — no duplicate logic.

Run by GitHub Actions daily after US market close.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    # ── Validate environment ──────────────────────────────────────────────────
    if not os.getenv("HF_TOKEN"):
        log.error("HF_TOKEN not set — cannot read/write HF Dataset")
        sys.exit(1)

    if not os.getenv("FRED_API_KEY"):
        log.warning("FRED_API_KEY not set — macro features may be stale")

    from data_manager_hf import (
        load_dataset_from_hf,
        save_dataset_to_hf,
        incremental_update,
    )

    # ── Load existing dataset from HF ────────────────────────────────────────
    log.info("Loading existing dataset from Hugging Face …")
    df_existing = load_dataset_from_hf()

    if df_existing is None or df_existing.empty:
        log.error(
            "No existing dataset found on HF. "
            "Run full build first via: python train_hf.py --force-refresh"
        )
        sys.exit(1)

    log.info(f"Existing dataset: {len(df_existing)} rows | "
             f"last date: {df_existing.index[-1].date()}")

    # ── Run incremental update ────────────────────────────────────────────────
    log.info("Running incremental update …")
    df_updated = incremental_update(df_existing)

    new_rows = len(df_updated) - len(df_existing)

    if new_rows == 0:
        log.info("Dataset is already up to date — nothing to push.")
        sys.exit(0)

    # ── Push updated dataset back to HF ──────────────────────────────────────
    log.info(f"Pushing updated dataset (+{new_rows} rows) to Hugging Face …")
    ok = save_dataset_to_hf(df_updated)

    if ok:
        log.info(
            f"✅ Done. Dataset now covers "
            f"{df_updated.index[0].date()} → {df_updated.index[-1].date()} "
            f"({len(df_updated)} rows)"
        )
    else:
        log.error("❌ Failed to push updated dataset to HF")
        sys.exit(1)


if __name__ == "__main__":
    main()
