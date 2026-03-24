# app.py — P2-ETF-REGIME-PREDICTOR (HF Dataset Version)
# Streamlit dashboard with single‑year on‑demand training.

import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

import config as cfg

# ------------------------------------------------------------
# Constants and helpers
# ------------------------------------------------------------
GITHUB_REPO = os.getenv("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
GITHUB_TOKEN = os.getenv("GH_PAT", "")   # Personal Access Token

TARGET_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]

def trigger_workflow(workflow_file: str, inputs: dict = None) -> tuple:
    """Trigger a GitHub Actions workflow_dispatch. Returns (success, message)."""
    if not GITHUB_TOKEN:
        return False, "GH_PAT secret not set in Streamlit"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches"
    payload = {"ref": "main"}
    if inputs:
        payload["inputs"] = inputs
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                 "Accept": "application/vnd.github+json"},
        json=payload,
        timeout=15,
    )
    if resp.status_code == 204:
        return True, "✅ Triggered successfully"
    return False, f"❌ GitHub API returned {resp.status_code}: {resp.text[:200]}"

def load_single_year_result(year: int) -> pd.DataFrame:
    """Load precomputed single‑year walk‑forward predictions from HF."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"data/wf_single_year_{year}.parquet",
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        elif df.index.name != "Date":
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return pd.DataFrame()

# ------------------------------------------------------------
# Sidebar (unchanged from original, but keep it)
# ------------------------------------------------------------
# ... (keep your sidebar code) ...

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
st.title("📈 P2-ETF Regime-Aware Rotation Model")
st.caption("Wasserstein k‑means regime detection · Momentum Ranking (RoC + OBV + Breakout) · ETFs: TLT · VNQ · SLV · GLD · LQD · HYG")

# Load data and models (cached)
@st.cache_data(ttl=3600)
def load_wf_predictions() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename="data/wf_mom_pred_history.parquet",
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            force_download=True,
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_sweep_results():
    # ... (existing sweep loading logic) ...
    pass

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Single‑Year Results", "🔄 Multi-Year Consensus Sweep"])

# =================== TAB 1 ===================
with tab1:
    st.subheader("Single‑Year Walk‑Forward Analysis")
    st.caption("Train on all data before the year, test on that year. Results are cached.")

    # Year selection
    available_years = list(range(2008, 2026))   # adjust to your data range
    selected_year = st.selectbox("Select test year", available_years, index=len(available_years)-1)

    # Check if results exist for this year
    df_result = load_single_year_result(selected_year)

    if df_result.empty:
        st.info(f"No results for {selected_year}. Click below to run the walk‑forward for this year.")
        if st.button(f"🚀 Run Walk‑Forward for {selected_year}"):
            ok, msg = trigger_workflow("single_year_walkforward.yml", {"year": str(selected_year)})
            if ok:
                st.success(f"Workflow triggered! Refresh this page in a few minutes to see the results.")
                st.info("You can check progress in the GitHub Actions tab.")
            else:
                st.error(msg)
    else:
        # Display results
        # Compute metrics from df_result (e.g., daily returns, cumulative curve, etc.)
        # ... (same as you would for the walk‑forward OOS predictions) ...
        st.success(f"✅ Results loaded for {selected_year} ({len(df_result)} trading days)")
        # Example: show top pick per day, cumulative return, etc.
        # You can reuse your existing logic from the walk‑forward tab.

        # For brevity, we'll show a simple cumulative curve
        if "Rank_Score" in df_result.columns:
            # Derive daily signals from the top score
            signals = df_result["Rank_Score"].idxmax(axis=1)
            # ... (backtest) ...
            st.subheader(f"Performance for {selected_year}")
            # ... (plot equity curve) ...

# =================== TAB 2 ===================
with tab2:
    # Existing consensus sweep logic (unchanged)
    st.subheader("Multi‑Year Consensus Sweep")
    # ... (your existing sweep code) ...
