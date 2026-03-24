# app.py — P2-ETF-REGIME-PREDICTOR
# ==================================
# Streamlit UI for the Wasserstein Regime-Aware ETF Rotation Model.
# 
# (Original content preserved; added single‑year on‑demand feature.)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import json
import re
from datetime import datetime, timedelta, timezone
from huggingface_hub import hf_hub_download
import sys

# ── Hugging Face Dataset Configuration ───────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-regime-predictor")
HF_TOKEN        = os.environ.get("HF_TOKEN", None)
GH_PAT          = os.environ.get("GH_PAT", None)
GITHUB_REPO     = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")

# ── Imports from local modules ────────────────────────────────────────────────
try:
    from data_manager_hf import (
        load_momentum_ranker_from_hf,
        load_wf_momentum_predictions_from_hf,
        load_momentum_predictions_from_hf,
        load_wf_ensemble_predictions_from_hf,
        load_model_from_hf,
        load_feature_list_from_hf,
    )
    from regime_detection import RegimeDetector
    from models import MomentumRanker
    from strategy import (
        execute_strategy,
        calculate_metrics,
        calculate_benchmark_metrics,
    )
except ImportError as e:
    st.error(f"Failed to import local modules: {e}")
    sys.exit(1)

# ── Local helper functions (originally in app.py) ────────────────────────────
def _today_est():
    import pytz
    return datetime.now(pytz.timezone("US/Eastern")).date()

def next_trading_day_from_today():
    return _next_trading_day(pd.Timestamp(_today_est()))

def _next_trading_day(last_date):
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (last_date + timedelta(days=14)).strftime("%Y-%m-%d")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            return pd.Timestamp(sched.index[0])
    except ImportError:
        pass
    nxt = last_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt

def regime_colour(regime_name):
    colours = {
        "Risk-On":            "#16a34a",
        "Risk-Off":           "#dc2626",
        "Rate-Rising":        "#d97706",
        "Rate-Falling":       "#3b82f6",
        "Crisis":             "#8b5cf6",
        "Risk-On-Commodity":  "#f97316",
    }
    return colours.get(str(regime_name), "#6b7280")

def conviction_colour(label):
    colours = {
        "Very High": "#16a34a",
        "High":      "#22c55e",
        "Moderate":  "#f97316",
        "Low":       "#dc2626",
    }
    return colours.get(label, "#dc2626")

def compute_conviction(p_beat_cash):
    mean = np.mean(p_beat_cash)
    std  = np.std(p_beat_cash)
    if std < 1e-9:
        return int(np.argmax(p_beat_cash)), 0.0, "Low"
    best = int(np.argmax(p_beat_cash))
    z = float((p_beat_cash[best] - mean) / std)
    label = ("Very High" if z >= 2.0 else
             "High"      if z >= 1.0 else
             "Moderate"  if z >= 0.5 else "Low")
    return best, z, label

# ── Cached loaders (from original) ───────────────────────────────────────────
@st.cache_resource(ttl=3600)
def cached_load_detector():
    try:
        bytes_data = load_model_from_hf("regime_detector.pkl")
        if bytes_data:
            return RegimeDetector.from_bytes(bytes_data)
    except Exception:
        pass
    return None

@st.cache_resource(ttl=3600)
def cached_load_ranker():
    try:
        bytes_data = load_momentum_ranker_from_hf()
        if bytes_data:
            return MomentumRanker.from_bytes(bytes_data)
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def cached_load_signals():
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="signals/signals.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ── Single‑year on‑demand helpers ────────────────────────────────────────────
def load_single_year_result(year: int) -> pd.DataFrame:
    """Load pre‑computed single‑year walk‑forward predictions from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"data/wf_single_year_{year}.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        elif df.index.name != "Date":
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

def trigger_single_year_workflow(year: int) -> tuple[bool, str]:
    """Trigger GitHub Actions workflow for a specific year."""
    if not GH_PAT:
        return False, "GH_PAT secret not set in Streamlit"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/single_year_walkforward.yml/dispatches"
    payload = {"ref": "main", "inputs": {"year": str(year)}}
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {GH_PAT}",
                     "Accept": "application/vnd.github+json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code == 204:
            return True, "✅ Triggered successfully"
        return False, f"❌ GitHub API returned {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"❌ Request failed: {e}"

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_COLS = {"SPY": "SPY_Ret", "AGG": "AGG_Ret"}
DEFAULT_Z_MIN  = 1.0
TARGET_ETFS    = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
SWEEP_YEARS    = [2008, 2013, 2015, 2017, 2019, 2021]

# ── Sidebar (original) ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.title("FTRL Engine")
    st.caption("Wasserstein Regime Detection\nMomentum Ranking (RoC+OBV+Breakout)")
    st.divider()

    st.subheader("⚙️ Parameters")
    start_year = st.slider("Start Year", 2008, 2024, 2010)
    stop_loss  = st.slider("Stop Loss (%)", -30, -5, -12, step=1) / 100.0
    z_reentry  = st.slider("Z Re‑entry Threshold", 0.5, 2.0, 1.0, step=0.1)
    fee_bps    = st.slider("Transaction Cost (bps)", 0, 20, 5)
    st.divider()

    st.subheader("📊 Display")
    benchmark = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    st.divider()

    st.subheader("📊 Backtest Mode")
    bt_mode = st.radio(
        "Validation Method",
        ["Walk‑Forward OOS (Monthly Refresh)", "In‑Sample (Full History)"],
        index=0,
        help="Walk‑Forward: OOS predictions from monthly refresh. "
             "In‑Sample: full fit on all data (may overfit)."
    )
    use_wf = "Walk-Forward" in bt_mode

    run_btn     = st.button("🚀 Run Model", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Force Data Refresh", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF Regime-Aware Rotation Model")
st.caption("Wasserstein k‑means regime detection · Momentum Ranking (RoC + OBV + Breakout) · ETFs: TLT · VNQ · SLV · GLD · LQD · HYG")

# Force refresh handling (original)
if refresh_btn:
    with st.spinner("🔄 Triggering pipeline via GitHub Actions..."):
        try:
            if not GH_PAT:
                raise ValueError("GH_PAT secret not set in Streamlit")
            resp = requests.post(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/daily_pipeline.yml/dispatches",
                headers={"Authorization": f"Bearer {GH_PAT}",
                         "Accept": "application/vnd.github+json"},
                json={"ref": "main"},
                timeout=15,
            )
            if resp.status_code == 204:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Pipeline triggered — GitHub Actions is fetching fresh data. "
                           "Check back in ~5 minutes then click **Run Model** to see updated results.")
            else:
                st.error(f"GitHub API returned {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Trigger failed: {e}")

# ── Sweep helpers (from original) ────────────────────────────────────────────
def _trigger_sweep(years, gh_token, gh_repo):
    ok_all = True
    for y in years:
        url = f"https://api.github.com/repos/{gh_repo}/actions/workflows/single_year_walkforward.yml/dispatches"
        payload = {"ref": "main", "inputs": {"year": str(y)}}
        resp = requests.post(url, headers={"Authorization": f"Bearer {gh_token}",
                                            "Accept": "application/vnd.github+json"},
                             json=payload, timeout=15)
        if resp.status_code != 204:
            ok_all = False
    return ok_all

@st.cache_data(ttl=60, show_spinner=False)
def _load_sweep_cache(date_str, _bust=0):
    cache = {}
    try:
        for yr in SWEEP_YEARS:
            fname = f"sweep/sweep_{yr}_{date_str}.json"
            try:
                path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname,
                                       repo_type="dataset", token=HF_TOKEN)
                with open(path) as f:
                    cache[yr] = json.load(f)
            except Exception:
                pass
    except Exception:
        pass
    return cache

@st.cache_data(ttl=60, show_spinner=False)
def _load_sweep_any(_bust=0):
    found, best_date = {}, None
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
        if not files:
            return found, best_date
        year_best = {}
        for name in files:
            if name.startswith("sweep/sweep_") and name.endswith(".json"):
                parts = name.replace("sweep/sweep_", "").replace(".json", "").split("_")
                if len(parts) >= 2:
                    try:
                        yr = int(parts[0])
                        dt = parts[1]
                        if yr not in year_best or dt > year_best[yr]:
                            year_best[yr] = dt
                    except Exception:
                        pass
        for yr in SWEEP_YEARS:
            if yr not in year_best:
                continue
            best_date = year_best[yr] if best_date is None or year_best[yr] > best_date else best_date
            fname = f"sweep/sweep_{yr}_{year_best[yr]}.json"
            try:
                path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname,
                                       repo_type="dataset", token=HF_TOKEN)
                with open(path) as f:
                    found[yr] = json.load(f)
            except Exception:
                pass
    except Exception:
        pass
    return found, best_date

def _compute_consensus(sweep_data):
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year":       yr,
            "signal":     sig.get("signal", "?"),
            "ann_return": float(sig.get("ann_return", 0.0)),
            "z_score":    float(sig.get("z_score", 0.0)),
            "sharpe":     float(sig.get("sharpe", 0.0)),
            "max_dd":     float(sig.get("max_dd", 0.0)),
            "conviction": sig.get("conviction", "?"),
            "regime":     sig.get("regime", "?"),
        })
    if not rows:
        return {}
    df_c = pd.DataFrame(rows)
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df_c["wtd"] = (0.40 * _mm(df_c["ann_return"]) + 0.20 * _mm(df_c["z_score"]) +
                   0.20 * _mm(df_c["sharpe"])      + 0.20 * _mm(-df_c["max_dd"]))
    etf_agg = {}
    for _, row in df_c.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years": [], "scores": [], "returns": [],
                                "zs": [], "sharpes": [], "dds": []})
        etf_agg[e]["years"].append(row["year"])
        etf_agg[e]["scores"].append(row["wtd"])
        etf_agg[e]["returns"].append(row["ann_return"])
        etf_agg[e]["zs"].append(row["z_score"])
        etf_agg[e]["sharpes"].append(row["sharpe"])
        etf_agg[e]["dds"].append(row["max_dd"])
    summary = {}
    total = sum(v["cum_score"] for v in etf_agg.values())
    for e, v in etf_agg.items():
        summary[e] = {
            "cum_score":   float(sum(v["scores"])),
            "score_share": round(sum(v["scores"]) / total, 3) if total > 0 else 0,
            "n_years":     len(v["years"]),
            "years":       v["years"],
            "avg_return":  round(float(np.mean(v["returns"])), 4),
            "avg_z":       round(float(np.mean(v["zs"])), 3),
            "avg_sharpe":  round(float(np.mean(v["sharpes"])), 3),
            "avg_max_dd":  round(float(np.mean(v["dds"])), 4),
        }
    winner_etf = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner": winner_etf, "etf_summary": summary,
            "per_year": df_c.to_dict("records"), "n_years": len(rows)}

ETF_COLORS_SW = {
    "TLT": "#4e79a7", "VNQ": "#76b7b2", "SLV": "#edc948", "GLD": "#b07aa1",
    "LQD": "#59a14f", "HYG": "#e15759", "CASH": "#aaaaaa",
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single‑Year Results", "🔄 Multi‑Year Consensus Sweep"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single‑Year Results (original global backtest + new on‑demand)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Single‑year on‑demand runner (new) ───────────────────────────────────
    available_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    selected_year = st.selectbox(
        "🔍 Walk‑Forward for a specific year (cache will be used if available)",
        options=["(Global backtest)"] + [str(y) for y in available_years],
        index=0,
        help="Select a test year to see the walk‑forward results for that year only. "
             "If not yet computed, you can trigger a run. This does not affect the global backtest below."
    )

    if selected_year != "(Global backtest)":
        year = int(selected_year)
        st.subheader(f"📅 Walk‑Forward Results – Test Year {year}")
        df_year = load_single_year_result(year)

        if df_year.empty:
            st.info(f"No cached results for {year}. Click below to run the walk‑forward for this year.")
            if st.button(f"🚀 Run Walk‑Forward for {year}"):
                ok, msg = trigger_single_year_workflow(year)
                if ok:
                    st.success("Workflow triggered! Refresh this page in a few minutes to see the results.")
                    st.info("You can check progress in the GitHub Actions tab.")
                else:
                    st.error(msg)
        else:
            st.success(f"✅ Cached results loaded for {year} ({len(df_year)} trading days)")
            # (You may add a simple summary here; for now, just a placeholder)
            st.markdown("**Example: Cumulative Return (placeholder – full backtest will be added)**")
        st.divider()

    # ── Global backtest (original, unchanged) ─────────────────────────────────
    # This is the exact code from the original app.py that appears after the earlier "run_btn" block.
    # It runs the backtest on the full dataset and displays the hero banner, metrics, etc.
    # We must include it exactly as it was.

    # ── Run model logic (original) ───────────────────────────────────────────
    if run_btn or st.session_state.get("auto_run", False):
        st.session_state.auto_run = False
        with st.spinner("Running backtest..."):
            # Load models (cached)
            detector = cached_load_detector()
            if detector is None:
                st.error("Regime detector not loaded — run Manual Retrain first")
                st.stop()
            ranker = cached_load_ranker()
            if ranker is None:
                st.error("Momentum ranker not loaded — run Manual Retrain first")
                st.stop()

            # Load data (cache or fresh)
            try:
                from data_manager_hf import get_data
                df = get_data(start_year=start_year, force_refresh=False)
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()

            # Add regime labels
            df = detector.add_regime_to_df(df)

            # Load predictions
            if use_wf:
                pred_history = load_wf_momentum_predictions_from_hf()
                src_label = "Walk‑Forward OOS"
            else:
                pred_history = load_momentum_predictions_from_hf()
                src_label = "In‑Sample"

            if pred_history is None:
                st.error("Predictions not found — run training first.")
                st.stop()

            # Align indices
            common = pred_history.index.intersection(df.index)
            pred_bt = pred_history.loc[common]
            df_bt = df.loc[common]

            # Daily returns for backtest
            ret_cols = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df_bt.columns]
            daily_rets = df_bt[ret_cols]

            # Risk‑free rate (assume DTB3 column present)
            if "DTB3" in df_bt.columns:
                rf_rate = float(df_bt["DTB3"].iloc[-1] / 100) if not df_bt.empty else 0.045
            else:
                rf_rate = 0.045

            # Execute strategy
            try:
                (strat_rets, audit_trail, _model_next_date, next_signal,
                 conviction_z, conviction_label, last_p) = execute_strategy(
                    predictions_df=pred_bt,
                    daily_ret_df=daily_rets,
                    rf_rate=rf_rate,
                    z_reentry=z_reentry,
                    stop_loss_pct=stop_loss,
                    fee_bps=fee_bps,
                    regime_series=df_bt["Regime_Name"],
                )
                metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

            # Store in session state for later display
            st.session_state.strat_rets = strat_rets
            st.session_state.audit_trail = audit_trail
            st.session_state.metrics = metrics
            st.session_state.next_signal = next_signal
            st.session_state.conviction_z = conviction_z
            st.session_state.conviction_label = conviction_label
            st.session_state.last_p = last_p
            st.session_state.df_bt = df_bt
            st.session_state.pred_bt = pred_bt
            st.session_state.rf_rate = rf_rate
            st.session_state.stop_loss = stop_loss
            st.session_state.z_reentry = z_reentry
            st.session_state.fee_bps = fee_bps
            st.session_state.benchmark = benchmark
            st.session_state.use_wf = use_wf
            st.session_state.src_label = src_label

    # ── Display results if available ──────────────────────────────────────────
    if "strat_rets" in st.session_state:
        strat_rets = st.session_state.strat_rets
        metrics = st.session_state.metrics
        audit_trail = st.session_state.audit_trail
        next_signal = st.session_state.next_signal
        conviction_z = st.session_state.conviction_z
        conviction_label = st.session_state.conviction_label
        last_p = st.session_state.last_p
        df_bt = st.session_state.df_bt
        pred_bt = st.session_state.pred_bt
        rf_rate = st.session_state.rf_rate
        stop_loss = st.session_state.stop_loss
        z_reentry = st.session_state.z_reentry
        fee_bps = st.session_state.fee_bps
        benchmark = st.session_state.benchmark
        use_wf = st.session_state.use_wf
        src_label = st.session_state.src_label

        # Compute true next trading day from today
        true_next_date = next_trading_day_from_today()
        pred_last_date = pred_bt.index[-1].date()
        today_est_date = _today_est()
        days_stale = (today_est_date - pred_last_date).days

        if days_stale > 1:
            st.warning(
                f"⚠️ **Predictions are stale** — last update: **{pred_last_date}** "
                f"({days_stale} calendar days ago). "
                "The signal shown is based on the most recent available data.",
                icon="📅"
            )

        # ── Hero banner ───────────────────────────────────────────────────────
        st.divider()
        regime_name = df_bt["Regime_Name"].iloc[-1] if "Regime_Name" in df_bt.columns else "?"
        regime_col = regime_colour(regime_name)
        conv_col = conviction_colour(conviction_label)

        accent_col = ("#16a34a" if conviction_label in ("High", "Very High")
                      else "#d97706" if conviction_label == "Moderate"
                      else "#dc2626")

        st.markdown(f"""
        <div style="background:white; border-left:6px solid {accent_col};
                    border-radius:12px; padding:18px 24px; margin-bottom:24px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.1);">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:12px; color:#6b7280; letter-spacing:1px;">
                NEXT TRADING DAY SIGNAL — {true_next_date.strftime('%A %b %d, %Y')}
              </div>
              <div style="font-size:44px; font-weight:800; color:#1a1a2e; line-height:1.2;">
                {next_signal}
              </div>
              <div style="font-size:14px; color:#6b7280; margin-top:4px;">
                Conviction: {conviction_label} | Z = {conviction_z:.2f}σ
              </div>
            </div>
            <div style="text-align:right;">
              <div style="color:#6b7280; font-size:13px; margin-bottom:4px;">CURRENT REGIME</div>
              <div style="background:{regime_col}18; border:1px solid {regime_col};
                          border-radius:8px; padding:8px 20px; display:inline-block;">
                <span style="color:{regime_col}; font-size:20px; font-weight:700;">
                  {regime_name}
                </span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ETF probability bars ───────────────────────────────────────────────
        st.subheader("P(Beat Cash) — Next 5 Days")

        base_rates = {t: 0.5 for t in TARGET_ETFS}
        row1_etfs = TARGET_ETFS[:4]
        row2_etfs = TARGET_ETFS[4:]
        prob_row1 = st.columns(len(row1_etfs))
        prob_row2 = st.columns(len(row2_etfs))

        for row_etfs, row_cols in [(row1_etfs, prob_row1), (row2_etfs, prob_row2)]:
            for col, etf in zip(row_cols, row_etfs):
                idx = TARGET_ETFS.index(etf)
                p_val = last_p[idx] if idx < len(last_p) else 0.5
                base = base_rates.get(etf, 0.5)
                col.metric(
                    label=etf,
                    value=f"{p_val:.1%}",
                    delta=f"{p_val - base:+.1%} vs baseline",
                    delta_color="normal" if p_val > base else "inverse"
                )

        # ── Performance metrics ───────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Backtest Performance")

        excess = metrics.get("ann_return", 0) - rf_rate
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 Ann. Return", f"{metrics.get('ann_return',0)*100:.2f}%",
                  delta=f"{excess*100:+.1f}pp vs T-Bill")
        c2.metric("📊 Sharpe", f"{metrics.get('sharpe',0):.2f}",
                  delta="Above 1.0 ✓" if metrics.get("sharpe", 0) > 1 else "Below 1.0")
        c3.metric("🎯 Hit Ratio (Full)", f"{metrics.get('hit_ratio',0)*100:.0f}%",
                  delta="Strong" if metrics.get("hit_ratio", 0) > 0.6 else "Weak")
        c4.metric("📉 Max Drawdown", f"{metrics.get('max_dd',0)*100:.2f}%",
                  delta="Peak to Trough")

        c5, c6, c7, c8 = st.columns(4)
        dd_idx = metrics.get("max_dd_idx", 0)
        dd_date = df_bt.index[dd_idx].date() if dd_idx < len(df_bt) else "?"
        c5.metric("🏆 Calmar", f"{metrics.get('calmar',0):.2f}",
                  delta=f"MaxDD on {dd_date}")
        c6.metric("✅ Avg Win", f"{metrics.get('avg_win',0)*100:.2f}%", delta="Daily")
        c7.metric("❌ Avg Loss", f"{metrics.get('avg_loss',0)*100:.2f}%", delta="Daily")
        c8.metric("⚖️ Win/Loss", f"{metrics.get('win_loss_r',0):.2f}x", delta="Ratio")

        # ── Equity curve ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Equity Curve")
        cum_rets = metrics.get("cum_returns", np.array([1]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_bt.index[-len(cum_rets):],
            y=cum_rets,
            mode="lines",
            name="Strategy",
            line=dict(color="#0e9f6e", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(14,159,110,0.1)",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            template="plotly_white", height=420,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#eeeeee", title="Cumulative Return (×)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Regime timeline ──────────────────────────────────────────────────
        if "Regime_Name" in df_bt.columns:
            st.divider()
            st.subheader("🗺️ Market Regime Timeline")
            regime_df = df_bt[["Regime_Name"]].reindex(pred_bt.index).ffill()
            fig_r = go.Figure()
            regime_list = regime_df["Regime_Name"].unique()
            for rname in regime_list:
                mask = regime_df["Regime_Name"] == rname
                fig_r.add_trace(go.Scatter(
                    x=regime_df.index[mask],
                    y=[rname] * mask.sum(),
                    mode="markers",
                    marker=dict(symbol="square", size=6, color=regime_colour(str(rname))),
                    name=str(rname),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{rname}<extra></extra>",
                ))
            fig_r.update_layout(
                template="plotly_white", height=180,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Audit trail ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Audit Trail — Last 30 Trading Days")
        if audit_trail:
            audit_df = pd.DataFrame(audit_trail).tail(30)
            def style_signal(val):
                if val == "CASH":
                    return "color: #888888"
                colours = {"TLT": "#00bfff", "VNQ": "#ffd700", "SLV": "#c0c0c0",
                           "GLD": "#ffa500", "LQD": "#88ddff", "HYG": "#ff9966"}
                return f"color: {colours.get(val, '#ffffff')}; font-weight: 600"
            def style_regime(val):
                col = regime_colour(val)
                return f"color: {col}; font-weight: 500"
            def style_return(val):
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        v = float(val.replace("%", ""))
                        return "color: #00b894; font-weight:600" if v > 0 else "color: #d63031; font-weight:600"
                    except:
                        pass
                return ""
            st.dataframe(
                audit_df.style
                .applymap(style_signal, subset=["Signal", "Top_Pick"])
                .applymap(style_regime, subset=["Regime"])
                .applymap(style_return, subset=["Signal_Ret%", "TLT_Ret%", "VNQ_Ret%",
                                                "SLV_Ret%", "GLD_Ret%", "LQD_Ret%", "HYG_Ret%"])
                .set_properties(**{"text-align": "center"})
                .hide(axis="index"),
                use_container_width=True
            )
        else:
            st.info("No audit trail available yet.")

        # ── Momentum weights (optional) ──────────────────────────────────────
        st.divider()
        st.subheader("⚖️ Option B — Momentum Score Weights")
        weights_data = {
            "Component": [
                "RoC 5d (×0.40)",
                "RoC 10d (×0.30)",
                "RoC 21d (×0.20)",
                "RoC 63d (×0.10)",
                "OBV Accumulation 21d (×0.15)",
                "Breakout vs 20d Range (×0.15)",
            ],
            "Weight": [0.40, 0.30, 0.20, 0.10, 0.15, 0.15],
            "Category": ["Momentum", "Momentum", "Momentum", "Momentum", "Volume", "Breakout"],
        }
        wdf = pd.DataFrame(weights_data)
        fig_i = go.Figure(go.Bar(
            x=wdf["Weight"],
            y=wdf["Component"],
            orientation="h",
            marker_color=["#00d1b2"]*4 + ["#ffa500"] + ["#ff6b6b"],
            hovertemplate="%{y}<br>Weight: %{x:.2f}<extra></extra>",
        ))
        fig_i.update_layout(
            template="plotly_white", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Weight", showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(autorange="reversed", showgrid=False),
        )
        st.plotly_chart(fig_i, use_container_width=True)

        # ── Signal history from HF ──────────────────────────────────────────
        st.divider()
        st.subheader("📡 Signal History (from Hugging Face)")
        try:
            sig_df = cached_load_signals()
            if sig_df is not None and not sig_df.empty:
                sig_df = sig_df.tail(30).sort_index(ascending=False)
                st.dataframe(sig_df, use_container_width=True)
            else:
                st.info("No signal history found.")
        except Exception as e:
            st.info(f"Could not load signal history: {e}")

    else:
        st.info("Click **Run Model** in the sidebar to start the backtest.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi‑Year Consensus Sweep (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── Consensus sweep UI (original) ────────────────────────────────────────
    st.subheader("Multi‑Year Consensus Sweep")
    st.caption(
        "Runs the regime‑aware rotation model across 6 start years and aggregates signals into a weighted consensus.  \n"
        "Sweep years: 2008, 2013, 2015, 2017, 2019, 2021  ·  Score: 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)  \n"
        "Auto‑runs daily at 8pm EST. Results are date‑stamped — stale cache never shown."
    )

    # Sweep years and display (identical to original)
    today_sw = datetime.now(pytz.timezone("US/Eastern")).strftime("%Y%m%d")
    st.session_state.setdefault("sweep_bust", 0)

    today_cache          = _load_sweep_cache(today_sw, _bust=st.session_state.sweep_bust)
    all_cache, best_date = _load_sweep_any(_bust=st.session_state.sweep_bust)
    prev_cache           = {yr: v for yr, v in all_cache.items() if yr not in today_cache}
    display_cache        = {**prev_cache, **today_cache}
    display_date         = today_sw if today_cache else best_date
    sweep_complete       = len(today_cache) == len(SWEEP_YEARS)

    if prev_cache and not sweep_complete:
        st.warning(
            f"⚠️ Displaying cached sweep results from **{display_date}**. "
            "Today's run is still in progress (⏳)."
        )

    # Visual status per year
    col_status = st.columns(len(SWEEP_YEARS))
    for i, yr in enumerate(SWEEP_YEARS):
        with col_status[i]:
            if yr in today_cache:
                st.success(f"{yr} ✅")
            elif yr in prev_cache:
                st.info(f"{yr} 📅")
            else:
                st.warning(f"{yr} ⏳")
    st.caption("✅ today · 📅 previous · ⏳ not run")
    st.divider()

    _missing     = [y for y in SWEEP_YEARS if y not in today_cache]
    _force_rerun = st.checkbox("🔄 Force re-run all years", value=False,
                               help="Re-trains even if today's results already exist")
    _trigger_yrs = SWEEP_YEARS if _force_rerun else _missing

    _cb, _cr, _ci = st.columns([1, 1, 2])
    with _cb:
        _sweep_btn = st.button("▶️ Run Missing Years", use_container_width=True,
                                disabled=len(_trigger_yrs)==0 and not _force_rerun)
    with _cr:
        if st.button("🔄 Refresh Cache", use_container_width=True):
            st.session_state.sweep_bust += 1
            st.rerun()
    with _ci:
        if sweep_complete and not _force_rerun:
            st.success(f"✅ Today's sweep complete ({today_sw}) — all {len(SWEEP_YEARS)} years ready")
        else:
            st.info(f"**{len(today_cache)}/{len(SWEEP_YEARS)}** years done today · "
                    f"**{len(display_cache)}/{len(SWEEP_YEARS)}** total available.  \n"
                    f"Will trigger **{len(_trigger_yrs)}** jobs: {', '.join(str(y) for y in _trigger_yrs)}")

    if _sweep_btn and _trigger_yrs:
        try:
            if not GH_PAT:
                raise ValueError("GH_PAT secret not set in Streamlit")
            _ok = _trigger_sweep(_trigger_yrs, GH_PAT, GITHUB_REPO)
            if _ok:
                st.success(f"✅ Triggered {len(_trigger_yrs)} jobs: "
                           f"{', '.join(str(y) for y in _trigger_yrs)}. ~10-15 mins each.")
            else:
                st.error("❌ GitHub API returned non-204. Check GH_PAT permissions.")
        except Exception as _ex:
            st.error(f"❌ Trigger failed: {_ex}")

    # Display consensus if any data
    if display_cache:
        _cons = _compute_consensus(display_cache)
        if not _cons:
            st.warning("Could not compute consensus.")
        else:
            _w    = _cons["winner"]
            _wi   = _cons["etf_summary"][_w]
            _wc   = ETF_COLORS_SW.get(_w, "#0066cc")
            _sp   = _wi["score_share"] * 100
            _sp   = _sp if _sp == _sp else 0.0
            _slab = "⚠️ Split Signal" if _wi["score_share"] < 0.40 else "✅ Clear Consensus"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                        border-radius:16px; padding:28px 32px; margin:20px 0;">
              <div style="font-size:12px; letter-spacing:2px; color:#aaa;">WEIGHTED CONSENSUS · REGIME-PREDICTOR · {display_date}</div>
              <div style="font-size:60px; font-weight:900; color:{_wc}; line-height:1.1;">{_w}</div>
              <div style="font-size:14px; color:#ccc; margin-top:4px;">{_slab} · Score share {_sp:.0f}% · {_wi['n_years']}/{len(SWEEP_YEARS)} years</div>
              <div style="display:flex; gap:32px; flex-wrap:wrap; margin-top:20px;">
                <div><span style="color:#aaa;">Avg Return</span><br><span style="font-size:24px; font-weight:600;">{_wi['avg_return']*100:.1f}%</span></div>
                <div><span style="color:#aaa;">Avg Z</span><br><span style="font-size:24px; font-weight:600;">{_wi['avg_z']:.2f}σ</span></div>
                <div><span style="color:#aaa;">Avg Sharpe</span><br><span style="font-size:24px; font-weight:600;">{_wi['avg_sharpe']:.2f}</span></div>
                <div><span style="color:#aaa;">Avg MaxDD</span><br><span style="font-size:24px; font-weight:600;">{_wi['avg_max_dd']*100:.1f}%</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Top ETFs (others)
            _others = sorted([(e, v) for e, v in _cons["etf_summary"].items() if e != _w],
                             key=lambda x: -x[1]["cum_score"])
            if _others:
                _parts = [f'<span style="color:{ETF_COLORS_SW.get(e,"#888")};font-weight:600;">{e}</span> '
                          f'<span style="color:#aaa;">({v["cum_score"]:.2f})</span>' for e, v in _others]
                st.markdown('<div style="text-align:center;font-size:13px;margin-bottom:12px;">Also ranked: '
                            + ' &nbsp;|&nbsp; '.join(_parts) + '</div>', unsafe_allow_html=True)

            st.divider()

            # Charts
            import plotly.graph_objects as _go
            _c1, _c2 = st.columns(2)
            with _c1:
                st.markdown("**Weighted Score per ETF**")
                _es   = _cons["etf_summary"]
                _setf = sorted(_es.keys(), key=lambda e: -_es[e]["cum_score"])
                _fb   = _go.Figure(_go.Bar(
                    x=_setf,
                    y=[_es[e]["cum_score"] for e in _setf],
                    marker_color=[ETF_COLORS_SW.get(e,"#888") for e in _setf],
                    text=[f"{_es[e]['n_years']}yr · {_es[e]['score_share']*100:.0f}%"
                          for e in _setf],
                    textposition="outside",
                ))
                _fb.update_layout(template="plotly_dark", height=360,
                                  yaxis_title="Cumulative Score", showlegend=False,
                                  margin=dict(t=20,b=20))
                st.plotly_chart(_fb, use_container_width=True)

            with _c2:
                st.markdown("**Z-Score Conviction by Start Year**")
                _fs = _go.Figure()
                for _row in _cons["per_year"]:
                    _etf = _row["signal"]
                    _col = ETF_COLORS_SW.get(_etf, "#888")
                    _fs.add_trace(_go.Scatter(
                        x=[_row["year"]], y=[_row["z_score"]],
                        mode="markers+text",
                        marker=dict(size=18, color=_col, line=dict(color="white",width=1)),
                        text=[_etf], textposition="top center", showlegend=False,
                        hovertemplate=(f"<b>{_etf}</b><br>Year: {_row['year']}<br>"
                                       f"Z: {_row['z_score']:.2f}σ<br>"
                                       f"Return: {_row['ann_return']*100:.1f}%<extra></extra>"),
                    ))
                _fs.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
                _fs.update_layout(template="plotly_dark", height=360,
                                  xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
                                  margin=dict(t=20,b=20))
                st.plotly_chart(_fs, use_container_width=True)

            st.subheader("📋 Full Per-Year Breakdown")
            _tbl = []
            for _row in _cons["per_year"]:
                _tbl.append({
                    "Start Year":   _row["year"],
                    "Signal":       _row["signal"],
                    "Regime":       _row.get("regime", "?"),
                    "Conviction":   _row.get("conviction", "?"),
                    "Wtd Score":    round(_row["wtd"], 3),
                    "Z-Score":      f"{_row['z_score']:.2f}σ",
                    "Ann. Return":  f"{_row['ann_return']*100:.2f}%",
                    "Sharpe":       f"{_row['sharpe']:.2f}",
                    "Max DD":       f"{_row['max_dd']*100:.2f}%",
                })
            _tdf = pd.DataFrame(_tbl)
            def _ss(val):
                c = ETF_COLORS_SW.get(val, "#888")
                return f"background-color:{c}22;color:{c};font-weight:700;"
            def _sr(val):
                try:
                    v = float(str(val).replace("%", ""))
                    return "color:#00b894;font-weight:600" if v > 0 else "color:#d63031;font-weight:600"
                except Exception:
                    return ""
            st.dataframe(_tdf.style.applymap(_ss, subset=["Signal"])
                                   .applymap(_sr, subset=["Ann. Return"])
                                   .set_properties(**{"text-align": "center"})
                                   .hide(axis="index"),
                         use_container_width=True, height=280)
    else:
        st.info("No sweep results yet. Run the sweep above to start.")

# ── Footer (unchanged) ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FTRL Engine | Data: P2SAMAPA/p2-etf-regime-predictor | "
    "Methodology: Wasserstein k‑means regime detection (Horvath et al., 2021) + Momentum Ranking | "
    "⚠️ Not financial advice"
)
