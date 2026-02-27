"""
app.py — P2-ETF-REGIME-PREDICTOR
==================================
Streamlit UI for the Wasserstein Regime-Aware ETF Rotation Model.

Architecture:
  - Loads pre-trained models and signals from GitLab (read-only)
  - Runs backtest on historical predictions for equity curve
  - Displays next-day signal, regime, conviction, and audit trail
  - No training happens in the UI — all compute is in GitHub Actions

Reference:
  "Clustering Market Regimes Using the Wasserstein Distance"
  B. Horvath, Z. Issa, A. Muguruza (2021)
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905

Author: P2SAMAPA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pickle
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF Regime Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from data_manager import (
        get_data, build_forward_targets,
        load_signals_from_gitlab, load_model_from_gitlab,
        load_feature_list_from_gitlab, load_predictions_from_gitlab,
        load_momentum_ranker_from_gitlab, load_momentum_predictions_from_gitlab,
        TARGET_ETFS,
    )
    from regime_detection import RegimeDetector
    from models import RegimeModelBank, MomentumRanker
    from strategy import (
        execute_strategy, calculate_metrics,
        calculate_benchmark_metrics, TARGET_ETFS as STRAT_ETFS,
    )
    from utils import get_est_time, regime_colour, conviction_colour
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# ── Cached loaders (prevent re-running on every widget interaction) ──────────

@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_data(start_year, force=False):
    return get_data(start_year=start_year, force_refresh=force)

def cached_load_predictions():
    # Not cached — must filter by start_year, caching would return stale full history
    return load_predictions_from_gitlab()

@st.cache_resource(ttl=3600, show_spinner=False)
def cached_load_detector():
    data = load_model_from_gitlab("regime_detector.pkl")
    if data:
        return RegimeDetector.from_bytes(data)
    return None

@st.cache_resource(ttl=3600, show_spinner=False)
def cached_load_bank():
    data = load_model_from_gitlab("model_bank.pkl")
    if data:
        return RegimeModelBank.from_bytes(data)
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_feature_list():
    return load_feature_list_from_gitlab()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_signals():
    return load_signals_from_gitlab()

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_COLS = {"SPY": "SPY_Ret", "AGG": "AGG_Ret"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("P2-ETF Regime Predictor")
    st.caption("Wasserstein Regime Detection + LightGBM Ensemble")

    now_est = get_est_time()
    st.write(f"🕒 **EST:** {now_est.strftime('%a %b %d, %H:%M')}")
    st.divider()

    st.header("⚙️ Configuration")

    start_year = st.slider(
        "Training Start Year", min_value=2008, max_value=2025,
        value=2008, step=1,
        help="Earlier start captures more rate cycles"
    )
    st.divider()

    st.subheader("🛡️ Risk Controls")
    z_reentry = st.slider(
        "Re-entry after Stop (Z)", min_value=0.5, max_value=3.0,
        value=1.0, step=0.1,
        help="Conviction required to re-enter after stop-loss triggers"
    )
    stop_loss_pct = st.slider(
        "Stop-Loss Threshold (%)", min_value=-25, max_value=-3,
        value=-12, step=1,
        help="2-day cumulative loss to trigger CASH exit"
    ) / 100
    fee_bps = st.number_input(
        "Transaction Fee (bps)", min_value=0, max_value=50,
        value=5, step=1,
        help="One-way cost per trade in basis points"
    )
    st.divider()

    st.subheader("📊 Display")
    benchmark = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    show_wfcv = st.checkbox("Show Walk-Forward CV", value=False)
    st.divider()
    st.subheader("🧠 Layer 2 Mode")
    layer2_mode = st.radio(
        "ETF Ranking Method",
        options=["Option A — ML Ensemble", "Option B — Momentum"],
        index=0,
        help=(
            "**Option A — ML Ensemble**: XGBoost + Ridge + LambdaRank trained "
            "per market regime. Enters only when ≥2/3 models agree. "
            "Best when macro patterns are regime-consistent.\n\n"
            "**Option B — Momentum**: Pure Rate-of-Change ranking — 5d/10d/21d/63d "
            "RoC + volume accumulation + 20d breakout score. No ML, fully "
            "transparent. Best when trends are strong and persistent."
        )
    )
    use_momentum = "Option B" in layer2_mode

    st.divider()
    run_btn = st.button("🚀 Run Model", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Force Data Refresh", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF Regime-Aware Rotation Model")
mode_label = ("Option B — Momentum (RoC + Volume + Breakout)"
               if use_momentum else
               "Option A — ML Ensemble (XGBoost + Ridge + LambdaRank voting)")
st.caption(
    f"Wasserstein k-means regime detection • {mode_label} • "
    "ETFs: TLT · VNQ · SLV · GLD · LQD · HYG"
)

if "refresh_status" not in st.session_state:
    st.session_state.refresh_status = None

if refresh_btn:
    with st.spinner("🔄 Triggering pipeline via GitHub Actions..."):
        try:
            import requests, os
            gh_token = os.environ.get("GITHUB_PAT", "")
            gh_repo  = os.environ.get("GITHUB_REPO",
                        "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
            if not gh_token:
                raise ValueError("GITHUB_PAT secret not set in Streamlit")
            resp = requests.post(
                f"https://api.github.com/repos/{gh_repo}/actions/workflows/daily_pipeline.yml/dispatches",
                headers={
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github+json",
                },
                json={"ref": "main"},
                timeout=15,
            )
            if resp.status_code == 204:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.refresh_status = (
                    "ok",
                    "✅ Pipeline triggered — GitHub Actions is fetching fresh data. "
                    "Check back in ~5 minutes then click **Run Model** to see updated results."
                )
            else:
                raise ValueError(f"GitHub API returned {resp.status_code}: {resp.text}")
        except Exception as e:
            st.session_state.refresh_status = ("err", f"❌ Trigger failed: {e}")

if st.session_state.refresh_status:
    status, msg = st.session_state.refresh_status
    if status == "ok":
        st.success(msg)
    else:
        st.error(msg)

if not run_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Configure risk controls in the sidebar")
    with col2:
        st.info("**Step 2** — Click **Run Model** to load and display results")
    with col3:
        st.info("**Step 3** — Review signal, regime, and audit trail")
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
# ── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("📥 Loading dataset from GitLab..."):
    try:
        df = cached_get_data(start_year)
        st.success(f"✅ Dataset: {len(df):,} rows × {df.shape[1]} cols "
                   f"({df.index[0].date()} → {df.index[-1].date()})")
    except Exception as e:
        st.error(f"❌ Data load failed: {e}")
        st.stop()

# ── Refresh button handler ───────────────────────────────────────────────────

# ── Load models from GitLab ───────────────────────────────────────────────────
with st.spinner("🧠 Loading models from GitLab..."):
    detector = cached_load_detector()

    if detector:
        st.success(f"✅ Regime detector loaded (k={detector.optimal_k_})")
    else:
        st.warning("⚠️ Regime detector not found — run pipeline first")

    if use_momentum:
        # Layer 2B — Momentum
        mom_bytes = load_momentum_ranker_from_gitlab()
        if mom_bytes:
            momentum_ranker = MomentumRanker.from_bytes(mom_bytes)
            bank = None
            st.success("✅ Momentum ranker loaded (Option B — RoC + Volume + Breakout)")
        else:
            st.error("⚠️ Momentum ranker not found — run Manual Retrain first")
            momentum_ranker = None
            bank = None
    else:
        # Layer 2A — ML Ensemble
        bank = cached_load_bank()
        momentum_ranker = None
        if bank:
            n_regime = len(getattr(bank, "models_", {}))
            st.success(f"✅ ML Ensemble loaded — Option A "
                       f"({n_regime} regime models, XGBoost+Ridge+LambdaRank voting)")
        else:
            st.warning("⚠️ ML model bank not found — run pipeline first")

if detector is None or (bank is None and momentum_ranker is None):
    st.error("Models not available. Trigger Manual Retrain in GitHub Actions first.")
    st.stop()

# ── Risk-free rate ─────────────────────────────────────────────────────────────
rf_rate  = 0.045
rf_label = "fallback 4.5%"
try:
    import pandas_datareader.data as web
    dtb3 = web.DataReader("DTB3", "fred", start="2024-01-01").dropna()
    if not dtb3.empty:
        rf_rate  = float(dtb3.iloc[-1].values[0]) / 100
        rf_label = f"FRED DTB3 ({dtb3.index[-1].date()})"
except Exception:
    if "DTB3" in df.columns:
        rf_rate  = float(df["DTB3"].dropna().iloc[-1]) / 100
        rf_label = "dataset DTB3"
st.caption(f"📊 Risk-free rate (3M T-Bill): **{rf_rate*100:.2f}%** — {rf_label}")

# ── Regime detection + feature prep ──────────────────────────────────────────
with st.spinner("🔍 Applying regime labels..."):
    try:
        df = detector.add_regime_to_df(df)
        df["Regime"]      = df["Regime"].fillna(0).astype(int)
        df["Regime_Name"] = df["Regime_Name"].fillna("Global")
    except Exception as e:
        st.warning(f"Regime labelling failed: {e} — using global regime")
        df["Regime"]      = 0
        df["Regime_Name"] = "Global"

    # Get current regime
    try:
        regime_int, regime_name = detector.get_current_regime(df)
    except Exception:
        regime_int, regime_name = 0, "Unknown"

# ── Generate prediction history ───────────────────────────────────────────────
with st.spinner("📡 Loading predictions from GitLab..."):
    try:
        feature_cols = cached_load_feature_list()
        if feature_cols is None:
            from models import get_feature_columns
            feature_cols = get_feature_columns(df)

        if use_momentum:
            pred_history = load_momentum_predictions_from_gitlab()
            if pred_history is None:
                st.warning("⚠️ Momentum predictions not found — generating now...")
                pred_history = momentum_ranker.predict_all_history(df)
        else:
            pred_history = cached_load_predictions()
            if pred_history is None:
                st.warning("⚠️ ML predictions not found — generating now (slow)...")
                feat_df = df.copy()
                feat_df[feature_cols] = (feat_df[feature_cols]
                                          .fillna(feat_df[feature_cols].median())
                                          .fillna(0.0))
                pred_history = bank.predict_all_history(feat_df)

        # Ensure DatetimeIndex and confirm load
        if pred_history is not None:
            pred_history.index = pd.to_datetime(pred_history.index)
            st.success(f"✅ Predictions loaded: {len(pred_history):,} rows "
                       f"({pred_history.index[0].date()} → {pred_history.index[-1].date()})")
    except Exception as e:
        st.error(f"❌ Prediction load failed: {e}")
        st.stop()

# ── Execute strategy ──────────────────────────────────────────────────────────
with st.spinner("📊 Running backtest..."):
    # Daily return columns
    ret_cols   = [f"{t}_Ret" for t in TARGET_ETFS if f"{t}_Ret" in df.columns]
    daily_rets = df[ret_cols]
    regime_ser = df["Regime_Name"] if "Regime_Name" in df.columns else None

    try:
        # Filter to start_year for backtest — pred_history covers full history
        cutoff = pd.Timestamp(f"{start_year}-01-01")

        # Ensure index is DatetimeIndex for filtering
        if not isinstance(pred_history.index, pd.DatetimeIndex):
            pred_history.index = pd.to_datetime(pred_history.index)
        if not isinstance(daily_rets.index, pd.DatetimeIndex):
            daily_rets.index = pd.to_datetime(daily_rets.index)

        pred_bt = pred_history[pred_history.index >= cutoff]
        rets_bt = daily_rets[daily_rets.index >= cutoff]
        reg_bt  = regime_ser[regime_ser.index >= cutoff]                   if regime_ser is not None else None

        st.info(
            f"DEBUG — start_year={start_year} | cutoff={cutoff.date()} | "
            f"pred_history rows={len(pred_history)} | "
            f"pred_history index type={type(pred_history.index).__name__} | "
            f"pred_history first={pred_history.index[0]} | "
            f"pred_bt rows={len(pred_bt)} | "
            f"pred_bt first={pred_bt.index[0].date() if len(pred_bt)>0 else 'EMPTY'}"
        )

        (strat_rets, audit_trail, next_date, next_signal,
         conviction_z, conviction_label, last_p) = execute_strategy(
            predictions_df = pred_bt,
            daily_ret_df   = rets_bt,
            rf_rate        = rf_rate,
            z_reentry      = z_reentry,
            stop_loss_pct  = stop_loss_pct,
            fee_bps        = fee_bps,
            regime_series  = reg_bt,
        )
        metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL BANNER
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
regime_col = regime_colour(regime_name)
conv_col   = conviction_colour(conviction_label)

banner_bg = ("#1a3a1a" if conviction_label in ("High", "Very High")
             else "#3a2a1a" if conviction_label == "Moderate"
             else "#2a1a1a")

st.markdown(f"""
<div style="background:{banner_bg};border-radius:12px;padding:24px 32px;
            margin-bottom:16px;border:1px solid #333;">
  <div style="display:flex;justify-content:space-between;align-items:center;
              flex-wrap:wrap;gap:16px;">
    <div>
      <div style="color:#888;font-size:13px;margin-bottom:4px;">
        NEXT TRADING DAY SIGNAL — {next_date.strftime('%A %b %d, %Y')}
      </div>
      <div style="font-size:42px;font-weight:800;color:#ffffff;
                  letter-spacing:2px;">{next_signal}</div>
      <div style="color:#aaa;font-size:13px;margin-top:4px;">
        Conviction: <span style="color:{conv_col};font-weight:600;">
        {conviction_label}</span> &nbsp;|&nbsp; Z = {conviction_z:+.2f}σ
      </div>
    </div>
    <div style="text-align:right;">
      <div style="color:#888;font-size:13px;margin-bottom:4px;">
        CURRENT REGIME</div>
      <div style="background:{regime_col}22;border:1px solid {regime_col};
                  border-radius:8px;padding:8px 20px;display:inline-block;">
        <span style="color:{regime_col};font-size:20px;font-weight:700;">
        {regime_name}</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ETF probability bars ──────────────────────────────────────────────────────
st.subheader("P(Beat Cash) — Next 5 Days")

# Get base rates from model bank if available
base_rates = getattr(bank, "base_rates_", {t: 0.5 for t in TARGET_ETFS})              if bank else {t: 0.5 for t in TARGET_ETFS}

# Split into two rows for 7 ETFs
row1_etfs = TARGET_ETFS[:4]
row2_etfs = TARGET_ETFS[4:]
prob_row1 = st.columns(len(row1_etfs))
prob_row2 = st.columns(len(row2_etfs))

for row_etfs, row_cols in [(row1_etfs, prob_row1), (row2_etfs, prob_row2)]:
    for col, etf in zip(row_cols, row_etfs):
        i    = TARGET_ETFS.index(etf)
        p    = float(last_p[i]) if i < len(last_p) else 0.5
        base = base_rates.get(etf, 0.5)
        adj  = p - base
        col.metric(
            label=f"{etf}  (base {base*100:.0f}%)",
            value=f"{p*100:.1f}%",
            delta=f"{adj*100:+.1f}pp vs base",
            delta_color="normal"
        )

# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 Backtest Performance")

c1, c2, c3, c4, c5 = st.columns(5)
excess = (metrics.get("ann_return", 0) - rf_rate) * 100
c1.metric("📈 Ann. Return",
          f"{metrics.get('ann_return',0)*100:.2f}%",
          delta=f"{excess:+.1f}pp vs T-Bill")
c2.metric("📊 Sharpe",
          f"{metrics.get('sharpe',0):.2f}",
          delta="Above 1.0 ✓" if metrics.get("sharpe",0) > 1 else "Below 1.0")
c3.metric("🎯 Hit Ratio (Full)",
          f"{metrics.get('hit_ratio',0)*100:.0f}%",
          delta="Strong" if metrics.get("hit_ratio",0) > 0.6 else "Weak")
c4.metric("📉 Max Drawdown",
          f"{metrics.get('max_dd',0)*100:.2f}%",
          delta="Peak to Trough")
c5.metric("⚠️ Worst Day",
          f"{metrics.get('max_daily_dd',0)*100:.2f}%",
          delta="Max Daily Loss")

# Secondary metrics row
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("🏆 Calmar",
          f"{metrics.get('calmar',0):.2f}",
          delta="Ann Return / MaxDD")
c7.metric("✅ Avg Win",
          f"{metrics.get('avg_win',0)*100:.2f}%",
          delta="Daily")
c8.metric("❌ Avg Loss",
          f"{metrics.get('avg_loss',0)*100:.2f}%",
          delta="Daily")
c9.metric("⚖️ Win/Loss",
          f"{metrics.get('win_loss_r',0):.2f}x",
          delta="Ratio")
c10.metric("📅 Test Days",
           f"{metrics.get('n_days',0):,}",
           delta="Trading Days")

# ═════════════════════════════════════════════════════════════════════════════
# EQUITY CURVE
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📈 Equity Curve")

plot_dates = pred_bt.index[:len(metrics.get("cum_returns", []))]
cum_rets   = metrics.get("cum_returns", np.array([]))

fig = go.Figure()

# Strategy line
fig.add_trace(go.Scatter(
    x=plot_dates, y=cum_rets,
    name="Strategy",
    line=dict(color="#00d1b2", width=2.5),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra>Strategy</extra>",
))

# Benchmark
if benchmark != "None":
    bm_col = BENCHMARK_COLS.get(benchmark)
    if bm_col and bm_col in df.columns:
        bm_rets = df.reindex(plot_dates)[bm_col].fillna(0).values
        bm_cum  = calculate_benchmark_metrics(bm_rets, rf_rate)
        if bm_cum:
            fig.add_trace(go.Scatter(
                x=plot_dates,
                y=bm_cum.get("cum_returns", []),
                name=benchmark,
                line=dict(color="#888888", width=1.5, dash="dot"),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:.3f}}x<extra>{benchmark}</extra>",
            ))

# Drawdown shading
cum_max = metrics.get("cum_max", cum_rets)
fig.add_trace(go.Scatter(
    x=np.concatenate([plot_dates, plot_dates[::-1]]),
    y=np.concatenate([cum_max, cum_rets[::-1]]),
    fill="toself",
    fillcolor="rgba(255,100,100,0.08)",
    line=dict(width=0),
    name="Drawdown",
    showlegend=True,
    hoverinfo="skip",
))

fig.update_layout(
    template="plotly_white",
    height=420,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#eeeeee",
               title="Cumulative Return (×)"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# REGIME TIMELINE
# ═════════════════════════════════════════════════════════════════════════════
if "Regime_Name" in df.columns:
    st.divider()
    st.subheader("🗺️ Market Regime Timeline")

    regime_df = df[["Regime_Name"]].reindex(plot_dates).ffill()
    fig_r     = go.Figure()

    regime_list = regime_df["Regime_Name"].unique()
    for rname in regime_list:
        mask = regime_df["Regime_Name"] == rname
        fig_r.add_trace(go.Scatter(
            x=regime_df.index[mask],
            y=[rname] * mask.sum(),
            mode="markers",
            marker=dict(
                symbol="square",
                size=6,
                color=regime_colour(str(rname)),
            ),
            name=str(rname),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{rname}<extra></extra>",
        ))

    fig_r.update_layout(
        template="plotly_white",
        height=180,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📋 Audit Trail — Last 30 Trading Days")

if audit_trail:
    audit_df = pd.DataFrame(audit_trail).tail(30)

    # Colour-code Signal column
    def style_signal(val):
        if val == "CASH":
            return "color: #888888"
        colours = {"TLT":"#00bfff","VNQ":"#ffd700","SLV":"#c0c0c0",
                   "GLD":"#ffa500","LQD":"#88ddff","HYG":"#ff9966"}
        return f"color: {colours.get(val, '#ffffff')}; font-weight: 600"

    def style_regime(val):
        return f"color: {regime_colour(str(val))}"

    def style_return(val):
        try:
            v = float(val)
            return "color: #00d1b2" if v > 0 else "color: #ff6b6b" if v < 0 else ""
        except Exception:
            return ""

    # Reorder columns for readability — only include columns that exist
    priority = ["Date", "Signal", "Top_Pick", "Regime", "Conviction_Z",
                "P_Top", "Signal_Ret%",
                "TLT_Ret%", "VNQ_Ret%", "SLV_Ret%", "GLD_Ret%",
                "LQD_Ret%", "HYG_Ret%",
                "Stop_Active", "Rotated", "Disagree"]
    audit_df = audit_df[[c for c in priority if c in audit_df.columns]]

    # Format numbers
    fmt = {}
    for col in audit_df.columns:
        if col == "Conviction_Z":
            fmt[col] = "{:+.2f}"
        elif col == "P_Top":
            fmt[col] = "{:.3f}"
        elif col.endswith("_Ret%") or col == "Signal_Ret%":
            fmt[col] = "{:+.3f}%"

    # Safe display without complex styling that can KeyError
    st.dataframe(
        audit_df.style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=500
    )
else:
    st.info("No audit trail available yet.")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE / MOMENTUM WEIGHTS
# ═════════════════════════════════════════════════════════════════════════════
st.divider()

if use_momentum:
    st.subheader("⚖️ Option B — Momentum Score Weights")
    weights_data = {
        "Component": [
            "RoC 5d (×0.40)", "RoC 10d (×0.30)",
            "RoC 21d (×0.20)", "RoC 63d (×0.10)",
            "OBV Accumulation 21d (×0.15)",
            "Breakout vs 20d Range (×0.15)",
        ],
        "Weight": [0.40, 0.30, 0.20, 0.10, 0.15, 0.15],
        "Category": ["Momentum","Momentum","Momentum","Momentum","Volume","Breakout"],
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
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Weight", showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    st.plotly_chart(fig_i, use_container_width=True)
else:
    st.subheader("🔍 Option A — Top 20 Features by Ensemble Gain")
    try:
        from models import aggregate_feature_importance
        imp_df = aggregate_feature_importance(bank).head(20)
        if not imp_df.empty:
            fig_i = go.Figure(go.Bar(
                x=imp_df["Mean_Gain"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color="#00d1b2",
                hovertemplate="%{y}<br>Mean Gain: %{x:.1f}<extra></extra>",
            ))
            fig_i.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Mean Gain (LambdaRank + XGBoost)",
                           showgrid=True, gridcolor="#eeeeee"),
                yaxis=dict(autorange="reversed", showgrid=False),
            )
            st.plotly_chart(fig_i, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# SIGNALS HISTORY
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📡 Signal History (from GitLab)")

try:
    sig_df = cached_load_signals()
    if sig_df is not None and not sig_df.empty:
        st.dataframe(sig_df.tail(20).sort_index(ascending=False),
                     use_container_width=True)
    else:
        st.info("No signal history yet.")
except Exception as e:
    st.info(f"Could not load signal history: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 Methodology", expanded=False):
    st.markdown(f"""
<div style="font-size:14px;line-height:1.7;color:#ccc;">

<h4 style="color:#00d1b2;margin-top:0;">
🏗️ Layer 1 — Wasserstein k-means Regime Detection</h4>
<p>Based on <em>"Clustering Market Regimes Using the Wasserstein Distance"</em>
(Horvath, Issa, Muguruza, 2021 —
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905"
style="color:#00d1b2;">SSRN 3947905</a>).</p>
<p>Each rolling {20}-day window of ETF returns is treated as an empirical
probability distribution. The <b>Wasserstein distance</b> (Earth Mover's Distance)
measures how far apart two such distributions are — accounting for shape,
not just mean and variance. This is mathematically superior to standard
k-means on moments and outperforms Hidden Markov Models on non-Gaussian
financial returns.</p>
<p>Optimal k (number of regimes) is auto-selected via MMD scoring.
Current model: <b>k={detector.optimal_k_ if detector else '?'}</b> regimes —
{', '.join(detector.regime_names_.values()) if detector and detector.regime_names_ else 'loading...'}</p>

<h4 style="color:#00d1b2;">🤖 Layer 2 — LightGBM + Logistic Regression Ensemble</h4>
<p>For each regime, <b>5 independent binary classifiers</b> are trained — one per ETF.
Each model answers: <em>"Will this ETF beat the 3M T-Bill rate over the next
5 trading days?"</em></p>
<ul>
  <li><b>LightGBM</b> captures non-linear feature interactions
      (60% ensemble weight)</li>
  <li><b>Logistic Regression (L1)</b> provides a sparse linear baseline
      (40% weight)</li>
  <li>When models <b>disagree</b> beyond {15}%, effective conviction is
      halved — natural uncertainty filter</li>
  <li>Falls back to global model when a regime has fewer than 150 rows</li>
</ul>

<h4 style="color:#00d1b2;">📊 Input Features</h4>
<p>~{len(feature_cols) if feature_cols else '?'} features across:</p>
<ul>
  <li><b>FRED macro:</b> DGS10, T10Y2Y, T10Y3M, DTB3, MORTGAGE30US, VIXCLS,
      DTWEXBGS, DCOILWTICO, BAMLC0A0CM, BAMLH0A0HYM2, UMCSENT, T10YIE</li>
  <li><b>Derived macro:</b> Real yield (DGS10−T10YIE), rate momentum (20d/60d),
      rising/falling flags, yield curve shape, inflation regime, VIX regime,
      credit stress, rolling Z-scores (60d)</li>
  <li><b>ETF signals:</b> Daily return, realised vol (10d/21d), momentum
      (5d/21d/63d), volume ratio, ATR14, relative strength vs SPY</li>
  <li><b>Regime label</b> — integer + one-hot encoded regime membership</li>
</ul>

<h4 style="color:#00d1b2;">⚡ Strategy Execution</h4>
<ul>
  <li><b>Conviction gate:</b> Regime-aware Z threshold — Risk-On ≥ 1.0σ, Rate-Rising ≥ 0.5σ, Crisis/Risk-Off ≥ 0.3σ — below threshold holds CASH
      earning {rf_rate*100:.2f}% annualised</li>
  <li><b>Stop-loss:</b> 2-day cumulative loss ≤ {stop_loss_pct*100:.0f}% → CASH
      until Z ≥ {z_reentry:.1f}σ</li>
  <li><b>Rotation:</b> if top pick 5-day cumulative return &lt; 0 →
      rotate to #2 ranked ETF until top pick has positive day</li>
  <li><b>Disagreement filter:</b> LightGBM vs LogReg disagreement &gt; 15%
      → conviction halved</li>
  <li><b>Transaction cost:</b> {fee_bps}bps per one-way trade</li>
</ul>

<h4 style="color:#00d1b2;">🔄 Daily Pipeline</h4>
<p>GitHub Actions runs at 6:30am EST every weekday:
fetch new data → detect regime → retrain models → generate signal →
push to GitLab. This UI is read-only — loads pre-trained models
and pre-computed signals from GitLab at runtime.</p>

<h4 style="color:#ff6b6b;">⚠️ Important Caveats</h4>
<ul>
  <li>Past performance does not guarantee future results</li>
  <li>TBT is a leveraged inverse ETF with structural decay — unsuitable
      for long holding periods</li>
  <li>Backtest includes transaction costs but not slippage or
      liquidity constraints</li>
  <li>This is a research tool, not investment advice</li>
</ul>

</div>
""", unsafe_allow_html=True)
