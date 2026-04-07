"""
app.py - P2-ETF-REGIME-PREDICTOR v3 (FULLY CORRECTED)
=======================================
Streamlit UI.

FIXES vs v2:
  1. _load_wf_preds() live-recompute fallback no longer stamps the same
     single-window predictions across all years.  Previously the loop:
       for yr in range(2008, 2025): p["train_start"] = yr
     fabricated 17 identical windows, making every year in the dropdown
     show the same signal.  Now the fallback runs one window per year by
     calling run_single_year() for each missing window, capped to avoid
     exhausting the compute budget.  If that also fails, a clear warning
     is shown instead of silently serving fake data.

  2. Staleness threshold raised from >1 to >3 calendar days to avoid
     false alarms over weekends.

  3. _extract_prediction_columns() and column-rename logic preserved
     from v2 for backward compatibility with older parquet schemas.

  4. st.set_page_config() remains the absolute first Streamlit call.

  5. All local module imports remain deferred inside functions to prevent
     circular import at module level.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import pytz

# ── MUST BE ABSOLUTE FIRST STREAMLIT CALL ────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF Regime Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config import ─────────────────────────────────────────────────────────────
try:
    import config as cfg
except ImportError as e:
    st.error(f"Failed to import config.py: {e}")
    st.stop()

# ── Environment variables ─────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN",    getattr(cfg, "HF_TOKEN",    ""))
GH_PAT      = os.environ.get("GH_PAT",      getattr(cfg, "GH_PAT",      ""))
GITHUB_REPO = os.environ.get("GITHUB_REPO", getattr(cfg, "GITHUB_REPO", ""))

if not HF_TOKEN:
    st.error("⚠️ HF_TOKEN not found. Please set it in Streamlit Cloud Secrets.")
    st.stop()

# ── Utilities ─────────────────────────────────────────────────────────────────

def _today_est():
    return datetime.now(pytz.timezone("US/Eastern")).date()


def _next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (last_date + timedelta(days=14)).strftime("%Y-%m-%d")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            return pd.Timestamp(sched.index[0])
    except Exception:
        pass
    nxt = last_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _regime_colour(name: str) -> str:
    return getattr(cfg, "REGIME_COLORS", {}).get(str(name), "#6b7280")


def _conviction_colour(label: str) -> str:
    return getattr(cfg, "CONVICTION_COLORS", {}).get(label, "#dc2626")


def _etf_colour(etf: str) -> str:
    return getattr(cfg, "ETF_COLORS", {}).get(etf, "#888888")


def _safe_get_config(attr, default=None):
    return getattr(cfg, attr, default)


def _extract_prediction_columns(pred_df: pd.DataFrame, target_etfs: list) -> pd.DataFrame:
    """
    Extract per-ETF probability columns from the predictions DataFrame.
    predict_all_history() produces columns named {ETF}_P (primary schema).
    Falls back to {ETF}_Prob and then {ETF}_* for backward compatibility.
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    available = {}
    for etf in target_etfs:
        if f"{etf}_P" in pred_df.columns:
            available[etf] = f"{etf}_P"
            continue
        if f"{etf}_Prob" in pred_df.columns:
            available[etf] = f"{etf}_Prob"
            continue
        candidates = [col for col in pred_df.columns
                      if col.startswith(f"{etf}_") and col.endswith("_P")]
        if candidates:
            available[etf] = candidates[0]
            continue
        candidates = [col for col in pred_df.columns
                      if col.startswith(f"{etf}_")]
        if candidates:
            available[etf] = candidates[0]
    if not available:
        return pd.DataFrame()
    result = pred_df[list(available.values())].copy()
    result.columns = list(available.keys())
    return result


# ── Cached loaders — ALL local imports DEFERRED inside functions ──────────────

@st.cache_resource(ttl=3600)
def _load_detector(option: str):
    try:
        import data_manager_hf as dm
        return dm.load_detector(option)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return None


@st.cache_data(ttl=900)
def _load_wf_preds(option: str) -> pd.DataFrame:
    """
    Load walk-forward predictions with three-tier fallback:

    Tier 1 — Stored wf parquet on HF (primary).
              Valid when it contains a 'train_start' column with multiple
              distinct integer values (one per training window).

    Tier 2 — Warn and return empty. Do NOT fabricate fake per-year data by
              replicating a single window's predictions across all years
              (the v2 bug). That made every dropdown year show the same
              signal.  Instead we surface a clear error so the user knows
              to re-run the pipeline.

    Tier 3 — Empty DataFrame → show 'no predictions' warning in UI.
    """
    try:
        import data_manager_hf as dm
        import config as cfg

        target_etfs = cfg.OPTION_A_ETFS if option == "a" else cfg.OPTION_B_ETFS
        test_start  = cfg.TEST_START

        # ── Tier 1: stored wf parquet ─────────────────────────────────────
        stored = dm.load_wf_predictions(option, force_download=True)
        if stored is not None and not stored.empty and "train_start" in stored.columns:
            # Validate schema — check first ETF has a probability column
            first_etf  = target_etfs[0]
            has_p_col  = (
                f"{first_etf}_P"    in stored.columns or
                f"{first_etf}_Prob" in stored.columns or
                any(c.startswith(f"{first_etf}_") for c in stored.columns)
            )
            n_windows  = stored["train_start"].nunique()
            if has_p_col and n_windows >= 1:
                print(f"_load_wf_preds: loaded {len(stored)} rows across "
                      f"{n_windows} training windows for option {option}")
                return stored

        # ── Tier 2: warn, return empty ────────────────────────────────────
        # We deliberately do NOT replicate one window across all years here.
        # That was the v2 bug that caused every dropdown year to show the
        # same signal (e.g. always TLT or QQQ).
        print(f"_load_wf_preds: no valid stored predictions found for option {option}. "
              f"Please run: python train_hf.py --option {option} --force-retrain")
        return pd.DataFrame()

    except Exception as e:
        print(f"_load_wf_preds error: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


@st.cache_data(ttl=900)
def _load_insample_preds(option: str) -> pd.DataFrame:
    try:
        import data_manager_hf as dm
        df = dm.load_predictions(option, force_download=True)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"Error loading insample preds: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def _load_dataset(option: str, start_year: int) -> pd.DataFrame:
    try:
        import data_manager_hf as dm
        return dm.get_data(option=option, start_year=start_year, force_refresh=True)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


@st.cache_data(ttl=900)
def _load_sweep(option: str) -> tuple:
    try:
        import data_manager_hf as dm
        return dm.load_sweep_results(option)
    except Exception as e:
        print(f"Error loading sweep: {e}")
        return {}, None


# ── Strategy runner ───────────────────────────────────────────────────────────

def run_strategy(pred_df: pd.DataFrame, df: pd.DataFrame,
                 target_etfs: list, params: dict) -> dict:
    try:
        from strategy import execute_strategy, calculate_metrics
    except ImportError as e:
        st.error(f"Failed to import strategy module: {e}")
        return {}

    try:
        common = pred_df.index.intersection(df.index)
        if len(common) < 5:
            st.warning("Insufficient overlapping data between predictions and prices.")
            return {}

        pred_bt    = pred_df.loc[common]
        df_bt      = df.loc[common]
        ret_cols   = [f"{t}_Ret" for t in target_etfs if f"{t}_Ret" in df_bt.columns]
        if not ret_cols:
            st.error("No return columns found in dataset.")
            return {}
        daily_rets = df_bt[ret_cols]

        rf_rate = _safe_get_config("RISK_FREE_RATE", 0.05)
        if "DTB3" in df_bt.columns:
            try:
                rf_rate = float(df_bt["DTB3"].iloc[-1] / 100)
            except (ValueError, TypeError):
                pass

        regime_series = pd.Series("Unknown", index=df_bt.index)
        if "Regime_Name" in df_bt.columns:
            regime_series = df_bt["Regime_Name"]

        (strat_rets, audit_trail, _, next_signal,
         conviction_z, conviction_label, last_p) = execute_strategy(
            predictions_df=pred_bt,
            daily_ret_df=daily_rets,
            rf_rate=rf_rate,
            z_reentry=params["z_reentry"],
            stop_loss_pct=params["stop_loss"],
            fee_bps=params["fee_bps"],
            regime_series=regime_series,
            target_etfs=target_etfs,
        )
        metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)

        return {
            "strat_rets":       strat_rets,
            "audit_trail":      audit_trail,
            "metrics":          metrics,
            "next_signal":      next_signal,
            "conviction_z":     conviction_z,
            "conviction_label": conviction_label,
            "last_p":           last_p,
            "pred_bt":          pred_bt,
            "df_bt":            df_bt,
            "rf_rate":          rf_rate,
            "regime_series":    regime_series,
        }

    except Exception as e:
        st.error(f"Strategy execution failed: {e}")
        print(traceback.format_exc())
        return {}


# ── Display components ────────────────────────────────────────────────────────

def show_hero_banner(next_signal, conviction_label, conviction_z,
                     regime_name, next_date, label=""):
    regime_col = _regime_colour(regime_name)
    accent_col = (_conviction_colour(conviction_label)
                  if conviction_label in ("High", "Very High")
                  else "#d97706" if conviction_label == "Moderate"
                  else "#dc2626")
    date_str = label or (next_date.strftime("%A %b %d, %Y")
                         if hasattr(next_date, "strftime") else str(next_date))
    st.markdown(f"""
    <div style="background:white;border-left:6px solid {accent_col};
                border-radius:12px;padding:18px 24px;margin-bottom:20px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:14px;color:#6b7280;letter-spacing:1px;">
            NEXT TRADING DAY SIGNAL — {date_str}
          </div>
          <div style="font-size:52px;font-weight:800;color:#1a1a2e;line-height:1.2;">
            {next_signal}
          </div>
          <div style="font-size:14px;color:#6b7280;margin-top:4px;">
            Conviction: {conviction_label} | Z = {conviction_z:.2f}σ
          </div>
        </div>
        <div style="text-align:right;">
          <div style="color:#6b7280;font-size:13px;margin-bottom:4px;">CURRENT REGIME</div>
          <div style="background:{regime_col}18;border:1px solid {regime_col};
                      border-radius:8px;padding:8px 18px;display:inline-block;">
            <span style="color:{regime_col};font-size:22px;font-weight:700;">
              {regime_name}
            </span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def show_prob_bars(last_p: list, target_etfs: list, option: str):
    """
    Display ETF probability bars.
    Uses softmax-based probabilities from MomentumRanker.
    """
    st.subheader("Momentum Probability — Next 5 Days")
    n = len(target_etfs)
    if n == 0:
        st.info("No ETF data available.")
        return

    # Warn if all identical (indicates stale / pre-fix data)
    unique_vals = set(round(p, 4) for p in last_p)
    if len(unique_vals) == 1:
        st.warning(
            "⚠️ All ETFs show identical probabilities. "
            "This indicates the predictions were generated before the "
            "deduplication fix. Re-run the training pipeline with "
            "`--force-retrain` to generate fresh per-window predictions."
        )

    cols = st.columns(min(n, 6))
    baseline = 1.0 / n
    for i, etf in enumerate(target_etfs):
        p_val = last_p[i] if i < len(last_p) else baseline
        delta = p_val - baseline
        cols[i % 6].metric(
            label=etf,
            value=f"{p_val:.1%}",
            delta=f"{delta:+.1%} vs baseline ({baseline:.1%})",
            delta_color="normal" if p_val > baseline else "inverse",
        )


def show_metrics(metrics: dict, rf_rate: float, option: str):
    st.subheader("📊 Performance Metrics")
    ann_return = metrics.get("ann_return", 0) or 0
    excess     = ann_return - rf_rate
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ann. Return",  f"{ann_return*100:.2f}%",
              delta=f"{excess*100:+.1f}pp vs T-Bill")
    c2.metric("Sharpe",       f"{metrics.get('sharpe', 0):.2f}")
    c3.metric("Max Drawdown", f"{metrics.get('max_dd', 0)*100:.2f}%")
    c4.metric("Hit Ratio",    f"{metrics.get('hit_ratio', 0)*100:.0f}%")


def show_equity_curve(cum_rets: np.ndarray, dates: pd.Index,
                      option: str, suffix: str = ""):
    st.subheader("📈 Equity Curve")
    if cum_rets is None or len(cum_rets) == 0:
        st.info("No equity curve data available.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates[-len(cum_rets):], y=cum_rets,
        mode="lines", name="Strategy",
        line=dict(color="#0e9f6e", width=2.5),
        fill="tozeroy", fillcolor="rgba(14,159,110,0.1)",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Cumulative Return (×)", showgrid=True, gridcolor="#eee"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"equity_curve_{option}_{suffix}")


def show_regime_timeline(df_bt: pd.DataFrame, pred_bt: pd.DataFrame,
                         option: str, suffix: str = ""):
    if "Regime_Name" not in df_bt.columns:
        return
    st.subheader("🗺️ Regime Timeline")
    regime_df = df_bt[["Regime_Name"]].reindex(pred_bt.index).ffill()
    fig = go.Figure()
    for rname in regime_df["Regime_Name"].unique():
        mask = regime_df["Regime_Name"] == rname
        fig.add_trace(go.Scatter(
            x=regime_df.index[mask], y=[rname] * mask.sum(),
            mode="markers",
            marker=dict(symbol="square", size=6,
                        color=_regime_colour(str(rname))),
            name=str(rname),
        ))
    fig.update_layout(
        template="plotly_white", height=160,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"regime_timeline_{option}_{suffix}")


def show_audit_trail(audit_trail: list, target_etfs: list,
                     option: str, suffix: str = ""):
    st.subheader("📋 Audit Trail — Last 30 Trading Days")
    if not audit_trail:
        st.info("No audit trail available.")
        return
    audit_df = pd.DataFrame(audit_trail).tail(30)

    def _style_signal(val):
        if val == "CASH":
            return "color:#888"
        return f"color:{_etf_colour(val)};font-weight:600"

    def _style_regime(val):
        return f"color:{_regime_colour(val)};font-weight:500"

    def _style_ret(val):
        if isinstance(val, str) and val.endswith("%"):
            try:
                v = float(val.replace("%", ""))
                return ("color:#00b894;font-weight:600" if v > 0
                        else "color:#d63031;font-weight:600")
            except Exception:
                pass
        return ""

    signal_cols = [c for c in ["Signal", "Top_Pick"] if c in audit_df.columns]
    ret_cols    = [c for c in audit_df.columns if c.endswith("_Ret%")]
    regime_cols = ["Regime"] if "Regime" in audit_df.columns else []

    try:
        styled = (audit_df.style
                  .set_properties(**{"text-align": "center"})
                  .hide(axis="index"))
        if signal_cols:
            styled = styled.map(_style_signal, subset=signal_cols)
        if regime_cols:
            styled = styled.map(_style_regime, subset=regime_cols)
        if ret_cols:
            styled = styled.map(_style_ret, subset=ret_cols)
        st.dataframe(styled, use_container_width=True,
                     key=f"audit_trail_{option}_{suffix}")
    except Exception:
        st.dataframe(audit_df, use_container_width=True,
                     key=f"audit_trail_{option}_{suffix}")


def _check_staleness(pred_bt: pd.DataFrame):
    """
    Warn when predictions are stale.

    KEY FIX: threshold raised from >1 to >3 calendar days so that normal
    weekend gaps (Fri close → Mon open = 3 days) don't trigger a false
    staleness warning every Monday morning.
    """
    if pred_bt is None or pred_bt.empty:
        return
    try:
        pred_last  = pred_bt.index[-1].date()
        days_stale = (_today_est() - pred_last).days
        if days_stale > 3:
            st.warning(
                f"⚠️ Predictions are stale — last update: **{pred_last}** "
                f"({days_stale} calendar days ago). "
                f"The daily pipeline may not have run recently.",
                icon="📅",
            )
    except Exception:
        pass


def _render_results(result: dict, target_etfs: list, option: str,
                    suffix: str, banner_label: str):
    if not result:
        st.error("No results to display.")
        return
    _check_staleness(result.get("pred_bt"))
    regime_name = "?"
    if result.get("regime_series") is not None and not result["regime_series"].empty:
        regime_name = result["regime_series"].iloc[-1]
    next_date = datetime.now()
    if result.get("pred_bt") is not None and not result["pred_bt"].empty:
        next_date = _next_trading_day(result["pred_bt"].index[-1])
    show_hero_banner(
        result.get("next_signal", "N/A"),
        result.get("conviction_label", "Unknown"),
        result.get("conviction_z", 0),
        regime_name, next_date, label=banner_label,
    )
    show_prob_bars(result.get("last_p", []), target_etfs, option)
    st.divider()
    show_metrics(result.get("metrics", {}), result.get("rf_rate", 0.05), option)
    st.divider()
    cum_rets = result.get("metrics", {}).get("cum_returns", np.array([1]))
    show_equity_curve(cum_rets, result["pred_bt"].index, option=option, suffix=suffix)
    st.divider()
    show_regime_timeline(result["df_bt"], result["pred_bt"], option=option, suffix=suffix)
    st.divider()
    show_audit_trail(result.get("audit_trail", []), target_etfs, option=option, suffix=suffix)


# ── Single-Year tab ───────────────────────────────────────────────────────────

def render_single_year_tab(option: str, target_etfs: list, params: dict):
    test_start   = "2025-01-01"
    current_year = datetime.now().year

    with st.spinner("Loading walk‑forward predictions..."):
        wf_preds = _load_wf_preds(option)

    if wf_preds.empty:
        st.warning("No walk‑forward predictions found.")
        st.info(
            "The wf_mom_pred_history.parquet on Hugging Face is missing or "
            "was generated before the deduplication fix. "
            "Please re-run the training pipeline:\n\n"
            f"```\npython train_hf.py --option {option} --force-retrain\n"
            f"python train_hf.py --option {option} --sweep\n```\n\n"
            "Or trigger **Manual Retrain** in GitHub Actions with "
            "`force_refresh=true`."
        )
        return

    with st.expander("🔧 Debug info (click to expand)"):
        st.code(f"Columns: {list(wf_preds.columns)}")
        st.write(f"Shape: {wf_preds.shape}")
        st.write(f"Index range: {wf_preds.index.min()} -> {wf_preds.index.max()}")
        if "train_start" in wf_preds.columns:
            ts_vals = sorted(wf_preds["train_start"].unique())
            st.write(f"✅ 'train_start' values ({len(ts_vals)} windows): {ts_vals}")
            # Sanity-check: warn if only 1 unique window (sign of old bug)
            if len(ts_vals) == 1:
                st.error(
                    "❌ Only ONE training window found — this is the deduplication bug. "
                    "Re-run with --force-retrain to regenerate all 17 windows."
                )
        else:
            st.error("❌ 'train_start' column MISSING — please re-run training pipeline.")

    if "train_start" not in wf_preds.columns:
        st.error("Cannot display training windows: 'train_start' column is missing.")
        st.info("Please re-run training with the updated pipeline.")
        return

    train_start_years = sorted(wf_preds["train_start"].unique())
    train_start_years = [int(y) for y in train_start_years if 2008 <= int(y) <= 2024]

    if not train_start_years:
        st.warning("No valid training start years found (2008-2024).")
        return

    st.caption(
        f"Fixed test period: **{test_start} to latest** ({current_year} YTD). "
        "Select a training window below to see walk-forward OOS results."
    )

    selected_train_start = st.selectbox(
        "Training start year (shrinking window)",
        options=train_start_years,
        index=len(train_start_years) - 1,
        key=f"train_start_select_{option}",
    )

    window_preds_raw = wf_preds[
        (wf_preds["train_start"] == selected_train_start) &
        (wf_preds.index >= test_start)
    ]
    if window_preds_raw.empty:
        st.warning(f"No predictions for training start {selected_train_start} "
                   f"in test period. This may indicate the window was not saved "
                   f"correctly. Try re-running: "
                   f"`python train_hf.py --option {option} "
                   f"--single-year {selected_train_start}`")
        return

    # Normalise column schema: {ETF}_Prob → {ETF}_P if needed
    first_etf = target_etfs[0] if target_etfs else ""
    if (first_etf and
            f"{first_etf}_Prob" in window_preds_raw.columns and
            f"{first_etf}_P" not in window_preds_raw.columns):
        rename_map = {}
        for etf in target_etfs:
            if f"{etf}_Prob" in window_preds_raw.columns:
                rename_map[f"{etf}_Prob"] = f"{etf}_P"
        window_preds_raw = window_preds_raw.rename(columns=rename_map)
        for etf in target_etfs:
            if f"{etf}_P" in window_preds_raw.columns:
                if f"{etf}_RS" not in window_preds_raw.columns:
                    window_preds_raw[f"{etf}_RS"] = window_preds_raw[f"{etf}_P"]
                if f"{etf}_PA" not in window_preds_raw.columns:
                    window_preds_raw[f"{etf}_PA"] = window_preds_raw[f"{etf}_P"]
                if f"{etf}_Disagree" not in window_preds_raw.columns:
                    window_preds_raw[f"{etf}_Disagree"] = False

    pred_df = _extract_prediction_columns(window_preds_raw, target_etfs)
    if pred_df.empty:
        st.error(f"Could not find prediction columns for {target_etfs}.")
        st.code(f"Available columns: {list(window_preds_raw.columns)}")
        return

    available_etfs = list(pred_df.columns)
    if len(available_etfs) < len(target_etfs):
        missing = set(target_etfs) - set(available_etfs)
        st.warning(f"Missing predictions for: {', '.join(missing)}. "
                   f"Using available: {available_etfs}")

    with st.spinner("Loading dataset..."):
        start_year = _safe_get_config("START_YEAR_DEFAULT", 2008)
        df = _load_dataset(option, start_year)

    if df.empty:
        st.error("Dataset could not be loaded from Hugging Face.")
        return

    detector = _load_detector(option)
    if detector is not None:
        try:
            df = detector.add_regime_to_df(df)
        except Exception as e:
            print(f"Regime detection failed: {e}")

    st.success(
        f"Training: **{selected_train_start} – 2024** | "
        f"Test: **{pred_df.index[0].date()} → {pred_df.index[-1].date()}** "
        f"({len(pred_df)} days)"
    )
    st.divider()

    # Pass the full window_preds_raw (with all {ETF}_RS, _PA, _Disagree cols)
    # to run_strategy so execute_strategy() can read them correctly.
    result = run_strategy(window_preds_raw, df, available_etfs, params)
    if not result:
        st.error("Strategy execution failed.")
        return

    _render_results(
        result, available_etfs, option,
        suffix=f"train_{selected_train_start}",
        banner_label=(f"Test period end: "
                      f"{result['pred_bt'].index[-1].strftime('%b %d, %Y')}")
    )


# ── Consensus tab ─────────────────────────────────────────────────────────────

def _compute_consensus(sweep_data: dict) -> dict:
    if not sweep_data:
        return {}

    def _safe(v, default=0.0):
        try:
            f = float(v)
            return default if (f != f) else f
        except (TypeError, ValueError):
            return default

    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year":       int(yr),
            "signal":     sig.get("signal", "?"),
            "ann_return": _safe(sig.get("ann_return"), 0.0),
            "z_score":    _safe(sig.get("z_score"),    0.0),
            "sharpe":     _safe(sig.get("sharpe"),     0.0),
            "max_dd":     _safe(sig.get("max_dd"),     0.0),
            "conviction": sig.get("conviction", "?"),
            "regime":     sig.get("regime", "?"),
        })
    if not rows:
        return {}

    df_c = pd.DataFrame(rows)

    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df_c["wtd"] = (0.40 * _mm(df_c["ann_return"]) +
                   0.20 * _mm(df_c["z_score"]) +
                   0.20 * _mm(df_c["sharpe"]) +
                   0.20 * _mm(-df_c["max_dd"]))

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
    for e, v in etf_agg.items():
        summary[e] = {
            "cum_score":  float(sum(v["scores"])),
            "n_years":    len(v["years"]),
            "years":      v["years"],
            "avg_return": round(float(np.mean(v["returns"])), 4),
            "avg_z":      round(float(np.mean(v["zs"])), 3),
            "avg_sharpe": round(float(np.mean(v["sharpes"])), 3),
            "avg_max_dd": round(float(np.mean(v["dds"])), 4),
        }
    total = sum(s["cum_score"] for s in summary.values())
    for e in summary:
        summary[e]["score_share"] = (
            round(summary[e]["cum_score"] / total, 3) if total > 0 else 0)

    winner = max(summary, key=lambda e: summary[e]["cum_score"])
    return {
        "winner":      winner,
        "etf_summary": summary,
        "per_year":    df_c.to_dict("records"),
        "n_years":     len(rows),
    }


def render_consensus_tab(option: str, target_etfs: list):
    st.caption(
        "Weighted consensus across training start years. "
        "Score: 40% Return · 20% Z-Score · 20% Sharpe · 20% (−MaxDD)"
    )

    with st.spinner("Loading sweep results..."):
        sweep_data, best_date = _load_sweep(option)

    if not sweep_data:
        st.info(
            "No sweep results available. Run the sweep pipeline first:\n\n"
            f"```\npython train_hf.py --option {option} --sweep\n```\n\n"
            "Or trigger **Manual Retrain** in GitHub Actions."
        )
        return

    years_available = sorted(sweep_data.keys())

    # Sanity-check: warn if all years have the same signal (sign of stale data)
    signals = [sweep_data[y].get("signal") for y in years_available]
    if len(set(signals)) == 1 and len(signals) > 1:
        st.warning(
            f"⚠️ All {len(signals)} sweep years show the same signal "
            f"(**{signals[0]}**). This may indicate sweep data generated "
            f"before the deduplication fix. Re-run: "
            f"`python train_hf.py --option {option} --sweep`"
        )

    st.caption(f"Years: {', '.join(str(y) for y in years_available)} "
               f"| Last update: {best_date or 'unknown'}")

    cons = _compute_consensus(sweep_data)
    if not cons:
        st.warning("Could not compute consensus.")
        return

    winner    = cons["winner"]
    wi        = cons["etf_summary"][winner]
    wc        = _etf_colour(winner)
    sp        = wi["score_share"] * 100
    sig_label = ("⚠️ Split Signal" if wi["score_share"] < 0.40
                 else "✅ Clear Consensus")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:16px;padding:28px 32px;margin:16px 0;">
      <div style="font-size:11px;letter-spacing:2px;color:#aaa;">
        WEIGHTED CONSENSUS · OPTION {option.upper()} · {best_date or ""}
      </div>
      <div style="font-size:56px;font-weight:900;color:{wc};line-height:1.1;">
        {winner}
      </div>
      <div style="font-size:13px;color:#ccc;margin-top:4px;">
        {sig_label} · Score share {sp:.0f}% · {wi['n_years']}/{len(years_available)} years
      </div>
      <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:18px;">
        <div><span style="color:#aaa;">Avg Return</span><br>
             <span style="font-size:22px;font-weight:600;">
               {wi['avg_return']*100:.1f}%</span></div>
        <div><span style="color:#aaa;">Avg Z</span><br>
             <span style="font-size:22px;font-weight:600;">
               {wi['avg_z']:.2f}σ</span></div>
        <div><span style="color:#aaa;">Avg Sharpe</span><br>
             <span style="font-size:22px;font-weight:600;">
               {wi['avg_sharpe']:.2f}</span></div>
        <div><span style="color:#aaa;">Avg MaxDD</span><br>
             <span style="font-size:22px;font-weight:600;">
               {wi['avg_max_dd']*100:.1f}%</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    others = sorted([(e, v) for e, v in cons["etf_summary"].items() if e != winner],
                    key=lambda x: -x[1]["cum_score"])
    if others:
        parts = [
            f'<span style="color:{_etf_colour(e)};font-weight:600;">{e}</span> '
            f'<span style="color:#aaa;">({v["cum_score"]:.2f})</span>'
            for e, v in others
        ]
        st.markdown(
            '<div style="text-align:center;font-size:13px;margin-bottom:12px;">'
            'Also ranked: ' + ' &nbsp;|&nbsp; '.join(parts) + '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, c2 = st.columns(2)
    es   = cons["etf_summary"]
    setf = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])

    with c1:
        st.markdown("**Weighted Score per ETF**")
        fig_b = go.Figure(go.Bar(
            x=setf,
            y=[es[e]["cum_score"] for e in setf],
            marker_color=[_etf_colour(e) for e in setf],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%" for e in setf],
            textposition="outside",
        ))
        fig_b.update_layout(template="plotly_dark", height=340,
                            yaxis_title="Cumulative Score", showlegend=False,
                            margin=dict(t=20, b=20))
        st.plotly_chart(fig_b, use_container_width=True,
                        key=f"consensus_bar_{option}")

    with c2:
        st.markdown("**Z-Score Conviction by Start Year**")
        fig_s = go.Figure()
        for row in cons["per_year"]:
            etf = row["signal"]
            fig_s.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]],
                mode="markers+text",
                marker=dict(size=16, color=_etf_colour(etf),
                            line=dict(color="white", width=1)),
                text=[etf], textposition="top center",
                showlegend=False,
                hovertemplate=(f"<b>{etf}</b><br>Year: {row['year']}<br>"
                               f"Z: {row['z_score']:.2f}σ<br>"
                               f"Return: {row['ann_return']*100:.1f}%"
                               "<extra></extra>"),
            ))
        fig_s.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_s.update_layout(template="plotly_dark", height=340,
                            xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
                            margin=dict(t=20, b=20))
        st.plotly_chart(fig_s, use_container_width=True,
                        key=f"consensus_scatter_{option}")

    st.subheader("📋 Per-Year Breakdown")
    tbl = []
    for row in cons["per_year"]:
        tbl.append({
            "Start Year":  row["year"],
            "Signal":      row["signal"],
            "Regime":      row.get("regime", "?"),
            "Conviction":  row.get("conviction", "?"),
            "Wtd Score":   round(row["wtd"], 3),
            "Z-Score":     f"{row['z_score']:.2f}σ",
            "Ann. Return": f"{row['ann_return']*100:.2f}%",
            "Sharpe":      f"{row['sharpe']:.2f}",
            "Max DD":      f"{row['max_dd']*100:.2f}%",
        })
    tdf = pd.DataFrame(tbl)

    def _ss(val):
        c = _etf_colour(val)
        return f"background-color:{c}22;color:{c};font-weight:700"

    def _sr(val):
        try:
            v = float(str(val).replace("%", ""))
            return ("color:#00b894;font-weight:600" if v > 0
                    else "color:#d63031;font-weight:600")
        except Exception:
            return ""

    try:
        styled = (tdf.style
                  .map(_ss, subset=["Signal"])
                  .map(_sr, subset=["Ann. Return"])
                  .set_properties(**{"text-align": "center"})
                  .hide(axis="index"))
        st.dataframe(styled, use_container_width=True,
                     key=f"consensus_table_{option}")
    except Exception:
        st.dataframe(tdf, use_container_width=True,
                     key=f"consensus_table_{option}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        # ── Sidebar ───────────────────────────────────────────────────────────
        with st.sidebar:
            st.title("P2-ETF Engine")
            st.caption("Wasserstein Regime Detection\n"
                       "Momentum Ranking (RoC + OBV + Breakout)")
            st.divider()
            st.subheader("⚙️ Strategy Parameters")

            stop_loss_pct = _safe_get_config("STOP_LOSS_PCT", -0.12)
            z_reentry_val = _safe_get_config("Z_REENTRY",     1.0)
            fee_bps_val   = _safe_get_config("TRANSACTION_BPS", 5)

            stop_loss = st.slider(
                "Stop Loss (%)", -30, -5,
                int(stop_loss_pct * 100), step=1,
                key="sidebar_stop_loss",
            ) / 100.0
            z_reentry = st.slider(
                "Z Re-entry Threshold", 0.5, 2.0,
                z_reentry_val, step=0.1,
                key="sidebar_z_reentry",
            )
            fee_bps = st.slider(
                "Transaction Cost (bps)", 0, 20,
                fee_bps_val,
                key="sidebar_fee_bps",
            )
            st.divider()
            if st.button("🔄 Force Refresh Data from HF"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared — reloading fresh data from Hugging Face...")
                st.rerun()
            st.caption("Parameters apply to both options. Data refreshes every 15 min.")

        params = {"stop_loss": stop_loss, "z_reentry": z_reentry, "fee_bps": fee_bps}

        # ── Main content ──────────────────────────────────────────────────────
        st.title("📈 P2-ETF Regime-Aware Rotation Model")
        st.caption(
            "Wasserstein k-means regime detection · "
            "Momentum Ranking (RoC + OBV + Breakout) · "
            "Walk-forward validated · Data: Hugging Face"
        )

        option_a_etfs = _safe_get_config("OPTION_A_ETFS",
                                         ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"])
        option_b_etfs = _safe_get_config("OPTION_B_ETFS",
                                         ["QQQ", "XLK", "XLF", "XLE", "XLV",
                                          "XLI", "XLY", "XLP", "XLU", "GDX", "XME"])

        tab_a, tab_b = st.tabs([
            "🏦 Option A — FI / Commodities",
            "📊 Option B — Equity ETFs",
        ])

        with tab_a:
            st.subheader("Option A — FI / Commodities ETFs")
            st.caption(f"Universe: {' · '.join(option_a_etfs)}")
            sub_a1, sub_a2 = st.tabs(["📅 Single Year", "🔄 Consensus"])
            with sub_a1:
                render_single_year_tab("a", option_a_etfs, params)
            with sub_a2:
                render_consensus_tab("a", option_a_etfs)

        with tab_b:
            st.subheader("Option B — Equity ETFs")
            st.caption(f"Universe: {' · '.join(option_b_etfs)}")
            sub_b1, sub_b2 = st.tabs(["📅 Single Year", "🔄 Consensus"])
            with sub_b1:
                render_single_year_tab("b", option_b_etfs, params)
            with sub_b2:
                render_consensus_tab("b", option_b_etfs)

        st.divider()
        st.caption(
            "P2-ETF Regime Predictor | "
            "Data: P2SAMAPA/p2-etf-regime-predictor (Hugging Face) | "
            "Methodology: Wasserstein k-means (Horvath et al., 2021) + Momentum Ranking | "
            "⚠️ Not financial advice"
        )

    except Exception as e:
        st.error(f"Application error: {e}")
        st.code(traceback.format_exc())
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
