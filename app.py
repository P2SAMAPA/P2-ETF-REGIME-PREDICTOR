"""
app.py — P2-ETF-REGIME-PREDICTOR v2
=====================================
Streamlit UI.

Structure:
  Top-level tabs:
    Option A — FI / Commodities ETFs (TLT, VNQ, SLV, GLD, LQD, HYG)
    Option B — Equity ETFs (SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME)

  Within each top-level tab, two sub-tabs:
    Single Year  — select a test year, view walk-forward OOS backtest
    Consensus    — multi-year sweep with weighted consensus signal
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime, timedelta

import pytz
from huggingface_hub import hf_hub_download

import config as cfg

# ── Local module imports ──────────────────────────────────────────────────────
try:
    from data_manager_hf import (
        get_data,
        load_predictions, load_wf_predictions,
        load_detector, load_ranker,
        load_signals, load_sweep_results,
    )
    from regime_detection import RegimeDetector
    from models import MomentumRanker
    from strategy import execute_strategy, calculate_metrics
except ImportError as e:
    st.error(f"Failed to import local modules: {e}")
    sys.exit(1)

# ── Environment ───────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", cfg.HF_DATASET_REPO)
HF_TOKEN        = os.environ.get("HF_TOKEN", cfg.HF_TOKEN)
GH_PAT          = os.environ.get("GH_PAT", cfg.GH_PAT)
GITHUB_REPO     = os.environ.get("GITHUB_REPO", cfg.GITHUB_REPO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF Regime Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Utility functions ─────────────────────────────────────────────────────────

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
    except ImportError:
        pass
    nxt = last_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _next_trading_day_from_today() -> pd.Timestamp:
    return _next_trading_day(pd.Timestamp(_today_est()))


def _regime_colour(name: str) -> str:
    return cfg.REGIME_COLORS.get(str(name), "#6b7280")


def _conviction_colour(label: str) -> str:
    return cfg.CONVICTION_COLORS.get(label, "#dc2626")


def _etf_colour(etf: str) -> str:
    return cfg.ETF_COLORS.get(etf, "#888888")


# ── Cached loaders (per option, fresh on every tab load) ──────────────────────

@st.cache_resource(ttl=0)  # TTL=0: always re-fetch, never stale
def _load_detector(option: str):
    try:
        import pickle
        b = load_detector(option)
        return RegimeDetector.from_bytes(b) if b else None
    except Exception:
        return None


@st.cache_resource(ttl=0)
def _load_ranker(option: str):
    try:
        import pickle
        b = load_ranker(option)
        return MomentumRanker.from_bytes(b) if b else None
    except Exception:
        return None


@st.cache_data(ttl=0)
def _load_wf_preds(option: str) -> pd.DataFrame:
    df = load_wf_predictions(option, force_download=True)
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=0)
def _load_dataset(option: str, start_year: int) -> pd.DataFrame:
    return get_data(option=option, start_year=start_year, force_refresh=False)


@st.cache_data(ttl=0)
def _load_sweep(option: str) -> tuple:
    return load_sweep_results(option)


# ── Strategy runner ───────────────────────────────────────────────────────────

def run_strategy(pred_df: pd.DataFrame, df: pd.DataFrame,
                 target_etfs: list, params: dict) -> dict:
    """
    Execute strategy and return a results dict.
    params: {stop_loss, z_reentry, fee_bps}
    """
    common = pred_df.index.intersection(df.index)
    if len(common) < 5:
        return {}

    pred_bt = pred_df.loc[common]
    df_bt   = df.loc[common]
    ret_cols = [f"{t}_Ret" for t in target_etfs if f"{t}_Ret" in df_bt.columns]
    daily_rets = df_bt[ret_cols]
    rf_rate = (float(df_bt["DTB3"].iloc[-1] / 100)
               if "DTB3" in df_bt.columns else cfg.RISK_FREE_RATE)
    regime_series = (df_bt["Regime_Name"]
                     if "Regime_Name" in df_bt.columns
                     else pd.Series("Unknown", index=df_bt.index))

    try:
        (strat_rets, audit_trail, _, next_signal,
         conviction_z, conviction_label, last_p) = execute_strategy(
            predictions_df=pred_bt,
            daily_ret_df=daily_rets,
            rf_rate=rf_rate,
            z_reentry=params["z_reentry"],
            stop_loss_pct=params["stop_loss"],
            fee_bps=params["fee_bps"],
            regime_series=regime_series,
        )
        metrics = calculate_metrics(strat_rets, rf_rate=rf_rate)
    except Exception as e:
        st.error(f"Strategy execution failed: {e}")
        return {}

    return {
        "strat_rets":      strat_rets,
        "audit_trail":     audit_trail,
        "metrics":         metrics,
        "next_signal":     next_signal,
        "conviction_z":    conviction_z,
        "conviction_label": conviction_label,
        "last_p":          last_p,
        "pred_bt":         pred_bt,
        "df_bt":           df_bt,
        "rf_rate":         rf_rate,
        "regime_series":   regime_series,
    }


# ── Display components ────────────────────────────────────────────────────────

def show_hero_banner(next_signal: str, conviction_label: str, conviction_z: float,
                     regime_name: str, next_date: pd.Timestamp, label: str = "") -> None:
    regime_col = _regime_colour(regime_name)
    accent_col = (_conviction_colour(conviction_label)
                  if conviction_label in ("High", "Very High") else
                  "#d97706" if conviction_label == "Moderate" else "#dc2626")
    date_label = label or next_date.strftime("%A %b %d, %Y")
    st.markdown(f"""
    <div style="background:white;border-left:6px solid {accent_col};
                border-radius:12px;padding:18px 24px;margin-bottom:20px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:14px;color:#6b7280;letter-spacing:1px;">
            NEXT TRADING DAY SIGNAL — {date_label}
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


def show_prob_bars(last_p: list, target_etfs: list) -> None:
    st.subheader("P(Beat Cash) — Next 5 Days")
    n = len(target_etfs)
    cols = st.columns(min(n, 6))
    for i, etf in enumerate(target_etfs):
        p_val = last_p[i] if i < len(last_p) else 0.5
        cols[i % 6].metric(
            label=etf,
            value=f"{p_val:.1%}",
            delta=f"{p_val - 0.5:+.1%} vs baseline",
            delta_color="normal" if p_val > 0.5 else "inverse",
        )


def show_metrics(metrics: dict, rf_rate: float) -> None:
    st.subheader("📊 Performance Metrics")
    excess = metrics.get("ann_return", 0) - rf_rate
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ann. Return",  f"{metrics.get('ann_return', 0)*100:.2f}%",
              delta=f"{excess*100:+.1f}pp vs T-Bill")
    c2.metric("Sharpe",       f"{metrics.get('sharpe', 0):.2f}")
    c3.metric("Max Drawdown", f"{metrics.get('max_dd', 0)*100:.2f}%")
    c4.metric("Hit Ratio",    f"{metrics.get('hit_ratio', 0)*100:.0f}%")


def show_equity_curve(cum_rets: np.ndarray, dates: pd.Index) -> None:
    st.subheader("📈 Equity Curve")
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
    st.plotly_chart(fig, use_container_width=True)


def show_regime_timeline(df_bt: pd.DataFrame, pred_bt: pd.DataFrame) -> None:
    if "Regime_Name" not in df_bt.columns:
        return
    st.subheader("🗺️ Regime Timeline")
    regime_df = df_bt[["Regime_Name"]].reindex(pred_bt.index).ffill()
    fig = go.Figure()
    for rname in regime_df["Regime_Name"].unique():
        mask = regime_df["Regime_Name"] == rname
        fig.add_trace(go.Scatter(
            x=regime_df.index[mask],
            y=[rname] * mask.sum(),
            mode="markers",
            marker=dict(symbol="square", size=6,
                        color=_regime_colour(str(rname))),
            name=str(rname),
        ))
    fig.update_layout(
        template="plotly_white", height=160,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_audit_trail(audit_trail: list, target_etfs: list) -> None:
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

    signal_cols = ["Signal", "Top_Pick"] if "Top_Pick" in audit_df.columns else ["Signal"]
    ret_cols    = [c for c in audit_df.columns if c.endswith("_Ret%")]
    regime_cols = ["Regime"] if "Regime" in audit_df.columns else []

    styled = audit_df.style.set_properties(**{"text-align": "center"}).hide(axis="index")
    if signal_cols:
        styled = styled.applymap(_style_signal, subset=signal_cols)
    if regime_cols:
        styled = styled.applymap(_style_regime, subset=regime_cols)
    if ret_cols:
        styled = styled.applymap(_style_ret, subset=ret_cols)

    st.dataframe(styled, use_container_width=True)


# ── Staleness check ───────────────────────────────────────────────────────────

def _check_staleness(pred_bt: pd.DataFrame) -> None:
    pred_last  = pred_bt.index[-1].date()
    today_est  = _today_est()
    days_stale = (today_est - pred_last).days
    if days_stale > 1:
        st.warning(
            f"⚠️ Predictions are stale — last update: **{pred_last}** "
            f"({days_stale} calendar days ago).",
            icon="📅",
        )


# ── Single-Year sub-tab ────────────────────────────────────────────────────────

def render_single_year_tab(option: str, target_etfs: list, params: dict) -> None:
    """Render the Single Year sub-tab for a given option."""
    st.caption(
        "Select a test year to view the walk-forward OOS backtest "
        "for that year. Data is always fetched fresh from Hugging Face."
    )

    available_years = [int(w["test_year"]) for w in cfg.WINDOWS]
    selected_year   = st.selectbox(
        "Test year",
        options=[str(y) for y in available_years],
        index=len(available_years) - 1,
        key=f"year_select_{option}",
    )
    year = int(selected_year)

    # Load WF predictions fresh (ttl=0)
    with st.spinner("Loading walk-forward predictions from Hugging Face..."):
        wf_preds = _load_wf_preds(option)

    if wf_preds.empty:
        st.warning(
            "No walk-forward predictions found. "
            "Run the daily pipeline or manual retrain first."
        )
        return

    # Filter to selected year
    wf_year = wf_preds.loc[f"{year}-01-01":f"{year}-12-31"]
    if wf_year.empty:
        st.warning(f"No walk-forward predictions found for {year}.")
        return

    # Load dataset fresh
    with st.spinner("Loading dataset from Hugging Face..."):
        df = _load_dataset(option, cfg.START_YEAR_DEFAULT)

    if df.empty:
        st.error("Dataset could not be loaded.")
        return

    # Add regime labels
    detector = _load_detector(option)
    if detector is not None:
        try:
            df = detector.add_regime_to_df(df)
        except Exception:
            pass

    # Run strategy for the year
    result = run_strategy(wf_year, df, target_etfs, params)
    if not result:
        return

    st.success(f"Walk-forward OOS results for {year} — {len(wf_year)} trading days")
    st.divider()

    _check_staleness(result["pred_bt"])

    regime_name = result["regime_series"].iloc[-1] if not result["regime_series"].empty else "?"
    next_date   = _next_trading_day(result["pred_bt"].index[-1])

    show_hero_banner(
        result["next_signal"], result["conviction_label"], result["conviction_z"],
        regime_name, next_date, label=f"Year-End {year}",
    )
    show_prob_bars(result["last_p"], target_etfs)
    st.divider()
    show_metrics(result["metrics"], result["rf_rate"])
    st.divider()
    show_equity_curve(
        result["metrics"].get("cum_returns", np.array([1])),
        result["pred_bt"].index,
    )
    st.divider()
    show_regime_timeline(result["df_bt"], result["pred_bt"])
    st.divider()
    show_audit_trail(result["audit_trail"], target_etfs)


# ── Consensus sub-tab ─────────────────────────────────────────────────────────

def _compute_consensus(sweep_data: dict) -> dict:
    if not sweep_data:
        return {}
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year":       int(yr),
            "signal":     sig.get("signal", "?"),
            "ann_return": float(sig.get("ann_return", 0.0)),
            "z_score":    float(sig.get("z_score", 0.0)),
            "sharpe":     float(sig.get("sharpe", 0.0)),
            "max_dd":     float(sig.get("max_dd", 0.0)),
            "conviction": sig.get("conviction", "?"),
            "regime":     sig.get("regime", "?"),
        })
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
        summary[e]["score_share"] = round(summary[e]["cum_score"] / total, 3) if total > 0 else 0

    winner = max(summary, key=lambda e: summary[e]["cum_score"])
    return {
        "winner":      winner,
        "etf_summary": summary,
        "per_year":    df_c.to_dict("records"),
        "n_years":     len(rows),
    }


def render_consensus_tab(option: str, target_etfs: list) -> None:
    """Render the Consensus sub-tab for a given option."""
    st.caption(
        "Weighted consensus across multiple training start years. "
        "Score: 40% Ann. Return · 20% Z-Score · 20% Sharpe · 20% (−MaxDD). "
        "Data always fetched fresh from Hugging Face."
    )

    with st.spinner("Loading sweep results from Hugging Face..."):
        sweep_data, best_date = _load_sweep(option)

    if not sweep_data:
        st.info(
            "No sweep results available yet. "
            "Run the daily pipeline to generate consensus data."
        )
        return

    years_available = sorted(sweep_data.keys())
    st.caption(f"Years available: {', '.join(str(y) for y in years_available)} "
               f"| Last update: {best_date or 'unknown'}")

    cons = _compute_consensus(sweep_data)
    if not cons:
        st.warning("Could not compute consensus.")
        return

    winner    = cons["winner"]
    wi        = cons["etf_summary"][winner]
    wc        = _etf_colour(winner)
    sp        = wi["score_share"] * 100
    sig_label = "⚠️ Split Signal" if wi["score_share"] < 0.40 else "✅ Clear Consensus"

    # Hero banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:16px;padding:28px 32px;margin:16px 0;">
      <div style="font-size:11px;letter-spacing:2px;color:#aaa;">
        WEIGHTED CONSENSUS · OPTION {option.upper()} · {best_date or ""}
      </div>
      <div style="font-size:56px;font-weight:900;color:{wc};line-height:1.1;">{winner}</div>
      <div style="font-size:13px;color:#ccc;margin-top:4px;">
        {sig_label} · Score share {sp:.0f}% · {wi['n_years']}/{len(years_available)} years
      </div>
      <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:18px;">
        <div><span style="color:#aaa;">Avg Return</span><br>
             <span style="font-size:22px;font-weight:600;">{wi['avg_return']*100:.1f}%</span></div>
        <div><span style="color:#aaa;">Avg Z</span><br>
             <span style="font-size:22px;font-weight:600;">{wi['avg_z']:.2f}σ</span></div>
        <div><span style="color:#aaa;">Avg Sharpe</span><br>
             <span style="font-size:22px;font-weight:600;">{wi['avg_sharpe']:.2f}</span></div>
        <div><span style="color:#aaa;">Avg MaxDD</span><br>
             <span style="font-size:22px;font-weight:600;">{wi['avg_max_dd']*100:.1f}%</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Runner-up ETFs
    others = sorted([(e, v) for e, v in cons["etf_summary"].items() if e != winner],
                    key=lambda x: -x[1]["cum_score"])
    if others:
        parts = [f'<span style="color:{_etf_colour(e)};font-weight:600;">{e}</span> '
                 f'<span style="color:#aaa;">({v["cum_score"]:.2f})</span>'
                 for e, v in others]
        st.markdown('<div style="text-align:center;font-size:13px;margin-bottom:12px;">'
                    'Also ranked: ' + ' &nbsp;|&nbsp; '.join(parts) + '</div>',
                    unsafe_allow_html=True)

    st.divider()

    # Charts
    c1, c2 = st.columns(2)
    es     = cons["etf_summary"]
    setf   = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])

    with c1:
        st.markdown("**Weighted Score per ETF**")
        fig_b = go.Figure(go.Bar(
            x=setf,
            y=[es[e]["cum_score"] for e in setf],
            marker_color=[_etf_colour(e) for e in setf],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%"
                  for e in setf],
            textposition="outside",
        ))
        fig_b.update_layout(
            template="plotly_dark", height=340,
            yaxis_title="Cumulative Score", showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_b, use_container_width=True)

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
                               f"Return: {row['ann_return']*100:.1f}%<extra></extra>"),
            ))
        fig_s.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_s.update_layout(
            template="plotly_dark", height=340,
            xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Per-year breakdown table
    st.subheader("📋 Per-Year Breakdown")
    tbl = []
    for row in cons["per_year"]:
        tbl.append({
            "Start Year": row["year"],
            "Signal":     row["signal"],
            "Regime":     row.get("regime", "?"),
            "Conviction": row.get("conviction", "?"),
            "Wtd Score":  round(row["wtd"], 3),
            "Z-Score":    f"{row['z_score']:.2f}σ",
            "Ann. Return": f"{row['ann_return']*100:.2f}%",
            "Sharpe":     f"{row['sharpe']:.2f}",
            "Max DD":     f"{row['max_dd']*100:.2f}%",
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

    st.dataframe(
        tdf.style
        .applymap(_ss, subset=["Signal"])
        .applymap(_sr, subset=["Ann. Return"])
        .set_properties(**{"text-align": "center"})
        .hide(axis="index"),
        use_container_width=True, height=300,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("P2-ETF Engine")
    st.caption("Wasserstein Regime Detection\nMomentum Ranking (RoC + OBV + Breakout)")
    st.divider()

    st.subheader("⚙️ Strategy Parameters")
    stop_loss  = st.slider("Stop Loss (%)", -30, -5, int(cfg.STOP_LOSS_PCT * 100), step=1) / 100.0
    z_reentry  = st.slider("Z Re-entry Threshold", 0.5, 2.0, cfg.Z_REENTRY, step=0.1)
    fee_bps    = st.slider("Transaction Cost (bps)", 0, 20, cfg.TRANSACTION_BPS)
    st.divider()

    st.caption(
        "Parameters apply to both options and both sub-tabs. "
        "Data is always fetched fresh from Hugging Face — no stale cache."
    )

params = {"stop_loss": stop_loss, "z_reentry": z_reentry, "fee_bps": fee_bps}

# ── Main title ────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF Regime-Aware Rotation Model")
st.caption(
    "Wasserstein k-means regime detection · Momentum Ranking (RoC + OBV + Breakout) · "
    "Walk-forward validated · Data: Hugging Face"
)

# ── Top-level tabs: Option A / Option B ───────────────────────────────────────
tab_a, tab_b = st.tabs([
    "🏦 Option A — FI / Commodities",
    "📊 Option B — Equity ETFs",
])

# ── Option A ──────────────────────────────────────────────────────────────────
with tab_a:
    st.subheader("Option A — FI / Commodities ETFs")
    st.caption(f"Universe: {' · '.join(cfg.OPTION_A_ETFS)}")
    sub_a1, sub_a2 = st.tabs(["📅 Single Year", "🔄 Consensus"])
    with sub_a1:
        render_single_year_tab("a", cfg.OPTION_A_ETFS, params)
    with sub_a2:
        render_consensus_tab("a", cfg.OPTION_A_ETFS)

# ── Option B ──────────────────────────────────────────────────────────────────
with tab_b:
    st.subheader("Option B — Equity ETFs")
    st.caption(f"Universe: {' · '.join(cfg.OPTION_B_ETFS)}")
    sub_b1, sub_b2 = st.tabs(["📅 Single Year", "🔄 Consensus"])
    with sub_b1:
        render_single_year_tab("b", cfg.OPTION_B_ETFS, params)
    with sub_b2:
        render_consensus_tab("b", cfg.OPTION_B_ETFS)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "P2-ETF Regime Predictor | "
    "Data: P2SAMAPA/p2-etf-regime-predictor (Hugging Face) | "
    "Methodology: Wasserstein k-means (Horvath et al., 2021) + Momentum Ranking | "
    "⚠️ Not financial advice"
)
