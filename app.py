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
        load_wf_momentum_predictions_from_gitlab, load_wf_ensemble_predictions_from_gitlab,
        TARGET_ETFS,
    )
    from regime_detection import RegimeDetector
    from models import MomentumRanker
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
    st.caption("Wasserstein Regime Detection + Momentum Ranking")

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
    use_momentum = True  # Momentum (Option B) — sole ranking method
    st.divider()
    st.subheader("📊 Backtest Mode")
    bt_mode = st.radio(
        "Validation Method",
        options=["In-Sample (Full History)", "Walk-Forward OOS"],
        index=0,
        help=(
            "**In-Sample**: Model trained on full history, backtested on same data. "
            "Returns will be optimistic — use for model development only.\n\n"
            "**Walk-Forward OOS**: 3-year rolling training window, tested on "
            "following year only. Model never sees test data. "
            "This is the honest performance estimate."
        )
    )
    use_wf = "Walk-Forward" in bt_mode

    st.divider()
    run_btn = st.button("🚀 Run Model", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Force Data Refresh", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF Regime-Aware Rotation Model")
st.caption(
    "Wasserstein k-means regime detection • Momentum Ranking (RoC + OBV + Breakout) • "
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


# ── Sweep helpers ─────────────────────────────────────────────────────────────
import json as _json_sw, re as _re_sw, os as _os_sw

SWEEP_YEARS = [2008, 2013, 2015, 2017, 2019, 2021]

def _today_est():
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    return (_dt.now(_tz.utc) - _td(hours=5)).date()

def _trigger_sweep(years, gh_token, gh_repo):
    import requests as _req
    sweep_str = ",".join(str(y) for y in years)
    resp = _req.post(
        f"https://api.github.com/repos/{gh_repo}/actions/workflows/sweep.yml/dispatches",
        headers={"Authorization": f"Bearer {gh_token}",
                 "Accept": "application/vnd.github+json"},
        json={"ref": "main", "inputs": {"sweep_years": sweep_str}},
        timeout=15,
    )
    return resp.status_code == 204

@st.cache_data(ttl=60, show_spinner=False)
def _load_sweep_cache(date_str, _bust=0):
    """Load sweep files specifically dated today (date_str = YYYYMMDD)."""
    cache = {}
    try:
        import requests as _req
        gl_token = _os_sw.environ.get("GITLAB_API_TOKEN", "")
        if not gl_token:
            return cache
        proj    = "P2SAMAPA%2Fp2-etf-regime-predictor"
        gl_base = "https://gitlab.com"
        headers = {"PRIVATE-TOKEN": gl_token}
        for yr in SWEEP_YEARS:
            fname = f"data%2Fsweep_{yr}_{date_str}.json"
            r = _req.get(
                f"{gl_base}/api/v4/projects/{proj}/repository/files/{fname}/raw?ref=main",
                headers=headers, timeout=10,
            )
            if r.status_code == 200:
                try:
                    cache[yr] = r.json()
                except Exception:
                    pass
    except Exception:
        pass
    return cache

@st.cache_data(ttl=60, show_spinner=False)
def _load_sweep_any(_bust=0):
    """
    Load the most recent sweep result for EACH year independently.
    Fix: old version picked ONE global best_date — if year 2013 only had
    a 20260307 file but best_date was 20260308, it returned nothing for 2013.
    New version: per-year independent date lookup — always shows latest per year.
    Returns (found_dict, display_date) where display_date is the most
    recent date seen across any year.
    """
    found, best_date = {}, None
    try:
        import requests as _req
        from datetime import datetime as _dt2
        gl_token = _os_sw.environ.get("GITLAB_API_TOKEN", "")
        if not gl_token:
            return found, best_date
        proj    = "P2SAMAPA%2Fp2-etf-regime-predictor"
        gl_base = "https://gitlab.com"
        headers = {"PRIVATE-TOKEN": gl_token}
        r = _req.get(
            f"{gl_base}/api/v4/projects/{proj}/repository/tree?path=data&per_page=100&ref=main",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return found, best_date

        # Build per-year most-recent-date map
        year_best = {}
        for item in r.json():
            name = item.get("name", "")
            if name.startswith("sweep_") and name.endswith(".json"):
                parts = name.replace(".json", "").split("_")
                if len(parts) == 3:
                    try:
                        yr_int = int(parts[1])
                        dt     = _dt2.strptime(parts[2], "%Y%m%d").date()
                        if yr_int not in year_best or dt > year_best[yr_int]:
                            year_best[yr_int] = dt
                    except Exception:
                        pass

        # Fetch most recent file per year independently
        for yr in SWEEP_YEARS:
            if yr not in year_best:
                continue
            yr_date  = year_best[yr]
            date_str = yr_date.strftime("%Y%m%d")
            fname    = f"data%2Fsweep_{yr}_{date_str}.json"
            r2 = _req.get(
                f"{gl_base}/api/v4/projects/{proj}/repository/files/{fname}/raw?ref=main",
                headers=headers, timeout=10,
            )
            if r2.status_code == 200:
                try:
                    found[yr] = r2.json()
                    if best_date is None or yr_date > best_date:
                        best_date = yr_date
                except Exception:
                    pass
    except Exception:
        pass
    return found, best_date

def _compute_consensus(sweep_data):
    import numpy as _np
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year":       yr,
            "signal":     sig.get("signal","?"),
            "ann_return": float(sig.get("ann_return", 0.0)),
            "z_score":    float(sig.get("z_score", 0.0)),
            "sharpe":     float(sig.get("sharpe", 0.0)),
            "max_dd":     float(sig.get("max_dd", 0.0)),
            "conviction": sig.get("conviction","?"),
            "regime":     sig.get("regime","?"),
        })
    if not rows:
        return {}
    df_c = pd.DataFrame(rows)
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df_c["wtd"] = (0.40*_mm(df_c["ann_return"]) + 0.20*_mm(df_c["z_score"]) +
                   0.20*_mm(df_c["sharpe"])      + 0.20*_mm(-df_c["max_dd"]))
    etf_agg = {}
    for _, row in df_c.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years":[], "scores":[], "returns":[],
                                "zs":[], "sharpes":[], "dds":[]})
        etf_agg[e]["years"].append(row["year"])
        etf_agg[e]["scores"].append(row["wtd"])
        etf_agg[e]["returns"].append(row["ann_return"])
        etf_agg[e]["zs"].append(row["z_score"])
        etf_agg[e]["sharpes"].append(row["sharpe"])
        etf_agg[e]["dds"].append(row["max_dd"])
    total = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        summary[e] = {
            "cum_score":   round(cs, 4),
            "score_share": round(cs / total, 3),
            "n_years":     len(v["years"]),
            "years":       v["years"],
            "avg_return":  round(float(_np.mean(v["returns"])), 4),
            "avg_z":       round(float(_np.mean(v["zs"])), 3),
            "avg_sharpe":  round(float(_np.mean(v["sharpes"])), 3),
            "avg_max_dd":  round(float(_np.mean(v["dds"])), 4),
        }
    winner_etf = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner": winner_etf, "etf_summary": summary,
            "per_year": df_c.to_dict("records"), "n_years": len(rows)}

ETF_COLORS_SW = {
    "TLT":"#4e79a7","VNQ":"#76b7b2","SLV":"#edc948","GLD":"#b07aa1",
    "LQD":"#59a14f","HYG":"#e15759","CASH":"#aaaaaa",
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep (rendered BEFORE tab1 to avoid st.stop())
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔄 Multi-Year Consensus Sweep")
    st.markdown(
        "Runs the regime-aware rotation model across **6 start years** and aggregates "
        "signals into a weighted consensus.  \n"
        f"**Sweep years:** {', '.join(str(y) for y in SWEEP_YEARS)}  &nbsp;·&nbsp;  "
        "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)  \n"
        "Auto-runs daily at **8pm EST**. Results are date-stamped — stale cache never shown."
    )

    today_sw  = _today_est()
    today_str = today_sw.strftime("%Y%m%d")

    if "sweep_bust" not in st.session_state:
        st.session_state.sweep_bust = 0

    today_cache          = _load_sweep_cache(today_str, _bust=st.session_state.sweep_bust)
    all_cache, best_date = _load_sweep_any(_bust=st.session_state.sweep_bust)

    # prev_cache = years that have ANY result but were NOT run today
    # This replaces the old "wipe prev_cache if prev_date == today" logic which
    # caused 2013/2015 to disappear when some years ran today and some hadn't yet
    prev_cache = {yr: v for yr, v in all_cache.items() if yr not in today_cache}

    # Merged display: today takes priority, fill gaps with prev
    # Ensures consensus uses ALL 6 years, not just the 4 that ran today
    display_cache = {**prev_cache, **today_cache}
    display_date  = today_sw if today_cache else best_date

    sweep_complete = len(today_cache) == len(SWEEP_YEARS)

    if prev_cache and not sweep_complete:
        st.warning(
            f"⚠️ {len(prev_cache)} year(s) showing previous results "
            f"({', '.join(str(y) for y in sorted(prev_cache.keys()))}). "
            "Today's sweep hasn't run for these yet.", icon="📅"
        )

    # Status grid — per-year today vs prev vs not run
    _cols = st.columns(len(SWEEP_YEARS))
    for _i, _yr in enumerate(SWEEP_YEARS):
        with _cols[_i]:
            if _yr in today_cache:
                st.success(f"**{_yr}**\n✅ {today_cache[_yr].get('signal','?')}")
            elif _yr in prev_cache:
                st.warning(f"**{_yr}**\n📅 {prev_cache[_yr].get('signal','?')}")
            else:
                st.error(f"**{_yr}**\n⏳ Not run")
    st.caption("✅ today · 📅 previous · ⏳ not run")
    st.divider()

    # Run button
    _missing      = [y for y in SWEEP_YEARS if y not in today_cache]
    _force_rerun  = st.checkbox("🔄 Force re-run all years", value=False,
                                help="Re-trains even if today\'s results already exist")
    _trigger_yrs  = SWEEP_YEARS if _force_rerun else _missing

    _cb, _cr, _ci = st.columns([1, 1, 2])
    with _cb:
        _sweep_btn = st.button("🚀 Run Consensus Sweep", type="primary",
                               use_container_width=True,
                               disabled=(sweep_complete and not _force_rerun))
    with _cr:
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.session_state.sweep_bust += 1
            _load_sweep_cache.clear()
            _load_sweep_any.clear()
            st.rerun()
    with _ci:
        if sweep_complete and not _force_rerun:
            st.success(f"✅ Today\'s sweep complete ({today_sw}) — all {len(SWEEP_YEARS)} years ready")
        else:
            st.info(f"**{len(today_cache)}/{len(SWEEP_YEARS)}** years done today · "
                    f"**{len(display_cache)}/{len(SWEEP_YEARS)}** total available.  \n"
                    f"Will trigger **{len(_trigger_yrs)}** jobs: {', '.join(str(y) for y in _trigger_yrs)}")

    if _sweep_btn and _trigger_yrs:
        try:
            import os as _os_sw
            _gh_token = _os_sw.environ.get("GITHUB_PAT", "")
            _gh_repo  = _os_sw.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-REGIME-PREDICTOR")
            if not _gh_token:
                raise ValueError("GITHUB_PAT secret not set")
            _ok = _trigger_sweep(_trigger_yrs, _gh_token, _gh_repo)
            if _ok:
                st.success(f"✅ Triggered {len(_trigger_yrs)} jobs: "
                           f"{', '.join(str(y) for y in _trigger_yrs)}. ~10-15 mins each.")
            else:
                st.error("❌ GitHub API returned non-204. Check GITHUB_PAT permissions.")
        except Exception as _ex:
            st.error(f"❌ Trigger failed: {_ex}")

    if not display_cache:
        st.info("👆 No sweep results yet. Click **🚀 Run Consensus Sweep** or wait for 8pm EST.")
    else:
        _cons = _compute_consensus(display_cache)
        if not _cons:
            st.warning("Could not compute consensus.")
        else:
            _w     = _cons["winner"]
            _wi    = _cons["etf_summary"][_w]
            _wc    = ETF_COLORS_SW.get(_w, "#0066cc")
            _sp    = _wi["score_share"] * 100
            _sp    = _sp if _sp == _sp else 0.0
            _slab  = "⚠️ Split Signal" if _wi["score_share"] < 0.40 else "✅ Clear Consensus"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                        border:2px solid {_wc};border-radius:16px;
                        padding:32px;text-align:center;margin:16px 0;">
              <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
                WEIGHTED CONSENSUS · REGIME-PREDICTOR · {len(display_cache)} START YEARS · {display_date}
              </div>
              <div style="font-size:72px;font-weight:900;color:{_wc};
                          text-shadow:0 0 30px {_wc}88;">{_w}</div>
              <div style="font-size:14px;color:#ccc;margin-top:8px;">
                {_slab} · Score share {_sp:.0f}% · {_wi["n_years"]}/{len(SWEEP_YEARS)} years
              </div>
              <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#aaa;">Avg Return</div>
                  <div style="font-size:22px;font-weight:700;color:{'#00b894' if _wi['avg_return']>0 else '#d63031'};">
                    {_wi["avg_return"]*100:.1f}%</div></div>
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#aaa;">Avg Z</div>
                  <div style="font-size:22px;font-weight:700;color:#74b9ff;">{_wi["avg_z"]:.2f}σ</div></div>
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
                  <div style="font-size:22px;font-weight:700;color:#a29bfe;">{_wi["avg_sharpe"]:.2f}</div></div>
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
                  <div style="font-size:22px;font-weight:700;color:#fd79a8;">{_wi["avg_max_dd"]*100:.1f}%</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            _others = sorted([(e,v) for e,v in _cons["etf_summary"].items() if e != _w],
                             key=lambda x: -x[1]["cum_score"])
            if _others:
                _parts = [f'<span style="color:{ETF_COLORS_SW.get(e,"#888")};font-weight:600;">{e}</span> '
                          f'<span style="color:#aaa;">({v["cum_score"]:.2f})</span>' for e,v in _others]
                st.markdown('<div style="text-align:center;font-size:13px;margin-bottom:12px;">Also ranked: '
                            + ' &nbsp;|&nbsp; '.join(_parts) + '</div>', unsafe_allow_html=True)
            st.divider()

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
                                       f"Return: {_row['ann_return']*100:.1f}%<extra></extra>")
                    ))
                _fs.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
                _fs.update_layout(template="plotly_dark", height=360,
                                  xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
                                  margin=dict(t=20,b=20))
                st.plotly_chart(_fs, use_container_width=True)

            st.subheader("📋 Full Per-Year Breakdown")
            st.caption(f"40% Ann. Return + 20% Z + 20% Sharpe + 20% (–MaxDD), "
                       f"min-max normalised · Results: {display_date}")
            _tbl = []
            for _row in sorted(_cons["per_year"], key=lambda r: r["year"]):
                _tbl.append({
                    "Start Year":   _row["year"],
                    "Signal":       _row["signal"],
                    "Regime":       _row.get("regime","?"),
                    "Conviction":   _row.get("conviction","?"),
                    "Wtd Score":    round(_row["wtd"], 3),
                    "Z-Score":      f"{_row['z_score']:.2f}σ",
                    "Ann. Return":  f"{_row['ann_return']*100:.2f}%",
                    "Sharpe":       f"{_row['sharpe']:.2f}",
                    "Max Drawdown": f"{_row['max_dd']*100:.2f}%",
                    "Date": "✅ Today" if _row["year"] in today_cache else f"📅 {display_date}",
                })
            _tdf = pd.DataFrame(_tbl)
            def _ss(val):
                c = ETF_COLORS_SW.get(val,"#888")
                return f"background-color:{c}22;color:{c};font-weight:700;"
            def _sr(val):
                try:
                    v = float(str(val).replace("%",""))
                    return "color:#00b894;font-weight:600" if v>0 else "color:#d63031;font-weight:600"
                except Exception: return ""
            st.dataframe(_tdf.style.applymap(_ss,subset=["Signal"])
                                   .applymap(_sr,subset=["Ann. Return"])
                                   .set_properties(**{"text-align":"center"})
                                   .hide(axis="index"),
                         use_container_width=True, height=280)

with tab1:
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
    if detector is None or momentum_ranker is None:
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
            if use_wf:
                pred_history = load_wf_momentum_predictions_from_gitlab()
                src_label = "Walk-Forward OOS"
                if pred_history is None:
                    st.error("⚠️ Walk-Forward predictions not found. "
                             "Trigger Manual Retrain with run_wfcv=true.")
                    st.stop()
            else:
                pred_history = load_momentum_predictions_from_gitlab()
                src_label = "In-Sample"
                if pred_history is None:
                    st.warning("⚠️ Momentum predictions not found — generating now...")
                    pred_history = momentum_ranker.predict_all_history(df)

            if pred_history is not None:
                pred_history.index = pd.to_datetime(pred_history.index)
                st.success(f"✅ {src_label}: {len(pred_history):,} rows "
                           f"({pred_history.index[0].date()} → "
                           f"{pred_history.index[-1].date()})")
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

            st.caption(f"Backtest period: {pred_bt.index[0].date()} → "
                       f"{pred_bt.index[-1].date()} ({len(pred_bt):,} days)")

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

    base_rates = {t: 0.5 for t in TARGET_ETFS}

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
    worst_idx  = metrics.get("max_daily_idx", 0)
    worst_date = pred_bt.index[min(worst_idx, len(pred_bt)-1)].strftime("%d %b %Y")
    c5.metric("⚠️ Worst Day",
              f"{metrics.get('max_daily_dd',0)*100:.2f}%",
              delta=f"on {worst_date}")

    # Secondary metrics row
    c6, c7, c8, c9, c10 = st.columns(5)
    dd_idx  = metrics.get("max_dd_idx", 0)
    dd_date = pred_bt.index[min(dd_idx, len(pred_bt)-1)].strftime("%d %b %Y")
    c6.metric("🏆 Calmar",
              f"{metrics.get('calmar',0):.2f}",
              delta=f"MaxDD on {dd_date}")
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

    <h4 style="color:#00d1b2;">📈 Layer 2 — Momentum Ranking</h4>
    <p>A pure rules-based composite momentum score ranks all 6 ETFs each day.
    The top-ranked ETF with sufficient conviction (Z ≥ threshold) is selected.
    No ML — fully transparent and interpretable.</p>
    <ul>
      <li><b>Rate-of-Change (RoC):</b> 40% × 5d + 30% × 10d + 20% × 21d + 10% × 63d</li>
      <li><b>OBV accumulation:</b> 15% weight — volume-weighted momentum confirmation</li>
      <li><b>20d Breakout score:</b> 15% weight — position within rolling high/low range</li>
      <li>Scores Z-normalised cross-sectionally across all 6 ETFs each day</li>
    </ul>

    <h4 style="color:#00d1b2;">📊 Input Features</h4>
    <p>FRED macro signals used for regime detection only:</p>
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

      <li><b>Transaction cost:</b> {fee_bps}bps per one-way trade</li>
    </ul>

    <h4 style="color:#00d1b2;">🔄 Daily Pipeline</h4>
    <p>GitHub Actions runs at 6:30am EST every weekday:
    fetch new data → detect regime → fit momentum ranker → generate signal →
    push to GitLab. Walk-forward OOS predictions refresh monthly.
    This UI is read-only — loads pre-computed signals from GitLab at runtime.</p>

    <h4 style="color:#ff6b6b;">⚠️ Important Caveats</h4>
    <ul>
      <li>Past performance does not guarantee future results</li>
      <li>Backtest includes transaction costs but not slippage or
          liquidity constraints</li>
      <li>This is a research tool, not investment advice</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
