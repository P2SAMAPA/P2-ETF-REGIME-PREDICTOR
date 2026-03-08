"""
strategy.py — P2-ETF-REGIME-PREDICTOR
=======================================
Signal execution, backtesting, and performance metrics.

Strategy rules:
  1. Conviction gate    — only enter if Z-score >= z_min_entry
  2. Stop-loss          — if 2-day cumulative return <= stop_loss_pct,
                          move to CASH until z_reentry conviction
  3. Rotation           — if top pick 5-day cumulative return < 0,
                          rotate to #2 ranked ETF
  4. Disagreement filter— if LightGBM and LogReg disagree beyond
                          threshold, halve effective conviction
  5. CASH earns daily risk-free rate (3M T-Bill / 252)

Author: P2SAMAPA
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pytz
_EST = pytz.timezone("US/Eastern")

log = logging.getLogger(__name__)

TARGET_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]

# Fixed Z threshold — 0.7 sigma across all regimes
DEFAULT_Z_MIN = 0.7


def compute_conviction(p_beat_cash: np.ndarray) -> Tuple[int, float, str]:
    """
    Cross-sectional conviction from P(beat cash) array on a single day.
    Returns (best_idx, z, label).

    NOTE: This Z is cross-sectional (spread across 6 ETFs on one day).
    It is used for the live conviction gate and the signal banner only.
    Do NOT use this Z in the sweep payload — use compute_sweep_z() instead,
    which computes a meaningful time-series Z from backtest metrics.
    """
    mean = np.mean(p_beat_cash)
    std  = np.std(p_beat_cash)
    if std < 1e-9:
        return int(np.argmax(p_beat_cash)), 0.0, "Low"
    best  = int(np.argmax(p_beat_cash))
    z     = float((p_beat_cash[best] - mean) / std)
    label = ("Very High" if z >= 2.0 else
             "High"      if z >= 1.0 else
             "Moderate"  if z >= 0.5 else "Low")
    return best, z, label


def compute_sweep_z(strat_rets: np.ndarray, rf_rate: float = 0.045) -> Tuple[float, str]:
    """
    Compute a meaningful sweep conviction Z from the backtest return series.

    This is a t-statistic on whether the strategy's mean daily excess return
    is significantly different from zero:
        Z = (mean_excess / std_excess) * sqrt(n)

    Unlike compute_conviction() which produces an identical ~1.54σ for all
    sweep years (cross-sectional artifact), this varies meaningfully across
    start years because it reflects actual strategy performance.

    Returns (z_score, conviction_label).
    """
    if len(strat_rets) < 10:
        return 0.0, "Low"

    daily_rf = rf_rate / 252
    excess   = strat_rets - daily_rf
    mean_ex  = float(np.mean(excess))
    std_ex   = float(np.std(excess))

    if std_ex < 1e-9:
        return 0.0, "Low"

    z = float(mean_ex / std_ex * np.sqrt(len(excess)))

    label = ("Very High" if z >= 3.0 else
             "High"      if z >= 1.5 else
             "Moderate"  if z >= 0.5 else "Low")
    return round(z, 3), label


# Per-ETF stop multipliers applied to user's slider value
ETF_STOP_MULTIPLIER = {
    "TLT": 1.00,
    "VNQ": 1.00,
    "SLV": 0.75,
    "GLD": 0.75,
    "LQD": 1.00,
    "HYG": 0.85,
}


def execute_strategy(
    predictions_df:  pd.DataFrame,
    daily_ret_df:    pd.DataFrame,
    rf_rate:         float = 0.045,
    z_reentry:       float = 1.0,
    stop_loss_pct:   float = -0.12,
    fee_bps:         int   = 5,
    regime_series:   Optional[pd.Series] = None,
) -> Tuple[np.ndarray, List[Dict], pd.Timestamp, str, float, str, np.ndarray]:
    """
    Execute strategy on prediction history.

    Returns (strat_rets, audit_trail, next_date, next_signal,
             conviction_z, conviction_label, last_p_array)

    Note: conviction_z / conviction_label returned here are cross-sectional
    (from the last day's p_array). For sweep quality metrics, use
    compute_sweep_z(strat_rets) after calling this function.
    """
    daily_rf      = rf_rate / 252
    now_est       = datetime.now(_EST)
    today         = now_est.date()
    market_closed = now_est.hour >= 16
    fee           = fee_bps / 10_000
    strat_rets    = []
    audit_trail   = []
    recent_rets   = []
    etf_rets_buf  = []
    stop_active   = False
    cum_ret       = 1.0
    rotated_idx   = None
    p_array       = np.full(len(TARGET_ETFS), 0.5)

    common_dates = predictions_df.index.intersection(daily_ret_df.index)
    pred_a       = predictions_df.loc[common_dates]
    ret_a        = daily_ret_df.loc[common_dates]
    p_cols       = [f"{t}_P"        for t in TARGET_ETFS]
    rs_cols      = [f"{t}_RS"       for t in TARGET_ETFS]
    pa_cols      = [f"{t}_PA"       for t in TARGET_ETFS]
    dis_cols     = [f"{t}_Disagree" for t in TARGET_ETFS]

    for i, trade_date in enumerate(common_dates):
        row      = pred_a.iloc[i]
        p_array  = np.array([float(row.get(c, 0.5)) for c in p_cols])
        rs_array = np.array([float(row.get(c, 0.0)) for c in rs_cols])
        pa_array = np.array([float(row.get(c, p_array[j] - 0.5))
                              for j, c in enumerate(pa_cols)])
        dis_arr  = np.array([bool(row.get(c, False)) for c in dis_cols])

        rank_input = rs_array if rs_array.std() > 1e-6 else pa_array
        ranked     = np.argsort(rank_input)[::-1]
        best_idx   = int(ranked[0])
        _, day_z, day_label = compute_conviction(p_array)

        current_regime_name = ""
        if regime_series is not None and trade_date in regime_series.index:
            current_regime_name = str(regime_series.loc[trade_date])

        if current_regime_name == "Risk-On":
            slv_pa = float(row.get("SLV_PA", 0.0))
            gld_pa = float(row.get("GLD_PA", 0.0))
            if slv_pa > 0.05 or gld_pa > 0.05:
                current_regime_name = "Risk-On-Commodity"

        z_min_entry     = DEFAULT_Z_MIN
        top_ret_col     = f"{TARGET_ETFS[best_idx]}_Ret"
        top_actual      = float(ret_a.iloc[i].get(top_ret_col, 0.0))

        rotated_idx     = None
        active_idx      = best_idx
        etf_name        = TARGET_ETFS[active_idx]
        active_disagree = bool(dis_arr[active_idx])
        effective_z     = day_z * 0.5 if active_disagree else day_z

        act_ret_col = f"{etf_name}_Ret"
        realized    = float(ret_a.iloc[i].get(act_ret_col, 0.0))

        etf_stop = stop_loss_pct * ETF_STOP_MULTIPLIER.get(etf_name, 1.0)

        two_day_breach = (
            len(recent_rets) >= 2 and
            (1 + recent_rets[-2]) * (1 + recent_rets[-1]) - 1 <= etf_stop
        )

        hwm_breach = False
        if len(etf_rets_buf) >= 3:
            cum_window = np.cumprod([1 + r for r in etf_rets_buf])
            peak       = float(np.max(cum_window))
            current    = float(cum_window[-1])
            dd_window  = (current - peak) / (peak + 1e-9)
            hwm_breach = dd_window <= etf_stop * 0.75

        if stop_active:
            if effective_z >= z_reentry:
                stop_active  = False
                net_ret      = realized - fee
                trade_signal = etf_name
            else:
                net_ret      = daily_rf
                trade_signal = "CASH"
        else:
            if effective_z < z_min_entry:
                net_ret      = daily_rf
                trade_signal = "CASH"
            elif two_day_breach or hwm_breach:
                stop_active  = True
                net_ret      = daily_rf
                trade_signal = "CASH"
            else:
                net_ret      = realized - fee
                trade_signal = etf_name

        strat_rets.append(net_ret)
        cum_ret *= (1 + net_ret)

        if trade_signal != "CASH":
            recent_rets.append(realized)
            etf_rets_buf.append(realized)
        if len(recent_rets) > 2:
            recent_rets.pop(0)
        if len(etf_rets_buf) > 10:
            etf_rets_buf.pop(0)

        td_val = trade_date.date() if hasattr(trade_date, "date") else trade_date
        if td_val < today or (td_val == today and market_closed):
            rname = ""
            if regime_series is not None and trade_date in regime_series.index:
                rname = str(regime_series.loc[trade_date])

            etf_rets = {}
            for etf in TARGET_ETFS:
                rc = f"{etf}_Ret"
                etf_rets[f"{etf}_Ret%"] = round(
                    float(ret_a.iloc[i].get(rc, 0.0)) * 100, 3
                )

            signal_ret = (realized * 100 if trade_signal != "CASH"
                          else daily_rf * 100)

            entry = {
                "Date":         td_val.strftime("%Y-%m-%d"),
                "Signal":       trade_signal,
                "Top_Pick":     TARGET_ETFS[best_idx],
                "Regime":       rname,
                "Conviction_Z": round(day_z, 2),
                "P_Top":        round(float(p_array[best_idx]), 3),
                "Signal_Ret%":  round(signal_ret, 3),
                "Stop_Active":  stop_active,
                "Rotated":      rotated_idx is not None,
                "Disagree":     active_disagree,
            }
            entry.update(etf_rets)
            audit_trail.append(entry)

    strat_rets = np.array(strat_rets)

    if len(common_dates) > 0:
        last_date   = common_dates[-1]
        next_date   = _next_trading_day(last_date)
        last_row    = pred_a.iloc[-1]
        last_p      = np.array([float(last_row.get(c, 0.5)) for c in p_cols])
        nb, nz, nl  = compute_conviction(last_p)
        next_signal = TARGET_ETFS[nb]
    else:
        next_date   = pd.Timestamp(datetime.now())
        next_signal = "CASH"
        nz, nl      = 0.0, "Low"
        last_p      = np.full(len(TARGET_ETFS), 0.5)

    return strat_rets, audit_trail, next_date, next_signal, nz, nl, last_p


def calculate_metrics(strat_rets: np.ndarray, rf_rate: float = 0.045) -> Dict:
    """Full performance metrics from daily return array."""
    if len(strat_rets) == 0:
        return {}
    cum         = np.cumprod(1 + strat_rets)
    n           = len(strat_rets)
    ann_ret     = float(cum[-1] ** (252 / n) - 1)
    daily_rf    = rf_rate / 252
    excess      = strat_rets - daily_rf
    sharpe      = float(np.mean(excess)) / (float(np.std(excess)) + 1e-9) * np.sqrt(252)
    daily_rf_scalar = rf_rate / 252
    tol         = daily_rf_scalar * 0.01
    active_mask = np.abs(strat_rets - daily_rf_scalar) > tol
    active_days = strat_rets[active_mask]
    hit_ratio   = float(np.mean(active_days > 0)) if len(active_days) > 0 else 0.0
    recent_rets_15d = strat_rets[-15:]
    active_15d  = recent_rets_15d[np.abs(recent_rets_15d - daily_rf_scalar) > tol]
    recent_15d  = float(np.mean(active_15d > 0)) if len(active_15d) > 0 else 0.0
    cum_max        = np.maximum.accumulate(cum)
    dd             = (cum - cum_max) / (cum_max + 1e-9)
    max_dd         = float(np.min(dd))
    max_dd_idx     = int(np.argmin(dd))
    max_daily      = float(np.min(strat_rets))
    max_daily_idx  = int(np.argmin(strat_rets))
    calmar         = ann_ret / (abs(max_dd) + 1e-9)
    wins        = strat_rets[strat_rets > 0]
    losses      = strat_rets[strat_rets < 0]
    avg_win     = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    avg_loss    = float(np.mean(losses)) if len(losses) > 0 else 0.0
    return {
        "cum_returns":    cum,
        "ann_return":     ann_ret,
        "sharpe":         sharpe,
        "calmar":         calmar,
        "hit_ratio":      hit_ratio,
        "hit_ratio_15d":  recent_15d,
        "max_dd":         max_dd,
        "max_dd_idx":     max_dd_idx,
        "max_daily_dd":   max_daily,
        "max_daily_idx":  max_daily_idx,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "win_loss_r":     abs(avg_win / (avg_loss + 1e-9)),
        "cum_max":        cum_max,
        "n_days":         n,
    }


def calculate_benchmark_metrics(bench_rets: np.ndarray,
                                  rf_rate: float = 0.045) -> Dict:
    if len(bench_rets) == 0:
        return {}
    cum     = np.cumprod(1 + bench_rets)
    ann_ret = float(cum[-1] ** (252 / len(bench_rets)) - 1)
    sharpe  = ((float(np.mean(bench_rets)) - rf_rate / 252) /
               (float(np.std(bench_rets)) + 1e-9) * np.sqrt(252))
    cum_max = np.maximum.accumulate(cum)
    dd      = (cum - cum_max) / (cum_max + 1e-9)
    return {
        "cum_returns":  cum,
        "ann_return":   ann_ret,
        "sharpe":       sharpe,
        "max_dd":       float(np.min(dd)),
        "max_daily_dd": float(np.min(bench_rets)),
    }


def build_signal_row(next_date, next_signal, conviction_z, conviction_label,
                      p_array, regime_int, regime_name, metrics) -> pd.DataFrame:
    row = {
        "Signal":           next_signal,
        "Conviction_Z":     round(conviction_z, 3),
        "Conviction_Label": conviction_label,
        "Regime":           regime_int,
        "Regime_Name":      regime_name,
        "Ann_Return_Pct":   round(metrics.get("ann_return", 0) * 100, 2),
        "Sharpe":           round(metrics.get("sharpe", 0), 3),
        "Max_DD_Pct":       round(metrics.get("max_dd", 0) * 100, 2),
    }
    for i, etf in enumerate(TARGET_ETFS):
        row[f"{etf}_P"] = round(float(p_array[i]), 4)
    return pd.DataFrame([row], index=[next_date])


def _next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    """
    Return the next NYSE trading day after last_date.
    Uses pandas_market_calendars if available for holiday awareness.
    Falls back to weekend-only skip if not installed.
    """
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (last_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            return pd.Timestamp(sched.index[0])
    except ImportError:
        pass
    # Fallback: skip weekends only
    nxt = last_date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt
