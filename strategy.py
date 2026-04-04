strategy.py — P2-ETF-REGIME-PREDICTOR v2 (CORRECTED v2)
=========================================
Signal execution, backtesting, and performance metrics.

Fixes in v2:
- Ensure last_p array contains valid probabilities (no NaN/Inf)
- Better handling of edge cases in conviction calculation
- Added validation for input data

All functions accept target_etfs as a parameter — NO hardcoded ETF list.
This allows the same strategy logic to work for both:
 Option A: TLT, VNQ, SLV, GLD, LQD, HYG
 Option B: SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME

Strategy rules:
 1. Conviction gate — only enter if Z-score >= z_min_entry
 2. Stop-loss — if 2-day cumulative return <= stop_loss_pct,
    move to CASH until z_reentry conviction
 3. Disagreement filter— if models disagree beyond threshold,
    halve effective conviction
 4. CASH earns daily risk-free rate (3M T-Bill / 252)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pytz

_EST = pytz.timezone("US/Eastern")
log = logging.getLogger(__name__)

# Fixed Z threshold
DEFAULT_Z_MIN = 0.7

# Per-ETF stop multipliers (Option A ETFs only — others default to 1.0)
ETF_STOP_MULTIPLIER = {
    "TLT": 1.00,
    "VNQ": 1.00,
    "SLV": 0.75,
    "GLD": 0.75,
    "LQD": 1.00,
    "HYG": 0.85,
}


def compute_conviction(p_beat_cash: np.ndarray) -> Tuple[int, float, str]:
    """
    Cross-sectional conviction from P(beat cash) array on a single day.
    Returns (best_idx, z, label).
    Works for any ETF universe size.
    """
    # CORRECTED: Handle edge cases properly
    if len(p_beat_cash) == 0:
        return 0, 0.0, "Low"

    # Clean the array - replace NaN/Inf with 0.5
    p_clean = np.nan_to_num(p_beat_cash, nan=0.5, posinf=1.0, neginf=0.0)

    mean = np.mean(p_clean)
    std = np.std(p_clean)

    if std < 1e-9 or not np.isfinite(std):
        # All probabilities are essentially the same
        best = int(np.argmax(p_clean))
        return best, 0.0, "Low"

    best = int(np.argmax(p_clean))
    z = float((p_clean[best] - mean) / std)

    # Ensure z is finite
    if not np.isfinite(z):
        z = 0.0

    # CORRECTED: More granular conviction labels
    label = ("Very High" if z >= 2.5 else
             "High" if z >= 1.5 else
             "Moderate" if z >= 0.7 else
             "Low" if z >= 0.3 else "Very Low")
    return best, z, label

def compute_sweep_z(strat_rets: np.ndarray,
                    rf_rate: float = 0.045) -> Tuple[float, str]:
    """
    t-statistic conviction from backtest return series.
    Z = (mean_excess / std_excess) * sqrt(n)
    Varies meaningfully across sweep years — use this for sweep payloads.
    """
    if len(strat_rets) < 10:
        return 0.0, "Low"

    # CORRECTED: Filter out NaN values
    strat_rets = strat_rets[~np.isnan(strat_rets)]
    if len(strat_rets) < 10:
        return 0.0, "Low"

    daily_rf = rf_rate / 252
    excess = strat_rets - daily_rf
    mean_ex = float(np.mean(excess))
    std_ex = float(np.std(excess))

    if std_ex < 1e-9 or not np.isfinite(std_ex) or not np.isfinite(mean_ex):
        return 0.0, "Low"

    z = float(mean_ex / std_ex * np.sqrt(len(excess)))

    if not np.isfinite(z):
        return 0.0, "Low"

    label = ("Very High" if z >= 3.0 else
             "High" if z >= 1.5 else
             "Moderate" if z >= 0.5 else "Low")
    return round(z, 3), label

def execute_strategy(
    predictions_df: pd.DataFrame,
    daily_ret_df: pd.DataFrame,
    rf_rate: float = 0.045,
    z_reentry: float = 1.0,
    stop_loss_pct: float = -0.12,
    fee_bps: int = 5,
    regime_series: Optional[pd.Series] = None,
    target_etfs: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict], pd.Timestamp, str, float, str, np.ndarray]:
    """
    Execute strategy on prediction history.

    Parameters
    ----------
    predictions_df : DataFrame with columns {ETF}_P, {ETF}_RS, {ETF}_PA,
        {ETF}_Disagree for each ETF in target_etfs.
    daily_ret_df : DataFrame with columns {ETF}_Ret for each ETF.
    target_etfs : ETF universe list. MUST be provided explicitly.

    Returns
    -------
    (strat_rets, audit_trail, next_date, next_signal,
     conviction_z, conviction_label, last_p)

    CORRECTED: last_p is now guaranteed to contain valid probabilities (no NaN/Inf)
    """
    # CORRECTED: No default fallback — require explicit ETF list
    if target_etfs is None:
        raise ValueError(
            "target_etfs must be provided explicitly. "
            "Use cfg.OPTION_A_ETFS or cfg.OPTION_B_ETFS"
        )

    n_etfs = len(target_etfs)
    daily_rf = rf_rate / 252
    now_est = datetime.now(_EST)
    today = now_est.date()
    market_closed = now_est.hour >= 16
    fee = fee_bps / 10_000

    strat_rets = []
    audit_trail = []
    recent_rets = []
    etf_rets_buf = []
    stop_active = False
    cum_ret = 1.0
    p_array = np.full(n_etfs, 1.0 / n_etfs)  # Equal weight default

    # Build column name lists for this ETF universe
    p_cols = [f"{t}_P" for t in target_etfs]
    rs_cols = [f"{t}_RS" for t in target_etfs]
    pa_cols = [f"{t}_PA" for t in target_etfs]
    dis_cols = [f"{t}_Disagree" for t in target_etfs]
    ret_cols = [f"{t}_Ret" for t in target_etfs]

    common_dates = predictions_df.index.intersection(daily_ret_df.index)
    if len(common_dates) == 0:
        log.warning("execute_strategy: no common dates between predictions and returns")
        # CORRECTED: Return valid last_p even on error
        return (np.array([]), [], pd.Timestamp(datetime.now()),
                "CASH", 0.0, "Low", np.full(n_etfs, 1.0 / n_etfs))

    pred_a = predictions_df.loc[common_dates]
    ret_a = daily_ret_df.loc[common_dates]

    for i, trade_date in enumerate(common_dates):
        row = pred_a.iloc[i]
        ret_row = ret_a.iloc[i]

        # CORRECTED: Handle missing columns gracefully and ensure valid values
        p_array = []
        for c in p_cols:
            val = float(row.get(c, 1.0 / n_etfs))
            # Ensure valid probability
            if not np.isfinite(val):
                val = 1.0 / n_etfs
            p_array.append(val)
        p_array = np.array(p_array)

        # Normalize to ensure sum = 1
        p_sum = np.sum(p_array)
        if p_sum > 0 and np.isfinite(p_sum):
            p_array = p_array / p_sum
        else:
            p_array = np.full(n_etfs, 1.0 / n_etfs)

        rs_array = np.array([float(row.get(c, 0.0)) for c in rs_cols])
        pa_array = np.array([float(row.get(c, p_array[j] - 1.0/n_etfs))
                             for j, c in enumerate(pa_cols)])
        dis_arr = np.array([bool(row.get(c, False)) for c in dis_cols])

        # Use rank scores if they have variance, otherwise use adjusted probabilities
        rank_input = rs_array if np.std(rs_array) > 1e-6 else pa_array
        ranked = np.argsort(rank_input)[::-1]
        best_idx = int(ranked[0])
        _, day_z, day_label = compute_conviction(p_array)

        current_regime_name = ""
        if regime_series is not None and trade_date in regime_series.index:
            current_regime_name = str(regime_series.loc[trade_date])

        active_idx = best_idx
        etf_name = target_etfs[active_idx]
        active_disagree = bool(dis_arr[active_idx])

        # CORRECTED: Disagreement reduces conviction more gradually
        effective_z = day_z * 0.7 if active_disagree else day_z

        act_ret_col = f"{etf_name}_Ret"
        realized = float(ret_row.get(act_ret_col, 0.0))
        etf_stop = stop_loss_pct * ETF_STOP_MULTIPLIER.get(etf_name, 1.0)

        # Check stop-loss conditions
        two_day_breach = (
            len(recent_rets) >= 2 and
            (1 + recent_rets[-2]) * (1 + recent_rets[-1]) - 1 <= etf_stop
        )

        hwm_breach = False
        if len(etf_rets_buf) >= 3:
            cum_window = np.cumprod([1 + r for r in etf_rets_buf])
            peak = float(np.max(cum_window))
            current = float(cum_window[-1])
            dd_window = (current - peak) / (peak + 1e-9)
            hwm_breach = dd_window <= etf_stop * 0.75

        # Trading logic
        if stop_active:
            if effective_z >= z_reentry:
                stop_active = False
                net_ret = realized - fee
                trade_signal = etf_name
            else:
                net_ret = daily_rf
                trade_signal = "CASH"
        else:
            if effective_z < DEFAULT_Z_MIN:
                net_ret = daily_rf
                trade_signal = "CASH"
            elif two_day_breach or hwm_breach:
                stop_active = True
                net_ret = daily_rf
                trade_signal = "CASH"
            else:
                net_ret = realized - fee
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

        # Audit trail
        td_val = trade_date.date() if hasattr(trade_date, "date") else trade_date
        if td_val < today or (td_val == today and market_closed):
            rname = ""
            if regime_series is not None and trade_date in regime_series.index:
                rname = str(regime_series.loc[trade_date])

            etf_rets_audit = {}
            for etf in target_etfs:
                rc = f"{etf}_Ret"
                etf_rets_audit[f"{etf}_Ret%"] = round(
                    float(ret_row.get(rc, 0.0)) * 100, 3
                )

            signal_ret = (realized * 100 if trade_signal != "CASH"
                          else daily_rf * 100)

            entry = {
                "Date": td_val.strftime("%Y-%m-%d"),
                "Signal": trade_signal,
                "Top_Pick": target_etfs[best_idx],
                "Regime": rname,
                "Conviction_Z": round(day_z, 2),
                "Effective_Z": round(effective_z, 2),
                "P_Top": round(float(p_array[best_idx]), 3),
                "Signal_Ret%": round(signal_ret, 3),
                "Stop_Active": stop_active,
                "Disagree": active_disagree,
            }
            entry.update(etf_rets_audit)
            audit_trail.append(entry)

    strat_rets = np.array(strat_rets)

    # Next signal
    last_date = common_dates[-1]
    next_date = _next_trading_day(last_date)
    last_row = pred_a.iloc[-1]

    # CORRECTED: Build last_p with validation
    last_p = []
    for c in p_cols:
        val = float(last_row.get(c, 1.0 / n_etfs))
        if not np.isfinite(val):
            val = 1.0 / n_etfs
        last_p.append(val)
    last_p = np.array(last_p)

    # Normalize
    p_sum = np.sum(last_p)
    if p_sum > 0 and np.isfinite(p_sum):
        last_p = last_p / p_sum
    else:
        last_p = np.full(n_etfs, 1.0 / n_etfs)

    nb, nz, nl = compute_conviction(last_p)
    next_signal = target_etfs[nb]

    return strat_rets, audit_trail, next_date, next_signal, nz, nl, last_p

def calculate_metrics(strat_rets: np.ndarray,
                      rf_rate: float = 0.045) -> Dict:
    """Full performance metrics from daily return array."""
    if len(strat_rets) == 0:
        return {}

    # Drop NaN returns (warmup rows) before any calculation
    strat_rets = strat_rets[~np.isnan(strat_rets)]
    if len(strat_rets) == 0:
        return {}

    cum = np.cumprod(1 + strat_rets)
    n = len(strat_rets)

    # CORRECTED: Handle edge case of insufficient data
    if n < 2:
        return {
            "cum_returns": cum,
            "ann_return": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "hit_ratio": 0.0,
            "hit_ratio_15d": 0.0,
            "max_dd": 0.0,
            "max_dd_idx": 0,
            "max_daily_dd": 0.0,
            "max_daily_idx": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_r": 0.0,
            "cum_max": cum,
            "n_days": n,
        }

    ann_ret = float(cum[-1] ** (252 / n) - 1)
    daily_rf = rf_rate / 252
    excess = strat_rets - daily_rf

    # CORRECTED: Handle zero std case
    std_excess = float(np.std(excess))
    if std_excess < 1e-9 or not np.isfinite(std_excess):
        sharpe = 0.0
    else:
        sharpe = float(np.mean(excess) / std_excess * np.sqrt(252))

    tol = daily_rf * 0.01
    active_mask = np.abs(strat_rets - daily_rf) > tol
    active_days = strat_rets[active_mask]
    hit_ratio = float(np.mean(active_days > 0)) if len(active_days) > 0 else 0.0

    recent_15d = strat_rets[-15:]
    active_15d = recent_15d[np.abs(recent_15d - daily_rf) > tol]
    hit_15d = float(np.mean(active_15d > 0)) if len(active_15d) > 0 else 0.0

    cum_max = np.maximum.accumulate(cum)
    dd = (cum - cum_max) / (cum_max + 1e-9)
    max_dd = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))
    max_daily = float(np.min(strat_rets))
    max_daily_idx = int(np.argmin(strat_rets))

    calmar = ann_ret / (abs(max_dd) + 1e-9)
    wins = strat_rets[strat_rets > 0]
    losses = strat_rets[strat_rets < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return {
        "cum_returns": cum,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "calmar": calmar,
        "hit_ratio": hit_ratio,
        "hit_ratio_15d": hit_15d,
        "max_dd": max_dd,
        "max_dd_idx": max_dd_idx,
        "max_daily_dd": max_daily,
        "max_daily_idx": max_daily_idx,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_r": abs(avg_win / (avg_loss + 1e-9)),
        "cum_max": cum_max,
        "n_days": n,
    }

def calculate_benchmark_metrics(bench_rets: np.ndarray,
                                rf_rate: float = 0.045) -> Dict:
    if len(bench_rets) == 0:
        return {}

    bench_rets = bench_rets[~np.isnan(bench_rets)]
    if len(bench_rets) == 0:
        return {}

    cum = np.cumprod(1 + bench_rets)
    ann_ret = float(cum[-1] ** (252 / len(bench_rets)) - 1)

    std_rets = float(np.std(bench_rets))
    if std_rets < 1e-9 or not np.isfinite(std_rets):
        sharpe = 0.0
    else:
        sharpe = ((float(np.mean(bench_rets)) - rf_rate / 252) /
                  std_rets * np.sqrt(252))

    cum_max = np.maximum.accumulate(cum)
    dd = (cum - cum_max) / (cum_max + 1e-9)
    return {
        "cum_returns": cum,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_dd": float(np.min(dd)),
        "max_daily_dd": float(np.min(bench_rets)),
    }

def build_signal_row(next_date, next_signal, conviction_z, conviction_label,
                     p_array, regime_int, regime_name, metrics,
                     target_etfs=None) -> pd.DataFrame:
    if target_etfs is None:
        raise ValueError("target_etfs must be provided explicitly")

    # CORRECTED: Ensure p_array contains valid values
    p_clean = []
    for p in p_array:
        val = float(p)
        if not np.isfinite(val):
            val = 0.5
        p_clean.append(val)

    row = {
        "Signal": next_signal,
        "Conviction_Z": round(conviction_z, 3),
        "Conviction_Label": conviction_label,
        "Regime": regime_int,
        "Regime_Name": regime_name,
        "Ann_Return_Pct": round(metrics.get("ann_return", 0) * 100, 2),
        "Sharpe": round(metrics.get("sharpe", 0), 3),
        "Max_DD_Pct": round(metrics.get("max_dd", 0) * 100, 2),
    }
    for i, etf in enumerate(target_etfs):
        row[f"{etf}_P"] = round(p_clean[i], 4)
    return pd.DataFrame([row], index=[next_date])


def _next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (last_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            return pd.Timestamp(sched.index[0])
    except ImportError:
        pass
    nxt = last_date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt

def next_trading_day_from_today() -> pd.Timestamp:
    now_est = datetime.now(_EST)
    today_ts = pd.Timestamp(now_est.date())
    return _next_trading_day(today_ts)
