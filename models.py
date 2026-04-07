"""
models.py - P2-ETF-REGIME-PREDICTOR v3 (FULLY CORRECTED)
=========================================

KEY FIXES vs v2:
  1. MomentumRanker._active_features is never cached across different
     DataFrames.  In walk-forward CV, each window gets a fresh ranker
     object, so this was already safe — but the fallback in
     predict_all_history() now always calls _discover_features(df)
     from the current df rather than relying on a cached value that
     may have been set on a different training window's columns.

  2. predict_all_history() outputs the EXACT column schema that
     strategy.execute_strategy() reads:
       {ETF}_P, {ETF}_RS, {ETF}_PA, {ETF}_Disagree, Top_Pick, Regime
     This is unchanged from v2.

  3. RegimeDetector.add_regime_to_df() re-runs predict() on the full
     df every time so regime labels are correct for any window, not
     just for the window that was used during fit().

  4. _composite_score_row() and _discover_features() try multiple column
     name schemas for backward compat with legacy HF data (unchanged).

  5. Softmax-based probabilities produce real cross-sectional variance
     so different ETFs win on different days and Z-scores are non-zero.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _cross_z(arr: np.ndarray) -> np.ndarray:
    std = float(np.std(arr))
    if std < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - float(np.mean(arr))) / std


# ──────────────────────────────────────────────────────────────────────────────
# RegimeDetector
# ──────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """Detects market regimes using clustering on rolling return statistics."""

    def __init__(self, window: int = 20, k: Optional[int] = None):
        self.window = window
        self.k = k
        self.optimal_k_ = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = None
        self.regime_labels_ = None

    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        for col in df.columns:
            vol      = df[col].rolling(self.window).std()
            mean_ret = df[col].rolling(self.window).mean()
            skew     = df[col].rolling(self.window).skew()
            kurt     = df[col].rolling(self.window).kurt()
            roll_max = df[col].expanding().max()
            drawdown = (df[col] - roll_max) / (roll_max.abs() + 1e-9)
            features.extend([vol, mean_ret, skew, kurt, drawdown])
        feature_df = pd.concat(features, axis=1).dropna()
        return feature_df.values

    def _find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        from sklearn.metrics import silhouette_score
        scores = []
        for k in range(2, min(max_k + 1, len(features) - 1)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(features)
            if len(np.unique(km.labels_)) > 1:
                scores.append(silhouette_score(features, km.labels_))
            else:
                scores.append(-1)
        return int(np.argmax(scores)) + 2 if scores else 3

    def fit(self, df: pd.DataFrame, sweep_mode: bool = False,
            wf_mode: bool = False, fixed_k: Optional[int] = None):
        features        = self._create_features(df)
        features_scaled = self.scaler.fit_transform(features)
        features_pca    = self.pca.fit_transform(features_scaled)

        if fixed_k is not None:
            self.optimal_k_ = fixed_k
        elif sweep_mode or wf_mode:
            self.optimal_k_ = self.k if self.k else 3
        else:
            self.optimal_k_ = self._find_optimal_k(features_pca)

        self.kmeans = KMeans(
            n_clusters=self.optimal_k_, random_state=42, n_init=10
        )
        self.regime_labels_ = self.kmeans.fit_predict(features_pca)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features        = self._create_features(df)
        features_scaled = self.scaler.transform(features)
        features_pca    = self.pca.transform(features_scaled)
        return self.kmeans.predict(features_pca)

    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign regime labels to every row in df using the fitted KMeans model.
        Runs predict() fresh on each call so labels are always correct for the
        supplied df regardless of which window was used during fit().
        """
        regimes   = self.predict(df)
        result    = df.copy()
        # predict() works on rows that have enough lookback; align to tail
        n         = len(regimes)
        result["Regime"] = np.nan
        result.iloc[-n:, result.columns.get_loc("Regime")] = regimes

        regime_names = {
            0: "Low Volatility",
            1: "High Volatility",
            2: "Trending",
            3: "Mean Reverting",
            4: "Crisis",
        }
        result["Regime_Name"] = result["Regime"].map(
            lambda x: regime_names.get(x, f"Regime {int(x)}")
            if pd.notna(x) else "Unknown"
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# MomentumRanker
# ──────────────────────────────────────────────────────────────────────────────

# Ordered feature lookup table.
# Each entry: (column_suffix, weight)
# The ranker tries EVERY entry and uses whatever columns are present,
# renormalising weights to sum to 1.0 after filtering to found columns.
_FEATURE_CANDIDATES = [
    # ── New schema (built by data_manager_hf._build_dataset_inline) ─────────
    ("RoC_5d",       0.40),
    ("RoC_10d",      0.30),
    ("RoC_21d",      0.20),
    ("RoC_63d",      0.10),
    ("OBV_21d",      0.15),
    ("Breakout_20d", 0.15),
    # ── Legacy HF schema (built by original data_manager.compute_etf_features) ─
    ("Mom5d",        0.40),
    ("Mom21d",       0.30),
    ("Mom63d",       0.20),
    ("RelSPY21d",    0.15),
    ("RVol21d",      -0.10),
]


class MomentumRanker:
    """
    Ranks ETFs by composite momentum.

    Output column schema (used by strategy.execute_strategy()):
      {ETF}_P        — softmax probability (sums to 1 across ETFs each row)
      {ETF}_RS       — composite cross-sectional Z-score
      {ETF}_PA       — regime-adjusted probability (= _P for now)
      {ETF}_Disagree — False placeholder
      Top_Pick, Regime
    """

    def __init__(self, lookback: int = 63,
                 target_etfs: Optional[List[str]] = None):
        self.lookback         = lookback
        self.target_etfs      = target_etfs or []
        self.regime_rankings_ : Dict = {}
        self.regime_weights_  : Dict = {}
        # _active_features is set during fit() and refreshed per-call in
        # predict_all_history() to avoid stale column assumptions across windows.
        self._active_features : Optional[List[Tuple[str, float]]] = None

    # ── feature discovery ──────────────────────────────────────────────────
    def _discover_features(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Find which feature columns actually exist in df for the first ETF.
        Returns list of (suffix, weight) pairs with weights renormalised to
        sum to 1.  Falls back to raw daily return if nothing else is found.
        """
        probe = self.target_etfs[0] if self.target_etfs else ""
        found = []
        for suffix, w in _FEATURE_CANDIDATES:
            col = f"{probe}_{suffix}"
            if col in df.columns:
                found.append((suffix, w))

        if not found:
            log.warning(
                "MomentumRanker: no known feature columns found — "
                "falling back to raw daily return (_Ret)"
            )
            return [("Ret", 1.0)]

        pos_sum = sum(abs(w) for _, w in found)
        if pos_sum > 0:
            found = [(s, w / pos_sum) for s, w in found]

        log.info(f"MomentumRanker: using features {[s for s, _ in found]}")
        return found

    # ── composite score for one row ────────────────────────────────────────
    def _composite_score_row(self, row: pd.Series,
                              features: List[Tuple[str, float]]) -> np.ndarray:
        n      = len(self.target_etfs)
        scores = np.zeros(n)

        for suffix, weight in features:
            vals = np.array([
                float(row.get(f"{e}_{suffix}", np.nan))
                for e in self.target_etfs
            ], dtype=float)
            valid = np.isfinite(vals)
            if valid.sum() < 2:
                continue
            vals[~valid] = float(np.nanmean(vals[valid]))
            scores += weight * _cross_z(vals)

        return scores

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        if "Regime" not in df.columns:
            raise ValueError(
                "DataFrame must have 'Regime' column. Run RegimeDetector first."
            )
        # Discover features from the training data and cache for this ranker
        self._active_features = self._discover_features(df)

        for regime in df["Regime"].dropna().unique():
            sub      = df[df["Regime"] == regime]
            ret_cols = [f"{e}_Ret" for e in self.target_etfs
                        if f"{e}_Ret" in df.columns]
            n        = len(self.target_etfs)
            if len(sub) < self.lookback or not ret_cols:
                self.regime_rankings_[regime] = pd.Series(0.5, index=self.target_etfs)
                self.regime_weights_[regime]  = pd.Series(1.0 / n, index=self.target_etfs)
                continue
            scores = sub[ret_cols].rolling(self.lookback).mean().iloc[-1]
            scores.index = [c.replace("_Ret", "") for c in scores.index]
            scores_pos = scores - scores.min() + 0.01
            self.regime_rankings_[regime] = scores.sort_values(ascending=False)
            self.regime_weights_[regime]  = scores_pos / scores_pos.sum()
        return self

    # ── predict_all_history ────────────────────────────────────────────────
    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate per-row predictions for the full DataFrame.

        KEY FIX: always calls _discover_features(df) from the supplied df
        rather than relying solely on the cached _active_features from
        fit().  This ensures walk-forward windows that have slightly
        different column availability don't silently fall back to equal
        probabilities.

        Output columns:
          {ETF}_P        — softmax probability (sums to 1 across ETFs each row)
          {ETF}_RS       — composite cross-sectional Z-score
          {ETF}_PA       — regime-adjusted probability (= _P)
          {ETF}_Disagree — False (placeholder)
          Top_Pick, Regime
        """
        # Always re-discover features from this specific df to avoid stale cache
        features = self._discover_features(df)
        # Update cached value so single-row predict() stays in sync
        self._active_features = features

        records = []
        idx_out = []

        for i in range(self.lookback, len(df)):
            row    = df.iloc[i]
            scores = self._composite_score_row(row, features)
            probs  = _softmax(scores)

            record = {}
            for j, etf in enumerate(self.target_etfs):
                record[f"{etf}_P"]        = float(probs[j])
                record[f"{etf}_RS"]       = float(scores[j])
                record[f"{etf}_PA"]       = float(probs[j])
                record[f"{etf}_Disagree"] = False

            best_idx           = int(np.argmax(probs))
            record["Top_Pick"] = self.target_etfs[best_idx]
            record["Regime"]   = row.get("Regime", 0)

            records.append(record)
            idx_out.append(df.index[i])

        if records:
            out = pd.DataFrame(records, index=pd.DatetimeIndex(idx_out))
            out.index.name = "Date"
            return out
        return pd.DataFrame()

    # ── single-row predict (used by sweep + legacy code) ──────────────────
    def predict(self, row: pd.Series) -> Dict:
        features = self._active_features or [("Ret", 1.0)]
        scores   = self._composite_score_row(row, features)
        probs    = _softmax(scores)
        rankings = pd.Series(scores, index=self.target_etfs).sort_values(ascending=False)
        weights  = pd.Series({e: float(probs[i]) for i, e in enumerate(self.target_etfs)})
        return {
            "Rank_Score": rankings,
            "Weights":    weights,
            "Top_Pick":   self.target_etfs[int(np.argmax(probs))],
            "Regime":     int(row.get("Regime", 0)),
        }


# ──────────────────────────────────────────────────────────────────────────────
def calculate_conviction_z(probabilities: np.ndarray) -> Tuple[float, str]:
    if len(probabilities) == 0:
        return 0.0, "Low"
    probs   = np.clip(probabilities / (np.sum(probabilities) + 1e-10), 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_ent = np.log(len(probs))
    conv    = 1 - (entropy / max_ent) if max_ent > 0 else 0
    z       = conv * 3
    label   = "High" if z > 2.0 else "Medium" if z > 1.0 else "Low"
    return z, label
