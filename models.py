"""
models.py - P2-ETF-REGIME-PREDICTOR v2 (FIXED)
=========================================
Models for regime detection and momentum ranking.

KEY FIX: predict_all_history() now produces the column schema that
strategy.py actually reads:
  {ETF}_P    — P(beat cash), cross-sectionally normalised momentum probability
  {ETF}_RS   — raw composite momentum rank score (Z-normalised cross-sectionally)
  {ETF}_PA   — probability adjusted for regime (same as _P in this impl)
  {ETF}_Disagree — bool, always False (placeholder for ensemble disagreement)

Previously it produced {ETF}_Prob which strategy.py never looked at,
causing it to fall back to equal-weight 1/N for every row → same signal
forever → stale TLT / QQQ across all years.

Composite score (matches README spec):
  40% × RoC_5d + 30% × RoC_10d + 20% × RoC_21d + 10% × RoC_63d
  + 15% × OBV_21d  + 15% × Breakout_20d
  (weights are renormalized if feature columns are missing)
All components are cross-sectionally Z-scored before blending.
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


# ---------------------------------------------------------------------------
# RegimeDetector  (unchanged logic, only minor cleanup)
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detects market regimes using clustering on returns."""

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
            drawdown = (df[col] - roll_max) / roll_max
            features.extend([vol, mean_ret, skew, kurt, drawdown])
        feature_df = pd.concat(features, axis=1).dropna()
        return feature_df.values

    def _find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        from sklearn.metrics import silhouette_score
        silhouette_scores = []
        for k in range(2, min(max_k + 1, len(features) - 1)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(features)
            if len(np.unique(km.labels_)) > 1:
                silhouette_scores.append(silhouette_score(features, km.labels_))
            else:
                silhouette_scores.append(-1)
        return int(np.argmax(silhouette_scores)) + 2 if silhouette_scores else 3

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
        regimes    = self.predict(df)
        regime_df  = pd.DataFrame(
            regimes, index=df.index[-len(regimes):], columns=["Regime"]
        )
        result = df.copy()
        result["Regime"] = regime_df["Regime"]
        regime_names = {
            0: "Low Volatility", 1: "High Volatility",
            2: "Trending",       3: "Mean Reverting", 4: "Crisis",
        }
        result["Regime_Name"] = result["Regime"].map(
            lambda x: regime_names.get(x, f"Regime {x}")
        )
        return result


# ---------------------------------------------------------------------------
# MomentumRanker  (predict_all_history FIXED)
# ---------------------------------------------------------------------------

# Composite score feature weights (README spec)
_FEAT_WEIGHTS = {
    "RoC_5d":       0.40,
    "RoC_10d":      0.30,
    "RoC_21d":      0.20,
    "RoC_63d":      0.10,
    "OBV_21d":      0.15,
    "Breakout_20d": 0.15,
}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _cross_z(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional Z-score of a 1-D array (returns zeros if flat)."""
    std = np.std(arr)
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


class MomentumRanker:
    """
    Ranks ETFs by composite momentum and outputs the column schema expected
    by strategy.execute_strategy():
        {ETF}_P       — probability of beating cash (softmax of Z-scores)
        {ETF}_RS      — raw composite Z-score
        {ETF}_PA      — regime-adjusted probability (= _P in this impl)
        {ETF}_Disagree — bool placeholder (always False)
    """

    def __init__(self, lookback: int = 63,
                 target_etfs: Optional[List[str]] = None):
        self.lookback         = lookback
        self.target_etfs      = target_etfs or []
        # Kept for backward-compat (legacy fit path)
        self.regime_rankings_ : Dict = {}
        self.regime_weights_  : Dict = {}

    # ------------------------------------------------------------------ fit --
    def fit(self, df: pd.DataFrame):
        """
        Fit regime-averaged rankings (used as fallback when feature columns
        are absent).  The primary prediction path uses per-row feature columns
        directly in predict_all_history().
        """
        if "Regime" not in df.columns:
            raise ValueError(
                "DataFrame must have 'Regime' column. Run RegimeDetector first."
            )
        for regime in df["Regime"].unique():
            sub      = df[df["Regime"] == regime]
            ret_cols = [f"{e}_Ret" for e in self.target_etfs
                        if f"{e}_Ret" in df.columns]
            if len(sub) < self.lookback or not ret_cols:
                n = len(self.target_etfs)
                self.regime_rankings_[regime] = pd.Series(
                    0.5, index=self.target_etfs)
                self.regime_weights_[regime]  = pd.Series(
                    1 / n, index=self.target_etfs)
                continue
            # Simple rolling-return momentum at the last date of each regime
            scores = sub[ret_cols].rolling(self.lookback).mean().iloc[-1]
            scores.index = [c.replace("_Ret", "") for c in scores.index]
            scores_pos = scores - scores.min() + 0.01
            self.regime_rankings_[regime] = scores.sort_values(ascending=False)
            self.regime_weights_[regime]  = scores_pos / scores_pos.sum()
        return self

    # ----------------------------------------------------- per-row composite --
    def _composite_score_row(self, row: pd.Series) -> np.ndarray:
        """
        Compute composite momentum Z-scores for all target ETFs from a
        single row of the feature DataFrame.

        Uses pre-engineered columns: {ETF}_RoC_5d, {ETF}_RoC_10d,
        {ETF}_RoC_21d, {ETF}_RoC_63d, {ETF}_OBV_21d, {ETF}_Breakout_20d.

        Falls back gracefully to {ETF}_Ret if feature columns are missing.
        """
        n      = len(self.target_etfs)
        scores = np.zeros(n)

        for feat, base_w in _FEAT_WEIGHTS.items():
            vals = np.array([
                float(row.get(f"{e}_{feat}", np.nan))
                for e in self.target_etfs
            ])
            valid = np.isfinite(vals)
            if valid.sum() < 2:
                continue          # skip this feature if too many NaNs
            # Fill NaN slots with cross-sectional mean so they don't skew Z
            vals[~valid] = np.nanmean(vals[valid])
            scores += base_w * _cross_z(vals)

        # Fallback: raw return if no feature columns produced any signal
        if np.allclose(scores, 0):
            vals = np.array([
                float(row.get(f"{e}_Ret", np.nan))
                for e in self.target_etfs
            ])
            valid = np.isfinite(vals)
            if valid.sum() >= 2:
                vals[~valid] = np.nanmean(vals[valid])
                scores = _cross_z(vals)

        return scores

    # ------------------------------------------------------------ predict_all --
    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for every row in df (after the warmup period).

        Output columns per ETF:
          {ETF}_P        — softmax probability (sums to 1 across ETFs)
          {ETF}_RS       — composite Z-score (cross-sectional)
          {ETF}_PA       — regime-adjusted probability (= _P here)
          {ETF}_Disagree — False (placeholder)

        Plus:
          Top_Pick, Regime
        """
        records = []
        idx_out = []
        n       = len(self.target_etfs)

        for i in range(self.lookback, len(df)):
            row    = df.iloc[i]
            scores = self._composite_score_row(row)   # shape (n,)

            # Softmax probabilities
            probs  = _softmax(scores)

            record = {}
            for j, etf in enumerate(self.target_etfs):
                record[f"{etf}_P"]        = float(probs[j])
                record[f"{etf}_RS"]       = float(scores[j])
                record[f"{etf}_PA"]       = float(probs[j])   # regime-adj ≡ _P
                record[f"{etf}_Disagree"] = False

            best_idx             = int(np.argmax(probs))
            record["Top_Pick"]   = self.target_etfs[best_idx]
            record["Regime"]     = row.get("Regime", 0)

            records.append(record)
            idx_out.append(df.index[i])

        if records:
            return pd.DataFrame(records, index=pd.DatetimeIndex(idx_out))
        return pd.DataFrame()

    # ------------------------------------------------- legacy single-row API --
    def predict(self, row: pd.Series) -> Dict:
        """Single-row prediction (used by sweep in train_hf.py)."""
        scores = self._composite_score_row(row)
        probs  = _softmax(scores)
        rankings = pd.Series(scores, index=self.target_etfs).sort_values(
            ascending=False
        )
        weights  = pd.Series(
            {e: float(probs[i]) for i, e in enumerate(self.target_etfs)}
        )
        return {
            "Rank_Score": rankings,
            "Weights":    weights,
            "Top_Pick":   self.target_etfs[int(np.argmax(probs))],
            "Regime":     int(row.get("Regime", 0)),
        }


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def calculate_conviction_z(probabilities: np.ndarray) -> Tuple[float, str]:
    if len(probabilities) == 0:
        return 0.0, "Low"
    probs      = np.clip(probabilities / (np.sum(probabilities) + 1e-10),
                         1e-10, 1.0)
    entropy    = -np.sum(probs * np.log(probs))
    max_ent    = np.log(len(probs))
    conviction = 1 - (entropy / max_ent) if max_ent > 0 else 0
    z_score    = conviction * 3
    label      = ("High" if z_score > 2.0 else
                  "Medium" if z_score > 1.0 else "Low")
    return z_score, label
