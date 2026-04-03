models.py — P2-ETF-REGIME-PREDICTOR v2 (CORRECTED)
=======================================
Layer 2 Momentum Ranker — rules-based ETF ranking.

MomentumRanker accepts target_etfs at construction time so it works
correctly for both Option A (FI/Commodities) and Option B (Equity ETFs).
No hardcoded ETF list anywhere in this file.

Composite momentum score per ETF:
  40% × RoC 5d
  30% × RoC 10d
  20% × RoC 21d
  10% × RoC 63d
  15% × OBV accumulation (21d)
  15% × Breakout score (20d range position)

Scores are Z-normalised cross-sectionally. Top ETF with Z ≥ threshold enters.
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# REMOVED: No default ETF list — must be provided explicitly
# This prevents accidental use of wrong ETF universe

RANDOM_SEED = 42


class MomentumRanker:
    """
    Rules-based ETF ranking using Rate-of-Change momentum.

    Parameters
    ----------
    target_etfs : list of ETF tickers to rank. Must match the ETFs
        present in the dataset (i.e. Option A or Option B list).
        Stored at fit time and used in all predict calls.
    """

    ROC_WEIGHTS = {5: 0.40, 10: 0.30, 21: 0.20, 63: 0.10}
    OBV_WEIGHT = 0.15
    BREAKOUT_WEIGHT = 0.15
    MOMENTUM_WEIGHT = 0.70

    def __init__(self, target_etfs: Optional[List[str]] = None):
        # CORRECTED: No default fallback — must provide ETFs explicitly
        if target_etfs is None:
            raise ValueError(
                "target_etfs must be provided explicitly. "
                "Use cfg.OPTION_A_ETFS or cfg.OPTION_B_ETFS"
            )
        self.target_etfs_: List[str] = list(target_etfs)
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, **kwargs) -> "MomentumRanker":
        """
        Fit the ranker. Validates that expected feature columns exist
        for each ETF in target_etfs. No ML — pure rules-based.
        """
        missing = []
        for etf in self.target_etfs_:
            if f"{etf}_RoC5d" not in df.columns:
                missing.append(etf)
        if missing:
            log.warning(f"MomentumRanker: missing RoC5d columns for: {missing}. "
                        f"These ETFs will score 0.")
        self.fitted_ = True
        log.info(f"MomentumRanker fitted — universe: {self.target_etfs_}")
        return self

    def score_row(self, row: pd.Series) -> Dict[str, float]:
        """
        Compute momentum score per ETF for a single row.
        Returns dict {etf: score} — higher = stronger momentum.
        """
        scores = {}
        for etf in self.target_etfs_:
            # 1. RoC composite
            roc_score = 0.0
            for n, w in self.ROC_WEIGHTS.items():
                val = float(row.get(f"{etf}_RoC{n}d", 0.0) or 0.0)
                roc_score += w * val

            # 2. OBV normalised
            obv_raw = float(row.get(f"{etf}_OBV21d", 0.0) or 0.0)
            obv_norm = np.tanh(obv_raw / (abs(obv_raw) + 1e-6)) if obv_raw != 0 else 0.0

            # 3. Breakout score (0–1, 1 = at 20d high)
            breakout = float(row.get(f"{etf}_Breakout20d", 0.5) or 0.5)

            scores[etf] = (
                self.MOMENTUM_WEIGHT * roc_score +
                self.OBV_WEIGHT * obv_norm +
                self.BREAKOUT_WEIGHT * (breakout - 0.5)
            )
        return scores

    def predict(self, row: pd.Series) -> pd.DataFrame:
        """
        Returns DataFrame indexed by ETF with columns:
        Rank_Score, Conviction_P, P_Adjusted, Disagree, Source

        CORRECTED: Conviction_P now properly represents relative
        probability using softmax transformation instead of clamped linear.
        """
        scores = self.score_row(row)
        vals = np.array(list(scores.values()))

        # CORRECTED: Use softmax for proper probability distribution
        # This ensures probabilities sum to 1 and reflect relative strength
        if len(vals) > 0:
            # Subtract max for numerical stability
            exp_vals = np.exp(vals - np.max(vals))
            probs = exp_vals / (np.sum(exp_vals) + 1e-9)
        else:
            probs = np.ones(len(self.target_etfs_)) / len(self.target_etfs_)

        mean, std = vals.mean(), vals.std()

        rows = []
        for i, etf in enumerate(self.target_etfs_):
            s = scores[etf]
            z_score = (s - mean) / (std + 1e-9) if std > 1e-9 else 0.0
            rows.append({
                "ETF": etf,
                "Rank_Score": float(s),
                # CORRECTED: Proper probability using softmax
                "Conviction_P": float(np.clip(probs[i], 0.001, 0.999)),
                "P_Adjusted": float(z_score),  # Z-score for threshold checks
                "Disagree": False,
                "Source": "momentum",
            })
        return pd.DataFrame(rows).set_index("ETF")

    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions for every row in df — used for backtesting and WF.
        Returns DataFrame indexed by Date with columns:
        {ETF}_P, {ETF}_PA, {ETF}_RS, {ETF}_Disagree for each ETF.
        """
        rows = []
        for idx, row in df.iterrows():
            regime = int(row.get("Regime", 0)) if pd.notna(row.get("Regime")) else 0
            preds = self.predict(row)
            entry = {"Date": idx, "Regime": regime}
            for etf in self.target_etfs_:
                if etf in preds.index:
                    entry[f"{etf}_P"] = preds.loc[etf, "Conviction_P"]
                    entry[f"{etf}_PA"] = preds.loc[etf, "P_Adjusted"]
                    entry[f"{etf}_RS"] = preds.loc[etf, "Rank_Score"]
                    entry[f"{etf}_Disagree"] = False
            rows.append(entry)
        return pd.DataFrame(rows).set_index("Date")

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "MomentumRanker":
        return pickle.loads(data)
