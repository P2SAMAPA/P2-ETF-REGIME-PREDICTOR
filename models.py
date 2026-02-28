"""
models.py — P2-ETF-REGIME-PREDICTOR
=====================================
LightGBM LambdaRank + Logistic Regression ensemble for ETF rotation.

Architecture:
  - LambdaRank: directly optimises ranking of ETFs by forward return
    (which ETF will rank #1 over the next N days?)
  - Per-regime models trained on optimal horizon per (ETF, regime)
  - Optimal horizon (5/10/15/20 days) selected by Spearman rank
    correlation during training — model finds its own best horizon
  - Logistic Regression (L1) binary baseline per ETF for disagreement check
  - Ensemble: LambdaRank score for ranking + LogReg for conviction

Author: P2SAMAPA
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from scipy.stats import spearmanr

import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)

RANDOM_SEED            = 42
MIN_REGIME_ROWS        = 100
FORWARD_HORIZONS       = [5, 10, 15, 20]
DISAGREEMENT_THRESHOLD = 0.15
TARGET_ETFS            = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]

# XGBoost ranking params
XGB_RANK_PARAMS = {
    "objective":        "rank:pairwise",
    "eval_metric":      "ndcg",
    "learning_rate":    0.05,
    "max_depth":        4,
    "n_estimators":     300,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     RANDOM_SEED,
    "verbosity":        0,
}

# LambdaRank params — optimises ranking directly
LGBM_RANK_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     [1, 3, 5],
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples":20,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "random_state":     RANDOM_SEED,
}

LOGREG_PARAMS = {
    "penalty":      "l1",
    "solver":       "liblinear",
    "C":            0.1,
    "max_iter":     1000,
    "random_state": RANDOM_SEED,
}


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame,
                         exclude_cols: Optional[list] = None) -> list:
    if exclude_cols is None:
        exclude_cols = []
    always_exclude = {"Regime", "Regime_Name"}
    for t in TARGET_ETFS:
        for h in [5, 10, 15, 20]:
            always_exclude.add(f"{t}_FwdRet{h}d")
            always_exclude.add(f"{t}_BeatCash{h}d")
        always_exclude.add(f"{t}_FwdRet")
        always_exclude.add(f"{t}_BeatCash")
    raw_suffixes = ("_Open", "_High", "_Low", "_Close", "_Volume", "_Adj Close")
    feature_cols = [
        c for c in df.columns
        if c not in always_exclude
        and c not in exclude_cols
        and not any(c.endswith(s) for s in raw_suffixes)
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and df[c].nunique() > 5
    ]
    log.info(f"Selected {len(feature_cols)} feature columns")
    return feature_cols


# ── Optimal horizon selection ─────────────────────────────────────────────────

def select_optimal_horizon(X: np.ndarray,
                            fwd_df: pd.DataFrame,
                            valid_mask: np.ndarray,
                            regime: int) -> int:
    """
    For this regime's training data, find the forward horizon (5/10/15/20d)
    that maximises average Spearman rank correlation between a simple
    linear score and actual ETF forward returns.
    Returns the best horizon in days.
    """
    best_h     = 5
    best_score = -np.inf

    for h in FORWARD_HORIZONS:
        ret_cols = [f"{t}_FwdRet{h}d" for t in TARGET_ETFS]
        avail    = [c for c in ret_cols if c in fwd_df.columns]
        if len(avail) < len(TARGET_ETFS):
            continue

        # Use PCA-like: first principal component of returns at this horizon
        # as a simple proxy for "which horizon has most cross-ETF variation"
        rets = fwd_df.loc[valid_mask, avail].values
        if np.isnan(rets).mean() > 0.3:
            continue
        rets_clean = np.nan_to_num(rets, nan=0.0)

        # Cross-sectional std per day — higher = more differentiation
        cross_std = np.nanstd(rets_clean, axis=1).mean()
        if cross_std > best_score:
            best_score = cross_std
            best_h     = h

    log.info(f"  Regime {regime}: optimal horizon = {best_h}d "
             f"(cross-std={best_score:.4f})")
    return best_h


# ── Regime ranking model ──────────────────────────────────────────────────────

class RegimeRankingModel:
    """
    LambdaRank model for one regime.
    Ranks all 5 ETFs by predicted forward return for each day.
    Also trains per-ETF LogReg for conviction/disagreement signal.
    """

    def __init__(self, regime: int, regime_name: str = ""):
        self.regime       = regime
        self.regime_name  = regime_name
        self.ranker_      = None      # LightGBM LambdaRank
        self.xgb_ranker_  = None      # XGBoost pairwise ranking
        self.ridge_ranker_: Dict[str, Any] = {}  # Ridge per ETF
        self.logregs_     : Dict[str, LogisticRegression] = {}
        self.scaler_      = RobustScaler()
        self.feature_cols_= []
        self.horizon_     = 5         # optimal horizon selected during fit
        self.is_fitted_   = False
        self.val_ndcg_    = None
        self.n_train_     = 0

    def fit(self, X_train: np.ndarray, X_val: np.ndarray,
            fwd_train: pd.DataFrame, fwd_val: pd.DataFrame,
            valid_train: pd.Index, valid_val: pd.Index,
            feature_cols: list) -> "RegimeRankingModel":

        self.feature_cols_ = feature_cols
        self.n_train_      = len(X_train)

        # Select optimal horizon
        self.horizon_ = select_optimal_horizon(
            X_train, fwd_train, valid_train, self.regime
        )
        h = self.horizon_

        # Build ranking labels: rank ETFs by actual forward return each day
        # LightGBM lambdarank needs: relevance labels (higher = better)
        ret_cols = [f"{t}_FwdRet{h}d" for t in TARGET_ETFS]
        avail    = [c for c in ret_cols if c in fwd_train.columns]
        if len(avail) < 2:
            log.warning(f"Regime {self.regime}: insufficient forward return cols")
            return self

        # Scale features
        X_train_s = self.scaler_.fit_transform(X_train)
        X_val_s   = self.scaler_.transform(X_val)

        # Build LambdaRank dataset
        # Each "query" is one trading day, items are the 5 ETFs
        # We repeat X_train for each ETF and add ETF identity features
        n_etfs   = len(TARGET_ETFS)
        X_rank_tr, y_rank_tr, q_tr = [], [], []
        X_rank_va, y_rank_va, q_va = [], [], []

        etf_onehot = np.eye(n_etfs)

        for day_i in range(len(X_train_s)):
            day_rets = fwd_train.loc[valid_train[day_i],
                                     avail].values if valid_train[day_i] in fwd_train.index else None
            if day_rets is None or np.isnan(day_rets).all():
                continue
            # Rank ETFs by return (0=worst, n_etfs-1=best)
            ranks     = np.argsort(np.argsort(
                np.nan_to_num(day_rets, nan=-999)
            ))
            base_feat = X_train_s[day_i]
            for etf_j in range(n_etfs):
                feat = np.concatenate([base_feat, etf_onehot[etf_j]])
                X_rank_tr.append(feat)
                y_rank_tr.append(int(ranks[etf_j]))
                q_tr.append(day_i)

        for day_i in range(len(X_val_s)):
            day_rets = fwd_val.loc[valid_val[day_i],
                                   avail].values if valid_val[day_i] in fwd_val.index else None
            if day_rets is None or np.isnan(day_rets).all():
                continue
            ranks     = np.argsort(np.argsort(
                np.nan_to_num(day_rets, nan=-999)
            ))
            base_feat = X_val_s[day_i]
            for etf_j in range(n_etfs):
                feat = np.concatenate([base_feat, etf_onehot[etf_j]])
                X_rank_va.append(feat)
                y_rank_va.append(int(ranks[etf_j]))
                q_va.append(day_i)

        if len(X_rank_tr) < 50:
            log.warning(f"Regime {self.regime}: too few ranking samples")
            return self

        X_rank_tr = np.array(X_rank_tr)
        y_rank_tr = np.array(y_rank_tr)
        X_rank_va = np.array(X_rank_va)
        y_rank_va = np.array(y_rank_va)
        q_tr_arr  = np.array([n_etfs] * (len(X_rank_tr) // n_etfs))
        q_va_arr  = np.array([n_etfs] * (len(X_rank_va) // n_etfs))

        dtrain = lgb.Dataset(X_rank_tr, label=y_rank_tr, group=q_tr_arr)
        dval   = lgb.Dataset(X_rank_va, label=y_rank_va, group=q_va_arr,
                              reference=dtrain)

        try:
            self.ranker_ = lgb.train(
                LGBM_RANK_PARAMS, dtrain,
                num_boost_round=300,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            log.info(f"  Regime {self.regime} [{self.regime_name}]: "
                     f"n={self.n_train_} horizon={h}d "
                     f"rank_samples={len(X_rank_tr)}")
        except Exception as e:
            log.error(f"  LambdaRank fit failed regime {self.regime}: {e}")
            return self

        # ── XGBoost pairwise ranking ──────────────────────────────────────
        if HAS_XGB and len(X_rank_tr) >= 50:
            try:
                self.xgb_ranker_ = xgb.XGBRanker(**XGB_RANK_PARAMS)
                self.xgb_ranker_.fit(
                    X_rank_tr, y_rank_tr,
                    group=q_tr_arr,
                    eval_set=[(X_rank_va, y_rank_va)],
                    eval_group=[q_va_arr],
                    verbose=False,
                )
                log.info(f"  XGBoost ranker fitted: regime {self.regime}")
            except Exception as e:
                log.warning(f"  XGBoost fit failed: {e}")

        # ── Ridge regression ranker (one per ETF, predicts fwd return) ────
        for j, etf in enumerate(TARGET_ETFS):
            fwd_col = f"{etf}_FwdRet{h}d"
            if fwd_col not in fwd_train.columns:
                continue
            valid_mask = fwd_train[fwd_col].notna()
            if valid_mask.sum() < 50:
                continue
            y_ridge = fwd_train.loc[valid_mask, fwd_col].values
            n_ridge = min(len(X_train_s), valid_mask.sum())
            try:
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_s[:n_ridge], y_ridge[:n_ridge])
                self.ridge_ranker_[etf] = ridge
            except Exception as e:
                log.warning(f"  Ridge fit failed {etf}: {e}")

        # Per-ETF LogReg for conviction signal
        beat_cols = [f"{t}_BeatCash{h}d" for t in TARGET_ETFS]
        for j, etf in enumerate(TARGET_ETFS):
            bcol = f"{etf}_BeatCash{h}d"
            if bcol not in fwd_train.columns:
                continue
            valid_mask = fwd_train[bcol].notna()
            if valid_mask.sum() < 50:
                continue
            y_lr = fwd_train.loc[valid_mask, bcol].values.astype(int)
            # Get X rows matching valid_mask within training set
            tr_idx = [k for k, idx in enumerate(valid_train)
                      if idx in fwd_train.index and
                      fwd_train.loc[idx, bcol] == fwd_train.loc[idx, bcol]]
            if len(tr_idx) < 50 or len(np.unique(y_lr[:len(tr_idx)])) < 2:
                continue
            try:
                lr = LogisticRegression(**LOGREG_PARAMS)
                lr.fit(X_train_s[:len(tr_idx)], y_lr[:len(tr_idx)])
                self.logregs_[etf] = lr
            except Exception:
                pass

        self.is_fitted_ = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Majority voting across LambdaRank + XGBoost + Ridge.
        # Returns combined ranking scores. Shape: (n_etfs,)
        n_etfs  = len(TARGET_ETFS)
        X_s     = self.scaler_.transform(X)
        onehot  = np.eye(n_etfs)
        X_rank  = np.array([
            np.concatenate([X_s[0], onehot[j]])
            for j in range(n_etfs)
        ])

        model_scores = []
        model_tops   = []

        # LambdaRank
        if self.ranker_ is not None:
            s = self.ranker_.predict(X_rank)
            model_scores.append(s / (np.std(s) + 1e-9))
            model_tops.append(int(np.argmax(s)))

        # XGBoost
        if self.xgb_ranker_ is not None:
            try:
                s = self.xgb_ranker_.predict(X_rank)
                model_scores.append(s / (np.std(s) + 1e-9))
                model_tops.append(int(np.argmax(s)))
            except Exception:
                pass

        # Ridge (predict fwd return per ETF, normalised)
        if self.ridge_ranker_:
            ridge_scores = np.zeros(n_etfs)
            for j, etf in enumerate(TARGET_ETFS):
                if etf in self.ridge_ranker_:
                    ridge_scores[j] = float(
                        self.ridge_ranker_[etf].predict(X_s)[0]
                    )
            if ridge_scores.std() > 1e-9:
                model_scores.append(ridge_scores / (ridge_scores.std() + 1e-9))
                model_tops.append(int(np.argmax(ridge_scores)))

        if not model_scores:
            self.last_votes_    = []
            self.last_majority_ = False
            return np.zeros(n_etfs)

        combined            = np.mean(model_scores, axis=0)
        self.last_votes_    = model_tops
        top                 = int(np.argmax(combined))
        vote_counts         = np.bincount(model_tops, minlength=n_etfs)
        # Majority = top ETF agreed by >= half of models
        self.last_majority_ = bool(vote_counts[top] >= max(1, len(model_tops) / 2))

        return combined

    def predict_conviction(self, X: np.ndarray) -> Dict[str, float]:
        """Returns LogReg P(beat cash) per ETF for conviction overlay."""
        if not self.logregs_:
            return {etf: 0.5 for etf in TARGET_ETFS}
        X_s  = self.scaler_.transform(X)
        out  = {}
        for etf in TARGET_ETFS:
            if etf in self.logregs_:
                out[etf] = float(self.logregs_[etf].predict_proba(X_s)[0, 1])
            else:
                out[etf] = 0.5
        return out

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "RegimeRankingModel":
        return pickle.loads(data)


# ── Regime model bank ─────────────────────────────────────────────────────────

class RegimeModelBank:
    """
    One RegimeRankingModel per regime + one global fallback.
    """

    def __init__(self):
        self.models_:         Dict[int, RegimeRankingModel] = {}
        self.global_model_:   Optional[RegimeRankingModel]  = None
        self.feature_cols_:   List[str] = []
        self.regimes_:        List[int] = []
        self.regime_names_:   Dict[int, str] = {}
        self.base_rates_:     Dict[str, float] = {}
        self.fitted_:         bool = False

    def fit(self, df: pd.DataFrame, fwd_df: pd.DataFrame,
            feature_cols: Optional[list] = None,
            val_pct: float = 0.15) -> "RegimeModelBank":

        if "Regime" not in df.columns:
            raise ValueError("df must have Regime column")

        if feature_cols is None:
            feature_cols = get_feature_columns(df)
        self.feature_cols_ = feature_cols

        common_idx = df.index.intersection(fwd_df.index)
        df_a       = df.loc[common_idx]
        fwd_a      = fwd_df.loc[common_idx]

        self.regimes_     = sorted(df_a["Regime"].dropna().unique().astype(int).tolist())
        self.regime_names_= {r: str(r) for r in self.regimes_}

        # Base rates per ETF (5d default)
        for etf in TARGET_ETFS:
            col = f"{etf}_BeatCash5d"
            if col in fwd_a.columns:
                self.base_rates_[etf] = float(fwd_a[col].dropna().mean())
            else:
                self.base_rates_[etf] = 0.5
        log.info(f"Base rates: { {k: f'{v:.2%}' for k,v in self.base_rates_.items()} }")

        # Clean feature matrix
        df_a = df_a.copy()
        df_a[feature_cols] = (df_a[feature_cols]
                               .fillna(df_a[feature_cols].median())
                               .fillna(0.0))
        feat_matrix = df_a[feature_cols].values

        log.info(f"Training {len(self.regimes_)} regime ranking models...")

        # Global model
        log.info("Training global fallback ranking model...")
        self.global_model_ = self._train_one(
            regime=   -1,
            regime_name="Global",
            df=       df_a,
            fwd_df=   fwd_a,
            feat_mat= feat_matrix,
            feature_cols=feature_cols,
            val_pct=  val_pct,
        )

        # Per-regime models
        for regime in self.regimes_:
            mask   = df_a["Regime"] == regime
            r_df   = df_a[mask]
            r_fwd  = fwd_a[mask]
            r_feat = feat_matrix[mask.values]

            if len(r_df) < MIN_REGIME_ROWS:
                log.warning(f"Regime {regime}: {len(r_df)} rows — using global")
                continue

            log.info(f"Training regime {regime} [{self.regime_names_.get(regime)}]: "
                     f"{len(r_df)} rows")
            model = self._train_one(
                regime=      regime,
                regime_name= self.regime_names_.get(regime, str(regime)),
                df=          r_df,
                fwd_df=      r_fwd,
                feat_mat=    r_feat,
                feature_cols=feature_cols,
                val_pct=     val_pct,
            )
            if model.is_fitted_:
                self.models_[regime] = model

        self.fitted_ = True
        log.info(f"Trained {len(self.models_)} regime models + 1 global")
        return self

    def _train_one(self, regime, regime_name, df, fwd_df,
                   feat_mat, feature_cols, val_pct) -> RegimeRankingModel:
        model   = RegimeRankingModel(regime=regime, regime_name=regime_name)
        n       = len(df)
        vs      = max(1, int(n * val_pct))
        idx     = df.index

        X_tr    = feat_mat[:-vs]
        X_va    = feat_mat[-vs:]
        fwd_tr  = fwd_df.iloc[:-vs]
        fwd_va  = fwd_df.iloc[-vs:]
        idx_tr  = idx[:-vs]
        idx_va  = idx[-vs:]

        try:
            model.fit(X_tr, X_va, fwd_tr, fwd_va, idx_tr, idx_va, feature_cols)
        except Exception as e:
            log.error(f"Regime {regime} training failed: {e}")
        return model

    def predict(self, X: np.ndarray, regime: int) -> pd.DataFrame:
        """
        Predict ETF rankings and conviction scores for current regime.
        Returns DataFrame indexed by ETF with columns:
        Rank_Score, Conviction_P, P_Adjusted, Disagree
        """
        model = self.models_.get(regime, self.global_model_)
        if model is None or not model.is_fitted_:
            # Pure fallback
            rows = [{"ETF": etf, "Rank_Score": 0.0,
                     "Conviction_P": 0.5, "P_Adjusted": 0.0,
                     "Disagree": False, "Source": "none"}
                    for etf in TARGET_ETFS]
            return pd.DataFrame(rows).set_index("ETF")

        source     = f"regime_{regime}" if regime in self.models_ else "global"
        scores     = model.predict_scores(X)
        conv       = model.predict_conviction(X)

        # Majority voting disagreement
        majority   = getattr(model, "last_majority_", True)
        last_votes = getattr(model, "last_votes_", [])
        top_idx    = int(np.argmax(scores))

        rows = []
        for j, etf in enumerate(TARGET_ETFS):
            base     = self.base_rates_.get(etf, 0.5)
            cp       = conv.get(etf, 0.5)
            disagree = (j == top_idx) and not majority
            votes    = int(np.bincount(last_votes, minlength=len(TARGET_ETFS))[j]) if last_votes else 0
            rows.append({
                "ETF":          etf,
                "Rank_Score":   float(scores[j]),
                "Conviction_P": float(cp),
                "P_Adjusted":   float(cp - base),
                "Disagree":     bool(disagree),
                "Votes":        votes,
                "Source":       source,
            })

        return pd.DataFrame(rows).set_index("ETF")
        return pd.DataFrame(rows).set_index("ETF")

    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for every row — for backtesting."""
        if not self.fitted_:
            raise ValueError("Not fitted")
        rows        = []
        df_clean    = df.copy()
        df_clean[self.feature_cols_] = (
            df_clean[self.feature_cols_]
            .fillna(df_clean[self.feature_cols_].median())
            .fillna(0.0)
        )
        feat_matrix = df_clean[self.feature_cols_].values

        for i, (idx, row) in enumerate(df.iterrows()):
            regime     = int(row.get("Regime", 0)) if pd.notna(row.get("Regime")) else 0
            X          = feat_matrix[i:i+1]
            preds      = self.predict(X, regime)
            entry      = {"Date": idx, "Regime": regime}
            for etf in TARGET_ETFS:
                if etf in preds.index:
                    entry[f"{etf}_P"]        = preds.loc[etf, "Conviction_P"]
                    entry[f"{etf}_PA"]       = preds.loc[etf, "P_Adjusted"]
                    entry[f"{etf}_RS"]       = preds.loc[etf, "Rank_Score"]
                    entry[f"{etf}_Disagree"] = preds.loc[etf, "Disagree"]
            rows.append(entry)

        return pd.DataFrame(rows).set_index("Date")

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "RegimeModelBank":
        return pickle.loads(data)


# ── Feature importance ────────────────────────────────────────────────────────

def aggregate_feature_importance(bank: RegimeModelBank) -> pd.DataFrame:
    all_imp = {}
    for model in list(bank.models_.values()) + ([bank.global_model_]
                                                  if bank.global_model_ else []):
        if model.ranker_ is None:
            continue
        imp  = model.ranker_.feature_importance(importance_type="gain")
        # Feature names include ETF one-hot appended at end
        n_fc = len(model.feature_cols_)
        for k, val in enumerate(imp[:n_fc]):
            feat = model.feature_cols_[k]
            all_imp.setdefault(feat, []).append(val)
    rows = [{"Feature": f, "Mean_Gain": np.mean(v),
             "Std_Gain": np.std(v), "Count": len(v)}
            for f, v in all_imp.items()]
    return (pd.DataFrame(rows)
            .sort_values("Mean_Gain", ascending=False)
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2B — PURE MOMENTUM RANKER
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumRanker:
    """
    Layer 2B: Rules-based ETF ranking using Rate-of-Change momentum.

    Ranks ETFs by composite momentum score:
      - RoC 5d, 10d, 21d, 63d (weighted: recent > distant)
      - OBV accumulation (volume confirms price move)
      - Breakout score (price vs 20d range)

    No ML, no regime-specific models — fully transparent and rules-based.
    Z-score computed from cross-ETF momentum spread.
    """

    # RoC weights — heavier on recent
    ROC_WEIGHTS = {5: 0.40, 10: 0.30, 21: 0.20, 63: 0.10}
    OBV_WEIGHT       = 0.15
    BREAKOUT_WEIGHT  = 0.15
    MOMENTUM_WEIGHT  = 0.70   # ROC composite weight in final score

    def __init__(self):
        self.feature_cols_: List[str] = []
        self.fitted_       = False
        self.scaler_obv_   = RobustScaler()

    def fit(self, df: pd.DataFrame, **kwargs) -> "MomentumRanker":
        # Fit OBV scaler for normalisation
        obv_cols = [f"{t}_OBV21d" for t in TARGET_ETFS
                    if f"{t}_OBV21d" in df.columns]
        if obv_cols:
            self.scaler_obv_.fit(df[obv_cols].fillna(0).values)
        self.fitted_ = True
        log.info("MomentumRanker fitted")
        return self

    def score_row(self, row: pd.Series) -> Dict[str, float]:
        """
        Compute momentum score per ETF for a single row.
        Returns dict {etf: score} — higher = stronger momentum.
        """
        scores = {}
        for etf in TARGET_ETFS:
            # 1. RoC composite
            roc_score = 0.0
            for n, w in self.ROC_WEIGHTS.items():
                val = float(row.get(f"{etf}_RoC{n}d", 0.0) or 0.0)
                roc_score += w * val

            # 2. OBV normalised (positive = accumulation)
            obv_raw = float(row.get(f"{etf}_OBV21d", 0.0) or 0.0)
            obv_norm = np.tanh(obv_raw / (abs(obv_raw) + 1e-6)) if obv_raw != 0 else 0.0

            # 3. Breakout score (0-1, 1 = at 20d high)
            breakout = float(row.get(f"{etf}_Breakout20d", 0.5) or 0.5)

            scores[etf] = (
                self.MOMENTUM_WEIGHT  * roc_score +
                self.OBV_WEIGHT       * obv_norm  +
                self.BREAKOUT_WEIGHT  * (breakout - 0.5)
            )
        return scores

    def predict(self, row: pd.Series) -> pd.DataFrame:
        """
        Returns DataFrame indexed by ETF with columns:
        Rank_Score, P_Adjusted, Disagree, Source
        """
        scores    = self.score_row(row)
        vals      = np.array(list(scores.values()))
        mean, std = vals.mean(), vals.std()

        rows = []
        for etf in TARGET_ETFS:
            s = scores[etf]
            rows.append({
                "ETF":          etf,
                "Rank_Score":   float(s),
                "Conviction_P": float(np.clip(0.5 + (s - mean) / (std + 1e-9) * 0.1, 0, 1)),
                "P_Adjusted":   float(s - mean),
                "Disagree":     False,   # pure rules — no disagreement
                "Source":       "momentum",
            })
        return pd.DataFrame(rows).set_index("ETF")

    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in df.iterrows():
            regime = int(row.get("Regime", 0)) if pd.notna(row.get("Regime")) else 0
            preds  = self.predict(row)
            entry  = {"Date": idx, "Regime": regime}
            for etf in TARGET_ETFS:
                if etf in preds.index:
                    entry[f"{etf}_P"]        = preds.loc[etf, "Conviction_P"]
                    entry[f"{etf}_PA"]       = preds.loc[etf, "P_Adjusted"]
                    entry[f"{etf}_RS"]       = preds.loc[etf, "Rank_Score"]
                    entry[f"{etf}_Disagree"] = False
            rows.append(entry)
        return pd.DataFrame(rows).set_index("Date")

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "MomentumRanker":
        return pickle.loads(data)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(
    df:           pd.DataFrame,
    fwd_df:       pd.DataFrame,
    mode:         str   = "momentum",   # "momentum" or "ensemble"
    train_years:  int   = 3,
    test_years:   int   = 1,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Walk-forward out-of-sample validation for both Layer 2 modes.

    For each fold:
      - Fit fresh model on train window only
      - Predict on test window only (model never sees test data)
      - Roll forward by test_years

    Returns concatenated OOS predictions DataFrame (same schema as
    predict_all_history output) covering the full backtest period.

    Parameters
    ----------
    df           : full feature DataFrame with Regime column
    fwd_df       : forward return targets DataFrame
    mode         : "momentum" (Option B) or "ensemble" (Option A)
    train_years  : years of training data per fold
    test_years   : years of OOS test data per fold
    feature_cols : feature columns for ensemble mode
    """
    TRAIN_DAYS = int(train_years * 252)
    TEST_DAYS  = int(test_years  * 252)

    log.info(f"Walk-forward CV: mode={mode}, "
             f"train={train_years}y, test={test_years}y")

    if feature_cols is None and mode == "ensemble":
        feature_cols = get_feature_columns(df)

    all_preds = []
    n         = len(df)
    fold      = 0

    start = TRAIN_DAYS  # first test fold starts after first training window

    while start < n:
        end      = min(start + TEST_DAYS, n)
        tr_start = max(0, start - TRAIN_DAYS)

        df_train  = df.iloc[tr_start:start]
        df_test   = df.iloc[start:end]
        fwd_train = fwd_df.iloc[tr_start:start]

        if len(df_train) < 252 or len(df_test) < 5:
            start += TEST_DAYS
            continue

        fold += 1
        log.info(f"  Fold {fold}: train {df_train.index[0].date()} → "
                 f"{df_train.index[-1].date()} | "
                 f"test {df_test.index[0].date()} → "
                 f"{df_test.index[-1].date()} "
                 f"({len(df_train)} train, {len(df_test)} test days)")

        try:
            if mode == "momentum":
                model = MomentumRanker()
                model.fit(df_train)
                preds = model.predict_all_history(df_test)

            else:  # ensemble
                # Clean features
                df_tr_c = df_train.copy()
                df_tr_c[feature_cols] = (
                    df_tr_c[feature_cols]
                    .fillna(df_tr_c[feature_cols].median())
                    .fillna(0.0)
                )
                bank = RegimeModelBank()
                bank.fit(df_tr_c, fwd_train, feature_cols=feature_cols,
                         val_pct=0.15)

                df_te_c = df_test.copy()
                df_te_c[feature_cols] = (
                    df_te_c[feature_cols]
                    .fillna(df_tr_c[feature_cols].median())
                    .fillna(0.0)
                )
                preds = bank.predict_all_history(df_te_c)

            all_preds.append(preds)
            log.info(f"  Fold {fold} complete: {len(preds)} predictions")

        except Exception as e:
            log.error(f"  Fold {fold} failed: {e}")

        start += TEST_DAYS

    if not all_preds:
        log.error("Walk-forward CV produced no predictions")
        return pd.DataFrame()

    result = pd.concat(all_preds).sort_index()
    # Remove any duplicate dates (overlap at fold boundaries)
    result = result[~result.index.duplicated(keep="last")]
    log.info(f"Walk-forward complete: {len(result)} OOS days across {fold} folds")
    return result
