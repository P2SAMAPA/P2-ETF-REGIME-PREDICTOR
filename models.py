"""
models.py — P2-ETF-REGIME-PREDICTOR
=====================================
LightGBM + Logistic Regression ensemble for ETF rotation.

Architecture:
  - One LightGBM binary classifier per ETF per regime
  - One Logistic Regression (L1) binary classifier per ETF per regime
  - Confidence-weighted ensemble: high conviction when both agree,
    CASH signal when they strongly disagree
  - Walk-forward cross-validation for robust out-of-sample metrics

Target variable:
  Binary 1 if ETF beats 3M T-Bill over next 5 trading days, else 0.
  Trained separately per regime so each model learns regime-specific
  patterns rather than averaging across all market environments.

Author: P2SAMAPA
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED            = 42
MIN_REGIME_ROWS        = 150
FORWARD_DAYS           = 5
DISAGREEMENT_THRESHOLD = 0.15
TARGET_ETFS            = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
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
    "n_estimators":     300,
    "early_stopping_rounds": 30,
}

LOGREG_PARAMS = {
    "penalty":     "l1",
    "solver":      "liblinear",
    "C":           0.1,
    "max_iter":    1000,
    "random_state": RANDOM_SEED,
}


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame,
                         exclude_cols: Optional[list] = None) -> list:
    """Select model input features — keeps derived/Z-scored, drops raw OHLC."""
    if exclude_cols is None:
        exclude_cols = []

    always_exclude = {
        "Regime", "Regime_Name",
        *[f"{t}_BeatCash" for t in TARGET_ETFS],
        *[f"{t}_FwdRet"   for t in TARGET_ETFS],
    }
    raw_suffixes = ("_Open", "_High", "_Low", "_Close",
                    "_Volume", "_Adj Close")

    feature_cols = [
        c for c in df.columns
        if c not in always_exclude
        and c not in exclude_cols
        and not any(c.endswith(s) for s in raw_suffixes)
        and df[c].dtype in [np.float64, np.float32,
                             np.int64, np.int32, float, int]
        and df[c].nunique() > 5
    ]
    log.info(f"Selected {len(feature_cols)} feature columns")
    return feature_cols


# ── Single ETF binary classifier ─────────────────────────────────────────────

class ETFBinaryClassifier:
    """LightGBM + LogReg ensemble for one ETF in one regime."""

    def __init__(self, etf: str, regime: int, regime_name: str = ""):
        self.etf           = etf
        self.regime        = regime
        self.regime_name   = regime_name
        self.lgbm_         = None
        self.logreg_       = None
        self.scaler_       = RobustScaler()
        self.feature_cols_ = []
        self.is_fitted_    = False
        self.val_auc_      = None
        self.n_train_      = 0
        self.n_pos_frac_   = 0.0

    def fit(self, X_train, y_train, X_val, y_val,
            feature_cols: list) -> "ETFBinaryClassifier":
        self.feature_cols_ = feature_cols
        self.n_train_      = len(X_train)
        self.n_pos_frac_   = float(y_train.mean()) if len(y_train) > 0 else 0.5

        X_train_s = self.scaler_.fit_transform(X_train)
        X_val_s   = self.scaler_.transform(X_val)

        # LightGBM
        params = {k: v for k, v in LGBM_PARAMS.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}
        dtrain = lgb.Dataset(X_train_s, label=y_train)
        dval   = lgb.Dataset(X_val_s,   label=y_val, reference=dtrain)
        self.lgbm_ = lgb.train(
            params, dtrain,
            num_boost_round=LGBM_PARAMS["n_estimators"],
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(LGBM_PARAMS["early_stopping_rounds"],
                                   verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Logistic Regression
        self.logreg_ = LogisticRegression(**LOGREG_PARAMS)
        self.logreg_.fit(X_train_s, y_train)

        # Validation AUC
        if len(np.unique(y_val)) > 1:
            lgbm_p  = self.lgbm_.predict(X_val_s)
            lr_p    = self.logreg_.predict_proba(X_val_s)[:, 1]
            ens_p   = 0.6 * lgbm_p + 0.4 * lr_p
            self.val_auc_ = round(roc_auc_score(y_val, ens_p), 4)

        self.is_fitted_ = True
        log.info(f"  [{self.etf} | {self.regime_name}] "
                 f"n={self.n_train_} pos={self.n_pos_frac_:.1%} "
                 f"val_auc={self.val_auc_}")
        return self

    def predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ensemble_prob, lgbm_prob, logreg_prob)."""
        X_s       = self.scaler_.transform(X)
        lgbm_p    = self.lgbm_.predict(X_s)
        lr_p      = self.logreg_.predict_proba(X_s)[:, 1]
        ensemble  = 0.6 * lgbm_p + 0.4 * lr_p
        return ensemble, lgbm_p, lr_p

    def feature_importance(self) -> pd.Series:
        if self.lgbm_ is None:
            return pd.Series(dtype=float)
        imp = self.lgbm_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_cols_).sort_values(ascending=False)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "ETFBinaryClassifier":
        return pickle.loads(data)


# ── Regime-aware model bank ───────────────────────────────────────────────────

class RegimeModelBank:
    """
    One ETFBinaryClassifier per (ETF, regime).
    Falls back to global model when regime has insufficient data.
    """

    def __init__(self):
        self.classifiers_:        Dict[Tuple[str, int], ETFBinaryClassifier] = {}
        self.global_classifiers_: Dict[str, ETFBinaryClassifier] = {}
        self.feature_cols_:       List[str] = []
        self.regimes_:            List[int] = []
        self.regime_names_:       Dict[int, str] = {}
        self.val_metrics_:        Dict = {}
        self.fitted_:             bool = False

    def fit(self, df: pd.DataFrame, fwd_df: pd.DataFrame,
            feature_cols: Optional[list] = None,
            val_pct: float = 0.15) -> "RegimeModelBank":

        if "Regime" not in df.columns:
            raise ValueError("df must have Regime column")

        if feature_cols is None:
            feature_cols = get_feature_columns(df)
        self.feature_cols_ = feature_cols

        common_idx  = df.index.intersection(fwd_df.index)
        df_a        = df.loc[common_idx]
        fwd_a       = fwd_df.loc[common_idx]
        self.regimes_     = sorted(df_a["Regime"].dropna()
                                   .unique().astype(int).tolist())
        self.regime_names_= {r: str(r) for r in self.regimes_}

        log.info(f"Training {len(TARGET_ETFS)} ETFs × "
                 f"{len(self.regimes_)} regimes...")

        # Global fallback models
        self._train_global(df_a, fwd_a, feature_cols, val_pct)

        # Per-regime models
        for regime in self.regimes_:
            mask      = df_a["Regime"] == regime
            r_df      = df_a[mask]
            r_fwd     = fwd_a[mask]

            if len(r_df) < MIN_REGIME_ROWS:
                log.warning(f"Regime {regime}: {len(r_df)} rows — using global")
                continue

            for etf in TARGET_ETFS:
                tcol       = f"{etf}_BeatCash"
                if tcol not in r_fwd.columns:
                    continue
                valid      = r_fwd[tcol].notna()
                X_full     = r_df.loc[valid, feature_cols].values
                y_full     = r_fwd.loc[valid, tcol].values.astype(int)

                if len(X_full) < MIN_REGIME_ROWS or len(np.unique(y_full)) < 2:
                    continue

                vs         = max(1, int(len(X_full) * val_pct))
                clf        = ETFBinaryClassifier(
                    etf=etf, regime=regime,
                    regime_name=self.regime_names_.get(regime, str(regime))
                )
                try:
                    clf.fit(X_full[:-vs], y_full[:-vs],
                            X_full[-vs:],  y_full[-vs:],
                            feature_cols)
                    self.classifiers_[(etf, regime)]  = clf
                    self.val_metrics_[(etf, regime)]   = clf.val_auc_
                except Exception as e:
                    log.error(f"[{etf}|{regime}] fit failed: {e}")

        self.fitted_ = True
        self._log_summary()
        return self

    def _train_global(self, df, fwd_df, feature_cols, val_pct):
        for etf in TARGET_ETFS:
            tcol = f"{etf}_BeatCash"
            if tcol not in fwd_df.columns:
                continue
            valid  = fwd_df[tcol].notna()
            X      = df.loc[valid, feature_cols].values
            y      = fwd_df.loc[valid, tcol].values.astype(int)
            if len(X) < 100 or len(np.unique(y)) < 2:
                continue
            vs  = max(1, int(len(X) * val_pct))
            clf = ETFBinaryClassifier(etf=etf, regime=-1, regime_name="Global")
            try:
                clf.fit(X[:-vs], y[:-vs], X[-vs:], y[-vs:], feature_cols)
                self.global_classifiers_[etf] = clf
            except Exception as e:
                log.error(f"[{etf}|global] fit failed: {e}")

    def predict(self, X: np.ndarray, regime: int) -> pd.DataFrame:
        """Predict P(beat cash) for all ETFs given current regime."""
        rows = []
        for etf in TARGET_ETFS:
            key = (etf, regime)
            if key in self.classifiers_:
                clf    = self.classifiers_[key]
                source = f"regime_{regime}"
            elif etf in self.global_classifiers_:
                clf    = self.global_classifiers_[etf]
                source = "global"
            else:
                rows.append({"ETF": etf, "P_BeatCash": 0.5,
                              "LGBM_Prob": 0.5, "LogReg_Prob": 0.5,
                              "Disagree": False, "Source": "none"})
                continue

            try:
                ens, lgbm_p, lr_p = clf.predict_proba(X)
                rows.append({
                    "ETF":         etf,
                    "P_BeatCash":  float(ens[0]),
                    "LGBM_Prob":   float(lgbm_p[0]),
                    "LogReg_Prob": float(lr_p[0]),
                    "Disagree":    bool(abs(lgbm_p[0] - lr_p[0]) >
                                        DISAGREEMENT_THRESHOLD),
                    "Source":      source,
                })
            except Exception as e:
                log.error(f"predict [{etf}|{regime}]: {e}")
                rows.append({"ETF": etf, "P_BeatCash": 0.5,
                              "LGBM_Prob": 0.5, "LogReg_Prob": 0.5,
                              "Disagree": False, "Source": "error"})

        return pd.DataFrame(rows).set_index("ETF")

    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for every row — used for backtesting."""
        if not self.fitted_:
            raise ValueError("Not fitted")
        rows         = []
        feat_matrix  = np.nan_to_num(
            df[self.feature_cols_].values, nan=0.0)

        for i, (idx, row) in enumerate(df.iterrows()):
            regime = int(row.get("Regime", 0))
            X      = feat_matrix[i:i+1]
            preds  = self.predict(X, regime)
            entry  = {"Date": idx, "Regime": regime}
            for etf in TARGET_ETFS:
                if etf in preds.index:
                    entry[f"{etf}_P"]        = preds.loc[etf, "P_BeatCash"]
                    entry[f"{etf}_Disagree"] = preds.loc[etf, "Disagree"]
            rows.append(entry)

        return pd.DataFrame(rows).set_index("Date")

    def get_conviction_zscore(self,
                               preds_df: pd.DataFrame) -> Tuple[str, float, str]:
        """
        Returns (best_etf, z_score, label).
        Z = std deviations top ETF's P_BeatCash sits above cross-ETF mean.
        """
        p    = preds_df["P_BeatCash"].values
        mean = np.mean(p)
        std  = np.std(p)
        if std < 1e-9:
            return str(preds_df.index[0]), 0.0, "Low"
        best = int(np.argmax(p))
        z    = float((p[best] - mean) / std)
        lbl  = ("Very High" if z >= 2.0 else
                "High"      if z >= 1.0 else
                "Moderate"  if z >= 0.5 else "Low")
        return str(preds_df.index[best]), z, lbl

    def _log_summary(self):
        if not self.val_metrics_:
            return
        log.info("Val AUC summary:")
        for (etf, regime), auc in sorted(self.val_metrics_.items()):
            log.info(f"  {etf} | regime {regime}: AUC={auc}")

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "RegimeModelBank":
        return pickle.loads(data)


# ── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, fwd_df: pd.DataFrame,
                    n_splits: int = 5,
                    val_pct: float = 0.15) -> pd.DataFrame:
    """
    Walk-forward CV: n_splits chronological folds.
    Each fold trains on all prior data, tests on next window.
    Regime detection re-run per fold to avoid look-ahead bias.
    """
    from regime_detection import RegimeDetector

    common_idx  = df.index.intersection(fwd_df.index)
    df_a        = df.loc[common_idx]
    fwd_a       = fwd_df.loc[common_idx]
    N           = len(df_a)
    min_train   = int(N * 0.6)
    fold_size   = max(30, (N - min_train) // n_splits)
    results     = []

    for fold in range(n_splits):
        t_end  = min_train + fold * fold_size
        ts     = t_end
        te     = min(t_end + fold_size, N)
        if te <= ts:
            break

        tr_df  = df_a.iloc[:t_end]
        tr_fwd = fwd_a.iloc[:t_end]
        te_df  = df_a.iloc[ts:te]
        te_fwd = fwd_a.iloc[ts:te]

        log.info(f"WF fold {fold+1}: train={len(tr_df)} test={len(te_df)}")
        try:
            det  = RegimeDetector()
            det.fit(tr_df)
            tr_r = det.add_regime_to_df(tr_df)
            te_r = det.add_regime_to_df(te_df)

            fc   = get_feature_columns(tr_r)
            bank = RegimeModelBank()
            bank.fit(tr_r, tr_fwd, feature_cols=fc, val_pct=val_pct)

            row  = {"Fold": fold+1,
                    "Train_Start": tr_df.index[0].date(),
                    "Train_End":   tr_df.index[-1].date(),
                    "Test_Start":  te_df.index[0].date(),
                    "Test_End":    te_df.index[-1].date()}

            for etf in TARGET_ETFS:
                tcol = f"{etf}_BeatCash"
                if tcol not in te_fwd.columns:
                    continue
                valid = te_fwd[tcol].notna()
                X_te  = te_r.loc[valid, fc].values
                y_te  = te_fwd.loc[valid, tcol].values.astype(int)
                regs  = te_r.loc[valid, "Regime"].values.astype(int)
                if len(y_te) == 0 or len(np.unique(y_te)) < 2:
                    continue
                probs = np.array([
                    bank.predict(X_te[i:i+1],
                                 int(regs[i])).loc[etf, "P_BeatCash"]
                    if etf in bank.predict(X_te[i:i+1],
                                           int(regs[i])).index
                    else 0.5
                    for i in range(len(X_te))
                ])
                row[f"{etf}_Acc"] = round(
                    accuracy_score(y_te, (probs > 0.5).astype(int)), 4)
                row[f"{etf}_AUC"] = round(roc_auc_score(y_te, probs), 4)

            results.append(row)
        except Exception as e:
            log.error(f"Fold {fold+1} failed: {e}")

    return pd.DataFrame(results)


# ── Feature importance ────────────────────────────────────────────────────────

def aggregate_feature_importance(bank: RegimeModelBank) -> pd.DataFrame:
    """Aggregate LightGBM gain importance across all classifiers."""
    all_imp = {}
    for clf in bank.classifiers_.values():
        for feat, val in clf.feature_importance().items():
            all_imp.setdefault(feat, []).append(val)
    rows = [{"Feature": f, "Mean_Gain": np.mean(v),
             "Std_Gain": np.std(v), "Count": len(v)}
            for f, v in all_imp.items()]
    return (pd.DataFrame(rows)
            .sort_values("Mean_Gain", ascending=False)
            .reset_index(drop=True))
