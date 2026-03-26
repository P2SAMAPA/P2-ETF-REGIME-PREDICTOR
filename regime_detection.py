"""
regime_detection.py — P2-ETF-REGIME-PREDICTOR
==============================================
Wasserstein k-means regime detection.

Implements the clustering methodology from:
  "Clustering Market Regimes Using the Wasserstein Distance"
  B. Horvath, Z. Issa, A. Muguruza (2021)
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905

Author: P2SAMAPA
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_SIZE   = 20        # Rolling window (trading days) per distribution
K_MIN         = 3         # Minimum number of regimes to test
K_MAX         = 5         # Maximum number of regimes to test
N_INIT        = 5         # WK-means restarts — full pipeline
N_INIT_SWEEP  = 2         # WK-means restarts — sweep mode (speed)
MAX_ITER      = 50        # Max iterations per WK-means run
MAX_WINDOWS   = 1500      # Subsample windows — full pipeline
MAX_WINDOWS_SWEEP = 600   # Subsample windows — sweep mode (speed)
RANDOM_SEED   = 42

REGIME_NAMES  = {
    0: "Risk-On",
    1: "Risk-Off",
    2: "Rate-Rising",
    3: "Crisis",
    4: "Stagflation",
    5: "Recovery",
}


# ── Wasserstein distance utilities ───────────────────────────────────────────

def wasserstein_dist_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance, returning 0.0 if the calculation fails
    (e.g., due to degenerate distributions that cause division by zero).
    """
    try:
        return float(wasserstein_distance(u, v))
    except (ZeroDivisionError, ValueError):
        # If both distributions are identical (or degenerate), distance is 0
        return 0.0


def multi_asset_wasserstein(window_u: np.ndarray,
                             window_v: np.ndarray) -> float:
    """Average Wasserstein distance across all assets."""
    n_assets = window_u.shape[1]
    total    = 0.0
    for j in range(n_assets):
        total += wasserstein_dist_1d(window_u[:, j], window_v[:, j])
    return total / n_assets


# ── Distribution extraction ──────────────────────────────────────────────────

def extract_return_windows(df: pd.DataFrame,
                            ret_cols: list,
                            window: int = WINDOW_SIZE) -> Tuple[np.ndarray,
                                                                  pd.DatetimeIndex]:
    data = df[ret_cols].values
    T    = len(data)

    windows = []
    dates   = []

    for i in range(window, T + 1):
        w = data[i - window:i]
        # Skip windows with too many NaNs (e.g., >10% missing)
        if np.isnan(w).mean() > 0.1:
            continue
        w = np.nan_to_num(w, nan=0.0)
        windows.append(w)
        dates.append(df.index[i - 1])

    windows = np.array(windows)
    dates   = pd.DatetimeIndex(dates)
    log.info(f"Extracted {len(windows)} return windows "
             f"({window}d, {len(ret_cols)} assets)")
    return windows, dates


# ── Wasserstein k-means ──────────────────────────────────────────────────────

class WassersteinKMeans:
    """
    Wasserstein k-means clustering for empirical return distributions.
    Implements Algorithm 1 from Horvath, Issa, Muguruza (2021).
    """

    def __init__(self, k: int = 4, n_init: int = N_INIT,
                 max_iter: int = MAX_ITER, random_state: int = RANDOM_SEED):
        self.k            = k
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.random_state = random_state
        self.centroids_   = None
        self.labels_      = None
        self.inertia_     = None
        self.n_iter_      = 0

    def _compute_distance_matrix(self, windows: np.ndarray,
                                  centroids: np.ndarray) -> np.ndarray:
        N = len(windows)
        D = np.zeros((N, self.k))
        for i in range(N):
            for j in range(self.k):
                D[i, j] = multi_asset_wasserstein(windows[i], centroids[j])
        return D

    def _compute_medoid(self, windows: np.ndarray) -> np.ndarray:
        n = len(windows)
        if n == 1:
            return windows[0]
        total_dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_dists[i] += multi_asset_wasserstein(
                        windows[i], windows[j])
        return windows[np.argmin(total_dists)]

    def _single_run(self, windows: np.ndarray,
                    seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.RandomState(seed)
        N   = len(windows)

        # Initialisation: choose k random windows as centroids
        init_idx  = rng.choice(N, size=self.k, replace=False)
        centroids = windows[init_idx].copy()

        labels   = np.zeros(N, dtype=int)
        inertia  = float("inf")

        for iteration in range(self.max_iter):
            D          = self._compute_distance_matrix(windows, centroids)
            new_labels = np.argmin(D, axis=1)
            new_inertia = D[np.arange(N), new_labels].sum()

            if np.array_equal(new_labels, labels) and iteration > 0:
                break

            labels  = new_labels
            inertia = new_inertia

            # Update centroids (medoids) for each cluster
            new_centroids = centroids.copy()
            for j in range(self.k):
                cluster_windows = windows[labels == j]
                if len(cluster_windows) == 0:
                    # Empty cluster: pick a random window as new centroid
                    new_centroids[j] = windows[rng.randint(N)]
                else:
                    # Compute medoid – if all points are identical, the medoid is that point
                    new_centroids[j] = self._compute_medoid(cluster_windows)
            centroids = new_centroids

        self.n_iter_ = iteration + 1
        return labels, centroids, inertia

    def fit(self, windows: np.ndarray,
            max_windows_override: int = None) -> "WassersteinKMeans":
        """
        Fit WK-means. max_windows_override allows sweep mode to pass
        a smaller subsample without changing the global constant.
        """
        rng_sub = np.random.RandomState(self.random_state)
        N_orig  = len(windows)
        cap     = max_windows_override if max_windows_override else MAX_WINDOWS

        if N_orig > cap:
            sub_idx = rng_sub.choice(N_orig, size=cap, replace=False)
            sub_idx = np.sort(sub_idx)
            fit_windows = windows[sub_idx]
            log.info(f"Subsampled {N_orig} → {cap} windows for fitting")
        else:
            fit_windows = windows
            sub_idx     = np.arange(N_orig)

        log.info(f"WK-means fitting k={self.k}, "
                 f"{self.n_init} inits, {len(fit_windows)} windows...")

        best_labels    = None
        best_centroids = None
        best_inertia   = float("inf")

        for run in range(self.n_init):
            seed = self.random_state + run
            try:
                labels, centroids, inertia = self._single_run(fit_windows, seed)
                log.info(f"  Run {run+1}/{self.n_init}: "
                         f"inertia={inertia:.4f}, iters={self.n_iter_}")
                if inertia < best_inertia:
                    best_inertia   = inertia
                    best_labels    = labels
                    best_centroids = centroids
            except Exception as e:
                log.warning(f"  Run {run+1} failed: {e}")

        if best_centroids is None:
            raise RuntimeError(
                f"All {self.n_init} runs failed for k={self.k}. "
                "Check data for zero-variance windows or reduce number of assets."
            )

        self.centroids_ = best_centroids
        self.inertia_   = best_inertia

        if N_orig > cap:
            log.info(f"Predicting labels for all {N_orig} windows...")
            self.labels_ = self.predict(windows)
        else:
            self.labels_ = best_labels

        log.info(f"WK-means fitted: best inertia={best_inertia:.4f}")
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("Model not fitted — call fit() first")
        D = self._compute_distance_matrix(windows, self.centroids_)
        return np.argmin(D, axis=1)

    def fit_predict(self, windows: np.ndarray,
                    max_windows_override: int = None) -> np.ndarray:
        self.fit(windows, max_windows_override=max_windows_override)
        return self.labels_


# ── Optimal k selection via MMD scoring ──────────────────────────────────────

def mmd_score(windows: np.ndarray, labels: np.ndarray) -> float:
    k          = len(np.unique(labels))
    within     = 0.0
    between    = 0.0
    wc_count   = 0
    bc_count   = 0

    rng = np.random.RandomState(RANDOM_SEED)

    for j in range(k):
        cluster_j = windows[labels == j]
        if len(cluster_j) < 2:
            continue

        idx  = rng.choice(len(cluster_j),
                          size=min(len(cluster_j), 30), replace=False)
        sub  = cluster_j[idx]
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                within   += multi_asset_wasserstein(sub[a], sub[b])
                wc_count += 1

        for l in range(j + 1, k):
            cluster_l = windows[labels == l]
            if len(cluster_l) == 0:
                continue
            idx_j = rng.choice(len(cluster_j),
                                size=min(len(cluster_j), 15), replace=False)
            idx_l = rng.choice(len(cluster_l),
                                size=min(len(cluster_l), 15), replace=False)
            for a in idx_j:
                for b in idx_l:
                    between  += multi_asset_wasserstein(
                        cluster_j[a], cluster_l[b])
                    bc_count += 1

    if wc_count == 0 or bc_count == 0:
        return 0.0

    avg_within  = within  / wc_count
    avg_between = between / bc_count
    return avg_between / (avg_within + 1e-9)


def select_optimal_k(windows: np.ndarray,
                      k_min: int = K_MIN,
                      k_max: int = K_MAX,
                      sweep_mode: bool = False) -> Tuple[int, dict]:
    """
    Test k values from k_min to k_max, score each with MMD.
    In sweep_mode: skip k selection entirely, always return k=3
    (k=3 wins in every logged run — saves ~4 mins per sweep job).
    """
    if sweep_mode:
        log.info("Sweep mode: skipping k-selection, using k=3 (historically always optimal)")
        return 3, {3: 0.0}

    scores = {}
    log.info(f"Selecting optimal k from {k_min} to {k_max}...")

    rng = np.random.RandomState(RANDOM_SEED)
    k_max_windows = min(600, len(windows))
    if len(windows) > k_max_windows:
        kidx      = np.sort(rng.choice(len(windows), k_max_windows, replace=False))
        k_windows = windows[kidx]
        log.info(f"k-selection subsampled to {k_max_windows} windows")
    else:
        k_windows = windows

    for k in range(k_min, k_max + 1):
        try:
            model  = WassersteinKMeans(k=k, n_init=3, max_iter=30)
            labels = model.fit_predict(k_windows)
            score  = mmd_score(k_windows, labels)
            scores[k] = round(score, 4)
            log.info(f"  k={k}: MMD score={score:.4f}")
        except Exception as e:
            log.warning(f"  k={k} failed: {e}")
            scores[k] = 0.0

    optimal_k = max(scores, key=scores.get)
    log.info(f"Optimal k={optimal_k} (MMD={scores[optimal_k]:.4f})")
    return optimal_k, scores


# ── Regime characterisation ──────────────────────────────────────────────────

def characterise_regimes(df: pd.DataFrame,
                          labels: np.ndarray,
                          dates: pd.DatetimeIndex,
                          ret_cols: list) -> pd.DataFrame:
    label_series = pd.Series(labels, index=dates, name="Regime")
    aligned      = df.join(label_series, how="inner")

    rows = []
    for regime in sorted(aligned["Regime"].unique()):
        sub   = aligned[aligned["Regime"] == regime]
        row   = {"Regime": regime, "N_Days": len(sub),
                 "Pct_Days": f"{100*len(sub)/len(aligned):.1f}%"}

        for col in ret_cols:
            if col in sub.columns:
                row[f"Avg_{col}"] = round(sub[col].mean() * 252, 4)

        for macro in ["VIXCLS", "DGS10", "T10Y2Y", "T10YIE",
                      "BAMLH0A0HYM2"]:
            if macro in sub.columns:
                row[f"Avg_{macro}"] = round(sub[macro].mean(), 3)

        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime")


def label_regimes(characteristics: pd.DataFrame) -> dict:
    mapping  = {}
    assigned = set()

    vix_col = "Avg_VIXCLS" if "Avg_VIXCLS" in characteristics.columns else None
    hy_col  = "Avg_BAMLH0A0HYM2" if "Avg_BAMLH0A0HYM2" in characteristics.columns else None
    yc_col  = "Avg_T10Y2Y" if "Avg_T10Y2Y" in characteristics.columns else None
    inf_col = "Avg_T10YIE" if "Avg_T10YIE" in characteristics.columns else None

    regimes = list(characteristics.index)

    if vix_col:
        crisis_idx = characteristics[vix_col].idxmax()
        mapping[crisis_idx] = "Crisis"
        assigned.add(crisis_idx)

    if inf_col:
        for idx in regimes:
            if idx not in assigned:
                stag_idx = characteristics.loc[
                    [i for i in regimes if i not in assigned], inf_col
                ].idxmax()
                mapping[stag_idx] = "Stagflation"
                assigned.add(stag_idx)
                break

    if yc_col:
        for idx in regimes:
            if idx not in assigned:
                rr_idx = characteristics.loc[
                    [i for i in regimes if i not in assigned], yc_col
                ].idxmin()
                mapping[rr_idx] = "Rate-Rising"
                assigned.add(rr_idx)
                break

    remaining = [i for i in regimes if i not in assigned]
    if vix_col and len(remaining) >= 2:
        sub = characteristics.loc[remaining, vix_col]
        mapping[sub.idxmin()] = "Risk-On"
        assigned.add(sub.idxmin())
        remaining = [i for i in remaining if i not in assigned]

    for idx in remaining:
        if idx not in assigned:
            mapping[idx] = "Risk-Off"
            assigned.add(idx)

    log.info(f"Regime labels: {mapping}")
    return mapping


# ── Main pipeline ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Full regime detection pipeline.
    Wraps WK-means with optimal k selection, labelling, and
    incremental prediction for new data.
    """

    def __init__(self, window: int = WINDOW_SIZE,
                 k: Optional[int] = None):
        self.window          = window
        self.k               = k
        self.model_          = None
        self.optimal_k_      = None
        self.regime_names_   = {}
        self.characteristics_= None
        self.ret_cols_       = None
        self.dates_          = None
        self.scaler_         = StandardScaler()

    def fit(self, df: pd.DataFrame,
            ret_cols: Optional[list] = None,
            sweep_mode: bool = False) -> "RegimeDetector":
        """
        Fit regime detector on full historical dataset.

        sweep_mode=True activates the fast path:
          - Skips k-selection (always uses k=3)
          - Reduces n_init: 5 → 2
          - Reduces max_windows: 1500 → 600
        This cuts fitting time from ~25 mins to ~5 mins per sweep job,
        with negligible impact on regime quality (k=3 always wins,
        inertia difference between best-of-2 vs best-of-5 is <0.5%).
        """
        if ret_cols is None:
            # Fallback: use all columns that end with "_Ret"
            ret_cols = [c for c in df.columns if c.endswith("_Ret")]

        self.ret_cols_ = ret_cols

        windows, dates = extract_return_windows(df, ret_cols, self.window)
        self.dates_    = dates

        if sweep_mode:
            log.info("RegimeDetector: sweep_mode=True — fast path "
                     "(k=3, n_init=2, max_windows=600)")

        # k selection
        if self.k is None:
            self.optimal_k_, mmd_scores = select_optimal_k(
                windows, sweep_mode=sweep_mode)
            log.info(f"MMD scores: {mmd_scores}")
        else:
            self.optimal_k_ = self.k

        # Choose params based on mode
        n_init      = N_INIT_SWEEP  if sweep_mode else N_INIT
        max_windows = MAX_WINDOWS_SWEEP if sweep_mode else MAX_WINDOWS

        self.model_ = WassersteinKMeans(
            k=self.optimal_k_, n_init=n_init, max_iter=MAX_ITER
        )
        self.model_.fit(windows, max_windows_override=max_windows)

        self.characteristics_ = characterise_regimes(
            df, self.model_.labels_, dates, ret_cols
        )
        self.regime_names_ = label_regimes(self.characteristics_)

        log.info("RegimeDetector fitted:")
        log.info(f"\n{self.characteristics_[['N_Days','Pct_Days']].to_string()}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError("Not fitted — call fit() first")

        windows, dates = extract_return_windows(
            df, self.ret_cols_, self.window
        )
        raw_labels   = self.model_.predict(windows)
        label_series = pd.Series(raw_labels, index=dates, name="Regime")
        label_series = label_series.reindex(df.index, method="ffill")
        return label_series

    def predict_named(self, df: pd.DataFrame) -> pd.Series:
        numeric = self.predict(df)
        return numeric.map(self.regime_names_).fillna("Unknown")

    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Regime"]      = self.predict(df)
        df["Regime_Name"] = self.predict_named(df)
        return df

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, str]:
        recent = df.tail(self.window + 5)
        windows, dates = extract_return_windows(
            recent, self.ret_cols_, self.window
        )
        if len(windows) == 0:
            return 0, "Unknown"
        label = int(self.model_.predict(windows[-1:])[0])
        name  = self.regime_names_.get(label, "Unknown")
        return label, name

    def summary(self) -> str:
        if self.characteristics_ is None:
            return "Not fitted"
        lines = [f"Regime Detector — k={self.optimal_k_}",
                 f"Window: {self.window} days", ""]
        for idx, name in self.regime_names_.items():
            if idx in self.characteristics_.index:
                row = self.characteristics_.loc[idx]
                lines.append(
                    f"  Regime {idx} [{name}]: "
                    f"{row.get('N_Days','?')} days "
                    f"({row.get('Pct_Days','?')})"
                )
        return "\n".join(lines)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "RegimeDetector":
        return pickle.loads(data)
