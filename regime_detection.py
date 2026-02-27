"""
regime_detection.py — P2-ETF-REGIME-PREDICTOR
==============================================
Wasserstein k-means regime detection.

Implements the clustering methodology from:
  "Clustering Market Regimes Using the Wasserstein Distance"
  B. Horvath, Z. Issa, A. Muguruza (2021)
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905

Core idea:
  Each rolling window of returns is treated as an empirical probability
  distribution. The Wasserstein distance measures how far apart two such
  distributions are — accounting for shape, not just mean/variance.
  WK-means clusters these distributions into k market regimes
  (e.g. Risk-On, Risk-Off, Rate-Rising, Crisis) without assuming
  Gaussianity or any parametric form.

Regime labels are added as a feature column to the dataset and used
by the LightGBM + LogReg ensemble to train regime-specific models.

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
N_INIT        = 5         # WK-means restarts (take best)
MAX_ITER      = 50        # Max iterations per WK-means run
MAX_WINDOWS   = 1500      # Subsample windows — Wasserstein is O(n²)
RANDOM_SEED   = 42

# Human-readable regime names assigned post-clustering based on characteristics
# These are assigned dynamically after fitting — see label_regimes()
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
    1D Wasserstein distance between two empirical distributions u and v.
    Uses scipy's exact implementation (equivalent to Earth Mover's Distance).
    """
    return float(wasserstein_distance(u, v))


def multi_asset_wasserstein(window_u: np.ndarray,
                             window_v: np.ndarray) -> float:
    """
    Multi-asset Wasserstein distance between two return windows.
    window_u, window_v: shape (window_size, n_assets)

    Computes average of per-asset 1D Wasserstein distances.
    This is consistent with the marginal approach in Horvath et al. (2021)
    and is computationally tractable for daily retraining.
    """
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
    """
    Extract rolling return windows from dataset.

    Parameters
    ----------
    df       : full feature DataFrame with return columns
    ret_cols : list of return column names to use (e.g. TLT_Ret, SLV_Ret ...)
    window   : rolling window size in trading days

    Returns
    -------
    windows  : np.ndarray of shape (N, window, n_assets)
    dates    : DatetimeIndex of length N (date of last day in each window)
    """
    data = df[ret_cols].values   # (T, n_assets)
    T    = len(data)

    windows = []
    dates   = []

    for i in range(window, T + 1):
        w = data[i - window:i]   # (window, n_assets)
        # Skip windows with too many NaNs
        if np.isnan(w).mean() > 0.1:
            continue
        # Replace remaining NaNs with 0 (neutral return)
        w = np.nan_to_num(w, nan=0.0)
        windows.append(w)
        dates.append(df.index[i - 1])   # date of last day in window

    windows = np.array(windows)           # (N, window, n_assets)
    dates   = pd.DatetimeIndex(dates)
    log.info(f"Extracted {len(windows)} return windows "
             f"({window}d, {len(ret_cols)} assets)")
    return windows, dates


# ── Wasserstein k-means ──────────────────────────────────────────────────────

class WassersteinKMeans:
    """
    Wasserstein k-means clustering for empirical return distributions.

    Implements Algorithm 1 from Horvath, Issa, Muguruza (2021):
      1. Initialise k centroids (random windows)
      2. Assign each window to nearest centroid (Wasserstein distance)
      3. Update centroid = window that minimises total distance to cluster
         (Fréchet mean approximation — exact mean not tractable for
          empirical distributions, so we use the medoid)
      4. Repeat until convergence or max_iter

    Parameters
    ----------
    k         : number of regimes
    n_init    : number of random restarts
    max_iter  : maximum iterations per run
    random_state : random seed for reproducibility
    """

    def __init__(self, k: int = 4, n_init: int = N_INIT,
                 max_iter: int = MAX_ITER, random_state: int = RANDOM_SEED):
        self.k            = k
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.random_state = random_state
        self.centroids_   = None    # (k, window, n_assets)
        self.labels_      = None    # (N,) int array
        self.inertia_     = None    # total within-cluster Wasserstein distance
        self.n_iter_      = 0

    def _compute_distance_matrix(self, windows: np.ndarray,
                                  centroids: np.ndarray) -> np.ndarray:
        """
        Compute (N, k) distance matrix between all windows and all centroids.
        """
        N = len(windows)
        D = np.zeros((N, self.k))
        for i in range(N):
            for j in range(self.k):
                D[i, j] = multi_asset_wasserstein(windows[i], centroids[j])
        return D

    def _compute_medoid(self, windows: np.ndarray) -> np.ndarray:
        """
        Return the medoid of a set of windows — the window that minimises
        total Wasserstein distance to all others in the set.
        (Approximation of the Fréchet mean for empirical distributions.)
        """
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
        """One full WK-means run. Returns (labels, centroids, inertia)."""
        rng = np.random.RandomState(seed)
        N   = len(windows)

        # Initialise centroids as random distinct windows
        init_idx  = rng.choice(N, size=self.k, replace=False)
        centroids = windows[init_idx].copy()

        labels   = np.zeros(N, dtype=int)
        inertia  = float("inf")

        for iteration in range(self.max_iter):
            # Assignment step
            D          = self._compute_distance_matrix(windows, centroids)
            new_labels = np.argmin(D, axis=1)
            new_inertia = D[np.arange(N), new_labels].sum()

            # Check convergence
            if np.array_equal(new_labels, labels) and iteration > 0:
                break

            labels  = new_labels
            inertia = new_inertia

            # Update step — recompute centroids as cluster medoids
            new_centroids = centroids.copy()
            for j in range(self.k):
                cluster_windows = windows[labels == j]
                if len(cluster_windows) == 0:
                    # Empty cluster — reinitialise to random window
                    new_centroids[j] = windows[rng.randint(N)]
                else:
                    new_centroids[j] = self._compute_medoid(cluster_windows)
            centroids = new_centroids

        self.n_iter_ = iteration + 1
        return labels, centroids, inertia

    def fit(self, windows: np.ndarray) -> "WassersteinKMeans":
        """
        Fit WK-means to windows array of shape (N, window_size, n_assets).
        Runs n_init times and keeps the best (lowest inertia) result.
        """
        # Subsample for tractability — Wasserstein distance is O(n²)
        rng_sub = np.random.RandomState(self.random_state)
        N_orig  = len(windows)
        if N_orig > MAX_WINDOWS:
            sub_idx = rng_sub.choice(N_orig, size=MAX_WINDOWS, replace=False)
            sub_idx = np.sort(sub_idx)   # preserve time order
            fit_windows = windows[sub_idx]
            log.info(f"Subsampled {N_orig} → {MAX_WINDOWS} windows for fitting")
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

        self.centroids_ = best_centroids
        self.inertia_   = best_inertia

        # Predict labels for ALL original windows using fitted centroids
        if N_orig > MAX_WINDOWS:
            log.info(f"Predicting labels for all {N_orig} windows...")
            self.labels_ = self.predict(windows)
        else:
            self.labels_ = best_labels

        log.info(f"WK-means fitted: best inertia={best_inertia:.4f}")
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Assign new windows to nearest centroid."""
        if self.centroids_ is None:
            raise ValueError("Model not fitted — call fit() first")
        D = self._compute_distance_matrix(windows, self.centroids_)
        return np.argmin(D, axis=1)

    def fit_predict(self, windows: np.ndarray) -> np.ndarray:
        self.fit(windows)
        return self.labels_


# ── Optimal k selection via MMD scoring ──────────────────────────────────────

def mmd_score(windows: np.ndarray, labels: np.ndarray) -> float:
    """
    Maximum Mean Discrepancy-inspired score for cluster quality.
    Higher = better separated clusters.

    Approximation: ratio of between-cluster to within-cluster
    average Wasserstein distance.
    Per the approach described in Horvath et al. (2021) Section 4.
    """
    k          = len(np.unique(labels))
    N          = len(windows)
    within     = 0.0
    between    = 0.0
    wc_count   = 0
    bc_count   = 0

    # Sample for efficiency on large datasets
    max_pairs = 500
    rng       = np.random.RandomState(RANDOM_SEED)

    for j in range(k):
        cluster_j = windows[labels == j]
        if len(cluster_j) < 2:
            continue

        # Within-cluster distances (sample)
        idx  = rng.choice(len(cluster_j),
                          size=min(len(cluster_j), 30), replace=False)
        sub  = cluster_j[idx]
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                within   += multi_asset_wasserstein(sub[a], sub[b])
                wc_count += 1

        # Between-cluster distances (sample)
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
                      k_max: int = K_MAX) -> Tuple[int, dict]:
    """
    Test k values from k_min to k_max, score each with MMD,
    return optimal k and scores dict.
    """
    scores = {}
    log.info(f"Selecting optimal k from {k_min} to {k_max}...")

    # Subsample for k selection — use smaller set for speed
    rng = np.random.RandomState(RANDOM_SEED)
    k_max_windows = min(600, len(windows))
    if len(windows) > k_max_windows:
        kidx    = np.sort(rng.choice(len(windows), k_max_windows, replace=False))
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
    """
    Compute summary statistics per regime to aid interpretation.
    Returns a DataFrame with one row per regime showing:
    avg return, volatility, VIX level, yield curve, frequency.
    """
    # Align labels to df index
    label_series = pd.Series(labels, index=dates, name="Regime")
    aligned      = df.join(label_series, how="inner")

    rows = []
    for regime in sorted(aligned["Regime"].unique()):
        sub   = aligned[aligned["Regime"] == regime]
        row   = {"Regime": regime, "N_Days": len(sub),
                 "Pct_Days": f"{100*len(sub)/len(aligned):.1f}%"}

        # Average returns per ETF
        for col in ret_cols:
            if col in sub.columns:
                row[f"Avg_{col}"] = round(sub[col].mean() * 252, 4)

        # Macro context
        for macro in ["VIXCLS", "DGS10", "T10Y2Y", "T10YIE",
                      "BAMLH0A0HYM2"]:
            if macro in sub.columns:
                row[f"Avg_{macro}"] = round(sub[macro].mean(), 3)

        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime")


def label_regimes(characteristics: pd.DataFrame) -> dict:
    """
    Auto-assign human-readable names to regime indices based on
    macro characteristics. Returns dict {regime_idx: name}.

    Logic:
    - Highest VIX + widest HY spread → Crisis
    - Positive DGS10 momentum + inverted curve → Rate-Rising
    - Low VIX + positive ETF returns → Risk-On
    - Low VIX + negative ETF returns → Risk-Off
    - High inflation (T10YIE) → Stagflation
    - Remaining → Recovery
    """
    mapping = {}
    assigned = set()

    vix_col = "Avg_VIXCLS" if "Avg_VIXCLS" in characteristics.columns else None
    hy_col  = "Avg_BAMLH0A0HYM2" if "Avg_BAMLH0A0HYM2" in characteristics.columns else None
    yc_col  = "Avg_T10Y2Y" if "Avg_T10Y2Y" in characteristics.columns else None
    inf_col = "Avg_T10YIE" if "Avg_T10YIE" in characteristics.columns else None

    regimes = list(characteristics.index)

    # Crisis: highest VIX
    if vix_col:
        crisis_idx = characteristics[vix_col].idxmax()
        mapping[crisis_idx] = "Crisis"
        assigned.add(crisis_idx)

    # Stagflation: highest inflation (T10YIE)
    if inf_col:
        for idx in regimes:
            if idx not in assigned:
                stag_idx = characteristics.loc[
                    [i for i in regimes if i not in assigned], inf_col
                ].idxmax()
                mapping[stag_idx] = "Stagflation"
                assigned.add(stag_idx)
                break

    # Rate-Rising: most inverted / shallowest yield curve
    if yc_col:
        for idx in regimes:
            if idx not in assigned:
                rr_idx = characteristics.loc[
                    [i for i in regimes if i not in assigned], yc_col
                ].idxmin()
                mapping[rr_idx] = "Rate-Rising"
                assigned.add(rr_idx)
                break

    # Remaining: assign Risk-On (lowest VIX among remaining) and Risk-Off
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
        self.k               = k          # None = auto-select
        self.model_          = None       # fitted WassersteinKMeans
        self.optimal_k_      = None
        self.regime_names_   = {}         # {int: str}
        self.characteristics_= None
        self.ret_cols_       = None
        self.dates_          = None
        self.scaler_         = StandardScaler()

    def fit(self, df: pd.DataFrame,
            ret_cols: Optional[list] = None) -> "RegimeDetector":
        """
        Fit regime detector on full historical dataset.

        Parameters
        ----------
        df       : full feature DataFrame
        ret_cols : return columns to use for distribution windows.
                   Defaults to all TARGET_ETF returns.
        """
        from data_manager import TARGET_ETFS

        if ret_cols is None:
            ret_cols = [f"{t}_Ret" for t in TARGET_ETFS
                        if f"{t}_Ret" in df.columns]

        self.ret_cols_ = ret_cols

        # Extract rolling windows
        windows, dates = extract_return_windows(df, ret_cols, self.window)
        self.dates_    = dates

        # Select k if not specified
        if self.k is None:
            self.optimal_k_, mmd_scores = select_optimal_k(windows)
            log.info(f"MMD scores: {mmd_scores}")
        else:
            self.optimal_k_ = self.k

        # Fit final model with optimal k
        self.model_ = WassersteinKMeans(
            k=self.optimal_k_, n_init=N_INIT, max_iter=MAX_ITER
        )
        self.model_.fit(windows)

        # Characterise and label regimes
        self.characteristics_ = characterise_regimes(
            df, self.model_.labels_, dates, ret_cols
        )
        self.regime_names_ = label_regimes(self.characteristics_)

        log.info("RegimeDetector fitted:")
        log.info(f"\n{self.characteristics_[['N_Days','Pct_Days']].to_string()}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for all rows in df.
        Returns a Series aligned to df.index with integer regime labels.
        """
        if self.model_ is None:
            raise ValueError("Not fitted — call fit() first")

        windows, dates = extract_return_windows(
            df, self.ret_cols_, self.window
        )
        raw_labels  = self.model_.predict(windows)
        label_series = pd.Series(raw_labels, index=dates, name="Regime")

        # Reindex to full df index (forward-fill for days before first window)
        label_series = label_series.reindex(df.index, method="ffill")
        return label_series

    def predict_named(self, df: pd.DataFrame) -> pd.Series:
        """Same as predict() but returns human-readable regime names."""
        numeric = self.predict(df)
        return numeric.map(self.regime_names_).fillna("Unknown")

    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Regime (int) and Regime_Name (str) columns to df.
        This is the main function called by train.py before model training.
        """
        df = df.copy()
        df["Regime"]      = self.predict(df)
        df["Regime_Name"] = self.predict_named(df)
        return df

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Return regime for the most recent row in df."""
        recent = df.tail(self.window + 5)
        windows, dates = extract_return_windows(
            recent, self.ret_cols_, self.window
        )
        if len(windows) == 0:
            return 0, "Unknown"
        label     = int(self.model_.predict(windows[-1:])[0])
        name      = self.regime_names_.get(label, "Unknown")
        return label, name

    def summary(self) -> str:
        """Human-readable summary of fitted regimes."""
        if self.characteristics_ is None:
            return "Not fitted"
        lines = [f"Regime Detector — k={self.optimal_k_}",
                 f"Window: {self.window} days",
                 ""]
        for idx, name in self.regime_names_.items():
            if idx in self.characteristics_.index:
                row = self.characteristics_.loc[idx]
                lines.append(
                    f"  Regime {idx} [{name}]: "
                    f"{row.get('N_Days','?')} days "
                    f"({row.get('Pct_Days','?')})"
                )
        return "\n".join(lines)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialise to bytes for GitLab storage."""
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "RegimeDetector":
        """Deserialise from bytes loaded from GitLab."""
        return pickle.loads(data)
