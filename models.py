"""
models.py - P2-ETF-REGIME-PREDICTOR v2 (CORRECTED v2)
=========================================
Models for regime detection and momentum ranking.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings
from typing import Dict, List, Optional, Tuple, Union


class RegimeDetector:
    """Detects market regimes using clustering on returns."""
    
    def __init__(self, window: int = 20, k: Optional[int] = None):
        self.window = window
        self.k = k
        self.optimal_k_ = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.kmeans = None
        self.regime_labels_ = None
        
    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create features for regime detection from return data."""
        features = []
        
        for col in df.columns:
            # Rolling volatility
            vol = df[col].rolling(self.window).std()
            # Rolling mean return
            mean_ret = df[col].rolling(self.window).mean()
            # Skewness
            skew = df[col].rolling(self.window).skew()
            # Kurtosis
            kurt = df[col].rolling(self.window).kurt()
            # Maximum drawdown in window
            rolling_max = df[col].expanding().max()
            drawdown = (df[col] - rolling_max) / rolling_max
            
            features.extend([vol, mean_ret, skew, kurt, drawdown])
        
        # Stack features
        feature_df = pd.concat(features, axis=1)
        feature_df = feature_df.dropna()
        
        return feature_df.values
    
    def _find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_k + 1, len(features) - 1)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(features, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        # Find k that maximizes silhouette score
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
        else:
            optimal_k = 3  # Default
        
        return optimal_k
    
    def fit(self, df: pd.DataFrame, sweep_mode: bool = False, 
            wf_mode: bool = False, fixed_k: Optional[int] = None):
        """Fit the regime detector."""
        # Create features
        features = self._create_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Determine k
        if fixed_k is not None:
            self.optimal_k_ = fixed_k
        elif sweep_mode or wf_mode:
            # Use default or previously determined k
            self.optimal_k_ = self.k if self.k else 3
        else:
            self.optimal_k_ = self._find_optimal_k(features_pca)
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.optimal_k_, random_state=42, n_init=10)
        self.regime_labels_ = self.kmeans.fit_predict(features_pca)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regimes for new data."""
        features = self._create_features(df)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        return self.kmeans.predict(features_pca)
    
    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime predictions to dataframe."""
        # Get regime for each date
        regimes = self.predict(df)
        
        # Create regime dataframe aligned with original index
        regime_df = pd.DataFrame(regimes, index=df.index[-len(regimes):], columns=["Regime"])
        
        # Merge with original dataframe
        result = df.copy()
        result["Regime"] = regime_df["Regime"]
        
        # Add regime names
        regime_names = {
            0: "Low Volatility",
            1: "High Volatility",
            2: "Trending",
            3: "Mean Reverting",
            4: "Crisis"
        }
        result["Regime_Name"] = result["Regime"].map(lambda x: regime_names.get(x, f"Regime {x}"))
        
        return result


class MomentumRanker:
    """Ranks assets by momentum within each regime."""
    
    def __init__(self, lookback: int = 63, target_etfs: List[str] = None):
        self.lookback = lookback
        self.target_etfs = target_etfs or []
        self.regime_rankings_ = {}
        self.regime_weights_ = {}
        
    def _calculate_momentum(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate momentum scores for each asset."""
        # Calculate returns over lookback period
        momentum = df.pct_change(lookback)
        
        # Calculate Sharpe-like ratio (return / volatility)
        vol = df.rolling(lookback).std()
        sharpe = momentum / vol
        
        # Calculate trend strength (linear regression slope)
        def trend_strength(series):
            if len(series) < lookback:
                return np.nan
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope / series.std() if series.std() > 0 else 0
        
        trend = df.rolling(lookback).apply(trend_strength)
        
        # Combine metrics
        momentum_score = 0.4 * momentum + 0.3 * sharpe + 0.3 * trend
        
        return momentum_score
    
    def fit(self, df: pd.DataFrame):
        """Fit the ranker for each regime."""
        # Get unique regimes
        if "Regime" not in df.columns:
            raise ValueError("DataFrame must have 'Regime' column. Run RegimeDetector first.")
        
        regimes = df["Regime"].unique()
        
        for regime in regimes:
            regime_df = df[df["Regime"] == regime]
            
            # Get ETF return columns
            ret_cols = [f"{etf}_Ret" for etf in self.target_etfs if f"{etf}_Ret" in df.columns]
            
            if len(regime_df) < self.lookback:
                # Not enough data, use equal weights
                self.regime_rankings_[regime] = pd.Series(0.5, index=self.target_etfs)
                self.regime_weights_[regime] = pd.Series(1/len(self.target_etfs), index=self.target_etfs)
                continue
            
            # Calculate momentum scores
            momentum_scores = self._calculate_momentum(regime_df[ret_cols], self.lookback)
            
            # Get latest momentum
            latest_scores = momentum_scores.iloc[-1]
            
            # Normalize to probabilities
            scores_positive = latest_scores - latest_scores.min() + 0.01
            probs = scores_positive / scores_positive.sum()
            
            # Store rankings and weights
            self.regime_rankings_[regime] = latest_scores.sort_values(ascending=False)
            self.regime_weights_[regime] = probs
        
        return self
    
    def predict(self, row: pd.Series) -> Dict:
        """Predict rankings for a single row."""
        regime = row.get("Regime", 0)
        
        if regime in self.regime_rankings_:
            rankings = self.regime_rankings_[regime]
            weights = self.regime_weights_[regime]
        else:
            # Default to equal weights
            rankings = pd.Series(0, index=self.target_etfs)
            weights = pd.Series(1/len(self.target_etfs), index=self.target_etfs)
        
        # Get top pick
        top_pick = rankings.index[0] if len(rankings) > 0 else None
        
        return {
            "Rank_Score": rankings,
            "Weights": weights,
            "Top_Pick": top_pick,
            "Regime": regime
        }
    
    def predict_all_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for entire history."""
        records = []
        index   = []

        for idx in range(self.lookback, len(df)):
            row  = df.iloc[idx]
            pred = self.predict(row)

            # Build a plain dict for this row — avoids pd.Series(index=Timestamp)
            # which crashes in pandas >= 2.0 when a scalar is passed as index.
            record = {}
            for etf in self.target_etfs:
                record[f"{etf}_Prob"] = pred["Weights"].get(etf, 0)
            record["Top_Pick"] = pred["Top_Pick"]
            record["Regime"]   = pred["Regime"]

            records.append(record)
            index.append(df.index[idx])

        if records:
            return pd.DataFrame(records, index=pd.DatetimeIndex(index))
        else:
            return pd.DataFrame()


def calculate_conviction_z(probabilities: np.ndarray) -> Tuple[float, str]:
    """
    Calculate conviction score based on probability distribution.
    Returns Z-score and label (High/Medium/Low).
    """
    if len(probabilities) == 0:
        return 0.0, "Low"
    
    # Calculate entropy of distribution (lower entropy = higher conviction)
    probs = np.array(probabilities)
    probs = probs / (probs.sum() + 1e-10)  # Normalize
    
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    
    # Convert to Z-score (1 - normalized entropy)
    if max_entropy > 0:
        conviction = 1 - (entropy / max_entropy)
    else:
        conviction = 0
    
    # Scale to Z-score (0-3 range)
    z_score = conviction * 3
    
    # Label
    if z_score > 2.0:
        label = "High"
    elif z_score > 1.0:
        label = "Medium"
    else:
        label = "Low"
    
    return z_score, label
