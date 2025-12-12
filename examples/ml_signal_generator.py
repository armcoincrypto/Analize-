#!/usr/bin/env python3
"""
ML Signal Generator - Machine Learning Based Trading Signals

Features:
1. Feature engineering from OHLCV data
2. Multiple ML models (Random Forest, XGBoost-like, Ensemble)
3. Walk-forward validation
4. Feature importance analysis
5. Prediction confidence scores
6. Model performance tracking

Note: Uses sklearn-compatible implementations for portability.
For production, consider XGBoost, LightGBM, or neural networks.

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MLSignal:
    """Machine learning generated signal."""
    symbol: str
    timestamp: datetime
    prediction: str  # BUY, SELL, HOLD
    probability_up: float
    probability_down: float
    confidence: float
    features_used: int
    model_name: str


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    total_trades: int
    winning_trades: int


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_name: str
    importance: float
    rank: int


@dataclass
class PredictionResult:
    """Detailed prediction result."""
    symbol: str
    date: datetime
    actual_return: Optional[float]
    predicted_direction: str
    predicted_probability: float
    was_correct: Optional[bool]


# =============================================================================
# DATA FETCHER
# =============================================================================

class MLDataFetcher:
    """Fetch data for ML models."""

    def __init__(self):
        self.cache = {}

    def fetch_ohlcv(self, symbol: str, days: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data."""
        cache_key = f"{symbol}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Binance US first (works in geo-restricted regions)
        df = self._fetch_binance_us(symbol, days)
        if df.empty:
            df = self._fetch_binance(symbol, days)

        if not df.empty:
            self.cache[cache_key] = df

        return df

    def _fetch_binance(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "limit": 1000
            }

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])

                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)

                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            pass

        return pd.DataFrame()

    def _fetch_binance_us(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance US."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "limit": 1000
            }

            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])

                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)

                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            pass

        return pd.DataFrame()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Generate features from OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)

    def generate_all_features(self) -> pd.DataFrame:
        """Generate all features."""
        self._add_returns()
        self._add_volatility_features()
        self._add_trend_features()
        self._add_momentum_features()
        self._add_volume_features()
        self._add_pattern_features()
        self._add_seasonality_features()

        # Combine with original data
        result = pd.concat([self.df, self.features], axis=1)
        return result

    def _add_returns(self):
        """Add return features."""
        close = self.df["close"]

        # Simple returns
        self.features["return_1d"] = close.pct_change(1)
        self.features["return_3d"] = close.pct_change(3)
        self.features["return_5d"] = close.pct_change(5)
        self.features["return_10d"] = close.pct_change(10)
        self.features["return_20d"] = close.pct_change(20)

        # Log returns
        self.features["log_return_1d"] = np.log(close / close.shift(1))

    def _add_volatility_features(self):
        """Add volatility features."""
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        # Historical volatility
        returns = close.pct_change()
        self.features["volatility_5d"] = returns.rolling(5).std()
        self.features["volatility_10d"] = returns.rolling(10).std()
        self.features["volatility_20d"] = returns.rolling(20).std()

        # Volatility ratio
        self.features["vol_ratio_5_20"] = (
            self.features["volatility_5d"] / self.features["volatility_20d"]
        )

        # True Range
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1))
        }).max(axis=1)
        self.features["atr_14"] = tr.rolling(14).mean()

        # ATR ratio
        self.features["atr_ratio"] = self.features["atr_14"] / close

        # Bollinger Band width
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        self.features["bb_width"] = (4 * std_20) / sma_20

        # Price position in BB
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        self.features["bb_position"] = (close - lower_band) / (upper_band - lower_band)

    def _add_trend_features(self):
        """Add trend features."""
        close = self.df["close"]

        # Moving averages
        self.features["sma_5"] = close.rolling(5).mean()
        self.features["sma_10"] = close.rolling(10).mean()
        self.features["sma_20"] = close.rolling(20).mean()
        self.features["sma_50"] = close.rolling(50).mean()
        self.features["sma_200"] = close.rolling(200).mean()

        # EMA
        self.features["ema_12"] = close.ewm(span=12).mean()
        self.features["ema_26"] = close.ewm(span=26).mean()

        # Price vs MAs
        self.features["price_sma_5_ratio"] = close / self.features["sma_5"]
        self.features["price_sma_20_ratio"] = close / self.features["sma_20"]
        self.features["price_sma_50_ratio"] = close / self.features["sma_50"]
        self.features["price_sma_200_ratio"] = close / self.features["sma_200"]

        # MA crossovers (as continuous variable)
        self.features["ma_5_20_diff"] = (
            (self.features["sma_5"] - self.features["sma_20"]) / self.features["sma_20"]
        )
        self.features["ma_50_200_diff"] = (
            (self.features["sma_50"] - self.features["sma_200"]) / self.features["sma_200"]
        )

        # MACD
        self.features["macd"] = self.features["ema_12"] - self.features["ema_26"]
        self.features["macd_signal"] = self.features["macd"].ewm(span=9).mean()
        self.features["macd_histogram"] = self.features["macd"] - self.features["macd_signal"]

        # Trend strength (ADX approximation)
        self.features["trend_strength"] = abs(self.features["ma_5_20_diff"]) * 100

    def _add_momentum_features(self):
        """Add momentum features."""
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        self.features["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic
        lowest_14 = low.rolling(14).min()
        highest_14 = high.rolling(14).max()
        self.features["stoch_k"] = (
            (close - lowest_14) / (highest_14 - lowest_14) * 100
        )
        self.features["stoch_d"] = self.features["stoch_k"].rolling(3).mean()

        # Rate of Change
        self.features["roc_5"] = (close / close.shift(5) - 1) * 100
        self.features["roc_10"] = (close / close.shift(10) - 1) * 100
        self.features["roc_20"] = (close / close.shift(20) - 1) * 100

        # Williams %R
        self.features["williams_r"] = (
            (highest_14 - close) / (highest_14 - lowest_14) * -100
        )

        # Momentum
        self.features["momentum_10"] = close - close.shift(10)
        self.features["momentum_20"] = close - close.shift(20)

    def _add_volume_features(self):
        """Add volume features."""
        volume = self.df["volume"]
        close = self.df["close"]

        # Volume moving averages
        self.features["volume_sma_5"] = volume.rolling(5).mean()
        self.features["volume_sma_20"] = volume.rolling(20).mean()

        # Volume ratio
        self.features["volume_ratio"] = volume / self.features["volume_sma_20"]

        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()
        self.features["obv"] = obv
        self.features["obv_sma_20"] = obv.rolling(20).mean()

        # Volume-Price Trend
        self.features["vpt"] = (volume * close.pct_change()).cumsum()

        # Money Flow Index
        typical_price = (self.df["high"] + self.df["low"] + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_sum = positive_flow.rolling(14).sum()
        negative_sum = negative_flow.rolling(14).sum()

        mfi = 100 - (100 / (1 + positive_sum / negative_sum))
        self.features["mfi_14"] = mfi

    def _add_pattern_features(self):
        """Add candlestick pattern features."""
        open_price = self.df["open"]
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # Body size
        body = abs(close - open_price)
        range_hl = high - low

        self.features["body_ratio"] = body / range_hl.replace(0, np.nan)

        # Upper/Lower shadow
        upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low

        self.features["upper_shadow_ratio"] = upper_shadow / range_hl.replace(0, np.nan)
        self.features["lower_shadow_ratio"] = lower_shadow / range_hl.replace(0, np.nan)

        # Bullish/Bearish candle
        self.features["is_bullish"] = (close > open_price).astype(int)

        # Consecutive up/down days
        direction = np.sign(close - close.shift(1))
        self.features["consecutive_up"] = (
            direction.groupby((direction != direction.shift()).cumsum()).cumsum()
        ).clip(lower=0)
        self.features["consecutive_down"] = (
            (-direction).groupby((direction != direction.shift()).cumsum()).cumsum()
        ).clip(lower=0)

        # Gap
        self.features["gap"] = (open_price - close.shift(1)) / close.shift(1)

    def _add_seasonality_features(self):
        """Add time-based features."""
        timestamps = self.df["timestamp"]

        # Day of week
        self.features["day_of_week"] = timestamps.dt.dayofweek
        self.features["is_monday"] = (timestamps.dt.dayofweek == 0).astype(int)
        self.features["is_friday"] = (timestamps.dt.dayofweek == 4).astype(int)

        # Month
        self.features["month"] = timestamps.dt.month
        self.features["is_january"] = (timestamps.dt.month == 1).astype(int)
        self.features["is_december"] = (timestamps.dt.month == 12).astype(int)

        # Day of month
        self.features["day_of_month"] = timestamps.dt.day
        self.features["is_month_start"] = (timestamps.dt.day <= 5).astype(int)
        self.features["is_month_end"] = (timestamps.dt.day >= 25).astype(int)


# =============================================================================
# ML MODELS (from scratch for portability)
# =============================================================================

class DecisionTreeClassifier:
    """Simple decision tree classifier."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree."""
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return np.array([self._predict_proba_single(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        """Recursively build tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree
        }

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find best split using Gini impurity."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gini = self._calculate_gini(y[left_mask], y[right_mask])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate weighted Gini impurity."""
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        def gini(y):
            if len(y) == 0:
                return 0
            probs = np.bincount(y) / len(y)
            return 1 - np.sum(probs ** 2)

        return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)

    def _calculate_leaf_value(self, y: np.ndarray) -> Dict:
        """Calculate class probabilities for leaf."""
        counts = np.bincount(y, minlength=2)
        probs = counts / counts.sum()
        return {"probs": probs, "class": np.argmax(counts)}

    def _predict_single(self, x: np.ndarray, node: Dict) -> int:
        """Predict single sample."""
        if node["leaf"]:
            return node["value"]["class"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(x, node["left"])
        return self._predict_single(x, node["right"])

    def _predict_proba_single(self, x: np.ndarray, node: Dict) -> np.ndarray:
        """Predict probabilities for single sample."""
        if node["leaf"]:
            return node["value"]["probs"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_proba_single(x, node["left"])
        return self._predict_proba_single(x, node["right"])


class RandomForestClassifier:
    """Random Forest classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 min_samples_split: int = 10, max_features: str = "sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit random forest."""
        n_samples, n_features = X.shape

        # Determine number of features to use
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        importances = np.zeros(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Random feature subset
            feature_idx = np.random.choice(n_features, max_features, replace=False)
            X_subset = X_boot[:, feature_idx]

            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, y_boot)

            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

            # Accumulate feature importances (simplified)
            for fi in feature_idx:
                importances[fi] += 1

        self.feature_importances_ = importances / importances.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        predictions = []

        for tree, feature_idx in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_idx]
            pred = tree.predict_proba(X_subset)
            predictions.append(pred)

        return np.mean(predictions, axis=0)


class GradientBoostingClassifier:
    """Simple gradient boosting classifier."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit gradient boosting."""
        # Initial prediction (log-odds)
        pos_rate = y.mean()
        self.initial_prediction = np.log(pos_rate / (1 - pos_rate)) if 0 < pos_rate < 1 else 0

        # Current predictions
        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            # Calculate probabilities
            probs = 1 / (1 + np.exp(-F))

            # Calculate residuals (negative gradient)
            residuals = y - probs

            # Fit tree to residuals (as regression)
            tree = self._fit_regression_tree(X, residuals)
            self.trees.append(tree)

            # Update predictions
            predictions = self._predict_tree(tree, X)
            F += self.learning_rate * predictions

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        F = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            F += self.learning_rate * self._predict_tree(tree, X)

        probs = 1 / (1 + np.exp(-F))
        return np.column_stack([1 - probs, probs])

    def _fit_regression_tree(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit a simple regression tree."""
        return self._build_reg_tree(X, y, depth=0)

    def _build_reg_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        """Build regression tree recursively."""
        if depth >= self.max_depth or len(y) < 10:
            return {"leaf": True, "value": y.mean()}

        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.percentile(X[:, feature], [25, 50, 75])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 5 or right_mask.sum() < 5:
                    continue

                mse = (np.var(y[left_mask]) * left_mask.sum() +
                       np.var(y[right_mask]) * right_mask.sum()) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return {"leaf": True, "value": y.mean()}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_reg_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_reg_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """Predict using regression tree."""
        return np.array([self._predict_single(tree, x) for x in X])

    def _predict_single(self, node: Dict, x: np.ndarray) -> float:
        """Predict single sample."""
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(node["left"], x)
        return self._predict_single(node["right"], x)


# =============================================================================
# ML SIGNAL GENERATOR
# =============================================================================

class MLSignalGenerator:
    """Generate trading signals using ML models."""

    def __init__(self):
        self.fetcher = MLDataFetcher()
        self.models = {}
        self.feature_names = []

    def train_models(self, symbol: str, lookback_days: int = 365,
                     prediction_horizon: int = 5) -> Dict[str, ModelPerformance]:
        """Train multiple ML models."""
        # Fetch and prepare data
        df = self.fetcher.fetch_ohlcv(symbol, lookback_days + 100)

        if len(df) < 200:
            return {}

        # Generate features
        engineer = FeatureEngineer(df)
        features_df = engineer.generate_all_features()

        # Create target (future returns > 0)
        features_df["target"] = (
            features_df["close"].shift(-prediction_horizon) /
            features_df["close"] - 1
        )
        features_df["target_class"] = (features_df["target"] > 0).astype(int)

        # Drop rows with NaN
        features_df = features_df.dropna()

        if len(features_df) < 100:
            return {}

        # Select feature columns
        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume",
                        "target", "target_class"]
        self.feature_names = [c for c in features_df.columns if c not in exclude_cols]

        # Prepare data
        X = features_df[self.feature_names].values
        y = features_df["target_class"].values

        # Train-test split (80-20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Handle NaN/Inf values
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # Train models
        performance = {}

        # Random Forest
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=5)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        performance["RandomForest"] = self._evaluate_model(rf, X_test, y_test, features_df.iloc[split_idx:])

        # Gradient Boosting
        print("  Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3)
        gb.fit(X_train, y_train)
        self.models["GradientBoosting"] = gb
        performance["GradientBoosting"] = self._evaluate_model(gb, X_test, y_test, features_df.iloc[split_idx:])

        return performance

    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                        test_df: pd.DataFrame) -> ModelPerformance:
        """Evaluate model performance."""
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # Basic metrics
        accuracy = (predictions == y_test).mean()

        # Precision/Recall for positive class
        true_pos = ((predictions == 1) & (y_test == 1)).sum()
        false_pos = ((predictions == 1) & (y_test == 0)).sum()
        false_neg = ((predictions == 0) & (y_test == 1)).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Trading performance
        returns = test_df["target"].values
        trade_returns = returns[predictions == 1]

        if len(trade_returns) > 0:
            winning = (trade_returns > 0).sum()
            profit_factor = (
                trade_returns[trade_returns > 0].sum() /
                abs(trade_returns[trade_returns < 0].sum())
            ) if (trade_returns < 0).sum() != 0 else float('inf')
        else:
            winning = 0
            profit_factor = 0

        return ModelPerformance(
            model_name=model.__class__.__name__,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            profit_factor=profit_factor,
            total_trades=len(trade_returns),
            winning_trades=winning
        )

    def generate_signal(self, symbol: str) -> Optional[MLSignal]:
        """Generate signal for current market conditions."""
        if not self.models:
            return None

        # Fetch recent data
        df = self.fetcher.fetch_ohlcv(symbol, days=100)

        if len(df) < 50:
            return None

        # Generate features
        engineer = FeatureEngineer(df)
        features_df = engineer.generate_all_features()

        # Get latest row
        latest = features_df.iloc[-1]

        # Prepare features
        X = latest[self.feature_names].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Get predictions from all models
        predictions = {}
        probabilities = {}

        for name, model in self.models.items():
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            predictions[name] = pred
            probabilities[name] = proba

        # Ensemble prediction (average probabilities)
        avg_prob_up = np.mean([p[1] for p in probabilities.values()])
        avg_prob_down = 1 - avg_prob_up

        # Determine signal
        if avg_prob_up > 0.6:
            prediction = "BUY"
        elif avg_prob_up < 0.4:
            prediction = "SELL"
        else:
            prediction = "HOLD"

        # Confidence based on agreement
        agreement = np.std([p[1] for p in probabilities.values()])
        confidence = 1 - min(1, agreement * 2)

        return MLSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction=prediction,
            probability_up=avg_prob_up,
            probability_down=avg_prob_down,
            confidence=confidence,
            features_used=len(self.feature_names),
            model_name="Ensemble"
        )

    def get_feature_importance(self) -> List[FeatureImportance]:
        """Get feature importance from models."""
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    feat_name = self.feature_names[i]
                    if feat_name not in importances:
                        importances[feat_name] = []
                    importances[feat_name].append(imp)

        # Average importances
        avg_importances = {
            feat: np.mean(imps)
            for feat, imps in importances.items()
        }

        # Sort and rank
        sorted_features = sorted(avg_importances.items(), key=lambda x: -x[1])

        return [
            FeatureImportance(
                feature_name=feat,
                importance=imp,
                rank=i + 1
            )
            for i, (feat, imp) in enumerate(sorted_features)
        ]


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """Walk-forward validation for ML models."""

    def __init__(self, train_months: int = 6, test_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months

    def validate(self, symbol: str) -> List[ModelPerformance]:
        """Perform walk-forward validation."""
        fetcher = MLDataFetcher()
        df = fetcher.fetch_ohlcv(symbol, days=500)

        if len(df) < 300:
            return []

        # Generate features
        engineer = FeatureEngineer(df)
        features_df = engineer.generate_all_features()

        # Create target
        features_df["target"] = (
            features_df["close"].shift(-5) / features_df["close"] - 1
        )
        features_df["target_class"] = (features_df["target"] > 0).astype(int)
        features_df = features_df.dropna()

        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume",
                        "target", "target_class"]
        feature_names = [c for c in features_df.columns if c not in exclude_cols]

        # Walk-forward periods
        results = []
        total_days = len(features_df)
        train_days = self.train_months * 21
        test_days = self.test_months * 21

        all_predictions = []
        all_actuals = []

        period = 0
        start_idx = 0

        while start_idx + train_days + test_days <= total_days:
            train_end = start_idx + train_days
            test_end = train_end + test_days

            # Get train/test data
            train_data = features_df.iloc[start_idx:train_end]
            test_data = features_df.iloc[train_end:test_end]

            X_train = train_data[feature_names].values
            y_train = train_data["target_class"].values
            X_test = test_data[feature_names].values
            y_test = test_data["target_class"].values

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
            X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

            # Train model
            model = RandomForestClassifier(n_estimators=30, max_depth=4)
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)

            # Move forward
            start_idx += test_days
            period += 1

        if not all_predictions:
            return []

        # Calculate overall performance
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        accuracy = (all_predictions == all_actuals).mean()
        true_pos = ((all_predictions == 1) & (all_actuals == 1)).sum()
        false_pos = ((all_predictions == 1) & (all_actuals == 0)).sum()
        false_neg = ((all_predictions == 0) & (all_actuals == 1)).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return [ModelPerformance(
            model_name="Walk-Forward RF",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            profit_factor=0,  # Would need returns data
            total_trades=all_predictions.sum(),
            winning_trades=true_pos
        )]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_ml_signal_generation():
    """Run ML signal generation."""
    print("=" * 70)
    print("ML SIGNAL GENERATOR - Machine Learning Trading Signals")
    print("=" * 70)
    print()

    generator = MLSignalGenerator()
    symbols = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {symbol}")
        print("=" * 70)

        # Train models
        print("\n[1] TRAINING MODELS...")
        print("-" * 50)

        performance = generator.train_models(symbol, lookback_days=365)

        if not performance:
            print(f"  Insufficient data for {symbol}")
            continue

        for name, perf in performance.items():
            print(f"\n  {name}:")
            print(f"    Accuracy: {perf.accuracy:.1%}")
            print(f"    Precision: {perf.precision:.1%}")
            print(f"    Recall: {perf.recall:.1%}")
            print(f"    F1 Score: {perf.f1_score:.2f}")
            print(f"    Total Trades: {perf.total_trades}")
            print(f"    Winning Trades: {perf.winning_trades}")
            if perf.profit_factor != float('inf'):
                print(f"    Profit Factor: {perf.profit_factor:.2f}")

        # Generate signal
        print("\n[2] CURRENT SIGNAL")
        print("-" * 50)

        signal = generator.generate_signal(symbol)

        if signal:
            signal_emoji = {
                "BUY": "🟢",
                "SELL": "🔴",
                "HOLD": "⚪"
            }

            print(f"  Signal: {signal_emoji.get(signal.prediction, '')} {signal.prediction}")
            print(f"  Probability Up: {signal.probability_up:.1%}")
            print(f"  Probability Down: {signal.probability_down:.1%}")
            print(f"  Confidence: {signal.confidence:.1%}")
            print(f"  Features Used: {signal.features_used}")

        # Feature importance
        print("\n[3] TOP FEATURES")
        print("-" * 50)

        importances = generator.get_feature_importance()

        for fi in importances[:10]:
            bar_len = int(fi.importance * 100)
            print(f"  {fi.rank:2d}. {fi.feature_name:25s} {'█' * bar_len}")

        # Walk-forward validation
        print("\n[4] WALK-FORWARD VALIDATION")
        print("-" * 50)

        validator = WalkForwardValidator(train_months=6, test_months=1)
        wf_results = validator.validate(symbol)

        if wf_results:
            wf = wf_results[0]
            print(f"  Accuracy: {wf.accuracy:.1%}")
            print(f"  Precision: {wf.precision:.1%}")
            print(f"  Recall: {wf.recall:.1%}")
            print(f"  F1 Score: {wf.f1_score:.2f}")
        else:
            print("  Insufficient data for walk-forward validation")

    # Summary
    print("\n" + "=" * 70)
    print("ML SIGNALS SUMMARY")
    print("=" * 70)

    for symbol in symbols:
        generator_temp = MLSignalGenerator()
        perf = generator_temp.train_models(symbol, lookback_days=365)

        if perf:
            signal = generator_temp.generate_signal(symbol)
            if signal:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
                print(f"  {symbol}: {emoji.get(signal.prediction, '')} {signal.prediction}")
                print(f"    Confidence: {signal.confidence:.0%}")
                print(f"    P(Up): {signal.probability_up:.0%}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nNote: ML models implemented from scratch for portability.")
    print("For production, consider using scikit-learn, XGBoost, or LightGBM.")


if __name__ == "__main__":
    run_ml_signal_generation()
