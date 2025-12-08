"""
Feature generator for creating ML-ready datasets.

Orchestrates indicator calculations across multiple timeframes
and combines them into feature sets for signals.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analize.features.indicators import TechnicalIndicators
from analize.models.signals import (
    CandleColor,
    HTFIndicators,
    LTFIndicators,
    OrderbookSnapshot,
    SignalRecord,
)


class FeatureGenerator:
    """Generates features for signal records."""

    # Default indicator parameters
    DEFAULT_PARAMS = {
        "ema_periods": [20, 50],
        "rsi_period": 14,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "volume_zscore_period": 20,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def compute_indicators(
        self,
        df: pd.DataFrame,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Compute all indicators for a DataFrame.

        Args:
            df: DataFrame with OHLCV columns
            prefix: Prefix for column names (e.g., "1h_")

        Returns:
            DataFrame with indicator columns added
        """
        result = df.copy()

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in result.columns:
                raise ValueError(f"Missing required column: {col}")

        # EMA
        for period in self.params["ema_periods"]:
            result[f"{prefix}ema_{period}"] = TechnicalIndicators.ema(result["close"], period)
            result[f"{prefix}ema_{period}_slope"] = TechnicalIndicators.ema_slope(
                result["close"], period
            )

        # RSI
        result[f"{prefix}rsi"] = TechnicalIndicators.rsi(
            result["close"], self.params["rsi_period"]
        )

        # ATR
        result[f"{prefix}atr"] = TechnicalIndicators.atr(
            result["high"],
            result["low"],
            result["close"],
            self.params["atr_period"],
        )
        result[f"{prefix}atr_pct"] = TechnicalIndicators.atr_percent(
            result["high"],
            result["low"],
            result["close"],
            self.params["atr_period"],
        )

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            result["close"],
            self.params["bb_period"],
            self.params["bb_std"],
        )
        result[f"{prefix}bb_upper"] = bb_upper
        result[f"{prefix}bb_middle"] = bb_middle
        result[f"{prefix}bb_lower"] = bb_lower
        result[f"{prefix}bb_width"] = TechnicalIndicators.bb_width(
            result["close"],
            self.params["bb_period"],
            self.params["bb_std"],
        )
        result[f"{prefix}bb_expand_rate"] = TechnicalIndicators.bb_expansion_rate(
            result["close"],
            self.params["bb_period"],
            self.params["bb_std"],
        )

        # Volume
        result[f"{prefix}vol_zscore"] = TechnicalIndicators.volume_zscore(
            result["volume"],
            self.params["volume_zscore_period"],
        )
        result[f"{prefix}vol_ratio"] = TechnicalIndicators.volume_ratio(
            result["volume"],
            self.params["volume_zscore_period"],
        )

        # VWAP
        result[f"{prefix}vwap"] = TechnicalIndicators.vwap(
            result["high"],
            result["low"],
            result["close"],
            result["volume"],
        )
        result[f"{prefix}vwap_dist"] = TechnicalIndicators.vwap_distance(
            result["close"],
            result["high"],
            result["low"],
            result["volume"],
        )

        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(result["close"])
        result[f"{prefix}macd"] = macd_line
        result[f"{prefix}macd_signal"] = signal_line
        result[f"{prefix}macd_hist"] = histogram

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            result["high"],
            result["low"],
            result["close"],
        )
        result[f"{prefix}stoch_k"] = stoch_k
        result[f"{prefix}stoch_d"] = stoch_d

        # Candle features
        result[f"{prefix}candle_body_pct"] = TechnicalIndicators.candle_body_percent(
            result["open"],
            result["high"],
            result["low"],
            result["close"],
        )
        result[f"{prefix}candle_color"] = TechnicalIndicators.candle_color(
            result["open"],
            result["close"],
        )

        # Price changes
        for periods in [1, 5, 10]:
            result[f"{prefix}price_change_{periods}"] = TechnicalIndicators.price_change_percent(
                result["close"], periods
            )

        return result

    def compute_htf_indicators(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute higher timeframe indicators.

        Args:
            df_1h: 1-hour OHLCV DataFrame
            df_4h: 4-hour OHLCV DataFrame (optional)

        Returns:
            DataFrame with HTF indicators
        """
        # Compute 1H indicators
        result = self.compute_indicators(df_1h, prefix="1h_")

        # Compute 4H indicators if provided
        if df_4h is not None:
            df_4h_ind = self.compute_indicators(df_4h, prefix="4h_")

            # Merge 4H indicators (forward fill to align with 1H)
            for col in df_4h_ind.columns:
                if col.startswith("4h_"):
                    result[col] = df_4h_ind[col].reindex(result.index, method="ffill")

        return result

    def compute_ltf_indicators(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Compute lower timeframe indicators.

        Args:
            df_1m: 1-minute OHLCV DataFrame

        Returns:
            DataFrame with LTF indicators
        """
        return self.compute_indicators(df_1m, prefix="1m_")

    def get_htf_at_timestamp(
        self,
        htf_df: pd.DataFrame,
        timestamp: datetime,
    ) -> HTFIndicators:
        """
        Get HTF indicators at a specific timestamp.

        Args:
            htf_df: DataFrame with HTF indicators
            timestamp: Target timestamp

        Returns:
            HTFIndicators object
        """
        # Find the most recent row at or before the timestamp
        mask = htf_df.index <= timestamp
        if not mask.any():
            return HTFIndicators()

        row = htf_df.loc[mask].iloc[-1]

        return HTFIndicators(
            ema_1h_20=row.get("1h_ema_20"),
            ema_1h_50=row.get("1h_ema_50"),
            ema_1h_slope=row.get("1h_ema_20_slope"),
            ema_4h_20=row.get("4h_ema_20"),
            rsi_1h=row.get("1h_rsi"),
            rsi_4h=row.get("4h_rsi"),
            atr_1h=row.get("1h_atr"),
            atr_1h_pct=row.get("1h_atr_pct"),
            atr_4h=row.get("4h_atr"),
            bb_1h_upper=row.get("1h_bb_upper"),
            bb_1h_lower=row.get("1h_bb_lower"),
            bb_1h_width=row.get("1h_bb_width"),
        )

    def get_ltf_at_timestamp(
        self,
        ltf_df: pd.DataFrame,
        timestamp: datetime,
    ) -> LTFIndicators:
        """
        Get LTF indicators at a specific timestamp.

        Args:
            ltf_df: DataFrame with LTF indicators
            timestamp: Target timestamp

        Returns:
            LTFIndicators object
        """
        mask = ltf_df.index <= timestamp
        if not mask.any():
            return LTFIndicators()

        row = ltf_df.loc[mask].iloc[-1]

        return LTFIndicators(
            rsi_1m=row.get("1m_rsi"),
            rsi_5m=row.get("5m_rsi"),
            atr_1m=row.get("1m_atr"),
            atr_5m=row.get("5m_atr"),
            bb_width=row.get("1m_bb_width"),
            bb_expand_rate=row.get("1m_bb_expand_rate"),
            volume_zscore_20=row.get("1m_vol_zscore"),
            volume_ratio=row.get("1m_vol_ratio"),
            vwap=row.get("1m_vwap"),
            vwap_distance_pct=row.get("1m_vwap_dist"),
        )

    def compute_orderbook_features(
        self,
        orderbook: OrderbookSnapshot,
    ) -> dict[str, float | None]:
        """
        Compute features from orderbook snapshot.

        Args:
            orderbook: OrderbookSnapshot object

        Returns:
            Dictionary of orderbook features
        """
        return {
            "spread": orderbook.spread,
            "spread_bps": orderbook.spread_bps,
            "bid": orderbook.bids[0].price if orderbook.bids else None,
            "ask": orderbook.asks[0].price if orderbook.asks else None,
            "orderbook_imbalance_top5": orderbook.imbalance_top5,
        }

    def enrich_signal_record(
        self,
        signal: SignalRecord,
        htf_df: pd.DataFrame | None = None,
        ltf_df: pd.DataFrame | None = None,
        orderbook: OrderbookSnapshot | None = None,
    ) -> SignalRecord:
        """
        Enrich a signal record with computed features.

        Args:
            signal: Signal record to enrich
            htf_df: DataFrame with HTF indicators
            ltf_df: DataFrame with LTF indicators
            orderbook: Orderbook snapshot at signal time

        Returns:
            Enriched SignalRecord
        """
        # Create a copy to avoid modifying original
        data = signal.model_dump()

        # Add HTF indicators
        if htf_df is not None and len(htf_df) > 0:
            htf = self.get_htf_at_timestamp(htf_df, signal.timestamp_utc)
            data["ema_1h_20"] = htf.ema_1h_20
            data["ema_1h_slope"] = htf.ema_1h_slope
            data["rsi_1h"] = htf.rsi_1h
            data["atr_1h"] = htf.atr_1h
            data["atr_1h_pct"] = htf.atr_1h_pct
            data["rsi_4h"] = htf.rsi_4h
            data["atr_4h"] = htf.atr_4h

        # Add LTF indicators
        if ltf_df is not None and len(ltf_df) > 0:
            ltf = self.get_ltf_at_timestamp(ltf_df, signal.timestamp_utc)
            data["rsi_1m"] = ltf.rsi_1m
            data["atr_1m"] = ltf.atr_1m
            data["bb_width"] = ltf.bb_width
            data["bb_expand_rate"] = ltf.bb_expand_rate
            data["vol_zscore_20"] = ltf.volume_zscore_20

        # Add orderbook features
        if orderbook is not None:
            ob_features = self.compute_orderbook_features(orderbook)
            data["spread"] = ob_features["spread"]
            data["spread_bps"] = ob_features["spread_bps"]
            data["bid"] = ob_features["bid"]
            data["ask"] = ob_features["ask"]
            data["orderbook_imbalance_top5"] = ob_features["orderbook_imbalance_top5"]

        # Add candle features
        if signal.price_high > 0 and signal.price_low > 0:
            range_val = signal.price_high - signal.price_low
            if range_val > 0:
                body = abs(signal.price_close - signal.price_open)
                data["candle_body_pct"] = (body / range_val) * 100

            if signal.price_close > signal.price_open:
                data["candle_color"] = CandleColor.GREEN
            elif signal.price_close < signal.price_open:
                data["candle_color"] = CandleColor.RED
            else:
                data["candle_color"] = CandleColor.DOJI

        return SignalRecord(**data)

    def process_signals_batch(
        self,
        signals: list[SignalRecord],
        market_data: dict[str, pd.DataFrame],
    ) -> list[SignalRecord]:
        """
        Process a batch of signals, enriching with features.

        Args:
            signals: List of signal records
            market_data: Dict of market data DataFrames keyed by timeframe
                         e.g., {"1m": df_1m, "1h": df_1h, "4h": df_4h}

        Returns:
            List of enriched signal records
        """
        enriched = []

        # Pre-compute indicators for each timeframe
        indicator_dfs = {}
        for tf, df in market_data.items():
            if df is not None and len(df) > 0:
                df_indexed = df.set_index("timestamp") if "timestamp" in df.columns else df
                indicator_dfs[tf] = self.compute_indicators(df_indexed, prefix=f"{tf}_")

        # Get HTF DataFrame
        htf_df = None
        if "1h" in indicator_dfs:
            htf_df = indicator_dfs["1h"]
            if "4h" in indicator_dfs:
                for col in indicator_dfs["4h"].columns:
                    if col.startswith("4h_"):
                        htf_df[col] = indicator_dfs["4h"][col].reindex(
                            htf_df.index, method="ffill"
                        )

        # Get LTF DataFrame
        ltf_df = indicator_dfs.get("1m")

        # Process each signal
        for signal in signals:
            enriched_signal = self.enrich_signal_record(
                signal,
                htf_df=htf_df,
                ltf_df=ltf_df,
            )
            enriched.append(enriched_signal)

        return enriched

    def generate_feature_matrix(
        self,
        signals: list[SignalRecord],
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate a feature matrix from signal records.

        Args:
            signals: List of signal records
            include_columns: Only include these columns (None = all)
            exclude_columns: Exclude these columns

        Returns:
            DataFrame with features as columns
        """
        # Convert to DataFrame
        records = [s.model_dump() for s in signals]
        df = pd.DataFrame(records)

        # Handle column filtering
        if include_columns:
            df = df[[c for c in include_columns if c in df.columns]]
        if exclude_columns:
            df = df.drop(columns=[c for c in exclude_columns if c in df.columns], errors="ignore")

        # Convert categorical columns
        categorical_cols = ["mode", "policy_mode", "exit_reason", "candle_color"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        return df
