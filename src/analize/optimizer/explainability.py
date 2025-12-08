"""
Explainability for optimizer suggestions.

Provides feature attribution and explains why parameter changes
are expected to improve performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analize.features.indicators import TechnicalIndicators


@dataclass
class FeatureAttribution:
    """Attribution of a feature to performance improvement."""

    feature_name: str
    importance_score: float  # 0-1, how important this feature is
    direction: str  # "positive" or "negative"
    condition: str  # Human-readable condition (e.g., "volume_z > 1.8")
    hit_rate: float  # % of winning trades matching this condition
    sample_size: int  # Number of trades matching condition
    confidence: float  # Statistical confidence (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "importance_score": round(self.importance_score, 4),
            "direction": self.direction,
            "condition": self.condition,
            "hit_rate": round(self.hit_rate, 2),
            "sample_size": self.sample_size,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class SuggestionExplanation:
    """Complete explanation for an optimizer suggestion."""

    suggestion_id: str
    parameter_name: str
    current_value: Any
    proposed_value: Any
    symbol: str | None

    # Performance impact
    expected_win_rate_delta: float
    expected_profit_factor_delta: float
    expected_trade_count_delta: int

    # Feature attributions (what drove the improvement)
    feature_attributions: list[FeatureAttribution]

    # Summary explanation
    primary_driver: str  # Main reason for improvement
    secondary_drivers: list[str]  # Additional contributing factors
    tradeoffs: list[str]  # Potential downsides

    # Confidence metrics
    overall_confidence: float
    sample_size: int
    backtest_periods: int

    # Generated explanation text
    explanation_text: str

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "symbol": self.symbol,
            "expected_impact": {
                "win_rate_delta": round(self.expected_win_rate_delta, 2),
                "profit_factor_delta": round(self.expected_profit_factor_delta, 4),
                "trade_count_delta": self.expected_trade_count_delta,
            },
            "feature_attributions": [fa.to_dict() for fa in self.feature_attributions],
            "primary_driver": self.primary_driver,
            "secondary_drivers": self.secondary_drivers,
            "tradeoffs": self.tradeoffs,
            "confidence": {
                "overall": round(self.overall_confidence, 4),
                "sample_size": self.sample_size,
                "backtest_periods": self.backtest_periods,
            },
            "explanation_text": self.explanation_text,
            "created_at": self.created_at.isoformat(),
        }


class SuggestionExplainer:
    """
    Generates explanations for optimizer suggestions.

    Analyzes which features drove the performance improvement and
    provides human-readable explanations.
    """

    # Features to analyze for attribution
    ANALYZABLE_FEATURES = [
        "volume_z",
        "spread_pct",
        "rsi_14",
        "atr_pct",
        "bb_position",
        "ema_trend",
        "macd_histogram",
        "orderbook_imbalance",
        "volatility_regime",
        "time_of_day",
        "day_of_week",
    ]

    # Thresholds for feature conditions
    FEATURE_THRESHOLDS = {
        "volume_z": [0.5, 1.0, 1.5, 2.0, 2.5],
        "spread_pct": [0.01, 0.02, 0.05, 0.1],
        "rsi_14": [20, 30, 40, 50, 60, 70, 80],
        "atr_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
        "bb_position": [-1.0, -0.5, 0, 0.5, 1.0],
        "ema_trend": [-2.0, -1.0, 0, 1.0, 2.0],
        "orderbook_imbalance": [-0.5, -0.2, 0, 0.2, 0.5],
    }

    def __init__(self, min_sample_size: int = 30, min_confidence: float = 0.6):
        """
        Initialize the explainer.

        Args:
            min_sample_size: Minimum trades to consider a condition significant
            min_confidence: Minimum confidence to include in explanation
        """
        self.min_sample_size = min_sample_size
        self.min_confidence = min_confidence

    def explain_suggestion(
        self,
        df: pd.DataFrame,
        parameter_name: str,
        current_value: Any,
        proposed_value: Any,
        current_metrics: dict[str, float],
        proposed_metrics: dict[str, float],
        symbol: str | None = None,
    ) -> SuggestionExplanation:
        """
        Generate explanation for a parameter suggestion.

        Args:
            df: DataFrame with trade data and features
            parameter_name: Name of the parameter being changed
            current_value: Current parameter value
            proposed_value: Proposed new value
            current_metrics: Performance metrics with current value
            proposed_metrics: Performance metrics with proposed value
            symbol: Optional symbol filter

        Returns:
            Complete explanation for the suggestion
        """
        # Filter by symbol if provided
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()

        # Calculate performance deltas
        win_rate_delta = proposed_metrics.get("win_rate", 0) - current_metrics.get("win_rate", 0)
        pf_delta = proposed_metrics.get("profit_factor", 0) - current_metrics.get("profit_factor", 0)
        trade_delta = int(proposed_metrics.get("trade_count", 0) - current_metrics.get("trade_count", 0))

        # Analyze feature attributions
        attributions = self._analyze_feature_attributions(
            df=df,
            parameter_name=parameter_name,
            current_value=current_value,
            proposed_value=proposed_value,
        )

        # Sort by importance
        attributions.sort(key=lambda x: x.importance_score, reverse=True)

        # Extract primary and secondary drivers
        primary_driver = self._extract_primary_driver(attributions, parameter_name, current_value, proposed_value)
        secondary_drivers = self._extract_secondary_drivers(attributions)
        tradeoffs = self._identify_tradeoffs(df, parameter_name, proposed_value, trade_delta)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            attributions=attributions,
            sample_size=len(df),
            win_rate_delta=win_rate_delta,
        )

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            parameter_name=parameter_name,
            current_value=current_value,
            proposed_value=proposed_value,
            symbol=symbol,
            win_rate_delta=win_rate_delta,
            pf_delta=pf_delta,
            trade_delta=trade_delta,
            attributions=attributions,
            tradeoffs=tradeoffs,
        )

        # Create suggestion ID
        suggestion_id = f"EXPL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return SuggestionExplanation(
            suggestion_id=suggestion_id,
            parameter_name=parameter_name,
            current_value=current_value,
            proposed_value=proposed_value,
            symbol=symbol,
            expected_win_rate_delta=win_rate_delta,
            expected_profit_factor_delta=pf_delta,
            expected_trade_count_delta=trade_delta,
            feature_attributions=attributions[:10],  # Top 10 attributions
            primary_driver=primary_driver,
            secondary_drivers=secondary_drivers[:3],
            tradeoffs=tradeoffs,
            overall_confidence=overall_confidence,
            sample_size=len(df),
            backtest_periods=1,
            explanation_text=explanation_text,
        )

    def _analyze_feature_attributions(
        self,
        df: pd.DataFrame,
        parameter_name: str,
        current_value: Any,
        proposed_value: Any,
    ) -> list[FeatureAttribution]:
        """Analyze which features drove the performance improvement."""
        attributions = []

        if "pnl_pct" not in df.columns:
            return attributions

        # Determine if we're tightening or loosening the parameter
        param_direction = self._get_parameter_direction(parameter_name, current_value, proposed_value)

        for feature in self.ANALYZABLE_FEATURES:
            if feature not in df.columns:
                continue

            feature_attrs = self._analyze_single_feature(
                df=df,
                feature=feature,
                param_direction=param_direction,
            )
            attributions.extend(feature_attrs)

        return attributions

    def _analyze_single_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        param_direction: str,
    ) -> list[FeatureAttribution]:
        """Analyze a single feature for attribution."""
        attributions = []

        thresholds = self.FEATURE_THRESHOLDS.get(feature, self._auto_thresholds(df[feature]))

        for i, threshold in enumerate(thresholds):
            # Test condition: feature > threshold
            mask_above = df[feature] > threshold
            if mask_above.sum() >= self.min_sample_size:
                attr = self._evaluate_condition(
                    df=df,
                    mask=mask_above,
                    feature=feature,
                    condition=f"{feature} > {threshold}",
                    direction="positive",
                )
                if attr and attr.confidence >= self.min_confidence:
                    attributions.append(attr)

            # Test condition: feature < threshold
            mask_below = df[feature] < threshold
            if mask_below.sum() >= self.min_sample_size:
                attr = self._evaluate_condition(
                    df=df,
                    mask=mask_below,
                    feature=feature,
                    condition=f"{feature} < {threshold}",
                    direction="negative",
                )
                if attr and attr.confidence >= self.min_confidence:
                    attributions.append(attr)

            # Test range conditions
            if i < len(thresholds) - 1:
                next_threshold = thresholds[i + 1]
                mask_range = (df[feature] >= threshold) & (df[feature] < next_threshold)
                if mask_range.sum() >= self.min_sample_size:
                    attr = self._evaluate_condition(
                        df=df,
                        mask=mask_range,
                        feature=feature,
                        condition=f"{threshold} <= {feature} < {next_threshold}",
                        direction="range",
                    )
                    if attr and attr.confidence >= self.min_confidence:
                        attributions.append(attr)

        return attributions

    def _evaluate_condition(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        feature: str,
        condition: str,
        direction: str,
    ) -> FeatureAttribution | None:
        """Evaluate a specific condition for attribution."""
        subset = df[mask]
        if len(subset) < self.min_sample_size:
            return None

        # Calculate win rate for this condition
        wins = (subset["pnl_pct"] > 0).sum()
        win_rate = wins / len(subset) * 100

        # Calculate overall win rate
        overall_wins = (df["pnl_pct"] > 0).sum()
        overall_win_rate = overall_wins / len(df) * 100

        # Calculate lift (how much better this condition performs)
        lift = (win_rate - overall_win_rate) / overall_win_rate if overall_win_rate > 0 else 0

        # Only include if this condition is meaningfully different
        if abs(lift) < 0.05:  # Less than 5% lift
            return None

        # Calculate importance score (combination of lift and sample size)
        sample_ratio = len(subset) / len(df)
        importance = abs(lift) * np.sqrt(sample_ratio)

        # Calculate confidence using binomial test approximation
        # Standard error of win rate
        se = np.sqrt(win_rate * (100 - win_rate) / len(subset))
        z_score = abs(win_rate - overall_win_rate) / se if se > 0 else 0
        confidence = min(0.99, 1 - np.exp(-z_score / 2))

        return FeatureAttribution(
            feature_name=feature,
            importance_score=min(1.0, importance),
            direction="positive" if lift > 0 else "negative",
            condition=condition,
            hit_rate=win_rate,
            sample_size=len(subset),
            confidence=confidence,
        )

    def _auto_thresholds(self, series: pd.Series) -> list[float]:
        """Generate automatic thresholds based on data distribution."""
        try:
            percentiles = [10, 25, 50, 75, 90]
            return [float(np.percentile(series.dropna(), p)) for p in percentiles]
        except Exception:
            return [0]

    def _get_parameter_direction(
        self,
        parameter_name: str,
        current_value: Any,
        proposed_value: Any,
    ) -> str:
        """Determine if parameter is being tightened or loosened."""
        try:
            current = float(current_value)
            proposed = float(proposed_value)

            # For TP/SL parameters, higher usually means looser
            if "tp" in parameter_name.lower() or "take_profit" in parameter_name.lower():
                return "looser" if proposed > current else "tighter"
            elif "sl" in parameter_name.lower() or "stop_loss" in parameter_name.lower():
                return "tighter" if proposed > current else "looser"
            else:
                return "increased" if proposed > current else "decreased"
        except (ValueError, TypeError):
            return "changed"

    def _extract_primary_driver(
        self,
        attributions: list[FeatureAttribution],
        parameter_name: str,
        current_value: Any,
        proposed_value: Any,
    ) -> str:
        """Extract the primary driver of improvement."""
        if not attributions:
            return f"Parameter adjustment from {current_value} to {proposed_value}"

        top_attr = attributions[0]
        return (
            f"Trades matching '{top_attr.condition}' have {top_attr.hit_rate:.1f}% win rate "
            f"(vs overall), occurring in {top_attr.sample_size} trades"
        )

    def _extract_secondary_drivers(
        self,
        attributions: list[FeatureAttribution],
    ) -> list[str]:
        """Extract secondary contributing factors."""
        secondary = []

        for attr in attributions[1:4]:  # Take 2nd through 4th
            desc = (
                f"{attr.feature_name}: {attr.condition} → "
                f"{attr.hit_rate:.1f}% win rate ({attr.sample_size} trades)"
            )
            secondary.append(desc)

        return secondary

    def _identify_tradeoffs(
        self,
        df: pd.DataFrame,
        parameter_name: str,
        proposed_value: Any,
        trade_delta: int,
    ) -> list[str]:
        """Identify potential tradeoffs of the proposed change."""
        tradeoffs = []

        # Trade frequency tradeoff
        if trade_delta < 0:
            reduction_pct = abs(trade_delta) / len(df) * 100 if len(df) > 0 else 0
            if reduction_pct > 10:
                tradeoffs.append(
                    f"Trade frequency reduced by ~{reduction_pct:.0f}% "
                    f"({abs(trade_delta)} fewer trades)"
                )

        # Parameter-specific tradeoffs
        param_lower = parameter_name.lower()

        if "tp" in param_lower or "take_profit" in param_lower:
            try:
                if float(proposed_value) > 1.0:
                    tradeoffs.append(
                        "Higher TP may result in more unrealized profits being given back"
                    )
            except (ValueError, TypeError):
                pass

        if "sl" in param_lower or "stop_loss" in param_lower:
            try:
                if float(proposed_value) > 1.5:
                    tradeoffs.append(
                        "Wider SL increases per-trade risk exposure"
                    )
            except (ValueError, TypeError):
                pass

        if "threshold" in param_lower:
            tradeoffs.append(
                "Threshold changes may affect signal quality in different market regimes"
            )

        if not tradeoffs:
            tradeoffs.append("Monitor for regime changes that may invalidate this optimization")

        return tradeoffs

    def _calculate_overall_confidence(
        self,
        attributions: list[FeatureAttribution],
        sample_size: int,
        win_rate_delta: float,
    ) -> float:
        """Calculate overall confidence in the suggestion."""
        if sample_size < self.min_sample_size:
            return 0.0

        # Base confidence from sample size
        sample_confidence = min(1.0, sample_size / 500)

        # Attribution confidence (average of top attributions)
        if attributions:
            attr_confidence = np.mean([a.confidence for a in attributions[:5]])
        else:
            attr_confidence = 0.5

        # Win rate delta confidence (larger deltas are more confident)
        delta_confidence = min(1.0, abs(win_rate_delta) / 10)

        # Weighted average
        overall = (
            0.3 * sample_confidence +
            0.4 * attr_confidence +
            0.3 * delta_confidence
        )

        return min(0.99, overall)

    def _generate_explanation_text(
        self,
        parameter_name: str,
        current_value: Any,
        proposed_value: Any,
        symbol: str | None,
        win_rate_delta: float,
        pf_delta: float,
        trade_delta: int,
        attributions: list[FeatureAttribution],
        tradeoffs: list[str],
    ) -> str:
        """Generate human-readable explanation text."""
        symbol_text = f" for {symbol}" if symbol else ""

        # Header
        lines = [
            f"## Parameter Suggestion Explanation{symbol_text}",
            "",
            f"**Recommendation**: Change `{parameter_name}` from `{current_value}` to `{proposed_value}`",
            "",
            "### Expected Impact",
            f"- Win Rate: {'+' if win_rate_delta >= 0 else ''}{win_rate_delta:.2f}%",
            f"- Profit Factor: {'+' if pf_delta >= 0 else ''}{pf_delta:.4f}",
            f"- Trade Count: {'+' if trade_delta >= 0 else ''}{trade_delta}",
            "",
        ]

        # Why this works
        if attributions:
            lines.extend([
                "### Why This Works",
                "",
            ])

            for i, attr in enumerate(attributions[:3], 1):
                direction_emoji = "📈" if attr.direction == "positive" else "📉"
                lines.append(
                    f"{i}. {direction_emoji} **{attr.condition}**: "
                    f"{attr.hit_rate:.1f}% win rate ({attr.sample_size} trades, "
                    f"{attr.confidence*100:.0f}% confidence)"
                )

            lines.append("")

        # Tradeoffs
        if tradeoffs:
            lines.extend([
                "### Tradeoffs to Consider",
                "",
            ])
            for tradeoff in tradeoffs:
                lines.append(f"- ⚠️ {tradeoff}")
            lines.append("")

        # Confidence
        if attributions:
            avg_confidence = np.mean([a.confidence for a in attributions[:5]])
            confidence_level = (
                "High" if avg_confidence > 0.8
                else "Medium" if avg_confidence > 0.6
                else "Low"
            )
            lines.extend([
                "### Confidence Assessment",
                "",
                f"Overall Confidence: **{confidence_level}** ({avg_confidence*100:.0f}%)",
                "",
                "---",
                "*This suggestion requires human approval before application.*",
            ])

        return "\n".join(lines)

    def explain_multiple_suggestions(
        self,
        df: pd.DataFrame,
        suggestions: list[dict[str, Any]],
    ) -> list[SuggestionExplanation]:
        """
        Generate explanations for multiple suggestions.

        Args:
            df: DataFrame with trade data and features
            suggestions: List of suggestion dictionaries

        Returns:
            List of explanations
        """
        explanations = []

        for suggestion in suggestions:
            try:
                explanation = self.explain_suggestion(
                    df=df,
                    parameter_name=suggestion["parameter_name"],
                    current_value=suggestion["current_value"],
                    proposed_value=suggestion["proposed_value"],
                    current_metrics=suggestion.get("current_metrics", {}),
                    proposed_metrics=suggestion.get("proposed_metrics", {}),
                    symbol=suggestion.get("symbol"),
                )
                explanations.append(explanation)
            except Exception as e:
                # Log error but continue with other suggestions
                print(f"Error explaining suggestion: {e}")
                continue

        return explanations

    def compare_feature_importance(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate a feature importance comparison table.

        Useful for understanding which features most influence trade outcomes.
        """
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()

        if "pnl_pct" not in df.columns:
            return pd.DataFrame()

        results = []

        for feature in self.ANALYZABLE_FEATURES:
            if feature not in df.columns:
                continue

            # Calculate correlation with outcome
            try:
                corr = df[feature].corr(df["pnl_pct"])
            except Exception:
                corr = 0

            # Calculate information gain (simplified)
            thresholds = self.FEATURE_THRESHOLDS.get(feature, self._auto_thresholds(df[feature]))
            best_lift = 0

            for threshold in thresholds:
                mask = df[feature] > threshold
                if mask.sum() >= self.min_sample_size:
                    subset_wr = (df[mask]["pnl_pct"] > 0).mean()
                    overall_wr = (df["pnl_pct"] > 0).mean()
                    if overall_wr > 0:
                        lift = (subset_wr - overall_wr) / overall_wr
                        best_lift = max(best_lift, abs(lift))

            results.append({
                "feature": feature,
                "correlation": round(corr, 4),
                "best_lift": round(best_lift, 4),
                "coverage": round((~df[feature].isna()).mean(), 4),
            })

        return pd.DataFrame(results).sort_values("best_lift", ascending=False)
