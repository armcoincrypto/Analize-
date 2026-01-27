"""
Conservative Maker Fill Model
=============================
Models maker order fills for realistic backtesting.

Problem: Backtesting maker orders is optimistic because:
1. Queue position: We assume we're first in queue (we're not)
2. Price touch: Assumes any price touch fills us (often doesn't)
3. Liquidity: Assumes infinite liquidity at our price (there isn't)

Solution: Conservative fill model that requires:
1. Price must CROSS our limit by X bps (not just touch)
2. OR use probabilistic model based on empirical fill rates

Usage:
    from hft_system.research.maker_fill_model import ConservativeFillModel

    model = ConservativeFillModel()

    # Check if order would fill
    filled = model.would_fill(
        side="buy",
        limit_price=100.00,
        market_low=99.95,
        market_high=100.05,
        time_at_price_ms=500
    )
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FillMode(Enum):
    """Fill model modes."""
    OPTIMISTIC = "optimistic"  # Touch = fill (unrealistic)
    CONSERVATIVE_CROSS = "conservative_cross"  # Price must cross by X bps
    CONSERVATIVE_PROBABILISTIC = "conservative_probabilistic"  # Random based on empirical rates
    VERY_CONSERVATIVE = "very_conservative"  # Cross + probabilistic


@dataclass
class FillModelConfig:
    """Configuration for maker fill model."""
    mode: FillMode = FillMode.CONSERVATIVE_CROSS

    # Cross-based fill requirements
    min_cross_bps: float = 2.0  # Price must cross limit by at least 2 bps
    min_time_at_price_ms: int = 100  # Must stay crossed for at least 100ms

    # Probabilistic fill rates (calibrated from live data)
    fill_rate_on_touch: float = 0.30  # 30% fill if price just touches
    fill_rate_on_small_cross: float = 0.60  # 60% if crosses by 1-3 bps
    fill_rate_on_large_cross: float = 0.85  # 85% if crosses by >3 bps

    # Queue position modeling
    queue_penalty_pct: float = 0.20  # 20% penalty for not being first in queue

    # Random seed for reproducibility
    random_seed: int = 42


class ConservativeFillModel:
    """
    Conservative model for simulating maker order fills.

    More realistic than "touch = fill" assumption used in naive backtests.
    """

    def __init__(self, config: FillModelConfig = None):
        self.config = config or FillModelConfig()
        random.seed(self.config.random_seed)

        # Track fill statistics for calibration
        self.fill_attempts = 0
        self.fills_granted = 0
        self.fills_denied = 0

    def would_fill(
        self,
        side: str,
        limit_price: float,
        market_low: float,
        market_high: float,
        time_at_price_ms: int = 0,
        queue_position: int = 0,
        volume_at_price: float = 0
    ) -> Tuple[bool, float, str]:
        """
        Determine if a maker order would fill under conservative model.

        Args:
            side: "buy" or "sell"
            limit_price: Order limit price
            market_low: Lowest price during period
            market_high: Highest price during period
            time_at_price_ms: Time price spent at or beyond limit
            queue_position: Estimated position in queue (0 = first)
            volume_at_price: Volume traded at this price level

        Returns:
            (would_fill: bool, fill_probability: float, reason: str)
        """
        self.fill_attempts += 1

        # Check basic price conditions
        if side == "buy":
            price_touched = market_low <= limit_price
            price_crossed = market_low < limit_price
            cross_depth_bps = ((limit_price - market_low) / limit_price * 10000) if price_crossed else 0
        else:  # sell
            price_touched = market_high >= limit_price
            price_crossed = market_high > limit_price
            cross_depth_bps = ((market_high - limit_price) / limit_price * 10000) if price_crossed else 0

        # No fill if price never reached limit
        if not price_touched:
            self.fills_denied += 1
            return False, 0.0, "price_never_touched"

        # Mode-specific logic
        if self.config.mode == FillMode.OPTIMISTIC:
            # Unrealistic: any touch = fill
            self.fills_granted += 1
            return True, 1.0, "optimistic_touch"

        elif self.config.mode == FillMode.CONSERVATIVE_CROSS:
            # Require price to cross by minimum bps
            if cross_depth_bps >= self.config.min_cross_bps:
                self.fills_granted += 1
                return True, 0.95, f"crossed_{cross_depth_bps:.1f}bps"
            else:
                self.fills_denied += 1
                return False, cross_depth_bps / self.config.min_cross_bps, f"insufficient_cross_{cross_depth_bps:.1f}bps"

        elif self.config.mode == FillMode.CONSERVATIVE_PROBABILISTIC:
            # Probabilistic based on cross depth
            if cross_depth_bps >= 3.0:
                fill_prob = self.config.fill_rate_on_large_cross
                reason = "large_cross"
            elif cross_depth_bps >= 1.0:
                fill_prob = self.config.fill_rate_on_small_cross
                reason = "small_cross"
            else:
                fill_prob = self.config.fill_rate_on_touch
                reason = "touch_only"

            # Apply queue penalty
            if queue_position > 0:
                fill_prob *= (1 - self.config.queue_penalty_pct * min(queue_position, 5))

            # Random determination
            if random.random() < fill_prob:
                self.fills_granted += 1
                return True, fill_prob, f"probabilistic_{reason}"
            else:
                self.fills_denied += 1
                return False, fill_prob, f"probabilistic_denied_{reason}"

        elif self.config.mode == FillMode.VERY_CONSERVATIVE:
            # Must cross AND pass probabilistic check
            if cross_depth_bps < self.config.min_cross_bps:
                self.fills_denied += 1
                return False, 0.0, "no_cross"

            # Apply probabilistic on top of cross requirement
            fill_prob = self.config.fill_rate_on_large_cross
            if time_at_price_ms < self.config.min_time_at_price_ms:
                fill_prob *= 0.5  # Penalty for quick price spike

            if random.random() < fill_prob:
                self.fills_granted += 1
                return True, fill_prob, "very_conservative_fill"
            else:
                self.fills_denied += 1
                return False, fill_prob, "very_conservative_denied"

        # Default deny
        self.fills_denied += 1
        return False, 0.0, "unknown_mode"

    def calculate_fill_price(
        self,
        side: str,
        limit_price: float,
        market_low: float,
        market_high: float,
        cross_depth_bps: float = 0
    ) -> float:
        """
        Calculate realistic fill price (may be better than limit due to price improvement).

        For conservative model, we assume fill at limit price (no improvement).
        """
        # Conservative: assume fill at limit price
        # In reality, there can be price improvement if price crosses significantly
        if self.config.mode in [FillMode.OPTIMISTIC]:
            # Optimistic: assume some price improvement
            if side == "buy":
                improvement = min(cross_depth_bps * 0.1, 1.0)  # Up to 0.1 bps improvement
                return limit_price * (1 - improvement / 10000)
            else:
                improvement = min(cross_depth_bps * 0.1, 1.0)
                return limit_price * (1 + improvement / 10000)

        # Conservative: fill at limit price
        return limit_price

    def estimate_time_to_fill(
        self,
        spread_pct: float,
        volatility_1m_pct: float,
        queue_position: int = 0
    ) -> Optional[int]:
        """
        Estimate time to fill in milliseconds.

        Based on:
        - Spread: Tighter spread = faster fills
        - Volatility: Higher vol = faster price movement
        - Queue: Deeper queue = longer wait

        Returns:
            Estimated time to fill in ms, or None if unlikely to fill
        """
        if spread_pct <= 0 or volatility_1m_pct <= 0:
            return None

        # Base time: spread / volatility
        # Intuition: if spread is 0.05% and 1m vol is 0.10%, expect cross in ~30s
        base_time_sec = (spread_pct / 2) / (volatility_1m_pct / 60)

        # Adjust for queue position
        queue_multiplier = 1 + queue_position * 0.2

        # Convert to ms
        time_ms = int(base_time_sec * queue_multiplier * 1000)

        # Cap at 5 minutes
        return min(time_ms, 300000)

    def get_fill_stats(self) -> dict:
        """Get fill model statistics."""
        total = self.fill_attempts
        if total == 0:
            return {"message": "No fill attempts yet"}

        return {
            "total_attempts": total,
            "fills_granted": self.fills_granted,
            "fills_denied": self.fills_denied,
            "fill_rate_pct": round(self.fills_granted / total * 100, 1),
            "mode": self.config.mode.value,
            "min_cross_bps": self.config.min_cross_bps
        }

    def reset_stats(self):
        """Reset fill statistics."""
        self.fill_attempts = 0
        self.fills_granted = 0
        self.fills_denied = 0


def calibrate_fill_model_from_db(db_path: str, days: int = 14) -> FillModelConfig:
    """
    Calibrate fill model parameters from actual maker order telemetry.

    Args:
        db_path: Path to database with maker_order_telemetry table
        days: Days of history to analyze

    Returns:
        Calibrated FillModelConfig
    """
    import sqlite3
    from datetime import datetime, timedelta

    config = FillModelConfig()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # Get fill rates by cross depth
        cursor.execute("""
            SELECT
                CASE
                    WHEN cross_depth_bps >= 3 THEN 'large_cross'
                    WHEN cross_depth_bps >= 1 THEN 'small_cross'
                    ELSE 'touch'
                END as cross_category,
                SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
                COUNT(*) as total
            FROM maker_order_telemetry
            WHERE posted_ts > ? AND price_crossed_limit IS NOT NULL
            GROUP BY cross_category
        """, (cutoff_ms,))

        for row in cursor.fetchall():
            category, filled, total = row
            if total > 10:  # Minimum sample size
                rate = filled / total
                if category == 'touch':
                    config.fill_rate_on_touch = rate
                elif category == 'small_cross':
                    config.fill_rate_on_small_cross = rate
                elif category == 'large_cross':
                    config.fill_rate_on_large_cross = rate

        # Get minimum cross depth that achieved 80% fill rate
        cursor.execute("""
            SELECT cross_depth_bps
            FROM maker_order_telemetry
            WHERE posted_ts > ? AND status = 'filled' AND cross_depth_bps > 0
            ORDER BY cross_depth_bps
        """, (cutoff_ms,))
        cross_depths = [row[0] for row in cursor.fetchall()]

        if cross_depths:
            # 20th percentile of successful fills
            p20_idx = int(len(cross_depths) * 0.2)
            config.min_cross_bps = cross_depths[p20_idx] if p20_idx < len(cross_depths) else 2.0

        conn.close()
        logger.info(f"Calibrated fill model: touch={config.fill_rate_on_touch:.2f}, "
                   f"small={config.fill_rate_on_small_cross:.2f}, "
                   f"large={config.fill_rate_on_large_cross:.2f}, "
                   f"min_cross={config.min_cross_bps:.1f}bps")

    except Exception as e:
        logger.warning(f"Could not calibrate fill model: {e}. Using defaults.")

    return config
