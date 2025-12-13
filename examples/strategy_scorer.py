#!/usr/bin/env python3
"""
Strategy Scoring Table - Auto-Rank by Stability

Ranks strategies by SURVIVAL and CONSISTENCY, not raw profit.

Scoring Priority (in order):
1. Drawdown stability (most important)
2. Consistency over time
3. Regime adaptability
4. Risk-adjusted return
5. Absolute PnL (least important)

"A strategy making +8% with 6% DD beats
 a strategy making +15% with 25% DD"

This is how prop firms, investors, and survivors evaluate strategies.
"""

import os
import json
import sqlite3
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum


class StrategyGrade(Enum):
    """Strategy quality grade."""
    A = "A"     # Elite - deploy with full size
    B = "B"     # Good - deploy with normal size
    C = "C"     # Marginal - deploy with reduced size
    D = "D"     # Poor - paper trade only
    F = "F"     # Failed - retire immediately


@dataclass
class StrategyScore:
    """Complete strategy scoring."""
    name: str
    grade: StrategyGrade
    total_score: float              # 0-100

    # Component scores (0-100 each)
    drawdown_score: float           # Weight: 30%
    consistency_score: float        # Weight: 25%
    regime_score: float             # Weight: 20%
    risk_adjusted_score: float      # Weight: 15%
    absolute_pnl_score: float       # Weight: 10%

    # Raw metrics
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    total_pnl_pct: float
    trade_count: int
    regimes_profitable: int
    regimes_total: int

    recommendation: str
    size_multiplier: float          # Recommended position size multiplier


@dataclass
class SymbolScore:
    """Score for a single trading symbol."""
    symbol: str
    trades: int
    win_rate: float
    profit_factor: float
    total_pnl_pct: float
    max_drawdown_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    best_regime: str
    worst_regime: str
    score: float
    grade: StrategyGrade
    deploy: bool


class StrategyScorer:
    """
    Auto-ranks strategies/symbols by stability and quality.

    Scoring weights (evidence-based prioritization):
    - Drawdown stability: 30%
    - Consistency: 25%
    - Regime adaptability: 20%
    - Risk-adjusted return: 15%
    - Absolute P&L: 10%
    """

    # Scoring weights
    WEIGHT_DRAWDOWN = 0.30
    WEIGHT_CONSISTENCY = 0.25
    WEIGHT_REGIME = 0.20
    WEIGHT_RISK_ADJUSTED = 0.15
    WEIGHT_ABSOLUTE = 0.10

    # Grade thresholds
    GRADE_THRESHOLDS = {
        StrategyGrade.A: 80,
        StrategyGrade.B: 65,
        StrategyGrade.C: 50,
        StrategyGrade.D: 35,
        StrategyGrade.F: 0
    }

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "evidence_collector.db")
        self.db_path = db_path
        self.trades = []

    def load_trades(self):
        """Load trades from evidence database."""
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT symbol, pnl_percent, pnl_absolute, regime,
                       timestamp, entry_signals
                FROM trades
                ORDER BY timestamp
            """)

            self.trades = []
            for row in cursor.fetchall():
                self.trades.append({
                    "symbol": row[0],
                    "pnl_pct": row[1],
                    "pnl_usd": row[2],
                    "regime": row[3] or "unknown",
                    "timestamp": row[4],
                    "signals": json.loads(row[5]) if row[5] else []
                })

            conn.close()
            print(f"Loaded {len(self.trades)} trades")

        except Exception as e:
            print(f"Error loading trades: {e}")

    def _calculate_drawdown_score(self, max_dd: float) -> float:
        """
        Score drawdown (lower is better).

        100 = DD < 5%
        80 = DD < 10%
        60 = DD < 15%
        40 = DD < 20%
        20 = DD < 25%
        0 = DD >= 25%
        """
        if max_dd < 5:
            return 100
        elif max_dd < 10:
            return 80 + (10 - max_dd) * 4
        elif max_dd < 15:
            return 60 + (15 - max_dd) * 4
        elif max_dd < 20:
            return 40 + (20 - max_dd) * 4
        elif max_dd < 25:
            return 20 + (25 - max_dd) * 4
        else:
            return max(0, 20 - (max_dd - 25) * 2)

    def _calculate_consistency_score(
        self,
        win_rate: float,
        profit_factor: float,
        max_loss_streak: int
    ) -> float:
        """
        Score consistency.

        Components:
        - Win rate contribution (40%)
        - Profit factor contribution (40%)
        - Loss streak penalty (20%)
        """
        # Win rate component (50-80% optimal)
        if win_rate >= 80:
            wr_score = 100
        elif win_rate >= 65:
            wr_score = 80 + (win_rate - 65) * 1.33
        elif win_rate >= 50:
            wr_score = 50 + (win_rate - 50) * 2
        else:
            wr_score = win_rate

        # Profit factor component (1.0-3.0 range)
        if profit_factor >= 3.0:
            pf_score = 100
        elif profit_factor >= 2.0:
            pf_score = 70 + (profit_factor - 2.0) * 30
        elif profit_factor >= 1.5:
            pf_score = 50 + (profit_factor - 1.5) * 40
        elif profit_factor >= 1.0:
            pf_score = (profit_factor - 1.0) * 100
        else:
            pf_score = 0

        # Loss streak penalty
        if max_loss_streak <= 3:
            streak_score = 100
        elif max_loss_streak <= 5:
            streak_score = 80
        elif max_loss_streak <= 7:
            streak_score = 50
        else:
            streak_score = max(0, 50 - (max_loss_streak - 7) * 10)

        return wr_score * 0.4 + pf_score * 0.4 + streak_score * 0.2

    def _calculate_regime_score(self, regimes_profitable: int, regimes_total: int) -> float:
        """
        Score regime adaptability.

        100 = Profitable in all regimes
        50 = Profitable in half
        0 = Profitable in none
        """
        if regimes_total == 0:
            return 50  # Unknown

        ratio = regimes_profitable / regimes_total
        return ratio * 100

    def _calculate_risk_adjusted_score(self, sharpe: float, sortino: float) -> float:
        """
        Score risk-adjusted returns.

        Sharpe > 2.0 = excellent
        Sharpe > 1.0 = good
        Sharpe > 0.5 = acceptable
        """
        # Use average of Sharpe and Sortino
        avg_ratio = (sharpe + sortino) / 2

        if avg_ratio >= 2.0:
            return 100
        elif avg_ratio >= 1.5:
            return 80 + (avg_ratio - 1.5) * 40
        elif avg_ratio >= 1.0:
            return 60 + (avg_ratio - 1.0) * 40
        elif avg_ratio >= 0.5:
            return 40 + (avg_ratio - 0.5) * 40
        elif avg_ratio >= 0:
            return avg_ratio * 80
        else:
            return 0

    def _calculate_absolute_score(self, total_pnl_pct: float) -> float:
        """
        Score absolute returns (least important).

        This is intentionally weighted low because raw profit
        without considering risk is misleading.
        """
        if total_pnl_pct >= 50:
            return 100
        elif total_pnl_pct >= 30:
            return 80 + (total_pnl_pct - 30) * 1
        elif total_pnl_pct >= 10:
            return 50 + (total_pnl_pct - 10) * 1.5
        elif total_pnl_pct >= 0:
            return total_pnl_pct * 5
        else:
            return 0

    def _determine_grade(self, score: float) -> StrategyGrade:
        """Determine letter grade from score."""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return StrategyGrade.F

    def _get_recommendation(self, grade: StrategyGrade, score: float) -> Tuple[str, float]:
        """Get deployment recommendation and size multiplier."""
        recommendations = {
            StrategyGrade.A: ("DEPLOY at full size - elite stability", 1.0),
            StrategyGrade.B: ("DEPLOY at normal size - good quality", 0.85),
            StrategyGrade.C: ("DEPLOY at reduced size - marginal edge", 0.5),
            StrategyGrade.D: ("PAPER TRADE only - needs improvement", 0.0),
            StrategyGrade.F: ("RETIRE immediately - failed validation", 0.0)
        }
        return recommendations.get(grade, ("UNKNOWN", 0.0))

    def score_symbol(self, symbol: str) -> Optional[SymbolScore]:
        """Score a single trading symbol."""
        symbol_trades = [t for t in self.trades if t["symbol"] == symbol]

        if len(symbol_trades) < 30:
            print(f"  {symbol}: Only {len(symbol_trades)} trades (need 30+)")
            return None

        # Calculate metrics
        wins = [t for t in symbol_trades if t["pnl_pct"] > 0]
        losses = [t for t in symbol_trades if t["pnl_pct"] <= 0]

        win_rate = len(wins) / len(symbol_trades) * 100
        total_pnl = sum(t["pnl_pct"] for t in symbol_trades)

        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

        gross_win = sum(t["pnl_pct"] for t in wins)
        gross_loss = abs(sum(t["pnl_pct"] for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 99.0

        # Calculate drawdown
        equity = 100
        peak = 100
        max_dd = 0
        for t in symbol_trades:
            equity += t["pnl_pct"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Loss streak
        max_streak = 0
        current_streak = 0
        for t in symbol_trades:
            if t["pnl_pct"] <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Regime analysis
        regimes = {}
        for t in symbol_trades:
            regime = t["regime"]
            if regime not in regimes:
                regimes[regime] = {"wins": 0, "total": 0}
            regimes[regime]["total"] += 1
            if t["pnl_pct"] > 0:
                regimes[regime]["wins"] += 1

        profitable_regimes = sum(1 for r in regimes.values() if r["wins"] / r["total"] > 0.5)
        best_regime = max(regimes.keys(), key=lambda r: regimes[r]["wins"] / regimes[r]["total"]) if regimes else "unknown"
        worst_regime = min(regimes.keys(), key=lambda r: regimes[r]["wins"] / regimes[r]["total"]) if regimes else "unknown"

        # Calculate component scores
        dd_score = self._calculate_drawdown_score(max_dd)
        consistency_score = self._calculate_consistency_score(win_rate, profit_factor, max_streak)
        regime_score = self._calculate_regime_score(profitable_regimes, len(regimes))

        # Simplified risk-adjusted (using profit factor as proxy)
        risk_score = self._calculate_risk_adjusted_score(profit_factor / 2, profit_factor / 1.5)
        pnl_score = self._calculate_absolute_score(total_pnl)

        # Weighted total
        total_score = (
            dd_score * self.WEIGHT_DRAWDOWN +
            consistency_score * self.WEIGHT_CONSISTENCY +
            regime_score * self.WEIGHT_REGIME +
            risk_score * self.WEIGHT_RISK_ADJUSTED +
            pnl_score * self.WEIGHT_ABSOLUTE
        )

        grade = self._determine_grade(total_score)
        deploy = grade in [StrategyGrade.A, StrategyGrade.B, StrategyGrade.C]

        return SymbolScore(
            symbol=symbol,
            trades=len(symbol_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl_pct=total_pnl,
            max_drawdown_pct=max_dd,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            best_regime=best_regime,
            worst_regime=worst_regime,
            score=total_score,
            grade=grade,
            deploy=deploy
        )

    def score_all_symbols(self) -> List[SymbolScore]:
        """Score all symbols and return ranked list."""
        if not self.trades:
            self.load_trades()

        symbols = set(t["symbol"] for t in self.trades)
        scores = []

        for symbol in symbols:
            score = self.score_symbol(symbol)
            if score:
                scores.append(score)

        # Sort by score (highest first)
        scores.sort(key=lambda s: s.score, reverse=True)

        return scores

    def print_ranking(self):
        """Print strategy ranking table."""
        scores = self.score_all_symbols()

        if not scores:
            print("No symbols with enough trades to score")
            return

        print("\n" + "=" * 80)
        print("STRATEGY SCORING TABLE - Ranked by Stability")
        print("=" * 80)
        print(f"{'Rank':<5} {'Symbol':<12} {'Grade':<6} {'Score':<8} {'Win%':<8} {'PF':<6} {'DD%':<8} {'Deploy':<8}")
        print("-" * 80)

        for i, score in enumerate(scores, 1):
            deploy = "YES" if score.deploy else "NO"
            print(f"{i:<5} {score.symbol:<12} {score.grade.value:<6} {score.score:<8.1f} {score.win_rate:<8.1f} {score.profit_factor:<6.2f} {score.max_drawdown_pct:<8.1f} {deploy:<8}")

        print("-" * 80)

        # Summary
        deployable = [s for s in scores if s.deploy]
        print(f"\nDeployable strategies: {len(deployable)}/{len(scores)}")

        if deployable:
            print("\n[DEPLOYMENT RECOMMENDATION]")
            for score in deployable:
                rec, mult = self._get_recommendation(score.grade, score.score)
                print(f"  {score.symbol}: {rec}")
                print(f"    Size multiplier: {mult:.2f}x")
                print(f"    Best regime: {score.best_regime}")
                print(f"    Avoid regime: {score.worst_regime}")

        # Failed strategies
        failed = [s for s in scores if not s.deploy]
        if failed:
            print("\n[DO NOT DEPLOY]")
            for score in failed:
                rec, _ = self._get_recommendation(score.grade, score.score)
                print(f"  {score.symbol}: {rec}")

        print("=" * 80)

    def get_deployment_list(self) -> List[Dict]:
        """Get list of deployable strategies with settings."""
        scores = self.score_all_symbols()
        deployable = []

        for score in scores:
            if score.deploy:
                _, mult = self._get_recommendation(score.grade, score.score)
                deployable.append({
                    "symbol": score.symbol,
                    "grade": score.grade.value,
                    "score": score.score,
                    "size_multiplier": mult,
                    "best_regime": score.best_regime,
                    "avoid_regime": score.worst_regime,
                    "win_rate": score.win_rate,
                    "profit_factor": score.profit_factor
                })

        return deployable


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STRATEGY SCORER")
    print("=" * 80)
    print()
    print("Scoring weights (prioritized by survival):")
    print(f"  Drawdown stability: {StrategyScorer.WEIGHT_DRAWDOWN * 100:.0f}%")
    print(f"  Consistency:        {StrategyScorer.WEIGHT_CONSISTENCY * 100:.0f}%")
    print(f"  Regime adaptability:{StrategyScorer.WEIGHT_REGIME * 100:.0f}%")
    print(f"  Risk-adjusted:      {StrategyScorer.WEIGHT_RISK_ADJUSTED * 100:.0f}%")
    print(f"  Absolute P&L:       {StrategyScorer.WEIGHT_ABSOLUTE * 100:.0f}%")
    print()
    print("Grade thresholds:")
    for grade, threshold in StrategyScorer.GRADE_THRESHOLDS.items():
        print(f"  {grade.value}: {threshold}+")

    scorer = StrategyScorer()
    scorer.print_ranking()

    # Show deployable list
    print("\n[EXPORT: Deployable Strategies]")
    deployable = scorer.get_deployment_list()
    print(json.dumps(deployable, indent=2))
