#!/usr/bin/env python3
"""
Performance Report Generator - Prop-Firm Style Reporting

Generates professional performance reports suitable for:
- Prop firm applications
- Investor presentations
- Self-evaluation
- Strategy validation

Metrics included:
- Absolute returns (P&L in $)
- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown analysis
- Win/Loss statistics
- Time-based analysis
- Consistency metrics

"Show me the numbers, not the story."
"""

import os
import json
import sqlite3
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class TradeData:
    """Single trade record."""
    timestamp: datetime.datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usd: float
    hold_time_minutes: float
    signals: List[str]
    regime: str


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""
    # Basic stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    total_pnl_usd: float
    total_pnl_pct: float
    avg_win_usd: float
    avg_loss_usd: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float

    # Drawdown
    max_drawdown_usd: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Consistency
    best_day_pnl: float
    worst_day_pnl: float
    profitable_days_pct: float
    avg_trades_per_day: float
    longest_win_streak: int
    longest_loss_streak: int

    # Time analysis
    avg_hold_time_minutes: float
    best_hour_utc: int
    worst_hour_utc: int
    best_day_of_week: str
    worst_day_of_week: str


class PerformanceReportGenerator:
    """
    Generates comprehensive performance reports.

    Can load data from:
    - Evidence collector database
    - Trade log JSON files
    - Direct trade list input
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.trades: List[TradeData] = []

    def load_from_evidence_db(self, db_path: str = None):
        """Load trades from evidence collector database."""
        if db_path is None:
            db_path = str(Path(__file__).parent / "evidence_collector.db")

        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            return

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, symbol, side, entry_price, exit_price,
                       pnl_percent, pnl_absolute, hold_time_minutes,
                       entry_signals, regime
                FROM trades
                ORDER BY timestamp
            """)

            for row in cursor.fetchall():
                signals = json.loads(row[8]) if row[8] else []
                trade = TradeData(
                    timestamp=datetime.datetime.fromisoformat(row[0]),
                    symbol=row[1],
                    side=row[2],
                    entry_price=row[3],
                    exit_price=row[4],
                    pnl_pct=row[5],
                    pnl_usd=row[6],
                    hold_time_minutes=row[7] or 0,
                    signals=signals,
                    regime=row[9] or "unknown"
                )
                self.trades.append(trade)

            conn.close()
            print(f"Loaded {len(self.trades)} trades from database")

        except Exception as e:
            print(f"Error loading from database: {e}")

    def load_from_json(self, json_path: str):
        """Load trades from JSON file."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for trade_data in data.get("trades", []):
                trade = TradeData(
                    timestamp=datetime.datetime.fromisoformat(trade_data["timestamp"]),
                    symbol=trade_data["symbol"],
                    side=trade_data["side"],
                    entry_price=trade_data["entry_price"],
                    exit_price=trade_data["exit_price"],
                    pnl_pct=trade_data["pnl_pct"],
                    pnl_usd=trade_data["pnl_usd"],
                    hold_time_minutes=trade_data.get("hold_time_minutes", 0),
                    signals=trade_data.get("signals", []),
                    regime=trade_data.get("regime", "unknown")
                )
                self.trades.append(trade)

            print(f"Loaded {len(self.trades)} trades from JSON")

        except Exception as e:
            print(f"Error loading from JSON: {e}")

    def add_trade(self, trade: TradeData):
        """Add a single trade."""
        self.trades.append(trade)

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        if not self.trades:
            return self._empty_metrics()

        # Basic stats
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.pnl_pct > 0]
        losers = [t for t in self.trades if t.pnl_pct <= 0]
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L calculations
        total_pnl_usd = sum(t.pnl_usd for t in self.trades)
        total_pnl_pct = (total_pnl_usd / self.initial_capital * 100) if self.initial_capital > 0 else 0

        avg_win_usd = sum(t.pnl_usd for t in winners) / len(winners) if winners else 0
        avg_loss_usd = sum(t.pnl_usd for t in losers) / len(losers) if losers else 0
        avg_win_pct = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
        avg_loss_pct = sum(t.pnl_pct for t in losers) / len(losers) if losers else 0

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in winners)
        gross_loss = abs(sum(t.pnl_usd for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate / 100 * avg_win_usd) + ((1 - win_rate / 100) * avg_loss_usd)

        # Drawdown analysis
        equity_curve = self._calculate_equity_curve()
        max_dd_usd, max_dd_pct, max_dd_duration = self._calculate_drawdown(equity_curve)
        avg_dd = self._calculate_avg_drawdown(equity_curve)

        # Daily P&L for Sharpe/Sortino
        daily_returns = self._calculate_daily_returns()

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)
        calmar = self._calculate_calmar(total_pnl_pct, max_dd_pct)

        # Consistency metrics
        daily_pnl = self._calculate_daily_pnl()
        best_day = max(daily_pnl.values()) if daily_pnl else 0
        worst_day = min(daily_pnl.values()) if daily_pnl else 0
        profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        profitable_days_pct = (profitable_days / len(daily_pnl) * 100) if daily_pnl else 0

        # Streaks
        win_streak, loss_streak = self._calculate_streaks()

        # Time analysis
        avg_hold = sum(t.hold_time_minutes for t in self.trades) / len(self.trades)
        best_hour, worst_hour = self._analyze_hours()
        best_dow, worst_dow = self._analyze_days_of_week()

        # Trades per day
        if self.trades:
            date_range = (self.trades[-1].timestamp - self.trades[0].timestamp).days + 1
            avg_trades_per_day = total_trades / max(date_range, 1)
        else:
            avg_trades_per_day = 0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_usd=total_pnl_usd,
            total_pnl_pct=total_pnl_pct,
            avg_win_usd=avg_win_usd,
            avg_loss_usd=avg_loss_usd,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown_usd=max_dd_usd,
            max_drawdown_pct=max_dd_pct,
            avg_drawdown_pct=avg_dd,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            best_day_pnl=best_day,
            worst_day_pnl=worst_day,
            profitable_days_pct=profitable_days_pct,
            avg_trades_per_day=avg_trades_per_day,
            longest_win_streak=win_streak,
            longest_loss_streak=loss_streak,
            avg_hold_time_minutes=avg_hold,
            best_hour_utc=best_hour,
            worst_hour_utc=worst_hour,
            best_day_of_week=best_dow,
            worst_day_of_week=worst_dow
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl_usd=0, total_pnl_pct=0, avg_win_usd=0, avg_loss_usd=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0, expectancy=0,
            max_drawdown_usd=0, max_drawdown_pct=0, avg_drawdown_pct=0,
            max_drawdown_duration_days=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, best_day_pnl=0, worst_day_pnl=0, profitable_days_pct=0,
            avg_trades_per_day=0, longest_win_streak=0, longest_loss_streak=0,
            avg_hold_time_minutes=0, best_hour_utc=0, worst_hour_utc=0,
            best_day_of_week="N/A", worst_day_of_week="N/A"
        )

    def _calculate_equity_curve(self) -> List[float]:
        """Calculate equity curve from trades."""
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade.pnl_usd)
        return equity

    def _calculate_drawdown(self, equity: List[float]) -> Tuple[float, float, int]:
        """Calculate max drawdown (USD, %, duration)."""
        if len(equity) < 2:
            return 0, 0, 0

        peak = equity[0]
        max_dd_usd = 0
        max_dd_pct = 0
        dd_start = 0
        max_dd_duration = 0

        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                dd_start = i

            dd_usd = peak - value
            dd_pct = (dd_usd / peak * 100) if peak > 0 else 0

            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
                max_dd_pct = dd_pct
                max_dd_duration = i - dd_start

        return max_dd_usd, max_dd_pct, max_dd_duration

    def _calculate_avg_drawdown(self, equity: List[float]) -> float:
        """Calculate average drawdown."""
        if len(equity) < 2:
            return 0

        drawdowns = []
        peak = equity[0]

        for value in equity:
            if value > peak:
                peak = value
            dd_pct = ((peak - value) / peak * 100) if peak > 0 else 0
            drawdowns.append(dd_pct)

        return sum(drawdowns) / len(drawdowns) if drawdowns else 0

    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns as percentages."""
        daily_pnl = self._calculate_daily_pnl()
        return [pnl / self.initial_capital * 100 for pnl in daily_pnl.values()]

    def _calculate_daily_pnl(self) -> Dict[str, float]:
        """Calculate P&L per day."""
        daily = {}
        for trade in self.trades:
            day = trade.timestamp.strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + trade.pnl_usd
        return daily

    def _calculate_sharpe(self, daily_returns: List[float], risk_free: float = 0) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(daily_returns) < 2:
            return 0

        avg_return = sum(daily_returns) / len(daily_returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns))

        if std_dev == 0:
            return 0

        daily_sharpe = (avg_return - risk_free) / std_dev
        return daily_sharpe * math.sqrt(252)  # Annualize

    def _calculate_sortino(self, daily_returns: List[float], target: float = 0) -> float:
        """Calculate Sortino ratio (annualized)."""
        if len(daily_returns) < 2:
            return 0

        avg_return = sum(daily_returns) / len(daily_returns)
        downside = [r for r in daily_returns if r < target]

        if not downside:
            return float('inf') if avg_return > 0 else 0

        downside_dev = math.sqrt(sum((r - target) ** 2 for r in downside) / len(downside))

        if downside_dev == 0:
            return 0

        daily_sortino = (avg_return - target) / downside_dev
        return daily_sortino * math.sqrt(252)

    def _calculate_calmar(self, total_return_pct: float, max_dd_pct: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd_pct == 0:
            return 0
        return total_return_pct / max_dd_pct

    def _calculate_streaks(self) -> Tuple[int, int]:
        """Calculate longest win and loss streaks."""
        if not self.trades:
            return 0, 0

        max_win = 0
        max_loss = 0
        current_win = 0
        current_loss = 0

        for trade in self.trades:
            if trade.pnl_pct > 0:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        return max_win, max_loss

    def _analyze_hours(self) -> Tuple[int, int]:
        """Find best and worst trading hours."""
        if not self.trades:
            return 0, 0

        hour_pnl = {}
        for trade in self.trades:
            hour = trade.timestamp.hour
            hour_pnl[hour] = hour_pnl.get(hour, 0) + trade.pnl_usd

        if not hour_pnl:
            return 0, 0

        best = max(hour_pnl, key=hour_pnl.get)
        worst = min(hour_pnl, key=hour_pnl.get)
        return best, worst

    def _analyze_days_of_week(self) -> Tuple[str, str]:
        """Find best and worst trading days."""
        if not self.trades:
            return "N/A", "N/A"

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_pnl = {}

        for trade in self.trades:
            dow = days[trade.timestamp.weekday()]
            day_pnl[dow] = day_pnl.get(dow, 0) + trade.pnl_usd

        if not day_pnl:
            return "N/A", "N/A"

        best = max(day_pnl, key=day_pnl.get)
        worst = min(day_pnl, key=day_pnl.get)
        return best, worst

    def generate_report(self) -> str:
        """Generate text performance report."""
        metrics = self.calculate_metrics()

        report = []
        report.append("=" * 70)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Trade Period: {self.trades[0].timestamp.strftime('%Y-%m-%d') if self.trades else 'N/A'} to {self.trades[-1].timestamp.strftime('%Y-%m-%d') if self.trades else 'N/A'}")
        report.append("=" * 70)

        # Summary
        report.append("\n[SUMMARY]")
        report.append("-" * 50)
        report.append(f"  Total Trades:     {metrics.total_trades}")
        report.append(f"  Win Rate:         {metrics.win_rate:.1f}%")
        report.append(f"  Total P&L:        ${metrics.total_pnl_usd:+,.2f} ({metrics.total_pnl_pct:+.1f}%)")
        report.append(f"  Profit Factor:    {metrics.profit_factor:.2f}")

        # Win/Loss Stats
        report.append("\n[WIN/LOSS STATISTICS]")
        report.append("-" * 50)
        report.append(f"  Winning Trades:   {metrics.winning_trades}")
        report.append(f"  Losing Trades:    {metrics.losing_trades}")
        report.append(f"  Avg Win:          ${metrics.avg_win_usd:+.2f} ({metrics.avg_win_pct:+.2f}%)")
        report.append(f"  Avg Loss:         ${metrics.avg_loss_usd:.2f} ({metrics.avg_loss_pct:.2f}%)")
        report.append(f"  Expectancy:       ${metrics.expectancy:.2f} per trade")
        report.append(f"  Win Streak:       {metrics.longest_win_streak}")
        report.append(f"  Loss Streak:      {metrics.longest_loss_streak}")

        # Risk Metrics
        report.append("\n[RISK METRICS]")
        report.append("-" * 50)
        report.append(f"  Max Drawdown:     ${metrics.max_drawdown_usd:.2f} ({metrics.max_drawdown_pct:.1f}%)")
        report.append(f"  Avg Drawdown:     {metrics.avg_drawdown_pct:.1f}%")
        report.append(f"  DD Duration:      {metrics.max_drawdown_duration_days} trades")

        # Risk-Adjusted Returns
        report.append("\n[RISK-ADJUSTED RETURNS]")
        report.append("-" * 50)
        report.append(f"  Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        report.append(f"  Sortino Ratio:    {metrics.sortino_ratio:.2f}")
        report.append(f"  Calmar Ratio:     {metrics.calmar_ratio:.2f}")

        # Interpretation
        report.append("\n  Interpretation:")
        if metrics.sharpe_ratio >= 2.0:
            report.append("    Sharpe >= 2.0: Excellent risk-adjusted returns")
        elif metrics.sharpe_ratio >= 1.0:
            report.append("    Sharpe >= 1.0: Good risk-adjusted returns")
        else:
            report.append("    Sharpe < 1.0: Returns don't adequately compensate for risk")

        # Consistency
        report.append("\n[CONSISTENCY]")
        report.append("-" * 50)
        report.append(f"  Best Day:         ${metrics.best_day_pnl:+.2f}")
        report.append(f"  Worst Day:        ${metrics.worst_day_pnl:+.2f}")
        report.append(f"  Profitable Days:  {metrics.profitable_days_pct:.1f}%")
        report.append(f"  Avg Trades/Day:   {metrics.avg_trades_per_day:.1f}")

        # Time Analysis
        report.append("\n[TIME ANALYSIS]")
        report.append("-" * 50)
        report.append(f"  Avg Hold Time:    {metrics.avg_hold_time_minutes:.0f} minutes")
        report.append(f"  Best Hour (UTC):  {metrics.best_hour_utc}:00")
        report.append(f"  Worst Hour (UTC): {metrics.worst_hour_utc}:00")
        report.append(f"  Best Day:         {metrics.best_day_of_week}")
        report.append(f"  Worst Day:        {metrics.worst_day_of_week}")

        # Prop Firm Assessment
        report.append("\n[PROP FIRM ASSESSMENT]")
        report.append("-" * 50)

        # Typical prop firm requirements
        passed = []
        failed = []

        if metrics.win_rate >= 50:
            passed.append(f"Win Rate: {metrics.win_rate:.1f}% (>50%)")
        else:
            failed.append(f"Win Rate: {metrics.win_rate:.1f}% (<50%)")

        if metrics.profit_factor >= 1.5:
            passed.append(f"Profit Factor: {metrics.profit_factor:.2f} (>1.5)")
        else:
            failed.append(f"Profit Factor: {metrics.profit_factor:.2f} (<1.5)")

        if metrics.max_drawdown_pct <= 10:
            passed.append(f"Max DD: {metrics.max_drawdown_pct:.1f}% (<10%)")
        else:
            failed.append(f"Max DD: {metrics.max_drawdown_pct:.1f}% (>10%)")

        if metrics.sharpe_ratio >= 1.0:
            passed.append(f"Sharpe: {metrics.sharpe_ratio:.2f} (>1.0)")
        else:
            failed.append(f"Sharpe: {metrics.sharpe_ratio:.2f} (<1.0)")

        if metrics.longest_loss_streak <= 5:
            passed.append(f"Loss Streak: {metrics.longest_loss_streak} (<5)")
        else:
            failed.append(f"Loss Streak: {metrics.longest_loss_streak} (>5)")

        report.append("  PASSED:")
        for p in passed:
            report.append(f"    + {p}")

        if failed:
            report.append("  NEEDS IMPROVEMENT:")
            for f in failed:
                report.append(f"    - {f}")

        overall = "PASSED" if len(failed) == 0 else f"NEEDS WORK ({len(failed)} items)"
        report.append(f"\n  Overall Assessment: {overall}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def print_report(self):
        """Print the performance report."""
        print(self.generate_report())

    def save_report(self, path: str = None):
        """Save report to file."""
        if path is None:
            path = f"performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.generate_report()
        with open(path, "w") as f:
            f.write(report)

        print(f"Report saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PERFORMANCE REPORT GENERATOR")
    print("=" * 70)

    generator = PerformanceReportGenerator(initial_capital=10000.0)

    # Try to load from evidence database
    db_path = Path(__file__).parent / "evidence_collector.db"
    if db_path.exists():
        generator.load_from_evidence_db(str(db_path))
    else:
        print("No evidence database found. Using sample data.")
        # Generate sample trades for demo
        import random
        base_time = datetime.datetime.utcnow() - datetime.timedelta(days=30)

        for i in range(100):
            is_win = random.random() < 0.73  # 73% win rate
            pnl_pct = random.uniform(1.5, 4.0) if is_win else random.uniform(-2.5, -1.0)
            pnl_usd = 200 * (pnl_pct / 100)

            trade = TradeData(
                timestamp=base_time + datetime.timedelta(hours=i * 4),
                symbol=random.choice(["ATOMUSDT", "SOLUSDT"]),
                side="BUY",
                entry_price=10.0,
                exit_price=10.0 * (1 + pnl_pct / 100),
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                hold_time_minutes=random.uniform(30, 180),
                signals=["BB", "MACD"],
                regime="sideways"
            )
            generator.add_trade(trade)

    generator.print_report()
