"""
HFT Trading Bot
===============
Main entry point for the HFT trading system.

Combines all components:
- WebSocket Manager: Real-time data feeds
- Signal Engine: Multi-condition signal detection
- Risk Controller: Position and risk management
- Execution Engine: Trade execution with TP/SL/Time stops
- Trade Logger: Persistent logging for analysis

Usage:
    python -m hft_system.hft_bot [--live] [--capital 10000]
"""

import asyncio
import logging
import signal
import sys
import argparse
from datetime import datetime

from .config import SYSTEM_CONFIG, ASSETS, TradingMode, get_asset_config
from .websocket_manager import WebSocketManager
from .signal_engine import SignalEngine
from .risk_controller import RiskController
from .execution_engine import ExecutionEngine
from .trade_logger import TradeLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class HFTBot:
    """
    High-Frequency Trading Bot.

    Monitors multiple assets via WebSocket and executes trades
    when 3+ of 5 conditions are met.
    """

    def __init__(self, mode: TradingMode = TradingMode.PAPER, capital: float = None):
        self.mode = mode
        self.capital = capital or SYSTEM_CONFIG.initial_capital
        self.running = False

        # Initialize components
        self.ws_manager = WebSocketManager()
        self.risk_controller = RiskController(initial_capital=self.capital)
        self.signal_engine = SignalEngine(self.ws_manager)
        self.execution_engine = ExecutionEngine(
            self.ws_manager,
            self.risk_controller,
            mode=self.mode
        )
        self.trade_logger = TradeLogger()

        # Wire up callbacks (don't use async callback for signal - handle in scan loop)
        self.execution_engine.on_trade = self._on_trade
        self.execution_engine.on_exit = self._on_exit

        # Status tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time = None

        # TASK 6: Track entry conditions for quality metrics
        self.entry_conditions: dict = {}  # trade_id -> {flow, orderbook, signal_price}

        # TASK 10: Market regime tracking
        self.current_regime: dict = {}  # symbol -> regime type
        self.regime_confidence: dict = {}  # symbol -> confidence
        # Preferred regimes for trading (can be configured)
        self.preferred_regimes = ["high_vol_trend", "mean_reversion"]
        self.avoid_regimes = ["liquidity_vacuum", "news_spike"]
        self.filter_by_regime = False  # Set True to skip signals in bad regimes

        # TASK 11: No-trade zone tracking
        self.enable_no_trade_zones = True  # Set False to disable blocking
        self.signals_blocked = 0

        logger.info(f"HFT Bot initialized")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  Capital: ${self.capital:,.2f}")
        logger.info(f"  Assets: {', '.join(SYSTEM_CONFIG.enabled_assets)}")
        logger.info(f"  Entry: {SYSTEM_CONFIG.min_entry_conditions}/5 conditions required")
        # Get exit settings from first asset config
        asset_config = get_asset_config(SYSTEM_CONFIG.enabled_assets[0])
        logger.info(f"  Exit: +{asset_config.take_profit_pct}% TP, -{asset_config.stop_loss_pct}% SL, {asset_config.time_stop_seconds}s time stop")
        logger.info(f"  Optimized: Orderbook + RSI combo priority (best performer)")

    async def _on_signal(self, signal):
        """Handle new signal from signal engine."""
        self.signals_generated += 1

        # TASK 10: Check market regime filter
        current_regime = self.current_regime.get(signal.symbol, "unknown")
        if self.filter_by_regime and current_regime in self.avoid_regimes:
            logger.info(f"Signal SKIPPED: {signal.symbol} - regime={current_regime} (avoided)")
            # Log as blocked signal with regime reason
            market_conditions = {"spread_pct": 0, "spread_change_1s": 0,
                                "delta_variance": 0, "ob_volume_instability": 0, "liquidity_depth": 0}
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="regime_avoided",
                block_details=f"Avoided regime: {current_regime}",
                market_conditions=market_conditions,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        # TASK 11: Check no-trade zone before entry
        if self.enable_no_trade_zones:
            is_blocked, block_reason, block_details, market_conditions = \
                self.ws_manager.detect_no_trade_zone(signal.symbol)

            if is_blocked:
                logger.info(f"NO-TRADE ZONE: {signal.symbol} - {block_reason}: {block_details}")
                self.trade_logger.log_blocked_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    block_reason=block_reason,
                    block_details=block_details,
                    market_conditions=market_conditions,
                    regime=current_regime
                )
                self.signals_blocked += 1
                return

        # Check risk
        risk_decision = self.risk_controller.check_risk(signal)

        # Log signal
        self.trade_logger.log_signal(signal, risk_decision)

        if risk_decision.approved:
            # Execute trade
            result = await self.execution_engine.execute_entry(signal, risk_decision)

            if result.success:
                self.trades_executed += 1
                self.trade_logger.log_trade_entry(result, signal)

                # Log MICROSTRUCTURE trade flow at entry
                trade_id = result.order_id or f"{result.symbol}_{result.timestamp}"
                trade_flow = self.ws_manager.get_trade_flow(signal.symbol)
                orderbook = self.ws_manager.get_orderbook(signal.symbol)
                self.trade_logger.log_trade_entry_flow(
                    trade_id=trade_id,
                    symbol=signal.symbol,
                    trade_flow=trade_flow,
                    orderbook_imbalance=orderbook.imbalance_ratio if orderbook else None,
                    spread_pct=orderbook.spread if orderbook else None,
                    snapshot_type="entry"
                )

                # TASK 6: Store entry conditions for quality metrics
                self.entry_conditions[trade_id] = {
                    "flow": trade_flow.copy(),
                    "orderbook": {
                        "imbalance": orderbook.imbalance_ratio if orderbook else 0,
                        "spread_pct": orderbook.spread if orderbook else 0
                    },
                    "signal_price": signal.entry_price,  # Expected entry price
                    "actual_price": result.entry_price,   # Actual fill price
                    "side": signal.signal_type.value
                }

                # TASK 8: Log trade causality - WHY this signal triggered
                ob_causality = self.ws_manager.get_orderbook_causality_data(signal.symbol)
                cvd_data = {
                    "delta_1s": trade_flow.get("delta_1s", 0),
                    "delta_3s": trade_flow.get("delta_3s", 0),
                    "delta_5s": trade_flow.get("net_delta", 0)  # Use net_delta as 5s proxy
                }
                price_velocity = self.ws_manager.get_price_velocity(signal.symbol)
                btc_data = self.ws_manager.get_btc_data()

                primary_cause = self.trade_logger.log_trade_causality(
                    trade_id=trade_id,
                    symbol=signal.symbol,
                    orderbook_data=ob_causality,
                    cvd_data=cvd_data,
                    price_velocity=price_velocity,
                    btc_data=btc_data
                )

                # Log primary cause in trade entry message
                logger.info(f"  Primary cause: {primary_cause}")

                # TASK 10: Log regime at entry
                entry_regime = self.current_regime.get(signal.symbol, "unknown")
                self.trade_logger.update_trade_regime(signal.symbol, entry_regime, is_entry=True)
                logger.info(f"  Market regime: {entry_regime}")
        else:
            logger.info(f"Signal rejected: {signal.symbol} - {risk_decision.message}")

    def _on_trade(self, result):
        """Handle trade execution result."""
        pass  # Already logged in _on_signal

    def _on_exit(self, result):
        """Handle position exit."""
        mfe = self.execution_engine.position_mfe.get(result.symbol, 0)
        mae = self.execution_engine.position_mae.get(result.symbol, 0)
        self.trade_logger.log_trade_exit(result, mfe, mae)

        # TASK 10: Log regime at exit (before status changes to closed)
        exit_regime = self.current_regime.get(result.symbol, "unknown")
        self.trade_logger.update_trade_regime(result.symbol, exit_regime, is_entry=False)

        # Log MICROSTRUCTURE trade flow at exit
        trade_id = f"{result.symbol}_{result.entry_time}"
        exit_flow = self.ws_manager.get_trade_flow(result.symbol)
        orderbook = self.ws_manager.get_orderbook(result.symbol)
        self.trade_logger.log_trade_entry_flow(
            trade_id=trade_id,
            symbol=result.symbol,
            trade_flow=exit_flow,
            orderbook_imbalance=orderbook.imbalance_ratio if orderbook else None,
            spread_pct=orderbook.spread if orderbook else None,
            snapshot_type="exit"
        )

        # TASK 6: Log consolidated trade quality metrics
        entry_cond = self.entry_conditions.pop(trade_id, None)
        if entry_cond:
            exit_orderbook = {
                "imbalance": orderbook.imbalance_ratio if orderbook else 0,
                "spread_pct": orderbook.spread if orderbook else 0
            }
            self.trade_logger.log_trade_quality_metrics(
                trade_id=trade_id,
                symbol=result.symbol,
                side=entry_cond.get("side", "unknown"),
                entry_flow=entry_cond.get("flow", {}),
                exit_flow=exit_flow,
                entry_orderbook=entry_cond.get("orderbook", {}),
                exit_orderbook=exit_orderbook,
                time_in_trade_sec=result.hold_time_sec,
                exit_reason=result.reason.value,
                pnl_pct=result.pnl_pct,
                expected_entry_price=entry_cond.get("signal_price", result.entry_price),
                actual_entry_price=entry_cond.get("actual_price", result.entry_price),
                expected_exit_price=result.exit_price,  # For paper, expected=actual
                actual_exit_price=result.exit_price
            )

        # TASK 9: Log edge validation metrics
        edge_data = self.execution_engine.get_edge_data(trade_id)
        if edge_data:
            self.trade_logger.log_trade_edge(
                trade_id=trade_id,
                symbol=result.symbol,
                side=edge_data.get("side", "unknown"),
                seconds_to_max_favorable=edge_data.get("seconds_to_max_favorable", 0),
                max_favorable_pct=edge_data.get("max_favorable_pct", 0),
                seconds_to_ob_decay=edge_data.get("seconds_to_ob_decay", 0),
                ob_decay_amount=edge_data.get("ob_decay_amount", 0),
                delta_persistence_sec=edge_data.get("delta_persistence_sec", 0),
                entry_imbalance=edge_data.get("entry_imbalance", 0),
                entry_delta=edge_data.get("entry_delta", 0),
                entry_spread=edge_data.get("entry_spread", 0),
                edge_duration_sec=edge_data.get("edge_duration_sec", 0),
                final_pnl_pct=result.pnl_pct,
                exit_reason=result.reason.value
            )

    async def _signal_scan_loop(self):
        """Periodically scan for signals."""
        # Wait for WebSocket data to populate
        logger.info("Waiting 30s for WebSocket data to populate...")
        await asyncio.sleep(30)

        logger.info("Starting signal scan loop")
        scan_count = 0

        while self.running:
            try:
                # Scan all assets for signals
                signals = self.signal_engine.scan_all_assets()

                # Process any valid signals
                for signal in signals:
                    await self._on_signal(signal)

                # Log indicator snapshot every scan (1 second)
                for symbol in SYSTEM_CONFIG.enabled_assets:
                    signal = self.signal_engine.check_all_conditions(symbol)
                    status = self.signal_engine.get_status().get(symbol, {})

                    # Add conditions_met to status
                    status["conditions_met"] = signal.conditions_met

                    # Log indicator values
                    self.trade_logger.log_indicator_snapshot(symbol, status)

                    # Log strategy signal for multi-strategy analysis
                    conditions = {c.name: c.triggered for c in signal.conditions}
                    self.trade_logger.log_strategy_signal(symbol, signal.entry_price, conditions)

                    # Log orderbook snapshot
                    orderbook = self.ws_manager.get_orderbook(symbol)
                    if orderbook:
                        self.trade_logger.log_orderbook_snapshot(symbol, {
                            "best_bid": orderbook.best_bid,
                            "best_ask": orderbook.best_ask,
                            "bid_volume": orderbook.bid_volume,
                            "ask_volume": orderbook.ask_volume,
                            "spread_pct": orderbook.spread,
                            "imbalance": orderbook.imbalance_ratio
                        })

                    # Update old signals with current price (for outcome tracking)
                    if signal.entry_price and signal.entry_price > 0:
                        self.trade_logger.update_signal_outcomes(symbol, signal.entry_price)

                    # Log CVD (Cumulative Volume Delta) snapshot
                    cvd_data = self.ws_manager.get_cvd(symbol)
                    self.trade_logger.log_cvd_snapshot(symbol, cvd_data)

                # Batch commit every 10 scans
                scan_count += 1
                if scan_count % 10 == 0:
                    self.trade_logger.batch_commit()

                # TASK 10: Classify market regime every 30 scans (30 seconds)
                if scan_count % 30 == 0:
                    for symbol in SYSTEM_CONFIG.enabled_assets:
                        regime, confidence, metrics = self.ws_manager.classify_market_regime(symbol)
                        self.current_regime[symbol] = regime
                        self.regime_confidence[symbol] = confidence
                        self.trade_logger.log_market_regime(symbol, regime, metrics, confidence)

                await asyncio.sleep(1)  # Scan every second

            except Exception as e:
                logger.error(f"Error in signal scan: {e}")
                await asyncio.sleep(5)

    async def _status_loop(self):
        """Periodically print status."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Every minute
                self._print_status()
            except Exception as e:
                logger.error(f"Error in status loop: {e}")

    def _print_status(self):
        """Print current bot status with condition debug info."""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600

        risk_status = self.risk_controller.get_status()
        ws_stats = self.ws_manager.get_connection_stats()
        positions = self.execution_engine.get_open_positions_status()

        logger.info("=" * 60)
        logger.info(f"STATUS UPDATE | Uptime: {uptime:.1f}h")
        logger.info(f"  Signals: {self.signals_generated} | Trades: {self.trades_executed} | Blocked: {self.signals_blocked}")
        logger.info(f"  Capital: ${risk_status['capital']['current']:,.2f} "
                   f"(Daily PnL: ${risk_status['capital']['daily_pnl']:+,.2f})")
        logger.info(f"  Daily: {risk_status['daily_stats']['wins']}W / "
                   f"{risk_status['daily_stats']['losses']}L | "
                   f"Losses remaining: {risk_status['daily_stats']['max_losses_remaining']}")
        logger.info(f"  WebSocket: {ws_stats['message_count']} msgs | "
                   f"Reconnects: {ws_stats['reconnect_count']}")

        # Show condition status for each asset
        logger.info("  CONDITIONS (need 3/5):")
        for symbol in SYSTEM_CONFIG.enabled_assets:
            signal = self.signal_engine.check_all_conditions(symbol)
            triggered = [c.name[:6] for c in signal.conditions if c.triggered]
            price = signal.entry_price
            regime = self.current_regime.get(symbol, "unknown")
            logger.info(f"    {symbol}: ${price:.4f} | {signal.conditions_met}/5 | {triggered} | regime={regime}")

        if positions:
            logger.info(f"  Open positions:")
            for symbol, pos in positions.items():
                logger.info(f"    {symbol}: {pos['pnl_pct']:+.2f}% | "
                          f"Hold: {pos['hold_time_sec']:.0f}s / {pos['time_stop_at']}s")

        # TASK 7: Microstructure performance metrics
        micro_stats = self.trade_logger.get_microstructure_stats()
        if "total_trades" in micro_stats and micro_stats["total_trades"] > 0:
            logger.info("  MICROSTRUCTURE METRICS:")
            logger.info(f"    Avg PnL: {micro_stats['avg_pnl_pct']:+.4f}% | "
                       f"Net Edge: {micro_stats['net_edge_after_costs_pct']:+.4f}%")
            logger.info(f"    Avg Hold: {micro_stats['avg_hold_time_sec']:.1f}s | "
                       f"Exit <5s: {micro_stats['pct_exited_within_5s']:.1f}%")
            logger.info(f"    Invalidation Exits: {micro_stats['pct_exited_by_invalidation']:.1f}% | "
                       f"Costs: {micro_stats['avg_total_cost_pct']:.4f}%")
            if "evaluation" in micro_stats:
                logger.info(f"    {micro_stats['evaluation']}")

        # TASK 9: Edge validation stats
        edge_stats = self.trade_logger.get_edge_stats()
        if "total_trades" in edge_stats and edge_stats["total_trades"] > 0:
            logger.info("  EDGE VALIDATION:")
            logger.info(f"    Real Edge: {edge_stats['pct_real_edge']:.1f}% | "
                       f"Fake Edge: {edge_stats['pct_fake_edge']:.1f}%")
            logger.info(f"    Avg Edge Duration: {edge_stats['avg_edge_duration_sec']:.1f}s | "
                       f"Time to MFE: {edge_stats['avg_time_to_max_favorable_sec']:.1f}s")
            if "edge_pnl_comparison" in edge_stats:
                real_pnl = edge_stats["edge_pnl_comparison"].get("real", {}).get("avg_pnl_pct", 0)
                fake_pnl = edge_stats["edge_pnl_comparison"].get("fake", {}).get("avg_pnl_pct", 0)
                logger.info(f"    Real Edge Avg PnL: {real_pnl:+.4f}% | Fake Edge Avg PnL: {fake_pnl:+.4f}%")

        # TASK 11: No-trade zone / blocked signal stats
        if self.signals_blocked > 0:
            blocked_stats = self.trade_logger.get_blocked_signal_stats()
            logger.info("  NO-TRADE ZONE STATS:")
            logger.info(f"    Blocked: {blocked_stats.get('total_blocked', 0)} | "
                       f"Block Rate: {blocked_stats.get('block_rate_pct', 0):.1f}%")
            if blocked_stats.get("reason_breakdown"):
                reasons = [f"{k}: {v['count']}" for k, v in blocked_stats["reason_breakdown"].items()]
                logger.info(f"    Reasons: {', '.join(reasons)}")
        logger.info("=" * 60)

    async def start(self):
        """Start the HFT bot."""
        self.running = True
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("HFT BOT STARTING")
        logger.info("=" * 60)

        # Start all tasks concurrently
        tasks = [
            asyncio.create_task(self.ws_manager.connect()),
            asyncio.create_task(self.execution_engine.start_monitoring()),
            asyncio.create_task(self._signal_scan_loop()),
            asyncio.create_task(self._status_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the HFT bot gracefully."""
        logger.info("Stopping HFT bot...")
        self.running = False

        # Stop components
        self.execution_engine.stop_monitoring()
        await self.ws_manager.disconnect()

        # Close any open positions
        positions = self.risk_controller.get_all_positions()
        if positions:
            logger.warning(f"Closing {len(positions)} open positions...")
            await self.execution_engine.emergency_close_all("Bot shutdown")

        # Print final summary
        summary = self.trade_logger.get_performance_summary()
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)

        if "message" not in summary:
            logger.info(f"  Total trades: {summary['total_trades']}")
            logger.info(f"  Win rate: {summary['win_rate']:.1f}%")
            logger.info(f"  Total PnL: ${summary['total_pnl']:.2f}")
            logger.info(f"  Profit factor: {summary['profit_factor']:.2f}")
            logger.info(f"  Avg hold time: {summary['avg_hold_time_sec']:.0f}s")
            logger.info(f"  Exit reasons: {summary['exit_reasons']}")
        else:
            logger.info(f"  {summary['message']}")

        # TASK 7: Microstructure final stats
        micro_stats = self.trade_logger.get_microstructure_stats()
        if "total_trades" in micro_stats and micro_stats["total_trades"] > 0:
            logger.info("-" * 60)
            logger.info("MICROSTRUCTURE PERFORMANCE (what matters)")
            logger.info(f"  Avg PnL per trade: {micro_stats['avg_pnl_pct']:+.4f}%")
            logger.info(f"  Net edge after costs: {micro_stats['net_edge_after_costs_pct']:+.4f}%")
            logger.info(f"  Avg hold time: {micro_stats['avg_hold_time_sec']:.1f}s")
            logger.info(f"  Exited within 5s: {micro_stats['pct_exited_within_5s']:.1f}%")
            logger.info(f"  Exited by invalidation: {micro_stats['pct_exited_by_invalidation']:.1f}%")
            logger.info(f"  Avg total cost: {micro_stats['avg_total_cost_pct']:.4f}%")
            logger.info(f"  Exit breakdown: {micro_stats.get('exit_breakdown', {})}")
            if "evaluation" in micro_stats:
                logger.info(f"  EVALUATION: {micro_stats['evaluation']}")

        # TASK 8: Causality analysis - WHY trades happened
        causality_stats = self.trade_logger.get_causality_stats()
        if "cause_breakdown" in causality_stats and causality_stats["cause_breakdown"]:
            logger.info("-" * 60)
            logger.info("TRADE CAUSALITY (WHY signals triggered)")
            for cause, data in causality_stats["cause_breakdown"].items():
                logger.info(f"  {cause}: {data['count']} trades (strength: {data['avg_strength']:.4f})")

        # TASK 9: Edge validation - WHEN edge is real vs fake
        edge_stats = self.trade_logger.get_edge_stats()
        if "total_trades" in edge_stats and edge_stats["total_trades"] > 0:
            logger.info("-" * 60)
            logger.info("EDGE VALIDATION (real vs fake signals)")
            logger.info(f"  Real edge: {edge_stats['pct_real_edge']:.1f}%")
            logger.info(f"  Fake edge: {edge_stats['pct_fake_edge']:.1f}%")
            logger.info(f"  Avg edge duration: {edge_stats['avg_edge_duration_sec']:.1f}s")
            logger.info(f"  Avg time to max favorable: {edge_stats['avg_time_to_max_favorable_sec']:.1f}s")
            logger.info(f"  Avg OB decay time: {edge_stats['avg_ob_decay_time_sec']:.1f}s")
            logger.info(f"  Avg delta persistence: {edge_stats['avg_delta_persistence_sec']:.1f}s")

            # Show PnL comparison between real and fake edge trades
            if "edge_pnl_comparison" in edge_stats:
                pnl_comp = edge_stats["edge_pnl_comparison"]
                if "real" in pnl_comp:
                    logger.info(f"  Real edge trades: {pnl_comp['real']['count']} @ avg {pnl_comp['real']['avg_pnl_pct']:+.4f}%")
                if "fake" in pnl_comp:
                    logger.info(f"  Fake edge trades: {pnl_comp['fake']['count']} @ avg {pnl_comp['fake']['avg_pnl_pct']:+.4f}%")

            # Show per-symbol breakdown
            if "symbol_breakdown" in edge_stats and edge_stats["symbol_breakdown"]:
                logger.info("  Per-symbol edge quality:")
                for symbol, data in edge_stats["symbol_breakdown"].items():
                    logger.info(f"    {symbol}: {data['pct_real_edge']:.1f}% real | "
                               f"avg {data['avg_edge_duration']:.1f}s duration")

        # TASK 10: Market regime analysis - WHICH market context
        regime_stats = self.trade_logger.get_regime_stats()
        if regime_stats.get("total_trades_with_regime", 0) > 0:
            logger.info("-" * 60)
            logger.info("MARKET REGIME ANALYSIS (which conditions work best)")

            # Show performance by regime
            if "regime_performance" in regime_stats and regime_stats["regime_performance"]:
                logger.info("  Performance by regime:")
                for regime, data in regime_stats["regime_performance"].items():
                    logger.info(f"    {regime}: {data['trades']} trades | "
                               f"avg PnL {data['avg_pnl_pct']:+.4f}% | "
                               f"win rate {data['win_rate']:.1f}%")

            # Show recent regime distribution
            if "recent_regime_distribution" in regime_stats and regime_stats["recent_regime_distribution"]:
                logger.info("  Recent regime distribution (last 1h):")
                for regime, count in regime_stats["recent_regime_distribution"].items():
                    logger.info(f"    {regime}: {count} samples")

        # TASK 11: No-trade zone analysis - WHEN NOT to trade
        blocked_stats = self.trade_logger.get_blocked_signal_stats()
        if blocked_stats.get("total_blocked", 0) > 0:
            logger.info("-" * 60)
            logger.info("NO-TRADE ZONE ANALYSIS (blocked signals)")
            logger.info(f"  Total blocked: {blocked_stats['total_blocked']}")
            logger.info(f"  Block rate: {blocked_stats['block_rate_pct']:.1f}%")

            # Show breakdown by reason
            if "reason_breakdown" in blocked_stats and blocked_stats["reason_breakdown"]:
                logger.info("  Block reasons:")
                for reason, data in blocked_stats["reason_breakdown"].items():
                    pnl_str = f" (would have been {data['avg_would_have_pnl']:+.4f}%)" if data["avg_would_have_pnl"] else ""
                    logger.info(f"    {reason}: {data['count']} signals{pnl_str}")

            # Show estimated savings
            if blocked_stats.get("bad_trades_avoided", 0) > 0:
                logger.info(f"  Bad trades avoided: {blocked_stats['bad_trades_avoided']}")
                logger.info(f"  Avg loss avoided: {blocked_stats['avg_loss_avoided_pct']:+.4f}%")
                logger.info(f"  Estimated total savings: {blocked_stats['estimated_savings_pct']:+.4f}%")

        logger.info("=" * 60)

        # Close database
        self.trade_logger.close()

        logger.info("HFT bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HFT Trading Bot")
    parser.add_argument("--live", action="store_true", help="Run in live mode (default: paper)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    args = parser.parse_args()

    mode = TradingMode.LIVE if args.live else TradingMode.PAPER

    bot = HFTBot(mode=mode, capital=args.capital)

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown_handler():
        logger.info("Shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
