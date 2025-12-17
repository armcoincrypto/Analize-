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

from .config import SYSTEM_CONFIG, ASSETS, TradingMode
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

        # Wire up callbacks
        self.signal_engine.on_signal = self._on_signal
        self.execution_engine.on_trade = self._on_trade
        self.execution_engine.on_exit = self._on_exit

        # Status tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time = None

        logger.info(f"HFT Bot initialized")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  Capital: ${self.capital:,.2f}")
        logger.info(f"  Assets: {', '.join(SYSTEM_CONFIG.enabled_assets)}")
        logger.info(f"  Entry: 3/5 conditions required")
        logger.info(f"  Exit: +2% TP, -1% SL, 90s time stop")

    async def _on_signal(self, signal):
        """Handle new signal from signal engine."""
        self.signals_generated += 1

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

    async def _signal_scan_loop(self):
        """Periodically scan for signals."""
        # Wait for WebSocket data to populate
        logger.info("Waiting 30s for WebSocket data to populate...")
        await asyncio.sleep(30)

        logger.info("Starting signal scan loop")

        while self.running:
            try:
                # Scan all assets for signals
                signals = self.signal_engine.scan_all_assets()

                # Log market snapshot periodically (every 60s)
                status = self.signal_engine.get_status()
                for symbol, data in status.items():
                    if data.get("price"):
                        self.trade_logger.log_market_snapshot(symbol, data)

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
        """Print current bot status."""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600

        risk_status = self.risk_controller.get_status()
        ws_stats = self.ws_manager.get_connection_stats()
        positions = self.execution_engine.get_open_positions_status()

        logger.info("=" * 60)
        logger.info(f"STATUS UPDATE | Uptime: {uptime:.1f}h")
        logger.info(f"  Signals: {self.signals_generated} | Trades: {self.trades_executed}")
        logger.info(f"  Capital: ${risk_status['capital']['current']:,.2f} "
                   f"(Daily PnL: ${risk_status['capital']['daily_pnl']:+,.2f})")
        logger.info(f"  Daily: {risk_status['daily_stats']['wins']}W / "
                   f"{risk_status['daily_stats']['losses']}L | "
                   f"Losses remaining: {risk_status['daily_stats']['max_losses_remaining']}")
        logger.info(f"  WebSocket: {ws_stats['message_count']} msgs | "
                   f"Reconnects: {ws_stats['reconnect_count']}")

        if positions:
            logger.info(f"  Open positions:")
            for symbol, pos in positions.items():
                logger.info(f"    {symbol}: {pos['pnl_pct']:+.2f}% | "
                          f"Hold: {pos['hold_time_sec']:.0f}s / {pos['time_stop_at']}s")
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
