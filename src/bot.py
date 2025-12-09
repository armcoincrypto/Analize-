"""
LongTrader Bot - Main Orchestrator

Automated trading bot implementing:
- Bollinger Band mean reversion strategy
- 5 Protection filters
- Position management
- Telegram notifications
"""

import json
import time
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .bybit_client import BybitClient
from .signal_generator import SignalGenerator, SignalType
from .database import Database
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor, TradeResult
from .notifier import TelegramNotifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class LongTraderBot:
    """
    Main trading bot class.

    Workflow:
    1. Load configuration
    2. Initialize components
    3. Check signals for each coin
    4. Execute trades if conditions met
    5. Monitor open positions for exit
    6. Send notifications
    7. Sleep and repeat
    """

    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.running = False

        # Initialize components
        self._init_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return json.load(f)

    def _init_components(self):
        """Initialize all bot components."""
        logger.info("Initializing LongTrader components...")

        # Exchange client
        exchange_config = self.config.get("exchange", {})
        self.client = BybitClient(
            api_key=exchange_config.get("api_key", ""),
            api_secret=exchange_config.get("api_secret", ""),
            testnet=exchange_config.get("testnet", True),
        )

        # Database
        self.db = Database("data/longtrader.db")

        # Position manager
        self.pm = PositionManager(self.db)

        # Signal generator
        self.signal_gen = SignalGenerator(self.config)

        # Risk manager
        self.rm = RiskManager(self.config)

        # Trade executor
        bot_config = self.config.get("bot", {})
        paper_mode = bot_config.get("mode", "paper") == "paper"
        self.executor = TradeExecutor(
            self.client, self.pm, self.rm, paper_mode=paper_mode
        )

        # Telegram notifier
        tg_config = self.config.get("notifications", {}).get("telegram", {})
        self.notifier = TelegramNotifier(
            bot_token=tg_config.get("bot_token", ""),
            chat_id=tg_config.get("chat_id", ""),
            enabled=tg_config.get("enabled", False),
        )

        # Set paper balance if needed
        if paper_mode:
            self.db.set_config("paper_balance", 1000.0)

        logger.info("All components initialized")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received...")
        self.running = False

    def _get_enabled_coins(self) -> Dict[str, Dict[str, Any]]:
        """Get list of enabled coins from config."""
        coins = self.config.get("coins", {})
        return {k: v for k, v in coins.items() if v.get("enabled", True)}

    def _fetch_market_data(self, symbol: str, days: int = 250):
        """Fetch OHLCV data for a symbol."""
        try:
            df = self.client.get_klines(symbol, "D", days)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _process_buy_signal(
        self,
        symbol: str,
        signal,
        coin_config: Dict[str, Any],
    ) -> Optional[TradeResult]:
        """Process a buy signal."""
        # Get balance and check risk
        balance = self.executor.get_balance()
        open_positions = self.pm.get_open_positions()

        risk_check = self.rm.can_open_position(balance, open_positions, symbol)

        if not risk_check.allowed:
            logger.info(f"{symbol}: Cannot open position - {risk_check.reason}")
            return None

        # Calculate position size
        qty, usdt_amount = self.rm.calculate_position_size(balance, signal.price)

        logger.info(f"{symbol}: Opening position - {qty} @ ${signal.price:.4f}")

        # Execute buy
        result = self.executor.execute_buy(symbol, usdt_amount, signal.price)

        if result.success:
            # Log signal
            self.pm.log_signal(
                symbol=symbol,
                signal_type=signal.signal_type.value,
                price=signal.price,
                lower_band=signal.lower_band,
                middle_band=signal.middle_band,
                protection_passed=signal.protection.passed_count,
                protection_details=str(signal.protection.failed_filters()),
                action_taken="BUY",
            )

            # Send notification
            paper_mode = self.config.get("bot", {}).get("mode", "paper") == "paper"
            self.notifier.notify_buy(
                symbol=symbol,
                qty=result.filled_qty,
                price=result.filled_price,
                usdt_amount=result.filled_price * result.filled_qty,
                protection_passed=signal.protection.passed_count,
                paper_mode=paper_mode,
            )

        return result

    def _process_exit_signal(
        self,
        symbol: str,
        df,
        position,
        coin_config: Dict[str, Any],
    ) -> Optional[TradeResult]:
        """Check and process exit conditions."""
        bb_period = coin_config.get("period", 20)
        bb_std = coin_config.get("std_dev", 1.5)

        should_exit, reason, current_price = self.signal_gen.check_exit(
            symbol=symbol,
            df=df,
            entry_price=position.entry_price,
            entry_date=position.entry_date,
            bb_period=bb_period,
            bb_std=bb_std,
        )

        if not should_exit:
            return None

        logger.info(f"{symbol}: Exit signal - {reason}")

        # Execute sell
        result = self.executor.execute_sell(symbol, position, reason)

        if result.success:
            # Calculate P&L
            pnl_usdt = (result.filled_price - position.entry_price) * position.entry_qty
            pnl_pct = ((result.filled_price - position.entry_price) / position.entry_price) * 100

            # Send notification
            paper_mode = self.config.get("bot", {}).get("mode", "paper") == "paper"
            self.notifier.notify_sell(
                symbol=symbol,
                qty=position.entry_qty,
                entry_price=position.entry_price,
                exit_price=result.filled_price,
                pnl_usdt=pnl_usdt,
                pnl_pct=pnl_pct,
                reason=reason,
                paper_mode=paper_mode,
            )

        return result

    def check_signals(self):
        """Check signals for all enabled coins."""
        coins = self._get_enabled_coins()

        for symbol, coin_config in coins.items():
            try:
                logger.info(f"Checking {symbol}...")

                # Fetch market data
                df = self._fetch_market_data(symbol)
                if df is None or df.empty:
                    continue

                # Check if we have open position
                position = self.pm.get_position(symbol)

                if position:
                    # Check exit conditions
                    self._process_exit_signal(symbol, df, position, coin_config)
                else:
                    # Generate signal
                    bb_period = coin_config.get("period", 20)
                    bb_std = coin_config.get("std_dev", 1.5)

                    signal = self.signal_gen.analyze(
                        symbol=symbol,
                        df=df,
                        bb_period=bb_period,
                        bb_std=bb_std,
                    )

                    # Log signal info
                    logger.info(
                        f"{symbol}: {signal.signal_type.value} | "
                        f"Price: ${signal.price:.4f} | "
                        f"Lower: ${signal.lower_band:.4f} | "
                        f"Protection: {signal.protection.passed_count}/4"
                    )

                    # Process strong buy signals
                    if signal.signal_type == SignalType.STRONG_BUY:
                        self._process_buy_signal(symbol, signal, coin_config)

                    # Optionally notify weak signals
                    elif signal.signal_type == SignalType.WEAK_BUY:
                        self.notifier.notify_signal(
                            symbol=symbol,
                            signal_type=signal.signal_type.value,
                            price=signal.price,
                            lower_band=signal.lower_band,
                            middle_band=signal.middle_band,
                            protection_passed=signal.protection.passed_count,
                            reason=signal.reason,
                        )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.notifier.notify_error(str(e), f"Processing {symbol}")

    def run(self):
        """Main bot loop."""
        logger.info("=" * 60)
        logger.info("LongTrader Bot Starting")
        logger.info("=" * 60)

        mode = self.config.get("bot", {}).get("mode", "paper")
        interval = self.config.get("bot", {}).get("check_interval_seconds", 3600)

        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Check interval: {interval}s")
        logger.info(f"Coins: {list(self._get_enabled_coins().keys())}")

        # Send startup notification
        self.notifier.notify_startup(self.config)

        self.running = True
        last_check = 0

        while self.running:
            try:
                current_time = time.time()

                if current_time - last_check >= interval:
                    logger.info("-" * 40)
                    logger.info(f"Signal check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    self.check_signals()

                    # Log stats
                    stats = self.pm.get_stats()
                    logger.info(
                        f"Stats: {stats['open_positions']} open, "
                        f"{stats['closed_positions']} closed, "
                        f"Win rate: {stats['win_rate']}%, "
                        f"Total P&L: ${stats['total_pnl_usdt']}"
                    )

                    last_check = current_time

                time.sleep(10)  # Sleep between checks

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.notifier.notify_error(str(e), "Main loop")
                time.sleep(60)  # Wait before retry

        # Shutdown
        logger.info("Shutting down...")
        self.notifier.notify_shutdown("Manual stop")
        logger.info("LongTrader Bot stopped")

    def run_once(self):
        """Run a single signal check (for testing)."""
        logger.info("Running single signal check...")
        self.check_signals()

        stats = self.pm.get_stats()
        logger.info(f"Stats: {stats}")

        return stats


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LongTrader Bot")
    parser.add_argument(
        "--config", "-c",
        default="config/config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )

    args = parser.parse_args()

    bot = LongTraderBot(config_path=args.config)

    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()
