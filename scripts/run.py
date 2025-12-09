#!/usr/bin/env python3
"""
LongTrader Runner Script

Usage:
    python scripts/run.py                   # Run bot continuously
    python scripts/run.py --once            # Single check
    python scripts/run.py --config path     # Custom config
    python scripts/run.py --test            # Test all components
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot import LongTraderBot, main


def test_components():
    """Test all components."""
    print("=" * 60)
    print("LongTrader Component Tests")
    print("=" * 60)

    # Test Bybit client
    print("\n1. Testing Bybit Client...")
    from src.bybit_client import BybitClient
    client = BybitClient(testnet=True)
    ticker = client.get_ticker("XRPUSDT")
    print(f"   XRP/USDT: ${ticker.last_price}")
    print("   Bybit Client OK")

    # Test Signal Generator
    print("\n2. Testing Signal Generator...")
    from src.signal_generator import SignalGenerator
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    prices = 100 + np.cumsum(np.random.randn(250) * 2)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 250),
    })

    config = {"strategy": {"trend_ma_period": 200}}
    gen = SignalGenerator(config)
    signal = gen.analyze("TESTUSDT", df)
    print(f"   Signal: {signal.signal_type.value}")
    print("   Signal Generator OK")

    # Test Database
    print("\n3. Testing Database...")
    from src.database import Database, Position
    from datetime import datetime
    import uuid

    db = Database("data/test_components.db")
    pos = Position(
        id=str(uuid.uuid4()),
        symbol="TESTUSDT",
        side="LONG",
        entry_price=100.0,
        entry_qty=10.0,
        entry_date=datetime.now(),
        entry_order_id="test",
    )
    db.create_position(pos)
    print(f"   Created position: {pos.id[:8]}...")
    print("   Database OK")

    # Test Position Manager
    print("\n4. Testing Position Manager...")
    from src.position_manager import PositionManager
    pm = PositionManager(db)
    open_pos = pm.get_open_positions()
    print(f"   Open positions: {len(open_pos)}")
    print("   Position Manager OK")

    # Test Risk Manager
    print("\n5. Testing Risk Manager...")
    from src.risk_manager import RiskManager
    rm = RiskManager({"risk": {"position_size_pct": 5.0}})
    check = rm.can_open_position(1000.0, [], "XRPUSDT")
    print(f"   Can open: {check.allowed}, Amount: ${check.suggested_usdt}")
    print("   Risk Manager OK")

    # Test Notifier (disabled mode)
    print("\n6. Testing Notifier...")
    from src.notifier import TelegramNotifier
    notifier = TelegramNotifier("test", "test", enabled=False)
    result = notifier.send_message("Test")
    print(f"   Send (disabled): {result}")
    print("   Notifier OK")

    print("\n" + "=" * 60)
    print("All components working!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LongTrader Runner")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--config", "-c", default="config/config.json", help="Config path")

    args = parser.parse_args()

    if args.test:
        test_components()
    else:
        # Run main bot
        sys.argv = ["run.py"]
        if args.once:
            sys.argv.append("--once")
        if args.config != "config/config.json":
            sys.argv.extend(["--config", args.config])

        main()
