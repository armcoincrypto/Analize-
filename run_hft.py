#!/usr/bin/env python3
"""
Run HFT Bot
===========
Simple launcher for the HFT trading system.

Usage:
    python run_hft.py              # Paper trading with $10,000
    python run_hft.py --capital 5000   # Paper trading with $5,000
    python run_hft.py --live       # Live trading (not recommended yet)

To run in background on VPS:
    screen -S hft
    python run_hft.py
    # Then press Ctrl+A, D to detach
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hft_system.hft_bot import main

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                HFT DATA COLLECTION BOT                        ║
╠══════════════════════════════════════════════════════════════╣
║  Assets: ATOM, SUI, XRP                                       ║
║  Entry: 2/5 conditions (for data collection)                  ║
║  Exit: +2% TP | -1% SL | 90s time stop                        ║
║  Data: strategy_signals, cvd, orderbook, indicators           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    main()
