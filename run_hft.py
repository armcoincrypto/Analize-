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
║           HFT OPTIMIZED BOT (Based on 70K signals)            ║
╠══════════════════════════════════════════════════════════════╣
║  Assets: ATOM, SUI, XRP                                       ║
║  Strategy: Orderbook + RSI focus (best combo: +0.024% at 5m)  ║
║  Exit: +1.5% TP | -0.8% SL | 180s time stop | Trailing stop   ║
║  Data: strategy_signals, cvd, orderbook, indicators           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    main()
