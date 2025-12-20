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
║       MICROSTRUCTURE BOT v2 (Event-Based Exits)               ║
╠══════════════════════════════════════════════════════════════╣
║  Assets: ATOM, SUI, XRP                                       ║
║  Entry: Orderbook imbalance >60% (1 condition only)           ║
║  Exits: TP +0.15% | SL -0.1% | Micro +0.05% on OB weaken      ║
║         OB Flip | Delta Negative | Spread 2x | No Movement    ║
║  Trade Flow: Aggressive buy/sell, delta 1s/3s tracking        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    main()
