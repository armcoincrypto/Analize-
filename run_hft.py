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
║       MICROSTRUCTURE BOT v3 (Liquidity Reaction Engine)       ║
╠══════════════════════════════════════════════════════════════╣
║  Entry: Orderbook imbalance >60% (1 condition only)           ║
║  Exits: TP +0.15% | SL -0.1% | Micro +0.05% on OB weaken      ║
║         OB Flip | Delta Negative | Spread 2x | No Movement    ║
╠══════════════════════════════════════════════════════════════╣
║  Trade Quality (TASK 6-7): Entry/Exit metrics, slippage       ║
║  Trade Causality (TASK 8): WHY signals triggered              ║
║  Edge Validation (TASK 9): Real vs Fake edge detection        ║
║   - Tracks: time to MFE, OB decay, delta persistence          ║
║   - real_edge_flag = 1 if edge > historical median            ║
║  Expected: 2-20s trades, small consistent edge                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    main()
