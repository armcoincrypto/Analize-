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
║       MICROSTRUCTURE BOT v5 (Liquidity Reaction Engine)       ║
╠══════════════════════════════════════════════════════════════╣
║  Entry: Orderbook imbalance >60% (1 condition only)           ║
║  Exits: TP +0.15% | SL -0.1% | Micro +0.05% on OB weaken      ║
║         OB Flip | Delta Negative | Spread 2x | No Movement    ║
╠══════════════════════════════════════════════════════════════╣
║  TASK 6-7: Trade Quality - Entry/Exit metrics, slippage       ║
║  TASK 8: Trade Causality - WHY signals triggered              ║
║  TASK 9: Edge Validation - Real vs Fake edge detection        ║
║  TASK 10: Market Regime - WHICH context (5 regimes)           ║
║  TASK 11: No-Trade Zones - WHEN NOT to trade                  ║
║   - Blocks: spread unstable, delta noise, OB unstable,        ║
║     low liquidity, bad regime                                 ║
║  Expected: 2-20s trades, small consistent edge                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    main()
