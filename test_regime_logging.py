#!/usr/bin/env python3
"""Quick test of regime logging"""
import sys
sys.path.insert(0, '/home/user/Analize-')

from hft_system.trade_logger import TradeLogger
from hft_system.websocket_manager import WebSocketManager
import asyncio
import time

async def test():
    logger = TradeLogger()
    ws = WebSocketManager()
    
    print("Connecting WebSocket...")
    # Start connection in background
    ws_task = asyncio.create_task(ws.connect())
    
    # Wait for data
    print("Waiting 35s for data + regime test...")
    await asyncio.sleep(35)
    
    # Test regime classification and logging
    for symbol in ["ATOM", "SUI", "XRP"]:
        regime, confidence, metrics = ws.classify_market_regime(symbol)
        print(f"\n{symbol}: regime={regime}, confidence={confidence:.2f}")
        print(f"  Metrics: vol_1m={metrics.get('volatility_1m', 0):.3f}%, trend={metrics.get('trend_strength', 0):.2f}")
        
        # Try to log
        try:
            logger.log_market_regime(symbol, regime, metrics, confidence)
            print(f"  ✓ Logged to market_regime table")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    # Check count
    cursor = logger.conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM market_regime").fetchone()[0]
    print(f"\nmarket_regime records: {count}")
    
    # Cleanup
    ws_task.cancel()
    logger.close()

asyncio.run(test())
