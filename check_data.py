#!/usr/bin/env python3
"""
Quick data verification script.
Run this while the bot is running to verify data collection.

Usage:
    python check_data.py
"""
import sqlite3
import time

DB_PATH = "hft_trades.db"

def check():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)

        tables = [
            "strategy_signals",
            "cvd_snapshots",
            "orderbook_snapshots",
            "indicator_snapshots",
            "trade_flow",
            "trades",
            "signals"
        ]

        print("\n" + "="*50)
        print("DATA COLLECTION STATUS")
        print("="*50)

        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table:25} : {count:,} records")
            except Exception as e:
                print(f"{table:25} : ERROR - {e}")

        # Check recent strategy signals with outcomes
        print("\n" + "="*50)
        print("STRATEGY SIGNALS (last 10)")
        print("="*50)

        try:
            rows = conn.execute("""
                SELECT conditions_met, price, pnl_60s, pnl_5m, pnl_10m
                FROM strategy_signals
                ORDER BY id DESC LIMIT 10
            """).fetchall()

            for row in rows:
                cond, price, pnl60, pnl5m, pnl10m = row
                pnl60_str = f"{pnl60:+.3f}%" if pnl60 else "pending"
                pnl5m_str = f"{pnl5m:+.3f}%" if pnl5m else "pending"
                pnl10m_str = f"{pnl10m:+.3f}%" if pnl10m else "pending"
                print(f"  {cond}/5 cond | ${price:.4f} | 60s:{pnl60_str} | 5m:{pnl5m_str} | 10m:{pnl10m_str}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Summary by conditions
        print("\n" + "="*50)
        print("PERFORMANCE BY CONDITIONS (completed signals)")
        print("="*50)

        try:
            rows = conn.execute("""
                SELECT
                    conditions_met,
                    COUNT(*) as total,
                    ROUND(AVG(pnl_60s), 3) as avg_60s,
                    ROUND(AVG(pnl_5m), 3) as avg_5m,
                    ROUND(AVG(pnl_10m), 3) as avg_10m
                FROM strategy_signals
                WHERE pnl_5m IS NOT NULL
                GROUP BY conditions_met
                ORDER BY conditions_met
            """).fetchall()

            for row in rows:
                cond, total, avg60, avg5m, avg10m = row
                print(f"  {cond}/5: {total:4} signals | 60s:{avg60:+.3f}% | 5m:{avg5m:+.3f}% | 10m:{avg10m:+.3f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

        conn.close()
        print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"\nERROR: Could not connect to database: {e}")
        print("Make sure the bot is running first!\n")

if __name__ == "__main__":
    check()
