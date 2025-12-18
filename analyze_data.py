#!/usr/bin/env python3
"""
Full Database Analysis Script
==============================
Run this to see everything collected:
    python analyze_data.py
"""
import sqlite3
from datetime import datetime

DB_PATH = "hft_trades.db"

def analyze():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    print("\n" + "="*70)
    print("1. DATABASE OVERVIEW - All Tables")
    print("="*70)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
            print(f"  {t[0]:25} : {count:,} records")
        except:
            pass

    print("\n" + "="*70)
    print("2. SIGNALS BY CONDITION COUNT (which threshold is best?)")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT conditions_met, COUNT(*) as total,
                   ROUND(AVG(pnl_60s),3) as avg_60s,
                   ROUND(AVG(pnl_90s),3) as avg_90s,
                   ROUND(AVG(pnl_2m),3) as avg_2m,
                   ROUND(AVG(pnl_5m),3) as avg_5m,
                   ROUND(AVG(pnl_10m),3) as avg_10m
            FROM strategy_signals
            WHERE pnl_5m IS NOT NULL
            GROUP BY conditions_met ORDER BY conditions_met
        """).fetchall()
        print(f"  {'Cond':6} {'Count':8} {'60s':10} {'90s':10} {'2m':10} {'5m':10} {'10m':10}")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in rows:
            pnl_60s = f"{r[2]:+.3f}%" if r[2] else "N/A"
            pnl_90s = f"{r[3]:+.3f}%" if r[3] else "N/A"
            pnl_2m = f"{r[4]:+.3f}%" if r[4] else "N/A"
            pnl_5m = f"{r[5]:+.3f}%" if r[5] else "N/A"
            pnl_10m = f"{r[6]:+.3f}%" if r[6] else "N/A"
            print(f"  {r[0]}/5    {r[1]:6}   {pnl_60s:10} {pnl_90s:10} {pnl_2m:10} {pnl_5m:10} {pnl_10m:10}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("3. BEST CONDITION COMBINATIONS (P=price V=volume O=orderbook F=funding R=rsi)")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT
                CASE WHEN price_cond=1 THEN 'P' ELSE '-' END ||
                CASE WHEN volume_cond=1 THEN 'V' ELSE '-' END ||
                CASE WHEN orderbook_cond=1 THEN 'O' ELSE '-' END ||
                CASE WHEN funding_cond=1 THEN 'F' ELSE '-' END ||
                CASE WHEN rsi_cond=1 THEN 'R' ELSE '-' END as combo,
                conditions_met,
                COUNT(*) as cnt,
                ROUND(AVG(pnl_60s),3) as avg_60s,
                ROUND(AVG(pnl_5m),3) as avg_5m,
                ROUND(AVG(pnl_10m),3) as avg_10m
            FROM strategy_signals
            WHERE pnl_5m IS NOT NULL AND conditions_met >= 1
            GROUP BY combo
            HAVING cnt >= 5
            ORDER BY avg_5m DESC
            LIMIT 15
        """).fetchall()
        print(f"  {'Combo':8} {'Cond':5} {'Count':8} {'Avg 60s':10} {'Avg 5m':10} {'Avg 10m':10}")
        print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for r in rows:
            print(f"  {r[0]:8} {r[1]}/5   {r[2]:6}   {r[3]:+.3f}%    {r[4]:+.3f}%    {r[5]:+.3f}%")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("4. ALL EXECUTED TRADES (most recent first)")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT symbol, side, entry_price, exit_price,
                   ROUND(pnl_pct,3) as pnl, exit_reason, hold_time_sec,
                   datetime(entry_time/1000, 'unixepoch', 'localtime') as time
            FROM trades WHERE status='closed'
            ORDER BY entry_time DESC LIMIT 30
        """).fetchall()
        print(f"  {'Time':20} {'Symbol':6} {'Side':6} {'Entry':10} {'Exit':10} {'PnL%':8} {'Hold':6} {'Reason':10}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*10}")
        for r in rows:
            exit_p = r[3] if r[3] else 0
            pnl = r[4] if r[4] else 0
            hold = int(r[6]) if r[6] else 0
            print(f"  {r[7]:20} {r[0]:6} {r[1]:6} ${r[2]:.4f}  ${exit_p:.4f}  {pnl:+.2f}%   {hold:4}s  {r[5] or 'open'}")

        # Summary
        wins = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl_pct > 0 AND status='closed'").fetchone()[0]
        losses = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl_pct <= 0 AND status='closed'").fetchone()[0]
        total_pnl = conn.execute("SELECT SUM(pnl_pct) FROM trades WHERE status='closed'").fetchone()[0] or 0
        print(f"\n  SUMMARY: {wins}W / {losses}L | Win Rate: {wins/(wins+losses)*100 if wins+losses > 0 else 0:.1f}% | Total PnL: {total_pnl:+.2f}%")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("5. CVD (Buy/Sell Pressure) - Recent Data")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT symbol,
                   ROUND(AVG(cvd_1m),0) as avg_cvd_1m,
                   ROUND(AVG(cvd_5m),0) as avg_cvd_5m,
                   ROUND(AVG(buy_volume_1m),0) as avg_buy,
                   ROUND(AVG(sell_volume_1m),0) as avg_sell,
                   COUNT(*) as samples
            FROM cvd_snapshots
            WHERE timestamp > (strftime('%s','now')*1000 - 3600000)
            GROUP BY symbol
        """).fetchall()
        for r in rows:
            pressure = "BUYING" if (r[1] or 0) > 0 else "SELLING"
            print(f"  {r[0]}: CVD_1m={r[1]:+.0f} | CVD_5m={r[2]:+.0f} | Buy={r[3]:.0f} | Sell={r[4]:.0f} | {pressure} pressure")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("6. ORDERBOOK IMBALANCE - Recent Data")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT symbol,
                   ROUND(AVG(imbalance),3) as avg_imb,
                   ROUND(AVG(spread_pct),4) as avg_spread,
                   COUNT(*) as samples
            FROM orderbook_snapshots
            WHERE timestamp > (strftime('%s','now')*1000 - 3600000)
            GROUP BY symbol
        """).fetchall()
        for r in rows:
            side = "BUY side" if (r[1] or 0.5) > 0.5 else "SELL side"
            print(f"  {r[0]}: Imbalance={r[1]:.3f} ({side}) | Spread={r[2]:.4f}% | {r[3]} samples")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("7. HOURLY SIGNAL DISTRIBUTION (last 24h)")
    print("="*70)
    try:
        rows = conn.execute("""
            SELECT strftime('%H', datetime(timestamp/1000, 'unixepoch', 'localtime')) as hour,
                   COUNT(*) as signals,
                   SUM(CASE WHEN conditions_met >= 2 THEN 1 ELSE 0 END) as strong_signals
            FROM strategy_signals
            WHERE timestamp > (strftime('%s','now')*1000 - 86400000)
            GROUP BY hour
            ORDER BY hour
        """).fetchall()
        for r in rows:
            bar = "█" * min(int(r[1]/10), 50)
            print(f"  {r[0]}:00 | {r[1]:4} signals ({r[2]:3} strong) {bar}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    try:
        # Find best condition count
        best = conn.execute("""
            SELECT conditions_met, ROUND(AVG(pnl_5m),3) as avg
            FROM strategy_signals WHERE pnl_5m IS NOT NULL
            GROUP BY conditions_met ORDER BY avg DESC LIMIT 1
        """).fetchone()
        if best:
            print(f"  Best condition count: {best[0]}/5 with avg {best[1]:+.3f}% at 5m")

        # Find best combo
        best_combo = conn.execute("""
            SELECT
                CASE WHEN price_cond=1 THEN 'P' ELSE '-' END ||
                CASE WHEN volume_cond=1 THEN 'V' ELSE '-' END ||
                CASE WHEN orderbook_cond=1 THEN 'O' ELSE '-' END ||
                CASE WHEN funding_cond=1 THEN 'F' ELSE '-' END ||
                CASE WHEN rsi_cond=1 THEN 'R' ELSE '-' END as combo,
                COUNT(*) as cnt, ROUND(AVG(pnl_5m),3) as avg
            FROM strategy_signals WHERE pnl_5m IS NOT NULL AND conditions_met >= 2
            GROUP BY combo HAVING cnt >= 10 ORDER BY avg DESC LIMIT 1
        """).fetchone()
        if best_combo:
            print(f"  Best combo: {best_combo[0]} with {best_combo[1]} signals, avg {best_combo[2]:+.3f}% at 5m")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*70 + "\n")
    conn.close()

if __name__ == "__main__":
    analyze()
