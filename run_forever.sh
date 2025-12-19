#!/bin/bash
#
# Auto-restart wrapper for HFT Bot
# Restarts the bot if it crashes
#
# Usage: ./run_forever.sh
#

cd "$(dirname "$0")"

echo "========================================"
echo "HFT BOT AUTO-RESTART WRAPPER"
echo "========================================"
echo "Bot will auto-restart if it crashes"
echo "Press Ctrl+C to stop completely"
echo "========================================"

while true; do
    echo ""
    echo "[$(date)] Starting HFT Bot..."
    echo ""

    python run_hft.py

    EXIT_CODE=$?
    echo ""
    echo "[$(date)] Bot stopped with exit code: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Clean exit. Not restarting."
        break
    fi

    echo "Restarting in 10 seconds... (Ctrl+C to cancel)"
    sleep 10
done
