#!/usr/bin/env python3
"""
Automated Strategy Rotation Service
====================================
Automated daily strategy discovery and challenger rotation.

This service:
1. Runs walk-forward optimization (3-way split)
2. Proposes challenger config if better than current
3. Updates strategies/registry.yml with new challenger
4. Tracks cooldown (max 1 challenger change per 24h per symbol)
5. NEVER auto-promotes to champion - only suggests

Usage:
    # Basic rotation
    python tools/auto_rotate.py --symbol XRPUSDT

    # With custom lookback
    python tools/auto_rotate.py --symbol XRPUSDT --lookback-days 45

    # Dry run (preview only)
    python tools/auto_rotate.py --symbol XRPUSDT --dry-run

    # JSON output (for cron)
    python tools/auto_rotate.py --symbol XRPUSDT --json

Cron example:
    # Daily at 2am UTC
    0 2 * * * cd ~/Analize- && python tools/auto_rotate.py --symbol XRPUSDT --json >> logs/auto_rotate.log 2>&1

Safety features:
    - 24h cooldown between challenger changes per symbol
    - NEVER auto-promotes to champion
    - Always preserves existing champion
    - Validates metrics before updating
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from hft_system.symbol_utils import normalize_symbol, normalize_db_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChallengerCandidate:
    """Candidate challenger configuration."""
    strategy_id: str
    symbol: str
    params: Dict[str, float]
    valid_pnl_pct: float = 0
    valid_win_rate: float = 0
    valid_trades: int = 0
    valid_sharpe: float = 0
    final_test_pnl_pct: float = 0
    final_test_win_rate: float = 0
    final_test_trades: int = 0
    final_test_sharpe: float = 0
    final_test_max_drawdown: float = 0
    stress_passed: bool = False
    stress_baseline_pnl: float = 0
    stress_combined_pnl: float = 0
    score: float = 0
    window_start: str = ""
    window_end: str = ""
    train_days: int = 14
    valid_days: int = 7
    final_days: int = 7


@dataclass
class RotationResult:
    """Result of the rotation service."""
    symbol: str
    timestamp: str
    action: str  # NO_CHANGE | NEW_CHALLENGER | UPDATED_CHALLENGER | SUGGEST_PROMOTE | COOLDOWN | ERROR
    reason: str
    current_champion_id: Optional[str] = None
    current_challenger_id: Optional[str] = None
    proposed_challenger: Optional[ChallengerCandidate] = None
    champion_metrics: Optional[Dict] = None
    challenger_metrics: Optional[Dict] = None
    improvement_pct: float = 0
    cooldown_remaining_hours: float = 0
    registry_updated: bool = False
    error: str = ""


# =============================================================================
# COOLDOWN MANAGEMENT
# =============================================================================

COOLDOWN_FILE = "strategies/.rotation_cooldown.json"
COOLDOWN_HOURS = 24


def load_cooldown_state() -> Dict[str, str]:
    """Load cooldown state from file."""
    cooldown_path = Path(COOLDOWN_FILE)
    if cooldown_path.exists():
        try:
            with open(cooldown_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cooldown_state(state: Dict[str, str]):
    """Save cooldown state to file."""
    cooldown_path = Path(COOLDOWN_FILE)
    cooldown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cooldown_path, 'w') as f:
        json.dump(state, f, indent=2)


def check_cooldown(symbol: str) -> tuple[bool, float]:
    """
    Check if symbol is in cooldown.

    Returns:
        (is_in_cooldown, remaining_hours)
    """
    state = load_cooldown_state()
    last_change = state.get(symbol)

    if not last_change:
        return False, 0

    try:
        last_change_dt = datetime.fromisoformat(last_change)
        elapsed = datetime.now() - last_change_dt
        remaining = timedelta(hours=COOLDOWN_HOURS) - elapsed

        if remaining.total_seconds() > 0:
            return True, remaining.total_seconds() / 3600
        return False, 0
    except (ValueError, TypeError):
        return False, 0


def update_cooldown(symbol: str):
    """Update cooldown timestamp for symbol."""
    state = load_cooldown_state()
    state[symbol] = datetime.now().isoformat()
    save_cooldown_state(state)


# =============================================================================
# REGISTRY MANAGEMENT
# =============================================================================

def load_registry(registry_path: str = "strategies/registry.yml") -> Dict:
    """Load registry YAML file."""
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed. Using simplified parsing.")
        return {"strategies": {}, "assignments": {}}

    if not Path(registry_path).exists():
        return {"strategies": {}, "assignments": {}, "defaults": {}}

    try:
        with open(registry_path, 'r') as f:
            return yaml.safe_load(f) or {"strategies": {}, "assignments": {}}
    except Exception as e:
        logger.error(f"Error loading registry: {e}")
        return {"strategies": {}, "assignments": {}}


def save_registry(registry: Dict, registry_path: str = "strategies/registry.yml"):
    """Save registry YAML file."""
    if not YAML_AVAILABLE:
        logger.error("PyYAML not installed. Cannot save registry.")
        return False

    try:
        # Create backup
        backup_path = registry_path + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if Path(registry_path).exists():
            import shutil
            shutil.copy(registry_path, backup_path)

        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return True
    except Exception as e:
        logger.error(f"Error saving registry: {e}")
        return False


def get_current_champion(registry: Dict, symbol: str) -> Optional[Dict]:
    """Get current champion strategy for symbol."""
    assignments = registry.get("assignments", {})
    symbol_assignment = assignments.get(symbol, {})
    champion_id = symbol_assignment.get("champion")

    if not champion_id:
        return None

    strategies = registry.get("strategies", {})
    return strategies.get(champion_id)


def get_current_challenger(registry: Dict, symbol: str) -> Optional[tuple[str, Dict]]:
    """Get current challenger strategy ID and config for symbol."""
    assignments = registry.get("assignments", {})
    symbol_assignment = assignments.get(symbol, {})
    challenger_id = symbol_assignment.get("challenger")

    if not challenger_id:
        return None

    strategies = registry.get("strategies", {})
    challenger = strategies.get(challenger_id)

    if challenger:
        return (challenger_id, challenger)
    return None


def generate_strategy_id(symbol: str) -> str:
    """Generate unique strategy ID."""
    base_symbol = normalize_db_symbol(symbol).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_symbol}_auto_{timestamp}"


def create_challenger_entry(candidate: ChallengerCandidate) -> Dict:
    """Create a complete strategy entry for the registry."""
    return {
        "symbol": candidate.symbol,
        "status": "challenger",
        "description": f"Auto-discovered challenger via walk-forward optimization",
        "regimes_allowed": ["high_vol_trend", "mean_reversion"],
        "regimes_blocked": ["low_vol_chop", "liquidity_vacuum", "news_spike"],
        "causes": {
            "enabled": [
                "ob_bullish_imbalance",
                "ob_bearish_imbalance",
                "cvd_buy_pressure",
                "cvd_sell_pressure",
                "price_momentum_up",
                "price_momentum_down"
            ],
            "disabled": ["btc_bullish", "btc_bearish"],
            "probe_causes": ["unknown"],
            "block_causes": []
        },
        "params": {
            "take_profit_pct": candidate.params.get("tp", 0.20),
            "stop_loss_pct": candidate.params.get("sl", 0.15),
            "time_stop_seconds": int(candidate.params.get("tstop", 90)),
            "min_conditions": 3,
            "min_imbalance": 0.60,
            "min_confidence": 0.6
        },
        "execution": {
            "mode": "auto",
            "auto_maker_spread_threshold_bps": 2.0,
            "max_spread_bps": 5.0,
            "maker_timeout_ms": 5000,
            "maker_price_offset_bps": 0.5
        },
        "risk": {
            "position_size_pct": 0.5,  # Start conservative
            "max_positions": 1,
            "max_daily_trades": 50,
            "max_daily_loss_pct": 2.0,
            "max_drawdown_pct": 5.0,
            "loss_streak_pause_trades": 5,
            "loss_streak_pause_minutes": 30
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "source": "auto_rotate",
            "walkforward_window": f"{candidate.window_start} to {candidate.window_end}",
            "train_days": candidate.train_days,
            "valid_days": candidate.valid_days,
            "final_days": candidate.final_days,
            "valid_pnl_pct": round(candidate.valid_pnl_pct, 4),
            "valid_win_rate": round(candidate.valid_win_rate, 2),
            "valid_trades": candidate.valid_trades,
            "final_test_pnl_pct": round(candidate.final_test_pnl_pct, 4),
            "final_test_sharpe": round(candidate.final_test_sharpe, 4),
            "final_test_win_rate": round(candidate.final_test_win_rate, 2),
            "final_test_trades": candidate.final_test_trades,
            "final_test_max_drawdown": round(candidate.final_test_max_drawdown, 4),
            "stress_test_passed": candidate.stress_passed,
            "combined_score": round(candidate.score, 4),
            "promotion_approved": False,
            "promotion_date": None,
            "notes": "Auto-discovered. Awaiting live validation before promotion."
        }
    }


def update_registry_challenger(
    registry: Dict,
    symbol: str,
    candidate: ChallengerCandidate,
    update_existing: bool = False
) -> tuple[bool, str]:
    """
    Update registry with new challenger.

    Args:
        registry: Registry dict to modify
        symbol: Trading symbol
        candidate: Challenger candidate
        update_existing: If True, update existing challenger instead of creating new

    Returns:
        (success, strategy_id)
    """
    assignments = registry.setdefault("assignments", {})
    strategies = registry.setdefault("strategies", {})

    # Check if there's an existing challenger to update
    current = get_current_challenger(registry, symbol)

    if current and update_existing:
        # Update existing challenger
        challenger_id, existing = current
        challenger_entry = create_challenger_entry(candidate)
        challenger_entry["metadata"]["created_at"] = existing.get("metadata", {}).get(
            "created_at", datetime.now().isoformat()
        )
        strategies[challenger_id] = challenger_entry
        logger.info(f"Updated existing challenger: {challenger_id}")
        return True, challenger_id
    else:
        # Create new challenger
        strategy_id = generate_strategy_id(symbol)
        candidate.strategy_id = strategy_id

        # Retire old challenger if exists
        if current:
            old_id, old_strategy = current
            old_strategy["status"] = "retired"
            old_strategy["metadata"]["retired_at"] = datetime.now().isoformat()
            old_strategy["metadata"]["retired_reason"] = f"Replaced by {strategy_id}"
            logger.info(f"Retired old challenger: {old_id}")

        # Add new challenger
        strategies[strategy_id] = create_challenger_entry(candidate)

        # Update assignment
        symbol_assignment = assignments.setdefault(symbol, {})
        symbol_assignment["challenger"] = strategy_id
        symbol_assignment["last_rotation"] = datetime.now().isoformat()

        logger.info(f"Added new challenger: {strategy_id}")
        return True, strategy_id


# =============================================================================
# DISCOVERY INTEGRATION
# =============================================================================

def run_discovery(
    symbol: str,
    start_date: str,
    end_date: str,
    train_days: int = 14,
    valid_days: int = 7,
    final_days: int = 7,
    top_k: int = 5,
    require_stress: bool = True,
    db_path: str = "hft_trades.db",
    grid: Optional[str] = None
) -> Optional[ChallengerCandidate]:
    """
    Run auto_discover and return best candidate.

    Returns:
        Best candidate or None if no viable candidate found
    """
    import subprocess

    cmd = [
        sys.executable, "tools/auto_discover.py",
        "--symbol", symbol,
        "--start", start_date,
        "--end", end_date,
        "--train-days", str(train_days),
        "--valid-days", str(valid_days),
        "--final-days", str(final_days),
        "--top-k", str(top_k),
        "--db", db_path,
        "--json",
        "--dry-run"  # Don't update registry from auto_discover
    ]

    if require_stress:
        cmd.append("--require-stress")

    if grid:
        cmd.extend(["--grid", grid])

    try:
        logger.info(f"Running discovery for {symbol}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(Path(__file__).parent.parent)
        )

        if result.returncode != 0:
            logger.error(f"Discovery failed: {result.stderr}")
            return None

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Try to find JSON in output
            for line in result.stdout.split('\n'):
                if line.strip().startswith('{'):
                    try:
                        data = json.loads(line)
                        break
                    except:
                        continue
            else:
                logger.error("No valid JSON in discovery output")
                return None

        if data.get("error"):
            logger.warning(f"Discovery warning: {data['error']}")
            return None

        best = data.get("best_candidate")
        if not best:
            logger.info("No viable candidate found")
            return None

        # Create candidate object
        candidate = ChallengerCandidate(
            strategy_id="",
            symbol=symbol,
            params=best.get("params", {}),
            valid_pnl_pct=best.get("valid_pnl_pct", 0),
            final_test_pnl_pct=best.get("final_test_pnl_pct", 0),
            final_test_sharpe=best.get("final_test_sharpe", 0),
            stress_passed=best.get("stress_passed", False),
            score=best.get("score", 0),
            window_start=start_date,
            window_end=end_date,
            train_days=train_days,
            valid_days=valid_days,
            final_days=final_days
        )

        return candidate

    except subprocess.TimeoutExpired:
        logger.error("Discovery timed out")
        return None
    except Exception as e:
        logger.error(f"Discovery error: {e}")
        return None


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def should_update_challenger(
    current_challenger: Optional[Dict],
    new_candidate: ChallengerCandidate,
    improvement_threshold: float = 10.0
) -> tuple[bool, str, float]:
    """
    Determine if new candidate should replace current challenger.

    Args:
        current_challenger: Current challenger strategy config
        new_candidate: New candidate from discovery
        improvement_threshold: Required improvement % to replace (default: 10%)

    Returns:
        (should_update, reason, improvement_pct)
    """
    if not current_challenger:
        return True, "No existing challenger", 100.0

    # Get current metrics
    current_metadata = current_challenger.get("metadata", {})
    current_pnl = current_metadata.get("final_test_pnl_pct", 0)
    current_sharpe = current_metadata.get("final_test_sharpe", 0)
    current_score = current_metadata.get("combined_score", 0)

    # Calculate improvement
    new_pnl = new_candidate.final_test_pnl_pct
    new_sharpe = new_candidate.final_test_sharpe
    new_score = new_candidate.score

    # Weighted improvement (50% PnL, 30% Sharpe, 20% Score)
    improvement = 0

    if current_pnl != 0:
        pnl_improvement = ((new_pnl - current_pnl) / abs(current_pnl)) * 100
    else:
        pnl_improvement = 100 if new_pnl > 0 else 0

    if current_sharpe != 0:
        sharpe_improvement = ((new_sharpe - current_sharpe) / abs(current_sharpe)) * 100
    else:
        sharpe_improvement = 100 if new_sharpe > 0 else 0

    if current_score != 0:
        score_improvement = ((new_score - current_score) / abs(current_score)) * 100
    else:
        score_improvement = 100 if new_score > 0 else 0

    improvement = (
        0.50 * pnl_improvement +
        0.30 * sharpe_improvement +
        0.20 * score_improvement
    )

    if improvement >= improvement_threshold:
        return True, f"Improvement of {improvement:.1f}% exceeds threshold of {improvement_threshold}%", improvement
    elif improvement > 0:
        return False, f"Improvement of {improvement:.1f}% below threshold of {improvement_threshold}%", improvement
    else:
        return False, f"New candidate worse by {abs(improvement):.1f}%", improvement


def should_suggest_promotion(
    champion: Optional[Dict],
    challenger: Dict,
    min_improvement: float = 15.0
) -> tuple[bool, str, float]:
    """
    Check if challenger should be suggested for promotion to champion.

    Note: This only SUGGESTS - actual promotion requires manual approval.

    Args:
        champion: Current champion config
        challenger: Current challenger config
        min_improvement: Minimum improvement % to suggest (default: 15%)

    Returns:
        (should_suggest, reason, improvement_pct)

    ECONOMICS CHECK: Refuses promotion if challenger's expected PnL doesn't cover costs.
    Taker costs ~0.26%, maker costs ~0.05%. Strategy must be profitable AFTER costs.
    """
    # =========================================================================
    # ECONOMICS CHECK: Must be profitable after costs
    # =========================================================================
    challenger_meta = challenger.get("metadata", {})
    challenger_exec = challenger.get("execution", {})

    challenger_pnl = challenger_meta.get("final_test_pnl_pct", 0)
    challenger_trades = challenger_meta.get("final_test_trades", 0)

    # Estimate cost based on execution mode
    exec_mode = challenger_exec.get("mode", "taker")
    if exec_mode == "maker":
        estimated_cost_per_trade_pct = 0.05  # ~0.05% for maker
    else:
        estimated_cost_per_trade_pct = 0.26  # ~0.26% for taker

    # Calculate average PnL per trade (minimum required profit after costs: 0.05%)
    min_pnl_after_costs = 0.05
    if challenger_trades and challenger_trades > 0:
        avg_pnl_per_trade = challenger_pnl / challenger_trades
        pnl_after_costs = avg_pnl_per_trade - estimated_cost_per_trade_pct
    else:
        avg_pnl_per_trade = 0
        pnl_after_costs = -estimated_cost_per_trade_pct  # No trades = no data = fail economics

    # Refuse if economics are broken (loses money after costs)
    if pnl_after_costs < min_pnl_after_costs:
        reason = (
            f"ECONOMICS BROKEN: avg_pnl={avg_pnl_per_trade:.3f}% - costs={estimated_cost_per_trade_pct:.3f}% = "
            f"{pnl_after_costs:.3f}% (need >= {min_pnl_after_costs:.3f}%)"
        )
        logger.warning(f"[ECONOMICS] {reason}")
        return False, reason, 0

    # =========================================================================
    # Check if there's no champion (first strategy)
    # =========================================================================
    if not champion:
        return True, "No current champion - challenger is economically viable candidate for first champion", 100.0

    champion_meta = champion.get("metadata", {})

    # Get key metrics
    champion_pnl = champion_meta.get("final_test_pnl_pct", 0)
    challenger_pnl = challenger_meta.get("final_test_pnl_pct", 0)

    champion_sharpe = champion_meta.get("final_test_sharpe", 0)
    challenger_sharpe = challenger_meta.get("final_test_sharpe", 0)

    # Calculate improvement
    if champion_pnl != 0:
        pnl_improvement = ((challenger_pnl - champion_pnl) / abs(champion_pnl)) * 100
    else:
        pnl_improvement = 100 if challenger_pnl > 0 else 0

    if champion_sharpe != 0:
        sharpe_improvement = ((challenger_sharpe - champion_sharpe) / abs(challenger_sharpe)) * 100
    else:
        sharpe_improvement = 100 if challenger_sharpe > 0 else 0

    improvement = 0.6 * pnl_improvement + 0.4 * sharpe_improvement

    if improvement >= min_improvement:
        return True, f"Challenger shows {improvement:.1f}% improvement (economics OK: {pnl_after_costs:.3f}% net)", improvement
    else:
        return False, f"Challenger shows only {improvement:.1f}% improvement (need {min_improvement}%)", improvement


# =============================================================================
# MAIN ROTATION SERVICE
# =============================================================================

def run_rotation(
    symbol: str,
    lookback_days: int = 45,
    train_days: int = 14,
    valid_days: int = 7,
    final_days: int = 7,
    top_k: int = 5,
    require_stress: bool = True,
    db_path: str = "hft_trades.db",
    grid: Optional[str] = None,
    improvement_threshold: float = 10.0,
    dry_run: bool = False,
    registry_path: str = "strategies/registry.yml"
) -> RotationResult:
    """
    Run the full rotation service.

    Args:
        symbol: Trading symbol
        lookback_days: Days of history for walk-forward
        train_days: Training window
        valid_days: Validation window
        final_days: Final test window
        top_k: Top candidates to evaluate
        require_stress: Require stress test pass
        db_path: Database path for stress test
        grid: Optional parameter grid
        improvement_threshold: Required improvement % to replace challenger
        dry_run: Preview only, don't modify registry
        registry_path: Path to registry YAML

    Returns:
        RotationResult with action and details
    """
    symbol = normalize_symbol(symbol)

    result = RotationResult(
        symbol=symbol,
        timestamp=datetime.now().isoformat(),
        action="NO_CHANGE",
        reason=""
    )

    # Step 1: Check cooldown
    in_cooldown, hours_remaining = check_cooldown(symbol)
    if in_cooldown and not dry_run:
        result.action = "COOLDOWN"
        result.reason = f"Symbol in cooldown. {hours_remaining:.1f} hours remaining."
        result.cooldown_remaining_hours = hours_remaining
        logger.info(f"[{symbol}] Cooldown active: {hours_remaining:.1f}h remaining")
        return result

    # Step 2: Load registry
    registry = load_registry(registry_path)

    # Get current champion and challenger
    champion = get_current_champion(registry, symbol)
    current_challenger = get_current_challenger(registry, symbol)

    if champion:
        assignments = registry.get("assignments", {})
        result.current_champion_id = assignments.get(symbol, {}).get("champion")
        result.champion_metrics = champion.get("metadata", {})

    if current_challenger:
        result.current_challenger_id = current_challenger[0]
        result.challenger_metrics = current_challenger[1].get("metadata", {})

    # Step 3: Run discovery
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    candidate = run_discovery(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        train_days=train_days,
        valid_days=valid_days,
        final_days=final_days,
        top_k=top_k,
        require_stress=require_stress,
        db_path=db_path,
        grid=grid
    )

    if not candidate:
        result.action = "NO_CHANGE"
        result.reason = "No viable candidate found from discovery"
        logger.info(f"[{symbol}] No viable candidate found")
        return result

    result.proposed_challenger = candidate

    # Step 4: Compare with current challenger
    current_challenger_config = current_challenger[1] if current_challenger else None
    should_update, update_reason, improvement = should_update_challenger(
        current_challenger_config,
        candidate,
        improvement_threshold
    )
    result.improvement_pct = improvement

    if not should_update:
        result.action = "NO_CHANGE"
        result.reason = update_reason
        logger.info(f"[{symbol}] No update needed: {update_reason}")

        # Check if current challenger should be promoted
        if current_challenger_config:
            should_promote, promote_reason, promote_improvement = should_suggest_promotion(
                champion, current_challenger_config
            )
            if should_promote:
                result.action = "SUGGEST_PROMOTE"
                result.reason = promote_reason
                result.improvement_pct = promote_improvement

        return result

    # Step 5: Update registry (unless dry run)
    if dry_run:
        result.action = "NEW_CHALLENGER" if not current_challenger else "UPDATED_CHALLENGER"
        result.reason = f"[DRY-RUN] Would update challenger: {update_reason}"
        logger.info(f"[{symbol}] [DRY-RUN] Would update challenger")
        return result

    # Actually update registry
    success, strategy_id = update_registry_challenger(
        registry, symbol, candidate,
        update_existing=False  # Always create new for audit trail
    )

    if not success:
        result.action = "ERROR"
        result.reason = "Failed to update registry"
        result.error = "Registry update failed"
        return result

    # Save registry
    if save_registry(registry, registry_path):
        result.registry_updated = True
        update_cooldown(symbol)

        result.action = "NEW_CHALLENGER" if not current_challenger else "UPDATED_CHALLENGER"
        result.reason = f"Updated challenger to {strategy_id}: {update_reason}"
        logger.info(f"[{symbol}] Updated challenger to {strategy_id}")

        # Check if new challenger should be suggested for promotion
        new_challenger = registry["strategies"].get(strategy_id)
        if new_challenger and champion:
            should_promote, promote_reason, _ = should_suggest_promotion(champion, new_challenger)
            if should_promote:
                result.reason += f" | PROMOTION SUGGESTED: {promote_reason}"
    else:
        result.action = "ERROR"
        result.reason = "Failed to save registry"
        result.error = "Registry save failed"

    return result


# =============================================================================
# CLI
# =============================================================================

def print_result_text(result: RotationResult):
    """Print result in human-readable format."""
    action_symbols = {
        "NO_CHANGE": "[-]",
        "NEW_CHALLENGER": "[+]",
        "UPDATED_CHALLENGER": "[~]",
        "SUGGEST_PROMOTE": "[!]",
        "COOLDOWN": "[C]",
        "ERROR": "[E]"
    }

    print(f"""
+================================================================================+
|                    AUTO ROTATION SERVICE RESULT                                 |
+================================================================================+
| Symbol:     {result.symbol}
| Timestamp:  {result.timestamp}
| Action:     {action_symbols.get(result.action, '[?]')} {result.action}
+================================================================================+

  Reason: {result.reason}
""")

    if result.current_champion_id:
        print(f"  Current Champion:    {result.current_champion_id}")
    if result.current_challenger_id:
        print(f"  Current Challenger:  {result.current_challenger_id}")

    if result.proposed_challenger:
        c = result.proposed_challenger
        print(f"""
  --- PROPOSED CHALLENGER ---
  Parameters:
    TP: {c.params.get('tp', 'N/A')}%
    SL: {c.params.get('sl', 'N/A')}%
    Time Stop: {c.params.get('tstop', 'N/A')}s

  Metrics:
    Final Test PnL:   {c.final_test_pnl_pct:+.4f}%
    Final Test Sharpe: {c.final_test_sharpe:.4f}
    Combined Score:    {c.score:.4f}
    Stress Passed:     {c.stress_passed}
""")

    if result.improvement_pct != 0:
        print(f"  Improvement: {result.improvement_pct:+.1f}%")

    if result.cooldown_remaining_hours > 0:
        print(f"  Cooldown Remaining: {result.cooldown_remaining_hours:.1f} hours")

    if result.registry_updated:
        print(f"  Registry Updated: Yes")

    if result.error:
        print(f"\n  [ERROR] {result.error}")

    print("""
+================================================================================+
""")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Strategy Rotation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic rotation
  python tools/auto_rotate.py --symbol XRPUSDT

  # With custom lookback
  python tools/auto_rotate.py --symbol XRPUSDT --lookback-days 60

  # Dry run (preview only)
  python tools/auto_rotate.py --symbol XRPUSDT --dry-run

  # JSON output for cron
  python tools/auto_rotate.py --symbol XRPUSDT --json

  # Multiple symbols
  for sym in XRPUSDT BTCUSDT ETHUSDT; do
    python tools/auto_rotate.py --symbol $sym --json
  done

Cron example (daily at 2am UTC):
  0 2 * * * cd ~/Analize- && python tools/auto_rotate.py --symbol XRPUSDT --json >> logs/auto_rotate.log 2>&1
        """
    )

    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., XRPUSDT or XRP)")
    parser.add_argument("--lookback-days", type=int, default=45, help="Days of history for walk-forward (default: 45)")
    parser.add_argument("--train-days", type=int, default=14, help="Training window days (default: 14)")
    parser.add_argument("--valid-days", type=int, default=7, help="Validation window days (default: 7)")
    parser.add_argument("--final-days", type=int, default=7, help="Final test window days (default: 7)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K candidates to evaluate (default: 5)")
    parser.add_argument("--improvement-threshold", type=float, default=10.0,
                        help="Required improvement %% to replace challenger (default: 10)")
    parser.add_argument("--no-stress", action="store_true", help="Skip stress test requirement")
    parser.add_argument("--db", type=str, default="hft_trades.db", help="Database path for stress test")
    parser.add_argument("--grid", type=str, default=None,
                        help="Parameter grid (e.g., 'tp=0.12,0.16,0.20; sl=0.08,0.10; tstop=30,60')")
    parser.add_argument("--registry", type=str, default="strategies/registry.yml",
                        help="Path to registry YAML (default: strategies/registry.yml)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify registry")
    parser.add_argument("--json", action="store_true", help="Output result as JSON (for cron)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Run rotation
    result = run_rotation(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        train_days=args.train_days,
        valid_days=args.valid_days,
        final_days=args.final_days,
        top_k=args.top_k,
        require_stress=not args.no_stress,
        db_path=args.db,
        grid=args.grid,
        improvement_threshold=args.improvement_threshold,
        dry_run=args.dry_run,
        registry_path=args.registry
    )

    # Output
    if args.json:
        output = {
            "symbol": result.symbol,
            "timestamp": result.timestamp,
            "action": result.action,
            "reason": result.reason,
            "current_champion_id": result.current_champion_id,
            "current_challenger_id": result.current_challenger_id,
            "proposed_challenger": {
                "strategy_id": result.proposed_challenger.strategy_id,
                "params": result.proposed_challenger.params,
                "final_test_pnl_pct": result.proposed_challenger.final_test_pnl_pct,
                "final_test_sharpe": result.proposed_challenger.final_test_sharpe,
                "score": result.proposed_challenger.score,
                "stress_passed": result.proposed_challenger.stress_passed
            } if result.proposed_challenger else None,
            "improvement_pct": result.improvement_pct,
            "cooldown_remaining_hours": result.cooldown_remaining_hours,
            "registry_updated": result.registry_updated,
            "error": result.error
        }
        print(json.dumps(output, indent=2))
    else:
        print_result_text(result)

    # Return code
    if result.action == "ERROR":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
