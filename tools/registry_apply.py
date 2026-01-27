#!/usr/bin/env python3
"""
Strategy Registry Apply Tool
=============================
Selects a strategy from the registry and generates an active config file.

Usage:
    # List all strategies
    python tools/registry_apply.py --list

    # Show strategy details
    python tools/registry_apply.py --strategy xrp_momentum_v1 --show

    # Generate config file (dry-run)
    python tools/registry_apply.py --symbol XRPUSDT --dry-run

    # Apply champion strategy for symbol
    python tools/registry_apply.py --symbol XRPUSDT --apply

    # Apply specific strategy
    python tools/registry_apply.py --strategy xrp_momentum_v2 --apply

    # Apply and generate custom output path
    python tools/registry_apply.py --symbol XRPUSDT --apply --output strategies/my_config.yml

Output:
    - Prints config diff
    - Generates strategies/generated_active.yml (or custom path)
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from hft_system.strategy_loader import (
    StrategyLoader,
    StrategyConfig,
    get_config_diff,
)


def print_strategy_list(loader: StrategyLoader):
    """Print list of all strategies."""
    strategies = loader.list_strategies()

    print("""
+================================================================================+
|                         STRATEGY REGISTRY                                      |
+================================================================================+
""")

    # Group by symbol
    by_symbol: Dict[str, list] = {}
    for sid, info in strategies.items():
        symbol = info["symbol"]
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append((sid, info))

    for symbol, strats in sorted(by_symbol.items()):
        # Get champion/challenger
        champion_id = loader.assignments.get(symbol, {}).get("champion")
        challenger_id = loader.assignments.get(symbol, {}).get("challenger")

        print(f"  {symbol}")
        print(f"  " + "-" * 70)

        for sid, info in strats:
            status_icon = {
                "champion": "[C]",
                "challenger": "[c]",
                "active": "[A]",
                "testing": "[T]",
                "retired": "[R]",
            }.get(info["status"], "[ ]")

            stress = "+" if info.get("stress_test_passed") else "-"
            pnl = info.get("final_test_pnl_pct")
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "N/A"

            role = ""
            if sid == champion_id:
                role = " <-- CHAMPION"
            elif sid == challenger_id:
                role = " <-- CHALLENGER"

            print(f"    {status_icon} {sid:30s} | Stress:{stress} | PnL:{pnl_str:>8s}{role}")
            print(f"        {info.get('description', '')[:60]}")

        print()

    print(f"  Total: {len(strategies)} strategies")
    print()


def print_strategy_details(strategy: StrategyConfig):
    """Print detailed strategy info."""
    print(f"""
+================================================================================+
|  STRATEGY: {strategy.strategy_id:63s} |
+================================================================================+

  Symbol:      {strategy.symbol}
  Status:      {strategy.status}
  Description: {strategy.description}

  --- REGIMES ---
  Allowed:  {', '.join(strategy.regimes_allowed) or 'all'}
  Blocked:  {', '.join(strategy.regimes_blocked) or 'none'}

  --- CAUSES ---
  Enabled:  {', '.join(strategy.causes.enabled[:5])}{'...' if len(strategy.causes.enabled) > 5 else ''}
  Disabled: {', '.join(strategy.causes.disabled[:5])}{'...' if len(strategy.causes.disabled) > 5 else ''}
  Probe:    {', '.join(strategy.causes.probe_causes) or 'none'}
  Block:    {', '.join(strategy.causes.block_causes) or 'none'}

  --- PARAMETERS ---
  Take Profit:   {strategy.params.take_profit_pct}%
  Stop Loss:     {strategy.params.stop_loss_pct}%
  Time Stop:     {strategy.params.time_stop_seconds}s
  Min Conditions:{strategy.params.min_conditions}
  Min Imbalance: {strategy.params.min_imbalance}
  Min Confidence:{strategy.params.min_confidence}

  --- EXECUTION ---
  Mode:          {strategy.execution.mode}
  Maker Spread:  <= {strategy.execution.auto_maker_spread_threshold_bps} bps
  Max Spread:    {strategy.execution.max_spread_bps} bps
  Maker Timeout: {strategy.execution.maker_timeout_ms} ms

  --- RISK ---
  Position Size: {strategy.risk.position_size_pct}%
  Max Positions: {strategy.risk.max_positions}
  Max Daily Loss:{strategy.risk.max_daily_loss_pct}%
  Max Drawdown:  {strategy.risk.max_drawdown_pct}%
  Loss Pause:    {strategy.risk.loss_streak_pause_trades} trades / {strategy.risk.loss_streak_pause_minutes} min

  --- METADATA ---
  Created:       {strategy.metadata.created_at or 'N/A'}
  Updated:       {strategy.metadata.updated_at or 'N/A'}
  Commit:        {strategy.metadata.commit_hash or 'N/A'}
  WF Run:        {strategy.metadata.walkforward_run_id or 'N/A'}

  --- VALIDATION ---
  Final Test PnL:   {strategy.metadata.final_test_pnl_pct or 'N/A'}%
  Final Test Sharpe:{strategy.metadata.final_test_sharpe or 'N/A'}
  Final Test WR:    {strategy.metadata.final_test_win_rate or 'N/A'}%
  Final Test Trades:{strategy.metadata.final_test_trades or 'N/A'}
  Stress Passed:    {'YES' if strategy.metadata.stress_test_passed else 'NO'}
  Promotion OK:     {'YES' if strategy.metadata.promotion_approved else 'NO'}

  Notes: {strategy.metadata.notes or 'None'}
""")


def print_config_diff(strategy: StrategyConfig, diff: Dict[str, tuple]):
    """Print configuration diff."""
    print("""
  --- CONFIG DIFF ---
""")

    if not diff:
        print("  No changes (strategy matches current config)")
        return

    print(f"  {'Parameter':<30s} {'Current':<15s} {'New':<15s}")
    print(f"  {'-' * 30} {'-' * 15} {'-' * 15}")

    for param, (current, new) in diff.items():
        current_str = str(current)[:14]
        new_str = str(new)[:14]
        print(f"  {param:<30s} {current_str:<15s} {new_str:<15s}")

    print()


def generate_config_file(
    strategy: StrategyConfig,
    output_path: str,
    format: str = "yaml"
) -> str:
    """Generate config file from strategy."""
    output_dict = {
        "_meta": {
            "source": f"registry:{strategy.strategy_id}",
            "generated_at": datetime.now().isoformat(),
            "symbol": strategy.symbol,
            "status": strategy.status,
        },
        "strategy": strategy.to_dict()
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "yaml" and YAML_AVAILABLE:
        with open(path, "w") as f:
            yaml.dump(output_dict, f, default_flow_style=False, sort_keys=False)
    else:
        with open(path, "w") as f:
            json.dump(output_dict, f, indent=2)

    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Registry Apply Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all strategies
  python tools/registry_apply.py --list

  # Show strategy details
  python tools/registry_apply.py --strategy xrp_momentum_v1 --show

  # Dry-run: show what would change
  python tools/registry_apply.py --symbol XRPUSDT --dry-run

  # Apply champion strategy
  python tools/registry_apply.py --symbol XRPUSDT --apply

  # Apply specific strategy
  python tools/registry_apply.py --strategy xrp_momentum_v2 --apply

  # Custom output path
  python tools/registry_apply.py --strategy xrp_momentum_v1 --apply --output my_config.yml
        """
    )

    parser.add_argument("--registry", default="strategies/registry.yml",
                        help="Path to registry file")
    parser.add_argument("--list", action="store_true",
                        help="List all strategies")
    parser.add_argument("--symbol", type=str,
                        help="Select champion strategy for symbol")
    parser.add_argument("--strategy", type=str,
                        help="Select specific strategy by ID")
    parser.add_argument("--show", action="store_true",
                        help="Show strategy details")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config diff without applying")
    parser.add_argument("--apply", action="store_true",
                        help="Generate config file")
    parser.add_argument("--output", type=str, default="strategies/generated_active.yml",
                        help="Output config file path")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml",
                        help="Output format")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON to stdout")

    args = parser.parse_args()

    # Find registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        for try_path in [
            Path.cwd() / "strategies" / "registry.yml",
            Path(__file__).parent.parent / "strategies" / "registry.yml",
            Path("/root/Analize-/strategies/registry.yml"),
        ]:
            if try_path.exists():
                registry_path = try_path
                break

    if not registry_path.exists():
        print(f"ERROR: Registry not found at {args.registry}")
        return 1

    # Load registry
    loader = StrategyLoader(str(registry_path))

    # List mode
    if args.list:
        if args.json:
            print(json.dumps(loader.list_strategies(), indent=2))
        else:
            print_strategy_list(loader)
        return 0

    # Determine which strategy to use
    strategy: Optional[StrategyConfig] = None

    if args.strategy:
        strategy = loader.get_strategy(args.strategy)
        if not strategy:
            print(f"ERROR: Strategy '{args.strategy}' not found")
            print(f"Available: {', '.join(loader.strategies.keys())}")
            return 1

    elif args.symbol:
        strategy = loader.get_champion(args.symbol)
        if not strategy:
            print(f"ERROR: No champion strategy for symbol '{args.symbol}'")
            # Show available
            available = loader.get_strategies_for_symbol(args.symbol)
            if available:
                print(f"Available strategies for {args.symbol}:")
                for s in available:
                    print(f"  - {s.strategy_id} ({s.status})")
            return 1

    # Show mode
    if args.show:
        if not strategy:
            print("ERROR: Specify --strategy or --symbol to show")
            return 1
        if args.json:
            print(json.dumps(strategy.to_dict(), indent=2))
        else:
            print_strategy_details(strategy)
        return 0

    # Dry-run or apply mode
    if args.dry_run or args.apply:
        if not strategy:
            print("ERROR: Specify --strategy or --symbol")
            return 1

        # Validate strategy
        errors = loader.validate_strategy(strategy)
        if errors:
            print(f"WARNING: Strategy has validation issues:")
            for err in errors:
                print(f"  - {err}")
            print()

        # Try to import current config for diff
        diff = {}
        try:
            from hft_system.config import SYSTEM_CONFIG, COST_MODEL
            diff = get_config_diff(strategy, SYSTEM_CONFIG, COST_MODEL)
        except ImportError:
            pass

        if args.json:
            output = {
                "strategy_id": strategy.strategy_id,
                "symbol": strategy.symbol,
                "status": strategy.status,
                "validation_errors": errors,
                "config_diff": {k: {"current": v[0], "new": v[1]} for k, v in diff.items()},
                "would_generate": args.output if args.apply else None,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"""
+================================================================================+
|  APPLYING STRATEGY: {strategy.strategy_id:55s} |
+================================================================================+

  Symbol:  {strategy.symbol}
  Status:  {strategy.status}
  Stress:  {'PASSED' if strategy.metadata.stress_test_passed else 'NOT PASSED'}
""")
            print_config_diff(strategy, diff)

        if args.apply:
            # Generate config file
            output_path = generate_config_file(strategy, args.output, args.format)

            if not args.json:
                print(f"""
  --- GENERATED ---
  Output: {output_path}

  To use this strategy, run:
    python hft_bot.py --strategy-file {output_path}

  Or set environment variable:
    export HFT_STRATEGY_FILE={output_path}
""")

        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
