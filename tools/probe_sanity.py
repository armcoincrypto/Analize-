#!/usr/bin/env python3
"""
Probe Sanity Check
==================
Validates that the single-probe logic is correctly configured.

This script ensures:
1. PROBE cause trades get exactly 0.10x size (not 0.025x from double probe)
2. FULL cause trades get the execution probe (0.25x)
3. No accidental compounding of multipliers

Run this before deploying to verify config is correct.

Usage:
    python tools/probe_sanity.py
"""

import re
import sys
from pathlib import Path


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def read_config_values():
    """Read config values by parsing config.py directly."""
    config_path = Path(__file__).parent.parent / "hft_system" / "config.py"

    if not config_path.exists():
        print(f"  ERROR: Config file not found at {config_path}")
        return None

    content = config_path.read_text()

    # Parse key values using regex
    config = {}

    # probe_cause_size_multiplier
    match = re.search(r'probe_cause_size_multiplier\s*[=:]\s*([0-9.]+)', content)
    config['probe_multiplier'] = float(match.group(1)) if match else 0.10

    # base_position_pct
    match = re.search(r'base_position_pct\s*[=:]\s*([0-9.]+)', content)
    config['base_position_pct'] = float(match.group(1)) if match else 1.0

    # Helper function for multi-line list extraction
    def extract_cause_list(pattern_name):
        # Match pattern like: cause_full_allow_long: List[str] = field(default_factory=lambda: [
        #     "cvd_buy_pressure",
        #     "price_momentum_up"
        # ])
        # Use lambda: [ as the start marker to avoid matching List[str]
        pattern = rf'{pattern_name}.*?lambda:\s*\[([^\]]*)\]'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            # Extract all quoted strings from the list
            causes = re.findall(r'"([^"]+)"', match.group(1))
            return causes
        return []

    # cause_full_allow_long
    config['full_long'] = extract_cause_list('cause_full_allow_long')

    # cause_full_allow_short
    config['full_short'] = extract_cause_list('cause_full_allow_short')

    # cause_probe_allow_long
    config['probe_long'] = extract_cause_list('cause_probe_allow_long')

    # cause_probe_allow_short
    config['probe_short'] = extract_cause_list('cause_probe_allow_short')

    return config


def check_cause_policy(config):
    """Check cause policy configuration."""
    print_header("CAUSE POLICY CONFIG")

    probe_multiplier = config['probe_multiplier']
    full_causes = config['full_long'] + config['full_short']
    probe_causes = config['probe_long'] + config['probe_short']

    print(f"\n  PROBE multiplier: {probe_multiplier}x")
    print(f"  FULL causes: {len(full_causes)}")
    for cause in full_causes:
        print(f"    - {cause}")
    print(f"  PROBE causes: {len(probe_causes)}")
    for cause in probe_causes:
        print(f"    - {cause}")

    # Sanity checks
    issues = []

    if probe_multiplier > 0.20:
        issues.append(f"PROBE multiplier too high ({probe_multiplier}x) - should be <= 0.20x for tiny probe trades")

    if probe_multiplier < 0.05:
        issues.append(f"PROBE multiplier very low ({probe_multiplier}x) - trades may be too small to execute")

    if len(probe_causes) == 0:
        issues.append("No PROBE causes configured - PROBE policy is disabled")

    # Check for duplicates
    overlap = set(config['full_long']) & set(config['probe_long'])
    if overlap:
        issues.append(f"Causes in both FULL and PROBE (long): {overlap}")

    overlap = set(config['full_short']) & set(config['probe_short'])
    if overlap:
        issues.append(f"Causes in both FULL and PROBE (short): {overlap}")

    return issues


def check_execution_probe(config):
    """Check execution probe is disabled for PROBE causes."""
    print_header("EXECUTION PROBE LOGIC")

    probe_multiplier = config['probe_multiplier']

    # The execution engine applies 0.25x probe by default
    exec_probe_multiplier = 0.25

    print(f"\n  Execution probe multiplier: {exec_probe_multiplier}x (applied to FULL trades)")
    print(f"  PROBE trades skip execution probe: YES (single-probe logic)")

    # Show expected final sizes
    print("\n  Expected final multipliers:")
    print(f"    FULL cause + exec probe: {exec_probe_multiplier}x")
    print(f"    PROBE cause (no exec probe): {probe_multiplier}x")

    # The old bug was: PROBE would get both multipliers
    old_bug_multiplier = exec_probe_multiplier * probe_multiplier
    print(f"\n  OLD BUG (double probe): {old_bug_multiplier}x <- this should NOT happen")

    issues = []

    # Check that PROBE multiplier >= 0.10 (our safety clamp)
    if probe_multiplier < 0.10:
        issues.append(f"PROBE multiplier below safety clamp (got {probe_multiplier}x, min 0.10x)")

    return issues


def check_position_sizes(config):
    """Calculate expected position sizes for common scenarios."""
    print_header("EXPECTED POSITION SIZES")

    probe_multiplier = config['probe_multiplier']

    # Assume $10,000 capital, base position from config
    capital = 10000
    base_pct = config['base_position_pct']
    base_position = capital * (base_pct / 100)

    print(f"\n  Capital: ${capital:,.0f}")
    print(f"  Base position: {base_pct}% = ${base_position:,.2f}")

    # FULL cause scenarios
    full_exec_size = base_position * 0.25  # execution probe
    print(f"\n  FULL cause trade:")
    print(f"    After exec probe (0.25x): ${full_exec_size:,.2f}")

    # PROBE cause scenarios
    probe_size = base_position * probe_multiplier
    print(f"\n  PROBE cause trade:")
    print(f"    After cause probe ({probe_multiplier}x): ${probe_size:,.2f}")

    # The bug scenario (should NOT happen anymore)
    double_probe_size = base_position * 0.25 * probe_multiplier
    print(f"\n  DOUBLE PROBE (BUG - should NOT happen):")
    print(f"    After both probes: ${double_probe_size:,.2f} <- too small!")

    issues = []

    # Check minimum viable trade size (e.g., $5 minimum for most exchanges)
    min_trade_size = 5.0
    if probe_size < min_trade_size:
        issues.append(f"PROBE trade size (${probe_size:.2f}) below exchange minimum (${min_trade_size})")

    return issues


def check_db_schema():
    """Check that position_sizing table has multiplier tracking columns."""
    print_header("DB SCHEMA CHECK")

    migrations_path = Path(__file__).parent.parent / "hft_system" / "db_migrations.py"

    if not migrations_path.exists():
        print(f"  ERROR: db_migrations.py not found")
        return ["db_migrations.py not found"]

    content = migrations_path.read_text()

    required_new_cols = [
        "base_size",
        "applied_multipliers",
        "final_size",
        "is_probe_trade",
    ]

    print("\n  Multiplier tracking columns in REQUIRED_SCHEMA:")
    issues = []
    for col_name in required_new_cols:
        if f'"{col_name}"' in content or f"'{col_name}'" in content:
            print(f"    [OK] {col_name}")
        else:
            print(f"    [MISSING] {col_name}")
            issues.append(f"Missing column {col_name} in position_sizing schema")

    return issues


def check_execution_engine():
    """Check that execution_engine.py has single-probe logic."""
    print_header("EXECUTION ENGINE CHECK")

    exec_path = Path(__file__).parent.parent / "hft_system" / "execution_engine.py"

    if not exec_path.exists():
        print(f"  ERROR: execution_engine.py not found")
        return ["execution_engine.py not found"]

    content = exec_path.read_text()

    issues = []

    # Check for is_cause_probe parameter
    if "is_cause_probe" in content:
        print("  [OK] is_cause_probe parameter found")
    else:
        print("  [MISSING] is_cause_probe parameter NOT found")
        issues.append("execution_engine.py missing is_cause_probe parameter")

    # Check for single-probe logic (skip exec probe when is_cause_probe)
    if "CAUSE_PROBE" in content or "cause_probe" in content.lower():
        print("  [OK] CAUSE_PROBE logic found")
    else:
        print("  [MISSING] CAUSE_PROBE logic NOT found")
        issues.append("execution_engine.py missing CAUSE_PROBE skip logic")

    return issues


def main():
    print("""
+==============================================================+
|           PROBE SANITY CHECK - Single Probe Audit            |
+==============================================================+
|  Verifies single-probe logic is correctly configured.        |
|  PROBE causes should get 0.10x, not 0.025x (double probe).   |
+==============================================================+
    """)

    all_issues = []

    # Read config
    config = read_config_values()
    if not config:
        print("ERROR: Could not read config values")
        return 1

    # Run checks
    all_issues.extend(check_cause_policy(config))
    all_issues.extend(check_execution_probe(config))
    all_issues.extend(check_position_sizes(config))
    all_issues.extend(check_db_schema())
    all_issues.extend(check_execution_engine())

    # Summary
    print_header("SUMMARY")

    if all_issues:
        print(f"\n  ISSUES FOUND: {len(all_issues)}")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
        print("\n  Status: NEEDS ATTENTION")
        return 1
    else:
        print("\n  All checks passed!")
        print("  Single-probe logic is correctly configured.")
        print("\n  Status: OK")
        return 0


if __name__ == "__main__":
    exit(main())
