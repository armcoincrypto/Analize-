#!/usr/bin/env python3
"""
Find Trade Blockers
===================
Search the codebase for common blocker tokens that might prevent trades.

This tool helps locate configuration knobs and filtering logic that
could be causing "0 trades" situations.

Usage:
    python tools/find_trade_blockers.py
    python tools/find_trade_blockers.py --pattern spread
    python tools/find_trade_blockers.py --verbose
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple


# Common blocker patterns to search for
DEFAULT_PATTERNS = [
    "winner",
    "gate",
    "imb",
    "imbalance",
    "spread",
    "no_trade",
    "blocked",
    "delta",
    "ob_",
    "orderbook",
    "strict",
    "confidence",
    "tier",
    "pocket",
    "min_",
    "max_",
    "threshold",
    "cooldown",
    "rate_limit",
    "avoid",
    "skip",
    "filter",
    "regime",
]

# File extensions to search
SEARCH_EXTENSIONS = {".py"}

# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".tox",
    "build",
    "dist",
    ".eggs",
}


def find_files(root: Path, extensions: set) -> List[Path]:
    """Find all files with given extensions under root."""
    files = []
    for item in root.rglob("*"):
        if item.is_file() and item.suffix in extensions:
            # Skip directories in SKIP_DIRS
            if not any(skip in item.parts for skip in SKIP_DIRS):
                files.append(item)
    return sorted(files)


def search_file(filepath: Path, patterns: List[str], case_insensitive: bool = True) -> List[Tuple[int, str, str]]:
    """
    Search a file for patterns.

    Returns list of (line_number, matched_pattern, line_content).
    """
    matches = []
    flags = re.IGNORECASE if case_insensitive else 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for pattern in patterns:
                    if re.search(pattern, line, flags):
                        matches.append((line_num, pattern, line.rstrip()))
                        break  # Only report first pattern match per line
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")

    return matches


def format_match(filepath: Path, line_num: int, pattern: str, line: str, verbose: bool) -> str:
    """Format a match for display."""
    # Relative path from cwd
    try:
        rel_path = filepath.relative_to(Path.cwd())
    except ValueError:
        rel_path = filepath

    if verbose:
        return f"{rel_path}:{line_num} [{pattern}]\n    {line.strip()}"
    else:
        return f"{rel_path}:{line_num} [{pattern}] {line.strip()[:80]}"


def main():
    parser = argparse.ArgumentParser(
        description="Find trade blocker patterns in codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for all default blocker patterns
  python tools/find_trade_blockers.py

  # Search for specific pattern
  python tools/find_trade_blockers.py --pattern spread

  # Search with multiple custom patterns
  python tools/find_trade_blockers.py --pattern "winner|gate|blocked"

  # Verbose output with full line content
  python tools/find_trade_blockers.py --verbose

  # Search specific directory
  python tools/find_trade_blockers.py --dir hft_system
        """
    )
    parser.add_argument("--pattern", "-p", type=str, default=None,
                        help="Custom pattern to search (regex). "
                             "Default: common blocker tokens")
    parser.add_argument("--dir", "-d", type=str, default=".",
                        help="Directory to search (default: current)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full line content")
    parser.add_argument("--list-patterns", action="store_true",
                        help="List default patterns and exit")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Case-sensitive search")
    args = parser.parse_args()

    if args.list_patterns:
        print("Default blocker patterns:")
        for p in DEFAULT_PATTERNS:
            print(f"  {p}")
        return 0

    # Determine patterns to search
    if args.pattern:
        patterns = [args.pattern]
    else:
        patterns = DEFAULT_PATTERNS

    # Find files
    root = Path(args.dir)
    if not root.exists():
        print(f"Error: Directory not found: {args.dir}")
        return 1

    files = find_files(root, SEARCH_EXTENSIONS)
    print(f"Searching {len(files)} Python files for blocker patterns...")
    print()

    # Search files
    total_matches = 0
    results_by_file = {}

    for filepath in files:
        matches = search_file(filepath, patterns, not args.case_sensitive)
        if matches:
            results_by_file[filepath] = matches
            total_matches += len(matches)

    # Display results grouped by file
    if results_by_file:
        print(f"Found {total_matches} matches in {len(results_by_file)} files:")
        print("=" * 70)

        for filepath, matches in sorted(results_by_file.items()):
            print(f"\n{filepath} ({len(matches)} matches):")
            print("-" * 40)
            for line_num, pattern, line in matches:
                print(format_match(filepath, line_num, pattern, line, args.verbose))

        print()
        print("=" * 70)
        print(f"Summary: {total_matches} blocker-related lines in {len(results_by_file)} files")
        print()
        print("Common adjustment locations:")

        # Find config files
        config_matches = [f for f in results_by_file.keys()
                         if "config" in f.name.lower()]
        if config_matches:
            print("  Config files (tune thresholds here):")
            for f in config_matches:
                print(f"    {f}")

        # Find main bot file
        bot_matches = [f for f in results_by_file.keys()
                      if "bot" in f.name.lower() or "hft" in f.name.lower()]
        if bot_matches:
            print("  Bot files (entry logic here):")
            for f in bot_matches:
                print(f"    {f}")

    else:
        print("No matches found.")

    return 0


if __name__ == "__main__":
    exit(main())
