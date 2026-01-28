"""
Symbol Normalization Utilities
==============================
Provides consistent symbol handling across the HFT system.

Problem: Database stores 'XRP' but Binance API uses 'XRPUSDT'.
Solution: Normalize symbols for consistent storage and queries.

Usage:
    from hft_system.symbol_utils import normalize_symbol, resolve_symbol, normalize_db_symbol

    # For storing in DB (always full symbol)
    symbol_to_store = normalize_symbol("XRP")  # -> "XRPUSDT"

    # For querying DB (handles both formats)
    base = normalize_db_symbol("XRPUSDT")  # -> "XRP" (base only, for LIKE queries)

    # For Binance API calls
    api_symbol = resolve_symbol("XRP")  # -> "XRPUSDT"
"""

from typing import Optional, Tuple

# Common quote currencies on Binance
QUOTE_CURRENCIES = ["USDT", "BUSD", "BTC", "ETH", "BNB", "USDC", "TUSD", "EUR", "GBP"]

# Default quote currency
DEFAULT_QUOTE = "USDT"


def normalize_symbol(symbol: str, default_quote: str = DEFAULT_QUOTE) -> str:
    """
    Normalize symbol to full format for storage (e.g., XRP -> XRPUSDT).

    Always returns the full trading pair symbol for consistent storage.

    Args:
        symbol: Input symbol (e.g., "XRP", "XRPUSDT", "xrp")
        default_quote: Quote currency to append if missing (default: "USDT")

    Returns:
        Normalized symbol in uppercase with quote currency (e.g., "XRPUSDT")

    Examples:
        >>> normalize_symbol("XRP")
        'XRPUSDT'
        >>> normalize_symbol("XRPUSDT")
        'XRPUSDT'
        >>> normalize_symbol("btc", "USDT")
        'BTCUSDT'
        >>> normalize_symbol("ETH", "BTC")
        'ETHBTC'
    """
    if not symbol:
        return symbol

    symbol = symbol.upper().strip()
    default_quote = default_quote.upper().strip()

    # Check if symbol already has a known quote currency
    for quote in QUOTE_CURRENCIES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol  # Already has a valid quote currency

    # Append default quote if not present
    return symbol + default_quote


def normalize_db_symbol(stored_symbol: str) -> str:
    """
    Normalize a stored DB symbol for LIKE queries.

    Extracts base currency for flexible matching.
    Use this when you need to match both "XRP" and "XRPUSDT" in queries.

    Args:
        stored_symbol: Symbol from database or user input

    Returns:
        Base currency only (e.g., "XRPUSDT" -> "XRP", "XRP" -> "XRP")

    Examples:
        >>> normalize_db_symbol("XRPUSDT")
        'XRP'
        >>> normalize_db_symbol("XRP")
        'XRP'
        >>> normalize_db_symbol("btcusdt")
        'BTC'
    """
    if not stored_symbol:
        return stored_symbol

    symbol = stored_symbol.upper().strip()

    # Remove known quote currencies
    for quote in QUOTE_CURRENCIES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[:-len(quote)]

    return symbol


def resolve_symbol(symbol: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Resolve symbol to Binance API format.

    Alias for normalize_symbol - use this for API calls.

    Args:
        symbol: Input symbol (e.g., "XRP", "XRPUSDT", "xrp")
        quote: Quote currency to append if missing (default: "USDT")

    Returns:
        Resolved symbol for Binance API (e.g., "XRPUSDT")
    """
    return normalize_symbol(symbol, quote)


def split_symbol(symbol: str) -> Tuple[str, Optional[str]]:
    """
    Split a trading pair symbol into base and quote currencies.

    Args:
        symbol: Trading pair symbol (e.g., "XRPUSDT", "ETHBTC")

    Returns:
        Tuple of (base, quote) or (symbol, None) if no quote found

    Examples:
        >>> split_symbol("XRPUSDT")
        ('XRP', 'USDT')
        >>> split_symbol("ETHBTC")
        ('ETH', 'BTC')
        >>> split_symbol("XRP")
        ('XRP', None)
    """
    if not symbol:
        return (symbol, None)

    symbol = symbol.upper().strip()

    for quote in QUOTE_CURRENCIES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[:-len(quote)]
            return (base, quote)

    return (symbol, None)


def symbols_match(symbol1: str, symbol2: str) -> bool:
    """
    Check if two symbols refer to the same trading pair.

    Handles both formats: "XRP" matches "XRPUSDT".

    Args:
        symbol1: First symbol
        symbol2: Second symbol

    Returns:
        True if symbols match (same base currency)

    Examples:
        >>> symbols_match("XRP", "XRPUSDT")
        True
        >>> symbols_match("XRPUSDT", "xrp")
        True
        >>> symbols_match("XRP", "ETH")
        False
    """
    if not symbol1 or not symbol2:
        return False

    base1 = normalize_db_symbol(symbol1)
    base2 = normalize_db_symbol(symbol2)

    return base1 == base2


def build_symbol_filter(symbol: str, column_name: str = "symbol") -> Tuple[str, str]:
    """
    Build SQL WHERE clause for symbol filtering.

    Handles both formats: "XRP" or "XRPUSDT" will match either in DB.

    Args:
        symbol: Symbol to filter by
        column_name: Name of the symbol column in SQL

    Returns:
        Tuple of (sql_clause, param_value)

    Examples:
        >>> build_symbol_filter("XRP")
        ('symbol LIKE ?', '%XRP%')
        >>> build_symbol_filter("XRPUSDT", "t.symbol")
        ('t.symbol LIKE ?', '%XRP%')
    """
    base = normalize_db_symbol(symbol)
    return (f"{column_name} LIKE ?", f"%{base}%")


# For backwards compatibility
def get_symbol_for_storage(symbol: str) -> str:
    """Alias for normalize_symbol - use for storing new records."""
    return normalize_symbol(symbol)


def get_symbol_for_query(symbol: str) -> str:
    """Alias for normalize_db_symbol - use for DB queries."""
    return normalize_db_symbol(symbol)


# Self-test
if __name__ == "__main__":
    print("Symbol Normalization Utils - Self Test")
    print("=" * 50)

    test_cases = [
        # (input, expected_normalized, expected_db_normalized)
        ("XRP", "XRPUSDT", "XRP"),
        ("xrp", "XRPUSDT", "XRP"),
        ("XRPUSDT", "XRPUSDT", "XRP"),
        ("xrpusdt", "XRPUSDT", "XRP"),
        ("BTC", "BTCUSDT", "BTC"),
        ("ETHBTC", "ETHBTC", "ETH"),
        ("SOL", "SOLUSDT", "SOL"),
        ("SUIUSDT", "SUIUSDT", "SUI"),
    ]

    all_passed = True
    for input_sym, expected_norm, expected_db in test_cases:
        norm = normalize_symbol(input_sym)
        db_norm = normalize_db_symbol(input_sym)

        norm_ok = norm == expected_norm
        db_ok = db_norm == expected_db

        if not norm_ok or not db_ok:
            all_passed = False

        status = "PASS" if norm_ok and db_ok else "FAIL"
        print(f"[{status}] '{input_sym}' -> normalize: '{norm}' (exp: '{expected_norm}'), "
              f"db_normalize: '{db_norm}' (exp: '{expected_db}')")

    print()
    print("Testing symbols_match:")
    match_cases = [
        ("XRP", "XRPUSDT", True),
        ("xrp", "XRPUSDT", True),
        ("XRP", "ETH", False),
        ("BTCUSDT", "btc", True),
    ]
    for s1, s2, expected in match_cases:
        result = symbols_match(s1, s2)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"[{status}] symbols_match('{s1}', '{s2}') = {result} (expected: {expected})")

    print()
    print("Testing split_symbol:")
    split_cases = [
        ("XRPUSDT", ("XRP", "USDT")),
        ("ETHBTC", ("ETH", "BTC")),
        ("XRP", ("XRP", None)),
    ]
    for sym, expected in split_cases:
        result = split_symbol(sym)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"[{status}] split_symbol('{sym}') = {result} (expected: {expected})")

    print()
    print("=" * 50)
    print(f"All tests: {'PASSED' if all_passed else 'FAILED'}")
