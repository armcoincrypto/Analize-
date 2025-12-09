"""
Database Manager for LongTrader

SQLite database for:
- Position tracking
- Order history
- Signal audit log
- Configuration
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class Position:
    """Trading position record."""
    id: str
    symbol: str
    side: str
    entry_price: float
    entry_qty: float
    entry_date: datetime
    entry_order_id: str
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_reason: Optional[str] = None
    exit_order_id: Optional[str] = None
    status: str = "OPEN"
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class OrderRecord:
    """Order record."""
    id: str
    position_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    filled_qty: float
    filled_price: float
    status: str
    bybit_order_id: str
    created_at: datetime


@dataclass
class SignalLog:
    """Signal audit log."""
    symbol: str
    signal_type: str
    price: float
    lower_band: float
    middle_band: float
    protection_passed: int
    protection_details: str
    action_taken: str
    created_at: datetime


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = "data/longtrader.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT DEFAULT 'LONG',
                entry_price REAL NOT NULL,
                entry_qty REAL NOT NULL,
                entry_date TEXT NOT NULL,
                entry_order_id TEXT,
                exit_price REAL,
                exit_date TEXT,
                exit_reason TEXT,
                exit_order_id TEXT,
                status TEXT DEFAULT 'OPEN',
                pnl_usdt REAL,
                pnl_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                position_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL,
                filled_qty REAL DEFAULT 0,
                filled_price REAL DEFAULT 0,
                status TEXT DEFAULT 'PENDING',
                bybit_order_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT
            )
        """)

        # Signals audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL NOT NULL,
                lower_band REAL,
                middle_band REAL,
                protection_passed INTEGER,
                protection_details TEXT,
                action_taken TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def create_position(self, position: Position) -> str:
        """Create a new position."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO positions (
                id, symbol, side, entry_price, entry_qty, entry_date,
                entry_order_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.id,
            position.symbol,
            position.side,
            position.entry_price,
            position.entry_qty,
            position.entry_date.isoformat(),
            position.entry_order_id,
            position.status,
        ))

        conn.commit()
        conn.close()
        return position.id

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        exit_order_id: str = "",
    ) -> Optional[Position]:
        """Close a position and calculate P&L."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get position
        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        # Calculate P&L
        entry_price = row["entry_price"]
        entry_qty = row["entry_qty"]
        pnl_usdt = (exit_price - entry_price) * entry_qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        # Update position
        cursor.execute("""
            UPDATE positions SET
                exit_price = ?,
                exit_date = ?,
                exit_reason = ?,
                exit_order_id = ?,
                status = 'CLOSED',
                pnl_usdt = ?,
                pnl_pct = ?
            WHERE id = ?
        """, (
            exit_price,
            datetime.now().isoformat(),
            exit_reason,
            exit_order_id,
            pnl_usdt,
            pnl_pct,
            position_id,
        ))

        conn.commit()

        # Get updated position
        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_position(row)

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_position(row)
        return None

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_position(row) for row in rows]

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM positions WHERE symbol = ? AND status = 'OPEN'",
            (symbol,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_position(row)
        return None

    def get_all_positions(self, limit: int = 100) -> List[Position]:
        """Get all positions (open and closed)."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM positions ORDER BY entry_date DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_position(row) for row in rows]

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """Convert database row to Position object."""
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            entry_price=row["entry_price"],
            entry_qty=row["entry_qty"],
            entry_date=datetime.fromisoformat(row["entry_date"]),
            entry_order_id=row["entry_order_id"] or "",
            exit_price=row["exit_price"],
            exit_date=datetime.fromisoformat(row["exit_date"]) if row["exit_date"] else None,
            exit_reason=row["exit_reason"],
            exit_order_id=row["exit_order_id"],
            status=row["status"],
            pnl_usdt=row["pnl_usdt"],
            pnl_pct=row["pnl_pct"],
        )

    # =========================================================================
    # ORDERS
    # =========================================================================

    def create_order(self, order: OrderRecord) -> str:
        """Create an order record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO orders (
                id, position_id, symbol, side, order_type, qty, price,
                filled_qty, filled_price, status, bybit_order_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.id,
            order.position_id,
            order.symbol,
            order.side,
            order.order_type,
            order.qty,
            order.price,
            order.filled_qty,
            order.filled_price,
            order.status,
            order.bybit_order_id,
            order.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()
        return order.id

    def update_order(self, order_id: str, **kwargs):
        """Update order fields."""
        conn = self._get_conn()
        cursor = conn.cursor()

        updates = []
        values = []
        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            values.append(value)

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(order_id)

        cursor.execute(
            f"UPDATE orders SET {', '.join(updates)} WHERE id = ?",
            values
        )

        conn.commit()
        conn.close()

    # =========================================================================
    # SIGNALS
    # =========================================================================

    def log_signal(self, signal: SignalLog):
        """Log a signal for audit."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, price, lower_band, middle_band,
                protection_passed, protection_details, action_taken, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.symbol,
            signal.signal_type,
            signal.price,
            signal.lower_band,
            signal.middle_band,
            signal.protection_passed,
            signal.protection_details,
            signal.action_taken,
            signal.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

    # =========================================================================
    # CONFIG
    # =========================================================================

    def set_config(self, key: str, value: Any):
        """Set a config value."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO config (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row["value"])
        return default

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Total positions
        cursor.execute("SELECT COUNT(*) as count FROM positions")
        total = cursor.fetchone()["count"]

        # Open positions
        cursor.execute("SELECT COUNT(*) as count FROM positions WHERE status = 'OPEN'")
        open_count = cursor.fetchone()["count"]

        # Closed positions
        cursor.execute("SELECT COUNT(*) as count FROM positions WHERE status = 'CLOSED'")
        closed_count = cursor.fetchone()["count"]

        # Wins/Losses
        cursor.execute("SELECT COUNT(*) as count FROM positions WHERE status = 'CLOSED' AND pnl_usdt > 0")
        wins = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM positions WHERE status = 'CLOSED' AND pnl_usdt <= 0")
        losses = cursor.fetchone()["count"]

        # Total P&L
        cursor.execute("SELECT SUM(pnl_usdt) as total FROM positions WHERE status = 'CLOSED'")
        total_pnl = cursor.fetchone()["total"] or 0

        conn.close()

        win_rate = (wins / closed_count * 100) if closed_count > 0 else 0

        return {
            "total_positions": total,
            "open_positions": open_count,
            "closed_positions": closed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl_usdt": round(total_pnl, 2),
        }


# Test function
def test_database():
    """Test the database."""
    import uuid

    print("Testing Database...")
    print("=" * 50)

    db = Database("data/test_longtrader.db")

    # Create a position
    pos = Position(
        id=str(uuid.uuid4()),
        symbol="XRPUSDT",
        side="LONG",
        entry_price=2.50,
        entry_qty=100,
        entry_date=datetime.now(),
        entry_order_id="test-order-1",
    )

    db.create_position(pos)
    print(f"Created position: {pos.id}")

    # Get open positions
    positions = db.get_open_positions()
    print(f"Open positions: {len(positions)}")

    # Close position
    closed = db.close_position(pos.id, 2.75, "TARGET", "test-order-2")
    print(f"Closed position: P&L = ${closed.pnl_usdt:.2f} ({closed.pnl_pct:.1f}%)")

    # Get stats
    stats = db.get_stats()
    print(f"Stats: {stats}")

    print("\nDatabase working!")


if __name__ == "__main__":
    test_database()
