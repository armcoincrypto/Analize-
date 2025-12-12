#!/usr/bin/env python3
"""
Governance & Human-in-the-Loop Approval System

Implements safety controls requiring human approval for:
1. Large trades (>X% of AUM)
2. New strategy deployments
3. Model updates
4. Risk parameter changes
5. Override mode for emergency actions

Features:
- Approval queue with timeout
- Audit trail for all approvals
- Role-based access control
- Emergency override with logging

Author: Cloud AI Analyzer
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pathlib import Path
from enum import Enum
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

GOVERNANCE_DB_PATH = Path(__file__).parent / "governance.db"


class ApprovalStatus(Enum):
    """Approval request status."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    AUTO_APPROVED = "AUTO_APPROVED"


class ActionType(Enum):
    """Types of actions requiring approval."""
    LARGE_TRADE = "large_trade"
    MODEL_DEPLOY = "model_deploy"
    RISK_PARAM_CHANGE = "risk_param_change"
    STRATEGY_ENABLE = "strategy_enable"
    STRATEGY_DISABLE = "strategy_disable"
    EMERGENCY_OVERRIDE = "emergency_override"
    WITHDRAWAL = "withdrawal"
    API_KEY_CHANGE = "api_key_change"


class Role(Enum):
    """User roles for access control."""
    VIEWER = "viewer"           # Read-only access
    TRADER = "trader"           # Can execute approved trades
    RISK_MANAGER = "risk_manager"  # Can approve trades
    ADMIN = "admin"             # Full access
    SYSTEM = "system"           # Automated system actions


@dataclass
class ApprovalRequest:
    """Approval request record."""
    request_id: str
    action_type: ActionType
    description: str
    requestor: str
    timestamp: datetime
    data: Dict
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    expires_at: Optional[datetime] = None
    auto_approve_after: Optional[timedelta] = None


@dataclass
class AuditEntry:
    """Audit trail entry."""
    entry_id: str
    timestamp: datetime
    action: str
    actor: str
    role: Role
    details: Dict
    ip_address: Optional[str] = None
    success: bool = True


@dataclass
class GovernanceConfig:
    """Governance configuration."""

    # Thresholds requiring approval
    large_trade_threshold_pct: float = 5.0    # Trades > 5% AUM need approval
    model_deploy_requires_approval: bool = True
    risk_change_requires_approval: bool = True

    # Timeouts
    approval_timeout_hours: int = 24
    auto_approve_low_risk_hours: int = 4      # Auto-approve low risk after 4h

    # Track record requirements
    min_track_record_months: int = 3
    min_trades_for_trust: int = 100

    # Emergency settings
    emergency_cooldown_hours: int = 72


# =============================================================================
# GOVERNANCE MANAGER
# =============================================================================

class GovernanceManager:
    """
    Human-in-the-loop governance system.

    Manages approvals, audit trails, and access control for
    high-risk trading operations.
    """

    def __init__(
        self,
        config: GovernanceConfig = None,
        db_path: str = None
    ):
        """Initialize governance manager."""
        self.config = config or GovernanceConfig()
        self.db_path = db_path or str(GOVERNANCE_DB_PATH)

        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approval_callbacks: Dict[str, Callable] = {}

        self._init_database()

        print("=" * 60)
        print("GOVERNANCE SYSTEM INITIALIZED")
        print("=" * 60)
        print(f"  Large Trade Threshold: {self.config.large_trade_threshold_pct}%")
        print(f"  Approval Timeout:      {self.config.approval_timeout_hours}h")
        print(f"  Min Track Record:      {self.config.min_track_record_months} months")
        print("=" * 60)

    def _init_database(self):
        """Initialize governance database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Approval requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approval_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                action_type TEXT NOT NULL,
                description TEXT NOT NULL,
                requestor TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT,
                status TEXT DEFAULT 'PENDING',
                approver TEXT,
                approval_timestamp TEXT,
                rejection_reason TEXT,
                expires_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Audit trail table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                role TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                success INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Users/roles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                api_key_hash TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_requests_status
            ON approval_requests(status, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor
            ON audit_trail(actor, timestamp)
        """)

        conn.commit()
        conn.close()

    # =========================================================================
    # APPROVAL MANAGEMENT
    # =========================================================================

    def request_approval(
        self,
        action_type: ActionType,
        description: str,
        requestor: str,
        data: Dict = None,
        callback: Callable = None,
        timeout_hours: int = None
    ) -> ApprovalRequest:
        """
        Create an approval request.

        Args:
            action_type: Type of action requiring approval
            description: Human-readable description
            requestor: Who is requesting
            data: Additional data about the request
            callback: Function to call when approved
            timeout_hours: Override default timeout

        Returns:
            Created approval request
        """
        request_id = f"REQ-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(description.encode()).hexdigest()[:6]}"

        timeout = timeout_hours or self.config.approval_timeout_hours
        expires_at = datetime.now() + timedelta(hours=timeout)

        request = ApprovalRequest(
            request_id=request_id,
            action_type=action_type,
            description=description,
            requestor=requestor,
            timestamp=datetime.now(),
            data=data or {},
            expires_at=expires_at,
        )

        self.pending_requests[request_id] = request

        if callback:
            self.approval_callbacks[request_id] = callback

        self._save_request(request)
        self._log_audit("approval_requested", requestor, Role.SYSTEM, {
            'request_id': request_id,
            'action_type': action_type.value,
            'description': description,
        })

        print(f"\n{'='*50}")
        print(f"⏳ APPROVAL REQUIRED")
        print(f"{'='*50}")
        print(f"  Request ID:  {request_id}")
        print(f"  Type:        {action_type.value}")
        print(f"  Description: {description}")
        print(f"  Requestor:   {requestor}")
        print(f"  Expires:     {expires_at.strftime('%Y-%m-%d %H:%M')}")

        return request

    def approve(
        self,
        request_id: str,
        approver: str,
        role: Role = Role.ADMIN
    ) -> bool:
        """
        Approve a pending request.

        Args:
            request_id: Request to approve
            approver: Who is approving
            role: Approver's role

        Returns:
            Success status
        """
        if request_id not in self.pending_requests:
            # Try to load from database
            request = self._load_request(request_id)
            if not request:
                print(f"  [GOV] Request {request_id} not found")
                return False
        else:
            request = self.pending_requests[request_id]

        # Check if still valid
        if request.status != ApprovalStatus.PENDING:
            print(f"  [GOV] Request {request_id} is not pending (status: {request.status.value})")
            return False

        if request.expires_at and datetime.now() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            self._update_request(request)
            print(f"  [GOV] Request {request_id} has expired")
            return False

        # Check role permissions
        if not self._can_approve(role, request.action_type):
            print(f"  [GOV] Role {role.value} cannot approve {request.action_type.value}")
            return False

        # Approve
        request.status = ApprovalStatus.APPROVED
        request.approver = approver
        request.approval_timestamp = datetime.now()

        self._update_request(request)
        self._log_audit("approval_granted", approver, role, {
            'request_id': request_id,
            'action_type': request.action_type.value,
        })

        # Execute callback if registered
        if request_id in self.approval_callbacks:
            try:
                self.approval_callbacks[request_id](request)
                del self.approval_callbacks[request_id]
            except Exception as e:
                print(f"  [GOV] Callback error: {e}")

        # Remove from pending
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]

        print(f"\n{'='*50}")
        print(f"✓ REQUEST APPROVED")
        print(f"{'='*50}")
        print(f"  Request ID:  {request_id}")
        print(f"  Approved by: {approver}")
        print(f"  At:          {request.approval_timestamp}")

        return True

    def reject(
        self,
        request_id: str,
        rejector: str,
        reason: str,
        role: Role = Role.ADMIN
    ) -> bool:
        """Reject a pending request."""

        if request_id not in self.pending_requests:
            request = self._load_request(request_id)
            if not request:
                return False
        else:
            request = self.pending_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.REJECTED
        request.approver = rejector
        request.approval_timestamp = datetime.now()
        request.rejection_reason = reason

        self._update_request(request)
        self._log_audit("approval_rejected", rejector, role, {
            'request_id': request_id,
            'reason': reason,
        })

        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
        if request_id in self.approval_callbacks:
            del self.approval_callbacks[request_id]

        print(f"\n{'='*50}")
        print(f"✗ REQUEST REJECTED")
        print(f"{'='*50}")
        print(f"  Request ID:  {request_id}")
        print(f"  Rejected by: {rejector}")
        print(f"  Reason:      {reason}")

        return True

    def _can_approve(self, role: Role, action_type: ActionType) -> bool:
        """Check if role can approve action type."""
        permissions = {
            Role.ADMIN: [a for a in ActionType],  # Can approve everything
            Role.RISK_MANAGER: [
                ActionType.LARGE_TRADE,
                ActionType.STRATEGY_ENABLE,
                ActionType.STRATEGY_DISABLE,
            ],
            Role.TRADER: [],  # Cannot approve
            Role.VIEWER: [],  # Cannot approve
        }
        return action_type in permissions.get(role, [])

    # =========================================================================
    # TRADE APPROVAL
    # =========================================================================

    def requires_trade_approval(self, size_usd: float, aum: float) -> bool:
        """Check if trade requires approval based on size."""
        trade_pct = (size_usd / aum) * 100 if aum > 0 else 100
        return trade_pct > self.config.large_trade_threshold_pct

    def request_trade_approval(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        aum: float,
        signals: List[str],
        requestor: str = "system",
        callback: Callable = None
    ) -> Optional[ApprovalRequest]:
        """
        Request approval for a large trade.

        Returns None if approval not required, otherwise returns the request.
        """
        if not self.requires_trade_approval(size_usd, aum):
            return None

        trade_pct = (size_usd / aum) * 100

        return self.request_approval(
            action_type=ActionType.LARGE_TRADE,
            description=f"{direction} {symbol} ${size_usd:,.0f} ({trade_pct:.1f}% of AUM)",
            requestor=requestor,
            data={
                'symbol': symbol,
                'direction': direction,
                'size_usd': size_usd,
                'aum': aum,
                'pct_of_aum': trade_pct,
                'signals': signals,
            },
            callback=callback,
        )

    # =========================================================================
    # OVERRIDE MODE
    # =========================================================================

    def emergency_override(
        self,
        action: str,
        reason: str,
        actor: str,
        role: Role = Role.ADMIN
    ) -> bool:
        """
        Execute an emergency override action.

        All overrides are logged and trigger alerts.
        """
        if role not in [Role.ADMIN]:
            print(f"  [GOV] Only ADMIN can execute emergency overrides")
            return False

        override_id = f"OVERRIDE-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._log_audit("emergency_override", actor, role, {
            'override_id': override_id,
            'action': action,
            'reason': reason,
        })

        print(f"\n{'!'*50}")
        print(f"⚠️  EMERGENCY OVERRIDE EXECUTED")
        print(f"{'!'*50}")
        print(f"  Override ID: {override_id}")
        print(f"  Action:      {action}")
        print(f"  Reason:      {reason}")
        print(f"  Actor:       {actor}")
        print(f"  Time:        {datetime.now()}")
        print(f"{'!'*50}")

        # TODO: Send notification alert

        return True

    # =========================================================================
    # AUDIT TRAIL
    # =========================================================================

    def _log_audit(
        self,
        action: str,
        actor: str,
        role: Role,
        details: Dict,
        success: bool = True
    ):
        """Log an audit entry."""
        entry_id = f"AUDIT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO audit_trail
            (entry_id, timestamp, action, actor, role, details, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            datetime.now().isoformat(),
            action,
            actor,
            role.value,
            json.dumps(details),
            1 if success else 0,
        ))

        conn.commit()
        conn.close()

    def get_audit_trail(
        self,
        actor: str = None,
        action: str = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """Get audit trail entries."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM audit_trail WHERE 1=1"
        params = []

        if actor:
            query += " AND actor = ?"
            params.append(actor)

        if action:
            query += " AND action = ?"
            params.append(action)

        since = (datetime.now() - timedelta(days=days)).isoformat()
        query += " AND timestamp >= ?"
        params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'entry_id': row['entry_id'],
                'timestamp': row['timestamp'],
                'action': row['action'],
                'actor': row['actor'],
                'role': row['role'],
                'details': json.loads(row['details']) if row['details'] else {},
                'success': bool(row['success']),
            })

        conn.close()
        return results

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    def _save_request(self, request: ApprovalRequest):
        """Save approval request to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO approval_requests
            (request_id, action_type, description, requestor, timestamp,
             data, status, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.request_id,
            request.action_type.value,
            request.description,
            request.requestor,
            request.timestamp.isoformat(),
            json.dumps(request.data),
            request.status.value,
            request.expires_at.isoformat() if request.expires_at else None,
        ))

        conn.commit()
        conn.close()

    def _update_request(self, request: ApprovalRequest):
        """Update approval request in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE approval_requests
            SET status = ?, approver = ?, approval_timestamp = ?,
                rejection_reason = ?
            WHERE request_id = ?
        """, (
            request.status.value,
            request.approver,
            request.approval_timestamp.isoformat() if request.approval_timestamp else None,
            request.rejection_reason,
            request.request_id,
        ))

        conn.commit()
        conn.close()

    def _load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load approval request from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM approval_requests WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return ApprovalRequest(
            request_id=row['request_id'],
            action_type=ActionType(row['action_type']),
            description=row['description'],
            requestor=row['requestor'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            data=json.loads(row['data']) if row['data'] else {},
            status=ApprovalStatus(row['status']),
            approver=row['approver'],
            approval_timestamp=datetime.fromisoformat(row['approval_timestamp']) if row['approval_timestamp'] else None,
            rejection_reason=row['rejection_reason'],
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
        )

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM approval_requests
            WHERE status = 'PENDING'
            ORDER BY timestamp DESC
        """)

        requests = []
        for row in cursor.fetchall():
            requests.append(ApprovalRequest(
                request_id=row['request_id'],
                action_type=ActionType(row['action_type']),
                description=row['description'],
                requestor=row['requestor'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                data=json.loads(row['data']) if row['data'] else {},
                status=ApprovalStatus(row['status']),
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
            ))

        conn.close()
        return requests

    def print_pending_requests(self):
        """Print all pending approval requests."""
        pending = self.get_pending_requests()

        print("\n" + "=" * 60)
        print("PENDING APPROVAL REQUESTS")
        print("=" * 60)

        if not pending:
            print("  No pending requests")
        else:
            for req in pending:
                time_left = ""
                if req.expires_at:
                    remaining = (req.expires_at - datetime.now()).total_seconds() / 3600
                    time_left = f" ({remaining:.1f}h remaining)"

                print(f"\n  [{req.request_id}]")
                print(f"    Type:        {req.action_type.value}")
                print(f"    Description: {req.description}")
                print(f"    Requestor:   {req.requestor}")
                print(f"    Time:        {req.timestamp}{time_left}")

        print("=" * 60)


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_governance():
    """Demonstrate governance functionality."""
    print("\n" + "=" * 70)
    print("GOVERNANCE SYSTEM DEMO")
    print("=" * 70)

    gov = GovernanceManager()

    # Request trade approval
    print("\n[1] Requesting large trade approval...")
    request = gov.request_trade_approval(
        symbol="BTCUSDT",
        direction="LONG",
        size_usd=600,
        aum=10000,
        signals=["RSI", "MACD", "Whale"],
        requestor="paper_trader",
    )

    if request:
        print(f"    Created request: {request.request_id}")

        # Approve the request
        print("\n[2] Approving request...")
        gov.approve(request.request_id, "admin_user", Role.ADMIN)

    # Show pending requests
    gov.print_pending_requests()

    # Show audit trail
    print("\n[3] Recent audit trail:")
    audit = gov.get_audit_trail(limit=5)
    for entry in audit:
        print(f"    {entry['timestamp'][:19]} | {entry['action']} | {entry['actor']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_governance()
