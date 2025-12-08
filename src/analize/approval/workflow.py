"""
Approval workflow for parameter change suggestions.

Enforces human review before any strategy changes are applied.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from analize.config import get_settings
from analize.models.reports import ParameterSuggestion


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    APPLIED = "APPLIED"


class ApprovalDecision(str, Enum):
    """Decision made on an approval request."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CHANGES = "REQUEST_CHANGES"


@dataclass
class ApprovalRequest:
    """
    Request for approval of parameter changes.

    This is the core unit of the approval workflow - every suggestion
    must go through this approval process before it can be applied.
    """

    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Suggestion details
    suggestion_id: UUID | None = None
    suggestion: ParameterSuggestion | None = None

    # What is being changed
    parameter_name: str = ""
    current_value: Any = None
    proposed_value: Any = None
    target_symbol: str | None = None

    # Expected impact
    expected_impact: dict[str, Any] = field(default_factory=dict)
    backtest_results: dict[str, Any] = field(default_factory=dict)

    # Approval metadata
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: str = "system"  # Who/what created this request
    approved_by: str | None = None
    approved_at: datetime | None = None

    # Review details
    decision: ApprovalDecision | None = None
    review_notes: str | None = None

    # Expiration
    expires_at: datetime | None = None

    # Git patch for applying changes
    patch_content: str | None = None
    patch_file_path: str | None = None

    # Audit trail
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def add_audit_entry(self, action: str, actor: str, details: dict[str, Any] | None = None) -> None:
        """Add an entry to the audit log."""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "actor": actor,
            "details": details or {},
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": str(self.request_id),
            "created_at": self.created_at.isoformat(),
            "suggestion_id": str(self.suggestion_id) if self.suggestion_id else None,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "target_symbol": self.target_symbol,
            "expected_impact": self.expected_impact,
            "backtest_results": self.backtest_results,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "decision": self.decision.value if self.decision else None,
            "review_notes": self.review_notes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "patch_content": self.patch_content,
            "patch_file_path": self.patch_file_path,
            "audit_log": self.audit_log,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequest":
        """Create from dictionary."""
        req = cls(
            request_id=UUID(data["request_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            parameter_name=data.get("parameter_name", ""),
            current_value=data.get("current_value"),
            proposed_value=data.get("proposed_value"),
            target_symbol=data.get("target_symbol"),
            expected_impact=data.get("expected_impact", {}),
            backtest_results=data.get("backtest_results", {}),
            status=ApprovalStatus(data.get("status", "PENDING")),
            requested_by=data.get("requested_by", "system"),
            approved_by=data.get("approved_by"),
            review_notes=data.get("review_notes"),
            patch_content=data.get("patch_content"),
            patch_file_path=data.get("patch_file_path"),
            audit_log=data.get("audit_log", []),
        )

        if data.get("suggestion_id"):
            req.suggestion_id = UUID(data["suggestion_id"])
        if data.get("approved_at"):
            req.approved_at = datetime.fromisoformat(data["approved_at"])
        if data.get("expires_at"):
            req.expires_at = datetime.fromisoformat(data["expires_at"])
        if data.get("decision"):
            req.decision = ApprovalDecision(data["decision"])

        return req


class ApprovalWorkflow:
    """
    Manages the approval workflow for parameter changes.

    CRITICAL: This class enforces that no changes are auto-applied.
    All changes must go through human review.
    """

    def __init__(self, storage_path: Path | None = None):
        settings = get_settings()
        self.storage_path = storage_path or (settings.storage.local_data_path / "approvals")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache of pending requests
        self._requests: dict[UUID, ApprovalRequest] = {}
        self._load_pending_requests()

    def _load_pending_requests(self) -> None:
        """Load pending requests from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                req = ApprovalRequest.from_dict(data)
                if req.status == ApprovalStatus.PENDING:
                    self._requests[req.request_id] = req
            except Exception:
                continue

    def _save_request(self, request: ApprovalRequest) -> None:
        """Save request to storage."""
        file_path = self.storage_path / f"{request.request_id}.json"
        with open(file_path, "w") as f:
            json.dump(request.to_dict(), f, indent=2, default=str)

    def create_approval_request(
        self,
        suggestion: ParameterSuggestion,
        backtest_results: dict[str, Any] | None = None,
        requested_by: str = "optimizer",
        expiration_hours: int = 168,  # 7 days default
    ) -> ApprovalRequest:
        """
        Create a new approval request for a parameter suggestion.

        Args:
            suggestion: The parameter suggestion to approve
            backtest_results: Results from backtesting the suggestion
            requested_by: Who/what created this request
            expiration_hours: Hours until request expires

        Returns:
            ApprovalRequest object
        """
        from datetime import timedelta

        request = ApprovalRequest(
            suggestion_id=suggestion.suggestion_id,
            suggestion=suggestion,
            parameter_name=suggestion.parameter_name,
            current_value=suggestion.current_value,
            proposed_value=suggestion.suggested_value,
            target_symbol=suggestion.symbol,
            expected_impact={
                "win_rate_delta": suggestion.expected_win_rate_delta,
                "profit_factor_delta": suggestion.expected_pf_delta,
                "trade_count_delta": suggestion.expected_trade_count_delta,
                "confidence": suggestion.confidence.value if suggestion.confidence else None,
                "tradeoff": suggestion.tradeoff,
            },
            backtest_results=backtest_results or {},
            requested_by=requested_by,
            expires_at=datetime.utcnow() + timedelta(hours=expiration_hours),
        )

        request.add_audit_entry(
            action="CREATED",
            actor=requested_by,
            details={"suggestion_id": str(suggestion.suggestion_id)},
        )

        self._requests[request.request_id] = request
        self._save_request(request)

        return request

    def get_pending_requests(self, symbol: str | None = None) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        # Check for expired requests
        now = datetime.utcnow()
        for req in list(self._requests.values()):
            if req.expires_at and req.expires_at < now:
                req.status = ApprovalStatus.EXPIRED
                req.add_audit_entry("EXPIRED", "system")
                self._save_request(req)

        pending = [
            req for req in self._requests.values()
            if req.status == ApprovalStatus.PENDING
        ]

        if symbol:
            pending = [req for req in pending if req.target_symbol == symbol]

        return sorted(pending, key=lambda x: x.created_at, reverse=True)

    def get_request(self, request_id: UUID) -> ApprovalRequest | None:
        """Get a specific approval request."""
        if request_id in self._requests:
            return self._requests[request_id]

        # Try loading from storage
        file_path = self.storage_path / f"{request_id}.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            return ApprovalRequest.from_dict(data)

        return None

    def approve(
        self,
        request_id: UUID,
        approved_by: str,
        notes: str | None = None,
    ) -> ApprovalRequest:
        """
        Approve a request.

        NOTE: Approving does NOT automatically apply the change.
        It only marks the request as approved and generates a patch.
        """
        request = self.get_request(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request is not pending (status: {request.status})")

        request.status = ApprovalStatus.APPROVED
        request.decision = ApprovalDecision.APPROVE
        request.approved_by = approved_by
        request.approved_at = datetime.utcnow()
        request.review_notes = notes

        request.add_audit_entry(
            action="APPROVED",
            actor=approved_by,
            details={"notes": notes},
        )

        # Generate patch
        from analize.approval.patch_generator import PatchGenerator
        generator = PatchGenerator()
        patch_content, patch_path = generator.generate_patch(request)
        request.patch_content = patch_content
        request.patch_file_path = str(patch_path) if patch_path else None

        self._save_request(request)

        return request

    def reject(
        self,
        request_id: UUID,
        rejected_by: str,
        reason: str,
    ) -> ApprovalRequest:
        """Reject a request."""
        request = self.get_request(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request is not pending (status: {request.status})")

        request.status = ApprovalStatus.REJECTED
        request.decision = ApprovalDecision.REJECT
        request.approved_by = rejected_by
        request.approved_at = datetime.utcnow()
        request.review_notes = reason

        request.add_audit_entry(
            action="REJECTED",
            actor=rejected_by,
            details={"reason": reason},
        )

        self._save_request(request)

        return request

    def mark_applied(
        self,
        request_id: UUID,
        applied_by: str,
        commit_hash: str | None = None,
    ) -> ApprovalRequest:
        """
        Mark an approved request as applied.

        This should be called AFTER manual application of the patch.
        """
        request = self.get_request(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        if request.status != ApprovalStatus.APPROVED:
            raise ValueError(f"Request must be approved before marking as applied")

        request.status = ApprovalStatus.APPLIED

        request.add_audit_entry(
            action="APPLIED",
            actor=applied_by,
            details={"commit_hash": commit_hash},
        )

        self._save_request(request)

        return request

    def get_audit_trail(self, request_id: UUID) -> list[dict[str, Any]]:
        """Get the full audit trail for a request."""
        request = self.get_request(request_id)
        if not request:
            return []
        return request.audit_log

    def get_statistics(self) -> dict[str, Any]:
        """Get approval workflow statistics."""
        all_requests = []
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                all_requests.append(ApprovalRequest.from_dict(data))
            except Exception:
                continue

        total = len(all_requests)
        by_status = {}
        for status in ApprovalStatus:
            by_status[status.value] = sum(1 for r in all_requests if r.status == status)

        # Calculate approval rate
        decided = by_status.get("APPROVED", 0) + by_status.get("REJECTED", 0)
        approval_rate = by_status.get("APPROVED", 0) / decided * 100 if decided > 0 else 0

        # Average time to decision
        decision_times = []
        for req in all_requests:
            if req.approved_at and req.created_at:
                delta = (req.approved_at - req.created_at).total_seconds() / 3600  # hours
                decision_times.append(delta)

        avg_decision_time = sum(decision_times) / len(decision_times) if decision_times else 0

        return {
            "total_requests": total,
            "by_status": by_status,
            "approval_rate_pct": round(approval_rate, 1),
            "avg_decision_time_hours": round(avg_decision_time, 1),
            "pending_count": by_status.get("PENDING", 0),
        }
