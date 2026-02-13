"""Tests for human-in-the-loop approval workflow."""

from datetime import datetime
from uuid import UUID

import pytest

from analize.approval.workflow import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalWorkflow,
)
from analize.approval.patch_generator import PatchGenerator
from analize.models.reports import ParameterSuggestion, ConfidenceLevel
from analize.utils.time import utcnow


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert ApprovalStatus.PENDING
        assert ApprovalStatus.APPROVED
        assert ApprovalStatus.REJECTED
        assert ApprovalStatus.EXPIRED
        assert ApprovalStatus.APPLIED


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_creation(self) -> None:
        """Test ApprovalRequest creation."""
        request = ApprovalRequest(
            parameter_name="tp_pct",
            current_value=1.5,
            proposed_value=2.0,
            expected_impact={"win_rate_delta": 2.5},
        )
        assert request.status == ApprovalStatus.PENDING
        assert request.current_value == 1.5
        assert request.proposed_value == 2.0

    def test_add_audit_entry(self) -> None:
        """Test audit log entries."""
        request = ApprovalRequest()
        request.add_audit_entry(
            action="TEST_ACTION",
            actor="test_user",
            details={"key": "value"},
        )
        assert len(request.audit_log) == 1
        assert request.audit_log[0]["action"] == "TEST_ACTION"

    def test_to_dict(self) -> None:
        """Test serialization."""
        request = ApprovalRequest(
            parameter_name="tp_pct",
            current_value=1.5,
            proposed_value=2.0,
        )
        d = request.to_dict()
        assert d["parameter_name"] == "tp_pct"
        assert d["status"] == "PENDING"


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow class."""

    @pytest.fixture
    def workflow(self, tmp_path) -> ApprovalWorkflow:
        """Create ApprovalWorkflow instance."""
        return ApprovalWorkflow(storage_path=tmp_path)

    @pytest.fixture
    def sample_suggestion(self) -> ParameterSuggestion:
        """Create sample ParameterSuggestion."""
        return ParameterSuggestion(
            parameter_name="tp_pct",
            current_value=1.5,
            suggested_value=2.0,
            change_description="Increase take profit target to capture larger moves",
            expected_win_rate_delta=2.5,
            expected_pf_delta=0.3,
            expected_trade_count_delta=-10,
            confidence=ConfidenceLevel.MEDIUM,
            tradeoff="Fewer trades but higher quality",
            symbol="BTC/USDT",
        )

    def test_workflow_initialization(self, tmp_path) -> None:
        """Test workflow can be initialized."""
        workflow = ApprovalWorkflow(storage_path=tmp_path)
        assert workflow.storage_path == tmp_path

    def test_create_approval_request(
        self, workflow: ApprovalWorkflow, sample_suggestion: ParameterSuggestion
    ) -> None:
        """Test creating an approval request."""
        request = workflow.create_approval_request(
            suggestion=sample_suggestion,
            requested_by="test_optimizer",
        )

        assert request is not None
        assert request.status == ApprovalStatus.PENDING
        assert request.parameter_name == "tp_pct"
        assert len(request.audit_log) >= 1

    def test_approve_request(
        self, workflow: ApprovalWorkflow, sample_suggestion: ParameterSuggestion
    ) -> None:
        """Test approving a request."""
        request = workflow.create_approval_request(suggestion=sample_suggestion)

        approved = workflow.approve(
            request_id=request.request_id,
            approved_by="trader1",
            notes="Looks good after review",
        )

        assert approved.status == ApprovalStatus.APPROVED
        assert approved.approved_by == "trader1"
        assert approved.approved_at is not None

    def test_reject_request(
        self, workflow: ApprovalWorkflow, sample_suggestion: ParameterSuggestion
    ) -> None:
        """Test rejecting a request."""
        request = workflow.create_approval_request(suggestion=sample_suggestion)

        rejected = workflow.reject(
            request_id=request.request_id,
            rejected_by="risk_manager",
            reason="Not suitable for current market conditions",
        )

        assert rejected.status == ApprovalStatus.REJECTED

    def test_get_pending_requests(
        self, workflow: ApprovalWorkflow, sample_suggestion: ParameterSuggestion
    ) -> None:
        """Test getting pending requests."""
        workflow.create_approval_request(suggestion=sample_suggestion)
        pending = workflow.get_pending_requests()
        assert len(pending) == 1


class TestPatchGenerator:
    """Tests for PatchGenerator class."""

    @pytest.fixture
    def generator(self, tmp_path) -> PatchGenerator:
        """Create PatchGenerator instance."""
        return PatchGenerator(patches_dir=tmp_path)

    @pytest.fixture
    def sample_request(self) -> ApprovalRequest:
        """Create sample approved request."""
        return ApprovalRequest(
            parameter_name="tp_pct",
            current_value=1.5,
            proposed_value=2.0,
            expected_impact={
                "win_rate_delta": 2.5,
                "profit_factor_delta": 0.3,
                "trade_count_delta": -10,
                "tradeoff": "Fewer trades but higher quality",
            },
            target_symbol="BTC/USDT",
            status=ApprovalStatus.APPROVED,
            approved_by="trader1",
            approved_at=utcnow(),
        )

    def test_generate_patch(
        self, generator: PatchGenerator, sample_request: ApprovalRequest
    ) -> None:
        """Test git patch generation."""
        patch_content, patch_path = generator.generate_patch(sample_request)

        assert patch_path.exists()
        assert "tp_pct" in patch_content

    def test_generate_shell_script(
        self, generator: PatchGenerator, sample_request: ApprovalRequest
    ) -> None:
        """Test shell script generation."""
        script, script_path = generator.generate_shell_script(sample_request)

        assert script_path.exists()
        assert "#!/bin/bash" in script

    def test_get_application_instructions(
        self, generator: PatchGenerator, sample_request: ApprovalRequest
    ) -> None:
        """Test application instructions generation."""
        instructions = generator.get_application_instructions(sample_request)
        assert "tp_pct" in instructions
