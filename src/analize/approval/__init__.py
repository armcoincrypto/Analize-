"""
Human-in-the-loop approval workflow for parameter changes.

Ensures no automatic deployment of strategy changes by enforcing:
- Suggestion review and approval process
- Git patch generation for manual application
- Audit logging of all approvals
- Configurable approval requirements
"""

from analize.approval.workflow import (
    ApprovalWorkflow,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalDecision,
)
from analize.approval.patch_generator import PatchGenerator

__all__ = [
    "ApprovalWorkflow",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalDecision",
    "PatchGenerator",
]
