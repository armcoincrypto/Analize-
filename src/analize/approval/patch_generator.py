"""
Git patch generator for approved parameter changes.

Generates patches that can be manually reviewed and applied
to the ScalperBot repository.
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from analize.approval.workflow import ApprovalRequest
from analize.config import get_settings
from analize.utils.time import utcnow, utcnow_iso


class PatchGenerator:
    """Generates git patches for parameter changes."""

    def __init__(self, patches_dir: Path | None = None):
        settings = get_settings()
        self.patches_dir = patches_dir or (settings.storage.local_data_path / "patches")
        self.patches_dir.mkdir(parents=True, exist_ok=True)

    def generate_patch(
        self,
        request: ApprovalRequest,
        config_file: str = "config/strategy.json",
    ) -> tuple[str, Path]:
        """
        Generate a git patch for a parameter change.

        Args:
            request: Approved approval request
            config_file: Path to the config file to modify

        Returns:
            Tuple of (patch_content, patch_file_path)
        """
        # Generate patch content
        patch_content = self._generate_patch_content(request, config_file)

        # Save patch file
        patch_filename = f"patch_{request.request_id}_{utcnow().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = self.patches_dir / patch_filename

        with open(patch_path, "w") as f:
            f.write(patch_content)

        # Also generate an info file with metadata
        info_path = self.patches_dir / f"{patch_filename}.info.json"
        with open(info_path, "w") as f:
            json.dump({
                "request_id": str(request.request_id),
                "parameter": request.parameter_name,
                "current_value": request.current_value,
                "proposed_value": request.proposed_value,
                "symbol": request.target_symbol,
                "expected_impact": request.expected_impact,
                "approved_by": request.approved_by,
                "approved_at": request.approved_at.isoformat() if request.approved_at else None,
                "generated_at": utcnow_iso(),
            }, f, indent=2, default=str)

        return patch_content, patch_path

    def _generate_patch_content(
        self,
        request: ApprovalRequest,
        config_file: str,
    ) -> str:
        """Generate the actual patch content."""
        now = utcnow().strftime("%Y-%m-%d %H:%M:%S +0000")

        # Generate header
        header = f"""From: Analize <analize@system>
Date: {now}
Subject: [PATCH] Update {request.parameter_name} parameter

Analize Parameter Change Suggestion

Request ID: {request.request_id}
Parameter: {request.parameter_name}
Symbol: {request.target_symbol or 'ALL'}

Change:
  Current: {request.current_value}
  Proposed: {request.proposed_value}

Expected Impact:
  Win Rate Delta: {request.expected_impact.get('win_rate_delta', 'N/A')}%
  Profit Factor Delta: {request.expected_impact.get('profit_factor_delta', 'N/A')}
  Trade Count Delta: {request.expected_impact.get('trade_count_delta', 'N/A')}
  Tradeoff: {request.expected_impact.get('tradeoff', 'N/A')}

Approved by: {request.approved_by}
Review Notes: {request.review_notes or 'None'}

---
"""

        # Generate diff
        # This is a simplified example - in production you'd generate actual diffs
        # based on the real config file format

        if request.target_symbol:
            old_value = f'"{request.parameter_name}": {json.dumps(request.current_value)}'
            new_value = f'"{request.parameter_name}": {json.dumps(request.proposed_value)}'
            context_key = f'"symbols": {{\n  "{request.target_symbol}":'
        else:
            old_value = f'"{request.parameter_name}": {json.dumps(request.current_value)}'
            new_value = f'"{request.parameter_name}": {json.dumps(request.proposed_value)}'
            context_key = '"global_parameters":'

        diff = f""" {config_file} | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/{config_file} b/{config_file}
index 1234567..abcdefg 100644
--- a/{config_file}
+++ b/{config_file}
@@ -10,7 +10,7 @@
   {context_key}
     ...
-    {old_value},
+    {new_value},
     ...
   }}
--
2.39.0
"""

        return header + diff

    def generate_env_patch(
        self,
        request: ApprovalRequest,
        env_var_mapping: dict[str, str] | None = None,
    ) -> tuple[str, Path]:
        """
        Generate a .env file patch for parameter changes.

        Args:
            request: Approved approval request
            env_var_mapping: Optional mapping of parameter names to env var names

        Returns:
            Tuple of (patch_content, patch_file_path)
        """
        mapping = env_var_mapping or {}

        # Convert parameter name to env var format
        param_upper = request.parameter_name.upper().replace(".", "_")
        env_var = mapping.get(request.parameter_name, f"STRATEGY_{param_upper}")

        if request.target_symbol:
            env_var = f"{env_var}_{request.target_symbol}"

        now = utcnow().strftime("%Y-%m-%d %H:%M:%S +0000")

        patch_content = f"""From: Analize <analize@system>
Date: {now}
Subject: [PATCH] Update {env_var} environment variable

Request ID: {request.request_id}
---
 .env | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -XX,Y +XX,Y @@
-{env_var}={request.current_value}
+{env_var}={request.proposed_value}
--
2.39.0
"""

        patch_filename = f"env_patch_{request.request_id}_{utcnow().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = self.patches_dir / patch_filename

        with open(patch_path, "w") as f:
            f.write(patch_content)

        return patch_content, patch_path

    def generate_shell_script(
        self,
        request: ApprovalRequest,
    ) -> tuple[str, Path]:
        """
        Generate a shell script for applying the change.

        This provides an alternative to git patches for simpler changes.
        """
        script = f"""#!/bin/bash
# Analize Parameter Change Script
# Request ID: {request.request_id}
# Generated: {utcnow_iso()}
#
# IMPORTANT: Review this script before running!
# This script was auto-generated by Analize.
#
# Parameter: {request.parameter_name}
# Symbol: {request.target_symbol or 'ALL'}
# Change: {request.current_value} -> {request.proposed_value}
#
# Approved by: {request.approved_by}
# Approved at: {request.approved_at}

set -e

echo "=== Analize Parameter Update ==="
echo "Request ID: {request.request_id}"
echo "Parameter: {request.parameter_name}"
echo "Change: {request.current_value} -> {request.proposed_value}"
echo ""

# Confirm before proceeding
read -p "Apply this change? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Backup current config
CONFIG_FILE="${{CONFIG_FILE:-config/strategy.json}}"
BACKUP_FILE="${{CONFIG_FILE}}.backup.$(date +%Y%m%d_%H%M%S)"

echo "Creating backup: $BACKUP_FILE"
cp "$CONFIG_FILE" "$BACKUP_FILE"

# Apply change using jq (assumes JSON config)
echo "Applying change..."
"""

        if request.target_symbol:
            script += f"""
jq '.symbols."{request.target_symbol}".{request.parameter_name} = {json.dumps(request.proposed_value)}' "$CONFIG_FILE" > "$CONFIG_FILE.tmp"
mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
"""
        else:
            script += f"""
jq '.global_parameters.{request.parameter_name} = {json.dumps(request.proposed_value)}' "$CONFIG_FILE" > "$CONFIG_FILE.tmp"
mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
"""

        script += f"""
echo "Change applied successfully!"
echo ""
echo "To revert, run:"
echo "  cp $BACKUP_FILE $CONFIG_FILE"
echo ""
echo "Don't forget to restart the bot to apply the new configuration."
"""

        script_filename = f"apply_{request.request_id}_{utcnow().strftime('%Y%m%d_%H%M%S')}.sh"
        script_path = self.patches_dir / script_filename

        with open(script_path, "w") as f:
            f.write(script)

        # Make executable
        script_path.chmod(0o755)

        return script, script_path

    def get_application_instructions(self, request: ApprovalRequest) -> str:
        """Generate human-readable instructions for applying a change."""
        instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ANALIZE PARAMETER CHANGE INSTRUCTIONS                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Request ID: {request.request_id}
Status: APPROVED ✓

┌─────────────────────────────────────────────────────────────────────────────┐
│ CHANGE DETAILS                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Parameter:      {request.parameter_name:<55} │
│ Symbol:         {(request.target_symbol or 'ALL'):<55} │
│ Current Value:  {str(request.current_value):<55} │
│ New Value:      {str(request.proposed_value):<55} │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPECTED IMPACT                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Win Rate Delta:       {str(request.expected_impact.get('win_rate_delta', 'N/A')) + '%':<46} │
│ Profit Factor Delta:  {str(request.expected_impact.get('profit_factor_delta', 'N/A')):<46} │
│ Trade Count Delta:    {str(request.expected_impact.get('trade_count_delta', 'N/A')):<46} │
│ Tradeoff:             {str(request.expected_impact.get('tradeoff', 'N/A')):<46} │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HOW TO APPLY                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Option 1: Apply the git patch                                                │
│   cd /path/to/scalperbot                                                     │
│   git apply {request.patch_file_path or 'patch_file.patch'}
│                                                                              │
│ Option 2: Run the shell script                                               │
│   chmod +x apply_script.sh                                                   │
│   ./apply_script.sh                                                          │
│                                                                              │
│ Option 3: Manual edit                                                        │
│   1. Open your strategy config file                                          │
│   2. Find the {request.parameter_name} parameter
│   3. Change {request.current_value} to {request.proposed_value}
│   4. Save and restart the bot                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ AFTER APPLYING                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ 1. Restart ScalperBot to pick up the new configuration                       │
│ 2. Monitor performance for the next 24-48 hours                              │
│ 3. Mark this request as APPLIED in Analize:                                  │
│                                                                              │
│    curl -X POST http://analize:8000/approvals/{request.request_id}/applied \\
│      -H "Content-Type: application/json" \\
│      -d '{{"applied_by": "your_name", "commit_hash": "abc123"}}'
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Approved by: {request.approved_by}
Review Notes: {request.review_notes or 'None'}

⚠️  IMPORTANT: Do not auto-apply. Always review changes manually before applying.
"""
        return instructions
