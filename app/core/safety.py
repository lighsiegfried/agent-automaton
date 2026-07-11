"""Safety layer: decides whether a tool may run.

Policy (see docs/SAFETY_RULES.md):
- SAFE actions run (or are simulated) without confirmation.
- SENSITIVE actions require confirm=true when REQUIRE_CONFIRMATION is enabled.
- DESTRUCTIVE actions are always blocked; enabling them later requires a
  deliberate code change, never just a flag or a confirmation click.
"""

from dataclasses import dataclass

from app.config import get_settings
from app.schemas.commands import ExecutionStatus, SafetyLevel


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    status: ExecutionStatus
    reason: str


def evaluate(level: SafetyLevel, confirmed: bool) -> SafetyDecision:
    """Return whether an action at the given safety level may proceed."""
    if level is SafetyLevel.DESTRUCTIVE:
        return SafetyDecision(
            allowed=False,
            status=ExecutionStatus.BLOCKED,
            reason="Destructive actions are blocked by default.",
        )

    if level is SafetyLevel.SENSITIVE:
        if get_settings().require_confirmation and not confirmed:
            return SafetyDecision(
                allowed=False,
                status=ExecutionStatus.NEEDS_CONFIRMATION,
                reason="Sensitive action: re-send the command with confirm=true to proceed.",
            )
        return SafetyDecision(
            allowed=True,
            status=ExecutionStatus.SIMULATED,
            reason="Sensitive action confirmed.",
        )

    return SafetyDecision(
        allowed=True,
        status=ExecutionStatus.SIMULATED,
        reason="Safe action.",
    )
