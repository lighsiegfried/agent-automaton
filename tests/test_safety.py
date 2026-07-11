from app.core.safety import evaluate
from app.schemas.commands import ExecutionStatus, SafetyLevel


def test_safe_is_allowed_without_confirmation(settings):
    settings.require_confirmation = True
    decision = evaluate(SafetyLevel.SAFE, confirmed=False)
    assert decision.allowed


def test_sensitive_needs_confirmation(settings):
    settings.require_confirmation = True
    decision = evaluate(SafetyLevel.SENSITIVE, confirmed=False)
    assert not decision.allowed
    assert decision.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_sensitive_allowed_when_confirmed(settings):
    settings.require_confirmation = True
    decision = evaluate(SafetyLevel.SENSITIVE, confirmed=True)
    assert decision.allowed


def test_sensitive_allowed_when_confirmation_disabled(settings):
    settings.require_confirmation = False
    decision = evaluate(SafetyLevel.SENSITIVE, confirmed=False)
    assert decision.allowed


def test_destructive_blocked_even_with_confirmation(settings):
    decision = evaluate(SafetyLevel.DESTRUCTIVE, confirmed=True)
    assert not decision.allowed
    assert decision.status is ExecutionStatus.BLOCKED
