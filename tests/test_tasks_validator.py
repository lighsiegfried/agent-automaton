"""Deterministic plan validation (Phase 5B, item 3): unknown intents, bad args,
unknown URLs, cycles, dependency and step-count limits, classification."""

# Importing the task package registers every domain service's intent schemas, so
# validate_service_args can check arguments.
import app.tasks  # noqa: F401
from app.core import errors
from app.tasks import models, validator


def _v(steps, max_steps=12):
    return validator.validate_plan(steps, task_id="task_x", max_steps=max_steps)


def test_empty_plan_rejected():
    assert _v([]).error_code == errors.EMPTY_PLAN


def test_too_many_steps_rejected():
    steps = [{"intent": "memory_search", "arguments": {"query": "x"}}] * 13
    assert _v(steps, max_steps=12).error_code == errors.PLAN_TOO_LARGE


def test_unknown_intent_rejected():
    result = _v([{"intent": "rm_rf", "arguments": {}}])
    assert result.error_code == errors.UNSUPPORTED_STEP
    assert result.blocked[0].intent == "rm_rf"


def test_disallowed_supported_but_unlisted_intent():
    # memory_propose exists as a service intent but is NOT a supported task step.
    assert _v([{"intent": "memory_propose", "arguments": {"content": "x"}}]).error_code \
        == errors.UNSUPPORTED_STEP
    # form fill / purchases / deletion / shell are simply not in the registry.
    for intent in ("browser_prepare_form", "email_confirm_send", "shell_exec"):
        assert _v([{"intent": intent, "arguments": {}}]).error_code == errors.UNSUPPORTED_STEP


def test_bad_arguments_rejected():
    # Unexpected argument keys (e.g. a shell/python payload) fail the schema check.
    assert _v([{"intent": "memory_search", "arguments": {"cmd": "rm -rf /"}}]).error_code \
        == errors.INVALID_STEP_ARGS
    assert _v([{"intent": "browser_read", "arguments": {"selector": "//div"}}]).error_code \
        == errors.INVALID_STEP_ARGS


def test_unknown_url_scheme_rejected():
    for url in ("file:///etc/passwd", "javascript:alert(1)", "data:text/html,x"):
        assert _v([{"intent": "browser_open", "arguments": {"url": url}}]).error_code \
            == errors.INVALID_STEP_ARGS
    # A normal https URL is fine.
    assert _v([{"intent": "browser_open", "arguments": {"url": "https://example.com"}}]).ok


def test_self_and_out_of_range_dependencies_rejected():
    assert _v([{"intent": "browser_read", "arguments": {}, "dependencies": [0]}]).error_code \
        == errors.INVALID_DEPENDENCY
    assert _v([{"intent": "browser_read", "arguments": {}, "dependencies": [5]}]).error_code \
        == errors.INVALID_DEPENDENCY


def test_cyclic_plan_rejected():
    steps = [
        {"intent": "browser_read", "arguments": {}, "dependencies": [1]},
        {"intent": "browser_summarize", "arguments": {}, "dependencies": [0]},
    ]
    assert _v(steps).error_code == errors.PLAN_HAS_CYCLE


def test_valid_plan_classified():
    steps = [
        {"intent": "memory_search", "arguments": {"query": "vs code"}},
        {"intent": "browser_search", "arguments": {"query": "playwright"}},
        {"intent": "browser_summarize", "arguments": {}, "dependencies": [1]},
        {"intent": "email_draft_new", "arguments": {"to": ["ana@x.com"], "body": "hi"}},
        {"intent": "email_prepare_send", "arguments": {}, "dependencies": [3]},
    ]
    result = _v(steps)
    assert result.ok
    kinds = [(s.intent, s.risk_level) for s in result.steps]
    assert kinds == [
        ("memory_search", models.READ_ONLY),
        ("browser_search", models.READ_ONLY),
        ("browser_summarize", models.READ_ONLY),
        ("email_draft_new", models.LOCAL),
        ("email_prepare_send", models.EFFECT),
    ]
    # The send step carries a recipient-specific expected confirmation phrase.
    send = result.steps[-1]
    assert send.requires_confirmation and "envío" in send.confirmation_phrase.lower()


def test_one_bad_step_blocks_whole_plan():
    steps = [
        {"intent": "memory_search", "arguments": {"query": "x"}},
        {"intent": "delete_everything", "arguments": {}},
    ]
    result = _v(steps)
    assert not result.ok and result.error_code == errors.UNSUPPORTED_STEP
    assert len(result.blocked) == 1
