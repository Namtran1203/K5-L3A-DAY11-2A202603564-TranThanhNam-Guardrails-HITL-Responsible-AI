"""Exercise the Blue runner and its artifacts without making provider calls."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.parametrize("local", [False, True])
def test_blue_pipeline_exports_observed_decisions(monkeypatch, tmp_path, local):
    from assignment import pipeline
    from core.config import BLUE_LOCAL_BASE_URL, BLUE_LOCAL_MODEL, BLUE_MODEL
    from core.openai_runtime import OpenAIRunner

    email = "fixture@vinbank.example"
    model_calls = []

    def complete(**kwargs):
        model_calls.append(kwargs)
        reply = f"Check your balance in the app. Email: {email}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    def build_client(runner):
        if local:
            assert runner.client_kwargs == {"base_url": BLUE_LOCAL_BASE_URL, "api_key": "local"}
        return client

    monkeypatch.setattr(OpenAIRunner, "_client", build_client)
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path)
    audit, monitor = pipeline.build_observability()
    result = asyncio.run(pipeline.run_assignment_suite({
        "plugins": pipeline.build_production_plugins(),
        "audit": audit,
        "monitor": monitor,
    }, local=local))

    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(saved, schema)
    assert saved == result
    assert result["llm_provider"] == ("llama.cpp" if local else "openrouter")
    assert result["llm_model"] == (BLUE_LOCAL_MODEL if local else BLUE_MODEL)
    assert all(not row["blocked"] for row in result["safe_queries"])
    assert all(row["layer"] == "output_guardrail" for row in result["safe_queries"])
    assert all(row["blocked"] for row in result["attack_queries"])
    assert result["rate_limit"] == {
        "max_requests": 10, "window_seconds": 60, "sent": 15, "passed": 10, "blocked": 5
    }
    assert len(model_calls) == 7
    assert len(audit.logs) == monitor.total_requests == 31
    assert monitor.rate_limit_hits == 5
    assert email not in (tmp_path / "audit_log.json").read_text(encoding="utf-8")
    assert email not in json.dumps(saved)


def test_egress_does_not_accept_host_lookalikes_or_sensitive_payloads():
    from assignment.pipeline import is_egress_allowed

    trusted = "https://api.vinbank.example/v1/transfers"
    ordinary_payload = "transfer amount 500000 VND"
    assert is_egress_allowed(trusted, ordinary_payload)
    assert not is_egress_allowed(trusted, "contact: fixture@vinbank.example")
    assert not is_egress_allowed(trusted, "contact: 0901234567")
    for destination in (
        "https://api.vinbank.example.evil.example/v1/transfers",
        "https://user@api.vinbank.example/v1/transfers",
        "https://api.vinbank.example:invalid/v1/transfers",
        "http://api.vinbank.example/v1/transfers",
    ):
        assert not is_egress_allowed(destination, ordinary_payload)


def test_rate_limit_tracks_users_and_expires_exactly_at_window_end(monkeypatch):
    from assignment import rate_limiter

    clock = iter([0.0, 1.0, 2.0, 3.0, 10.0])
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    limiter = rate_limiter.RateLimitPlugin(max_requests=1, window_seconds=10)

    async def run():
        for user, blocked in [
            ("alice", False), ("alice", True), ("bob", False), ("alice", True), ("alice", False)
        ]:
            context = SimpleNamespace(user_id=user)
            result = await limiter.on_user_message_callback(invocation_context=context, user_message=None)
            assert (result is not None) is blocked

    asyncio.run(run())
    assert limiter.total_count == 5
    assert limiter.blocked_count == 2
