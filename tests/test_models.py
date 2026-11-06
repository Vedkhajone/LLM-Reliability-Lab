import pytest

from reliability_lab.models import ModelSpec, RetryableError, build_client, load_specs
from reliability_lab.models.base import ModelClient, is_retryable
from reliability_lab.models.mock_client import MockClient
from reliability_lab.schema import ModelResponse
from reliability_lab.utils.cost_tracker import BudgetExceeded, CostTracker, estimate_cost


class Flaky(ModelClient):
    def __init__(self, spec, fail_times, exc, **kw):
        super().__init__(spec, **kw)
        self.fail_times, self.exc, self.calls = fail_times, exc, 0

    def _call(self, prompt, system, temperature, max_tokens, seed):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return ModelResponse(text="ok", input_tokens=3, output_tokens=2)


def spec(**kw):
    d = dict(key="t", provider="mock", model_id="m", max_retries=3, requests_per_minute=0)
    d.update(kw)
    return ModelSpec(**d)


def test_retry_then_success():
    sleeps = []
    c = Flaky(spec(), 2, RetryableError("429"), sleep=sleeps.append)
    r = c.generate("hi")
    assert r.text == "ok" and r.attempts == 3 and c.calls == 3 and len(sleeps) == 2


def test_gives_up_after_max_retries_and_records_error():
    c = Flaky(spec(max_retries=2), 99, RetryableError("503"), sleep=lambda s: None)
    r = c.generate("hi")
    assert r.error and "RetryableError" in r.error and c.calls == 3 and r.text == ""


def test_non_retryable_fails_fast():
    c = Flaky(spec(), 99, ValueError("bad request"), sleep=lambda s: None)
    r = c.generate("hi")
    assert r.error and c.calls == 1


def test_is_retryable_by_status_and_name():
    class E(Exception):
        status_code = 429
    class Timeout(Exception):
        pass
    assert is_retryable(E()) and is_retryable(Timeout()) and not is_retryable(KeyError("x"))


def test_rate_limit_spaces_requests():
    t = {"now": 0.0}
    slept = []
    def clock(): return t["now"]
    def sleep(s): slept.append(s); t["now"] += s
    c = MockClient(spec(requests_per_minute=60), sleep=sleep, clock=clock)
    c.generate("a"); c.generate("b")
    assert slept and slept[0] == pytest.approx(1.0)


def test_mock_records_calls_and_usage():
    c = build_client(spec())
    r = c.generate("Context:\nParis is in France. It is nice.\n\nQuestion: where?")
    assert r.input_tokens and r.output_tokens and r.latency_s is not None and c.calls[0]["temperature"] == 0.0


def test_cost_estimation():
    assert estimate_cost(1_000_000, 500_000, 3.0, 15.0) == pytest.approx(10.5)
    assert estimate_cost(100, 100, None, 1.0) is None


def test_budget_enforced():
    t = CostTracker(max_requests=2)
    t.record("m", 1, 1, None); t.record("m", 1, 1, None)
    with pytest.raises(BudgetExceeded):
        t.check_before_request()
    t2 = CostTracker(max_cost_usd=0.01)
    t2.record("m", 1, 1, 0.02)
    with pytest.raises(BudgetExceeded):
        t2.check_before_request()


def test_models_yaml_loads_and_has_no_secrets():
    specs = load_specs()
    assert {"claude", "openai", "gemini", "open_weight", "mock"} <= set(specs)
    assert specs["mock"].provider == "mock" and not specs["mock"].enabled
    for s in specs.values():
        assert not (s.api_key_env or "").startswith(("sk-", "AIza"))


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pytest.importorskip("anthropic")
    with pytest.raises(RuntimeError):
        build_client(spec(provider="anthropic", api_key_env="ANTHROPIC_API_KEY"))


def _anthropic_client(monkeypatch):
    pytest.importorskip("anthropic")
    from reliability_lab.models.anthropic_client import AnthropicClient
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return AnthropicClient(spec(provider="anthropic", api_key_env="ANTHROPIC_API_KEY", max_retries=0))


def test_anthropic_adapter_omits_unsupported_temperature_and_reports_it(monkeypatch):
    from types import SimpleNamespace
    c = _anthropic_client(monkeypatch)
    seen = {}

    def fake_create(kwargs):
        seen.clear(); seen.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="hi")], usage=SimpleNamespace(input_tokens=4, output_tokens=2),
                               stop_reason="end_turn")
    monkeypatch.setattr(c, "_create", fake_create)
    c._send_temperature = False                         # as with an SDK that has no `temperature`
    r = c.generate("x", temperature=0.7)
    assert "temperature" not in seen and r.params_applied["temperature"] is None and r.text == "hi"
    c._send_temperature = True
    r = c.generate("x", temperature=0.7)
    assert seen["temperature"] == 0.7 and r.params_applied["temperature"] == 0.7


def test_anthropic_adapter_retries_without_rejected_temperature(monkeypatch):
    from types import SimpleNamespace
    c = _anthropic_client(monkeypatch)
    c._send_temperature = True
    calls = []

    def fake_create(kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise ValueError("temperature is not supported for this model")
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], usage=None, stop_reason="end_turn")
    monkeypatch.setattr(c, "_create", fake_create)
    r = c.generate("x", temperature=0.3)
    assert r.text == "ok" and len(calls) == 2 and "temperature" not in calls[1] and r.params_applied["temperature"] is None
    assert c._send_temperature is False
