"""Unit tests for the Hy4 transport. No network calls, no real token."""
import pytest

from scripts import hy4_client


_ENV_KEYS = (
    "Hy4_TOKEN", "HY4_TOKEN", "HY4_ENDPOINT", "HY4_MODEL",
    "LLM_BASE_URL", "LLM_MODEL", "HY4_ENV_FILE",
)


@pytest.fixture(autouse=True)
def _clean_hy4_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_config_from_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("Hy4_TOKEN='tok-123'\nHY4_MODEL=hy4-test\n", encoding="utf-8")
    monkeypatch.setenv("HY4_ENV_FILE", str(env_file))

    cfg = hy4_client.load_config()

    assert cfg["token"] == "tok-123"
    assert cfg["model"] == "hy4-test"
    assert cfg["endpoint"] == hy4_client.DEFAULT_ENDPOINT


def test_process_env_wins_over_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("Hy4_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("HY4_ENV_FILE", str(env_file))
    monkeypatch.setenv("Hy4_TOKEN", "from-env")

    assert hy4_client.load_config()["token"] == "from-env"


def test_missing_token_raises(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    monkeypatch.setenv("HY4_ENV_FILE", str(env_file))

    with pytest.raises(hy4_client.Hy4Error, match="token not set"):
        hy4_client.load_config()


@pytest.mark.parametrize("endpoint,expected", [
    ("https://host/openapi/v1", "https://host/openapi/v1/chat/completions"),
    ("https://host/openapi/v1/", "https://host/openapi/v1/chat/completions"),
    ("https://host/openapi/v1/chat/completions",
     "https://host/openapi/v1/chat/completions"),
])
def test_normalize_endpoint(endpoint, expected):
    assert hy4_client.normalize_endpoint(endpoint) == expected


def test_extract_message_content_ignores_reasoning():
    payload = {
        "model": "hy4",
        "choices": [{"message": {"content": "answer", "reasoning_content": "cot"}}],
    }
    assert hy4_client.extract_message_content(payload) == "answer"
    assert hy4_client.response_model(payload, "fallback") == "hy4"


def test_extract_message_content_rejects_empty():
    payload = {"choices": [{"message": {"content": "   "}}]}
    with pytest.raises(hy4_client.Hy4Error, match="empty"):
        hy4_client.extract_message_content(payload)


def test_complete_with_retries_success(monkeypatch):
    monkeypatch.setattr(hy4_client, "complete", lambda *a, **k: "ok")
    assert hy4_client.complete_with_retries("p", max_retries=2) == ("ok", None, 1)


def test_complete_with_retries_exhausts_and_redacts(monkeypatch):
    monotonic = {"n": 0}

    def boom(*args, **kwargs):
        monotonic["n"] += 1
        raise hy4_client.Hy4Error("boom")

    monkeypatch.setattr(hy4_client, "complete", boom)
    monkeypatch.setattr(hy4_client.time, "sleep", lambda _s: None)

    text, err, attempts = hy4_client.complete_with_retries("p", max_retries=2)

    assert text is None
    assert err == "boom"
    assert attempts == 3
    assert monotonic["n"] == 3


def test_redact_hides_secret():
    assert hy4_client._redact("token abc leaked", "abc") == "token [redacted] leaked"
    assert hy4_client._redact("nothing", "abc") == "nothing"
