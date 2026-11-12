"""OpenAI-compatible chat client for the DeepSeek chat completions endpoint.

The API key is read from ``DEEPSEEK_API_KEY``, or from the repo-root
``.env`` when that variable is unset. It is never logged and never
written by this module. Override the URL with ``DEEPSEEK_API_BASE``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API_BASE = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
SYSTEM_PROMPT = "You are a helpful assistant."
EXTRA_BODY = {
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
}


class DeepSeekError(RuntimeError):
    """Transport or response-shape failure. ``rate_limited`` drives backoff."""

    def __init__(self, message: str, *, status: int | None = None, rate_limited: bool = False):
        super().__init__(message)
        self.status = status
        self.rate_limited = rate_limited


def load_repo_env(path: Path | None = None) -> None:
    """Fill missing environment variables from the repo-root ``.env``.

    Existing variables win. Values are never logged. Set ``REDACT_DOTENV=0``
    to skip the file (used by tests that assert a missing key).
    """
    if os.environ.get("REDACT_DOTENV") == "0":
        return
    env_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku"}


def prepare_deepseek_argv(*, default_output: str | None = None) -> None:
    """Rewrite ``sys.argv`` so a DeepSeek entry point can parse a Claude command.

    Drops ``--claude_bin``. Rewrites Claude model aliases to ``deepseek-flash``.
    Inserts ``default_output`` when ``--output_jsonl`` was not given.
    """
    import sys

    argv = sys.argv[1:]
    out: list[str] = []
    saw_output = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--claude_bin":
            i += 2
            continue
        if arg.startswith("--claude_bin="):
            i += 1
            continue
        if arg == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            if model in _CLAUDE_MODEL_ALIASES or model.startswith("claude"):
                model = DEFAULT_MODEL
            out.extend(["--model", model])
            i += 2
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            if model in _CLAUDE_MODEL_ALIASES or model.startswith("claude"):
                model = DEFAULT_MODEL
            out.append(f"--model={model}")
            i += 1
            continue
        if arg == "--output_jsonl" or arg.startswith("--output_jsonl="):
            saw_output = True
        out.append(arg)
        i += 1
    if default_output and not saw_output:
        out.extend(["--output_jsonl", default_output])
    sys.argv = [sys.argv[0], *out]


def require_api_key() -> str:
    """Return ``DEEPSEEK_API_KEY`` from the environment or the repo ``.env``."""
    load_repo_env()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not set")
    return key


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


def extract_message_content(payload: object) -> str:
    """Pull ``choices[0].message.content``. ``reasoning_content`` is ignored."""
    if not isinstance(payload, dict):
        raise DeepSeekError("response JSON is not an object")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekError("response missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekError("response message content is empty")
    return content


def complete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 180.0,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.0,
) -> str:
    """POST one non-streaming chat completion and return the assistant text.

    A missing key raises ``DeepSeekError`` and does not open a connection.
    """
    key = require_api_key() if api_key is None else api_key.strip()
    if not key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not set")
    base = (api_base or os.environ.get("DEEPSEEK_API_BASE") or DEFAULT_API_BASE).strip()
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": temperature,
            **EXTRA_BODY,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        snippet = exc.read().decode("utf-8", errors="replace")[:500]
        lowered = snippet.lower()
        rate_limited = exc.code == 429 or any(
            token in lowered for token in ("rate", "quota", "too many", "overloaded")
        )
        raise DeepSeekError(
            _redact(f"HTTP {exc.code}: {snippet}", key),
            status=exc.code,
            rate_limited=rate_limited,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(exc, TimeoutError) or "timed out" in str(reason).lower()
        message = f"timeout after {timeout}s" if timed_out else f"request failed: {reason}"
        raise DeepSeekError(_redact(message, key)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DeepSeekError("response was not JSON") from exc
    return extract_message_content(payload)


def complete_with_retries(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 180.0,
    max_retries: int = 2,
) -> tuple[str | None, str | None, int]:
    """Call ``complete`` up to ``max_retries + 1`` times.

    Returns ``(text, error, attempts)``. ``text`` is None when every
    attempt failed. ``error`` never includes the API key.
    """
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            return complete(prompt, model=model, timeout=timeout), None, attempt + 1
        except DeepSeekError as exc:
            last_err = str(exc)
            wait = 30 if exc.rate_limited else min(2 ** attempt, 30)
            if attempt < max_retries:
                time.sleep(wait)
    return None, last_err, max_retries + 1
