"""OpenAI-compatible chat client for the Hy4 endpoint.

Hy4 is this project's judge and reviewer transport (see
``unlearning-reproduction/references/reviewer-override.md``). The token is read
from ``Hy4_TOKEN`` / ``HY4_TOKEN`` or from ``$HY4_ENV_FILE`` (default
``/usr/local/workspace/.env``). It is never logged, never included in errors,
and never written to an artifact.

The endpoint may be given either as the full ``.../chat/completions`` URL or as
a base ending in ``/v1``; both are normalized to the full URL.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
DEFAULT_MODEL = "hy4"
DEFAULT_ENV_FILE = Path("/usr/local/workspace/.env")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

TOKEN_KEYS = ("Hy4_TOKEN", "HY4_TOKEN")
ENDPOINT_KEYS = ("HY4_ENDPOINT", "LLM_BASE_URL")
MODEL_KEYS = ("HY4_MODEL", "LLM_MODEL")


class Hy4Error(RuntimeError):
    """Transport or response-shape failure. ``rate_limited`` drives backoff."""

    def __init__(self, message: str, *, status: int | None = None,
                 rate_limited: bool = False):
        super().__init__(message)
        self.status = status
        self.rate_limited = rate_limited


def _read_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = Path(path or os.environ.get("HY4_ENV_FILE") or DEFAULT_ENV_FILE)
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
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
        if key:
            values[key] = value
    return values


def _first(values: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = (values.get(key) or "").strip()
        if value:
            return value
    return ""


def load_config(path: Path | None = None) -> dict[str, str]:
    """Resolve ``{"token", "endpoint", "model"}``; process env wins over file."""
    merged = _read_env_file(path)
    merged.update({k: v for k, v in os.environ.items() if v})
    token = _first(merged, TOKEN_KEYS)
    if not token:
        raise Hy4Error(
            f"Hy4 token not set: export Hy4_TOKEN or add it to "
            f"{path or os.environ.get('HY4_ENV_FILE') or DEFAULT_ENV_FILE}"
        )
    endpoint = normalize_endpoint(_first(merged, ENDPOINT_KEYS) or DEFAULT_ENDPOINT)
    model = _first(merged, MODEL_KEYS) or DEFAULT_MODEL
    return {"token": token, "endpoint": endpoint, "model": model}


def normalize_endpoint(endpoint: str) -> str:
    """Accept a full chat-completions URL or a ``/v1`` base."""
    url = endpoint.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def require_token() -> str:
    return load_config()["token"]


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "[redacted]") if secret and secret in text else text


def extract_message_content(payload: object) -> str:
    """Pull ``choices[0].message.content``. ``reasoning_content`` is ignored."""
    if not isinstance(payload, dict):
        raise Hy4Error("response JSON is not an object")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Hy4Error("response missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise Hy4Error("response message content is empty")
    return content


def response_model(payload: object, fallback: str) -> str:
    """Return the identity-bearing model name reported by the endpoint."""
    if isinstance(payload, dict):
        name = payload.get("model")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return fallback


def _post(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 180.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    thinking: bool = True,
    reasoning_effort: str = "high",
) -> tuple[dict, str, str]:
    """POST one non-streaming completion; return ``(payload, model, token)``."""
    cfg = load_config()
    key = cfg["token"]
    model = model or cfg["model"]
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": temperature,
    }
    if thinking:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = reasoning_effort

    request = urllib.request.Request(
        cfg["endpoint"],
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
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
        raise Hy4Error(
            _redact(f"HTTP {exc.code}: {snippet}", key),
            status=exc.code,
            rate_limited=rate_limited,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(exc, TimeoutError) or "timed out" in str(reason).lower()
        message = f"timeout after {timeout}s" if timed_out else f"request failed: {reason}"
        raise Hy4Error(_redact(message, key)) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Hy4Error("response was not JSON") from exc
    return payload, model, key


def complete(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 180.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    thinking: bool = True,
    reasoning_effort: str = "high",
) -> str:
    """POST one non-streaming chat completion and return the assistant text."""
    payload, _, _ = _post(
        prompt, model=model, timeout=timeout, system_prompt=system_prompt,
        temperature=temperature, thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    return extract_message_content(payload)


def complete_with_identity(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 180.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    thinking: bool = True,
    reasoning_effort: str = "high",
) -> tuple[str, str]:
    """Like :func:`complete`, but also return the endpoint-reported model name."""
    payload, fallback, _ = _post(
        prompt, model=model, timeout=timeout, system_prompt=system_prompt,
        temperature=temperature, thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    return extract_message_content(payload), response_model(payload, fallback)


def complete_with_retries(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 180.0,
    max_retries: int = 2,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str | None, str | None, int]:
    """Call :func:`complete` up to ``max_retries + 1`` times.

    Returns ``(text, error, attempts)``. ``text`` is None when every attempt
    failed. ``error`` never includes the token.
    """
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            text = complete(
                prompt, model=model, timeout=timeout, system_prompt=system_prompt,
            )
            return text, None, attempt + 1
        except Hy4Error as exc:
            last_err = str(exc)
            if attempt < max_retries:
                time.sleep(30 if exc.rate_limited else min(2 ** attempt, 30))
    return None, last_err, max_retries + 1
