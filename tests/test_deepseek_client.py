"""Unit tests for the DeepSeek chat client. No network."""
import json
from unittest.mock import patch

import pytest

from scripts.deepseek_client import (
    DeepSeekError,
    complete,
    prepare_deepseek_argv,
    require_api_key,
)


class TestPrepareArgv:
    def test_rewrites_claude_flags(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--model", "sonnet", "--claude_bin", "claude", "--split", "forget10"],
        )
        prepare_deepseek_argv(
            default_output="leak_detector_data_promptfmt/forget10/first_leak_labels.jsonl",
        )
        import sys
        assert sys.argv == [
            "prog",
            "--model",
            "deepseek-flash",
            "--split",
            "forget10",
            "--output_jsonl",
            "leak_detector_data_promptfmt/forget10/first_leak_labels.jsonl",
        ]


class TestRequireKey:
    def test_missing_key_does_not_open_a_connection(self, monkeypatch):
        monkeypatch.setenv("REDACT_DOTENV", "0")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with patch("scripts.deepseek_client.urllib.request.urlopen") as urlopen:
            with pytest.raises(DeepSeekError, match="DEEPSEEK_API_KEY"):
                require_api_key()
            with pytest.raises(DeepSeekError, match="DEEPSEEK_API_KEY"):
                complete("ping")
        urlopen.assert_not_called()


class TestComplete:
    def test_posts_deepseek_body_and_returns_content_only(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-logged")
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "final answer",
                            "reasoning_content": "hidden trace",
                        }
                    }
                ]
            }
        ).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        captured = {}

        def _urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode())
            return _Resp()

        with patch("scripts.deepseek_client.urllib.request.urlopen", _urlopen):
            text = complete("ping")

        assert text == "final answer"
        assert text != "hidden trace"
        assert captured["url"] == "https://api.deepseek.com/chat/completions"
        assert captured["body"]["model"] == "deepseek-flash"
        assert captured["body"]["stream"] is False
        assert captured["body"]["temperature"] == 0
        assert captured["body"]["thinking"] == {"type": "enabled"}
        assert captured["body"]["reasoning_effort"] == "high"
        assert captured["body"]["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "ping"},
        ]
        assert captured["auth"] == "Bearer test-key-not-logged"
        assert "test-key-not-logged" not in text

    def test_empty_content_is_an_error(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-logged")
        body = json.dumps(
            {"choices": [{"message": {"content": "  ", "reasoning_content": "trace"}}]}
        ).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch("scripts.deepseek_client.urllib.request.urlopen", return_value=_Resp()):
            with pytest.raises(DeepSeekError, match="empty"):
                complete("ping")
