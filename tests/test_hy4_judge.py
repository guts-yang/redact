"""Unit tests for the Hy4 AFE/CFE judge record construction. No network calls."""
from scripts import hy4_judge_afe_rtofu as afe
from scripts import hy4_judge_cfe_rtofu as cfe


def test_afe_judge_ok_record(monkeypatch):
    monkeypatch.setattr(
        afe, "complete_with_retries",
        lambda *a, **k: ('{"label": "LEAK", "leak_span": "Alice"}', None, 1),
    )

    rec = afe.judge_one("T__C__q0", "q", "gold", "gen", "hy4", 2, 10.0)

    assert rec["status"] == "ok"
    assert rec["label"] == "LEAK"
    assert rec["leak_span"] == "Alice"
    assert rec["model"] == "hy4"
    assert rec["attempts"] == 1


def test_afe_judge_forces_empty_span_for_non_leak(monkeypatch):
    monkeypatch.setattr(
        afe, "complete_with_retries",
        lambda *a, **k: ('{"label": "REFUSE", "leak_span": "should be dropped"}', None, 1),
    )

    rec = afe.judge_one("T__C__q0", "q", "gold", "gen", "hy4", 2, 10.0)

    assert rec["status"] == "ok"
    assert rec["label"] == "REFUSE"
    assert rec["leak_span"] == ""


def test_afe_judge_parse_failure(monkeypatch):
    monkeypatch.setattr(afe, "complete_with_retries", lambda *a, **k: ("not json", None, 2))

    rec = afe.judge_one("T__C__q0", "q", "gold", "gen", "hy4", 2, 10.0)

    assert rec["status"] == "parse_failed"
    assert rec["label"] is None
    assert rec["error"] == "parse_failed"


def test_afe_judge_transport_error(monkeypatch):
    monkeypatch.setattr(afe, "complete_with_retries", lambda *a, **k: (None, "boom", 3))

    rec = afe.judge_one("T__C__q0", "q", "gold", "gen", "hy4", 2, 10.0)

    assert rec["status"] == "error"
    assert rec["error"] == "boom"


def test_cfe_judge_ok_record(monkeypatch):
    monkeypatch.setattr(cfe, "complete_with_retries", lambda *a, **k: ("0.42", None, 1))

    rec = cfe.judge_one("T__C__q0", "q", "gold", "cot", "hy4", 2, 10.0)

    assert rec["status"] == "ok"
    assert rec["score"] == 0.42
    assert rec["model"] == "hy4"


def test_cfe_judge_rejects_out_of_range_score(monkeypatch):
    monkeypatch.setattr(cfe, "complete_with_retries", lambda *a, **k: ("1.42", None, 1))

    rec = cfe.judge_one("T__C__q0", "q", "gold", "cot", "hy4", 2, 10.0)

    assert rec["status"] == "parse_failed"
    assert rec["score"] is None


def test_cfe_judge_transport_error(monkeypatch):
    monkeypatch.setattr(cfe, "complete_with_retries", lambda *a, **k: (None, "timeout", 3))

    rec = cfe.judge_one("T__C__q0", "q", "gold", "cot", "hy4", 2, 10.0)

    assert rec["status"] == "error"
    assert rec["error"] == "timeout"
