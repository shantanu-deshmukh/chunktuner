"""AgenticStrategy truncation metadata and offset clamp logging."""

import logging

import pytest

from chunktuner.chunking.agentic import AgenticStrategy
from chunktuner.models import ChunkConfig, Document


def test_agentic_stamps_metadata_when_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chunktuner.chunking.agentic.MAX_CHARS", 10)

    class _Msg:
        content = '{"chunks": [{"start_offset": 0, "end_offset": 10}]}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    import litellm

    monkeypatch.setattr(litellm, "completion", lambda *a, **k: _Resp())

    doc = Document(id="d1", content="0123456789abcdefghij", content_type="text")
    strat = AgenticStrategy()
    chunks = strat.chunk(doc, ChunkConfig(name="agentic", params={"max_propositions": 5}))

    assert len(chunks) >= 1
    assert all(c.metadata.get("agentic_truncated") is True for c in chunks)
    assert all(c.metadata.get("agentic_truncated_at") == 10 for c in chunks)


def test_agentic_no_truncation_metadata_when_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When doc fits under MAX_CHARS, do not set truncation flags."""

    class _Msg:
        content = '{"chunks": [{"start_offset": 0, "end_offset": 5}]}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    import litellm

    monkeypatch.setattr(litellm, "completion", lambda *a, **k: _Resp())

    doc = Document(id="d2", content="short", content_type="text")
    strat = AgenticStrategy()
    chunks = strat.chunk(doc, ChunkConfig(name="agentic", params={"max_propositions": 5}))

    assert len(chunks) >= 1
    assert all(c.metadata.get("agentic_truncated") is None for c in chunks)


def test_agentic_logs_when_offsets_clamped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    class _Msg:
        content = '{"chunks": [{"start_offset": 0, "end_offset": 999}]}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    import litellm

    monkeypatch.setattr(litellm, "completion", lambda *a, **k: _Resp())

    doc = Document(id="d3", content="hello world", content_type="text")
    strat = AgenticStrategy()
    chunks = strat.chunk(doc, ChunkConfig(name="agentic", params={"max_propositions": 5}))

    assert len(chunks) >= 1
    assert "clamped LLM offsets" in caplog.text
    assert "d3" in caplog.text
