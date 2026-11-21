"""In-process MCP session tests via ``mcp.shared.memory`` (mcp >= 1.7)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests._extras import require_extra

require_extra("mcp", "mcp.shared.memory", reason="mcp with in-memory transport required")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from chunktuner.mcp.server import create_mcp_app  # noqa: E402


def _tool_payload(result: CallToolResult) -> Any:
    """Decode tool output.

    FastMCP may split structured results across several ``TextContent`` blocks.
    """
    assert result.content, "tool returned empty content"
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            raise AssertionError(f"unexpected content block type: {type(block)}")
    joined = "".join(parts).strip()
    if not joined:
        return None
    if not (joined.startswith("{") or joined.startswith("[")):
        return joined
    decoder = json.JSONDecoder()
    values: list[Any] = []
    idx = 0
    while idx < len(joined):
        while idx < len(joined) and joined[idx].isspace():
            idx += 1
        if idx >= len(joined):
            break
        obj, end = decoder.raw_decode(joined, idx)
        values.append(obj)
        idx = end
    if len(values) == 1:
        return values[0]
    return values


def test_mcp_session_lists_chunktuner_tools() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            assert names >= {
                "list_strategies",
                "preview_chunks",
                "evaluate_chunking",
                "recommend_config",
            }

    asyncio.run(body())


def test_mcp_session_list_strategies() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            raw = await session.call_tool("list_strategies", {"content_type": None})
            assert raw.isError is False
            rows = _tool_payload(raw)
            assert isinstance(rows, list)
            names = {r["name"] for r in rows}
            assert "fixed_tokens" in names
            assert "recursive_character" in names

    asyncio.run(body())


def test_mcp_session_preview_chunks() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            raw = await session.call_tool(
                "preview_chunks",
                {
                    "text": "alpha beta " * 20,
                    "strategy_name": "fixed_tokens",
                    "config": {"max_tokens": 16, "overlap_tokens": 0},
                },
            )
            assert raw.isError is False
            chunks = _tool_payload(raw)
            assert isinstance(chunks, list) and chunks
            assert "start_offset" in chunks[0]

    asyncio.run(body())


def test_mcp_session_recommend_rejects_path_outside_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.md"
    f.write_text("# Hi\n")
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(corpus))

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "x.md"
    outside.write_text("nope")

    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            ok = await session.call_tool(
                "recommend_config",
                {
                    "path": str(f),
                    "use_case": "rag_qa",
                    "strategies": ["fixed_tokens"],
                    "max_docs": 1,
                },
            )
            assert ok.isError is False

            bad = await session.call_tool(
                "recommend_config",
                {
                    "path": str(outside),
                    "use_case": "rag_qa",
                    "strategies": ["fixed_tokens"],
                    "max_docs": 1,
                },
            )
            assert bad.isError is True

    asyncio.run(body())


def test_mcp_session_evaluate_with_new_optional_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunktuner.mcp import service

    base = tmp_path / "base"
    corpus = base / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("Alpha sentence about apples. " * 40)
    (corpus / "b.txt").write_text("Beta sentence about rivers. " * 40)
    ds = base / "ds.json"
    ds.write_text(
        json.dumps(
            {
                "queries": [
                    {"question": "apples?", "document_id": "a.txt", "answer_spans": [[0, 40]]}
                ]
            }
        )
    )
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(base))
    monkeypatch.setattr(service, "mcp_cache_dir", lambda: tmp_path / "cache")

    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            listed = await session.list_tools()
            ev_tool = next(t for t in listed.tools if t.name == "evaluate_chunking")
            props = ev_tool.inputSchema["properties"]
            assert {"dataset_path", "generate_dataset", "retrieval_scope", "use_cache"} <= set(
                props
            )
            assert "does NOT affect metrics" in props["top_k"]["description"]
            preview_tool = next(t for t in listed.tools if t.name == "preview_chunks")
            assert "content_type" in preview_tool.inputSchema["properties"]

            raw = await session.call_tool(
                "evaluate_chunking",
                {
                    "path": str(corpus),
                    "strategies": ["fixed_tokens"],
                    "content_type": "text",
                    "dataset_path": str(ds),
                    "retrieval_scope": "corpus",
                    "use_cache": False,
                },
            )
            assert raw.isError is False
            out = _tool_payload(raw)
            assert out["dataset_summary"] == {"queries": 1, "source": "user_provided"}
            assert isinstance(out["warnings"], list)
            assert not any("trivial" in w for w in out["warnings"])
            assert out["results"]

            default = _tool_payload(
                await session.call_tool(
                    "evaluate_chunking", {"path": str(corpus), "strategies": ["fixed_tokens"]}
                )
            )
            assert any("trivial" in w for w in default["warnings"])

            bad = await session.call_tool(
                "evaluate_chunking",
                {"path": str(corpus), "retrieval_scope": "global"},
            )
            assert bad.isError is True

    asyncio.run(body())


def test_mcp_session_retrieval_mode_and_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunktuner.mcp import service

    require_extra("bm25", "bm25s")
    base = tmp_path / "base"
    corpus = base / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("Alpha sentence about apples. " * 40)
    (corpus / "b.txt").write_text("Beta sentence about rivers. " * 40)
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(base))
    monkeypatch.setattr(service, "mcp_cache_dir", lambda: tmp_path / "cache")

    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            listed = await session.list_tools()
            for name in ("evaluate_chunking", "recommend_config"):
                tool = next(t for t in listed.tools if t.name == name)
                props = tool.inputSchema["properties"]
                assert {"retrieval_mode", "reranker"} <= set(props)
                assert "retrieval_mode" not in tool.inputSchema.get("required", [])
                assert "reranker" not in tool.inputSchema.get("required", [])
                assert props["retrieval_mode"]["default"] == "dense"
                assert set(props["retrieval_mode"]["enum"]) == {"dense", "bm25", "hybrid"}
                assert "'bm25' extra" in props["retrieval_mode"]["description"]
                assert "local:<hf-model-id>" in props["reranker"]["description"]
                assert "may cost money" in props["reranker"]["description"]

            raw = await session.call_tool(
                "evaluate_chunking",
                {
                    "path": str(corpus),
                    "strategies": ["fixed_tokens"],
                    "retrieval_mode": "bm25",
                    "reranker": "dummy",
                    "use_cache": False,
                },
            )
            assert raw.isError is False
            out = _tool_payload(raw)
            assert out["results"]
            for r in out["results"]:
                assert r["metrics"]["retrieval_mode"] == "bm25"
                assert r["metrics"]["reranker"] == "dummy/rerank"

            bad_mode = await session.call_tool(
                "evaluate_chunking",
                {"path": str(corpus), "retrieval_mode": "sparse"},
            )
            assert bad_mode.isError is True

            bad_reranker = await session.call_tool(
                "evaluate_chunking",
                {"path": str(corpus), "strategies": ["fixed_tokens"], "reranker": "local:"},
            )
            assert bad_reranker.isError is True
            assert "local:" in "".join(
                b.text for b in bad_reranker.content if isinstance(b, TextContent)
            )

            outside = await session.call_tool(
                "recommend_config",
                {"path": str(tmp_path), "retrieval_mode": "hybrid", "reranker": "dummy"},
            )
            assert outside.isError is True

    asyncio.run(body())
