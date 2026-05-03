"""FastAPI routes mirroring MCP tool contracts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chunktuner.mcp.service import (
    DEFAULT_MAX_PREVIEW_CHARS,
    evaluate_chunking_impl,
    list_strategies_impl,
    preview_chunks_impl,
    recommend_config_impl,
)

router = APIRouter()


class PreviewChunksBody(BaseModel):
    text: str = Field(..., max_length=DEFAULT_MAX_PREVIEW_CHARS)
    strategy_name: str
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluateBody(BaseModel):
    path: str
    use_case: str = "rag_qa"
    content_type: str | None = None
    strategies: list[str] | None = None
    max_docs: int = 20
    top_k: int = 5
    dry_run: bool = False
    embedding_model: str | None = None


class RecommendConfigBody(BaseModel):
    path: str
    use_case: str = "rag_qa"
    content_type: str | None = None
    strategies: list[str] | None = None
    max_docs: int = 20
    top_k: int = 5
    embedding_model: str | None = None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/list_strategies")
def list_strategies(content_type: str | None = None) -> list[dict]:
    return list_strategies_impl(content_type)


@router.post("/preview_chunks")
def preview_chunks(body: PreviewChunksBody) -> list[dict]:
    try:
        return preview_chunks_impl(body.text, body.strategy_name, body.config)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/evaluate_chunking")
def evaluate_chunking(body: EvaluateBody) -> dict:
    try:
        return evaluate_chunking_impl(
            body.path,
            body.use_case,
            content_type=body.content_type,
            strategies=body.strategies,
            max_docs=body.max_docs,
            top_k=body.top_k,
            dry_run=body.dry_run,
            embedding_model=body.embedding_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/recommend_config")
def recommend_config(body: RecommendConfigBody) -> dict:
    try:
        return recommend_config_impl(
            body.path,
            body.use_case,
            content_type=body.content_type,
            strategies=body.strategies,
            max_docs=body.max_docs,
            top_k=body.top_k,
            embedding_model=body.embedding_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
