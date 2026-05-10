"""Application configuration, tokenizer profile, and score weight defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Default tiktoken encoding (GPT-4 / cl100k family)
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_BUDGET_TOKENS = 2000

# Rough throughput for wall-time estimates (tokens per minute)
THROUGHPUT_LOCAL_TOKENS_PER_MIN = 500_000
THROUGHPUT_API_TOKENS_PER_MIN = 200_000


class WorkspaceConfig(BaseModel):
    """Shape of `.autochunk.yaml` (user workspace)."""

    version: int = 1
    embedding_model: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL
    api_base: str | None = None
    api_key: str | None = None
    use_case: str = "rag_qa"
    max_docs: int = 100
    max_tokens_per_run: int = 250_000
    top_k: int = DEFAULT_TOP_K
    cache_dir: str = "~/.cache/chunktuner"
    log_level: str = "INFO"
    strip_patterns: list[str] = Field(default_factory=list)
    tokenizer_encoding: str = DEFAULT_TOKENIZER_ENCODING


def load_workspace_config(path: Path | None) -> WorkspaceConfig:
    if path is None or not path.is_file():
        return WorkspaceConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    return WorkspaceConfig.model_validate(raw)


def resolve_provider_config(ws: WorkspaceConfig) -> tuple[str | None, str | None]:
    """Return ``(api_base, api_key)`` from workspace config with env var fallback.

    Priority: workspace fields ``api_base`` / ``api_key``, then ``CHUNKTUNER_API_BASE`` /
    ``CHUNKTUNER_API_KEY`` environment variables.
    """
    api_base = ws.api_base or os.environ.get("CHUNKTUNER_API_BASE") or None
    api_key = ws.api_key or os.environ.get("CHUNKTUNER_API_KEY") or None
    return api_base, api_key


def default_cache_dir() -> Path:
    override = os.environ.get("CHUNKTUNER_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "chunktuner"


def score_profile_weights(use_case: str) -> dict[str, float]:
    """Default metric weights for each use-case profile.

    All keys must correspond to EvalMetrics fields that are already on a
    0–1 scale (token_recall, mrr, token_iou, recall_at_k, faithfulness,
    answer_relevancy, duplication_ratio).  Raw token-count fields such as
    avg_tokens_per_query, avg_chunk_length, and chunk_length_std are
    available on EvalMetrics for inspection but are not included here.
    """
    profiles: dict[str, dict[str, float]] = {
        "rag_qa": {
            "token_recall": 0.45,
            "mrr": 0.30,
            "token_iou": 0.15,
            "faithfulness": 0.10,
            "duplication_ratio": -0.10,
        },
        "search": {
            "recall_at_1": 0.50,
            "mrr": 0.35,
            "duplication_ratio": -0.15,
        },
        "summarization": {
            "token_recall": 0.60,
            "token_iou": 0.20,
            "duplication_ratio": -0.20,
        },
        "code_assist": {
            "token_recall": 0.50,
            "mrr": 0.35,
            "duplication_ratio": -0.15,
        },
    }
    return dict(profiles.get(use_case, profiles["rag_qa"]))


def default_init_yaml() -> dict[str, Any]:
    return {
        "version": 1,
        # embedding_model: null → DummyEmbeddingFunction (free). Set to any LiteLLM model id,
        # e.g. text-embedding-3-small (OpenAI), gemini/gemini-embedding-001 (Google),
        # or openai/<id> for local servers (LM Studio, Ollama, vLLM).
        "embedding_model": None,
        # llm_model: used for agentic chunking and generation metrics only.
        # Alternatives: claude-3-haiku-20240307 (Anthropic), gemini/gemini-2.0-flash (Google),
        # openai/<id> for local servers.
        "llm_model": DEFAULT_LLM_MODEL,
        "api_base": None,
        "use_case": "rag_qa",
        "max_docs": 100,
        "max_tokens_per_run": 250_000,
        "top_k": DEFAULT_TOP_K,
        "cache_dir": str(Path.home() / ".cache" / "chunktuner"),
        "log_level": "INFO",
        "strip_patterns": [],
    }
