"""RAGAS Faithfulness + Answer Relevancy (optional ``chunktuner[ragas]`` + compatible stack)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# RAGAS 0.3.x uses OpenAI by default for LLM-as-judge. Other providers require
# custom LangChain wrappers and manual ragas_bridge configuration.
_RAGAS_LLM_ENV_VARS = ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")

_UNAVAILABLE: dict[str, float | None] = {
    "faithfulness": None,
    "answer_relevancy": None,
}


class RagasBridge:
    """Thin adapter around RAGAS ``faithfulness`` and ``answer_relevancy`` (optional extra)."""

    def __init__(self, llm_client: object | None = None):
        self.llm_client = llm_client

    @staticmethod
    def is_configured() -> bool:
        """Return True if an API key RAGAS/LiteLLM typically reads is set.

        Includes ``CHUNKTUNER_API_KEY`` for the same OpenAI-compatible routing as the
        rest of chunktuner. RAGAS still does not support local-only (LM Studio / Ollama)
        judge flows without extra LangChain wiring — a key here means provider-style
        auth, not ``api_base`` alone.
        """
        return any(os.environ.get(k) for k in _RAGAS_LLM_ENV_VARS)

    def compute(
        self,
        questions: list[str],
        contexts: list[list[str]],
        answers: list[str],
        ground_truths: list[str],
    ) -> dict[str, float | None]:
        """Return averaged metrics, or sentinel Nones if RAGAS cannot run."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness
        except ImportError as exc:  # pragma: no cover
            logger.warning("RAGAS not installed (%s). Install: uv add 'chunktuner[ragas]'", exc)
            return dict(_UNAVAILABLE)
        except Exception as exc:  # pragma: no cover
            logger.warning("RAGAS import failed (%s); skipping generation metrics", exc)
            return dict(_UNAVAILABLE)

        try:
            rows = []
            for q, ctx, ans, gt in zip(questions, contexts, answers, ground_truths, strict=True):
                rows.append(
                    {
                        "user_input": q,
                        "retrieved_contexts": ctx,
                        "response": ans,
                        "reference": gt,
                    }
                )
            ds = Dataset.from_list(rows)
            result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
            df = result.to_pandas()
            out: dict[str, float | None] = dict(_UNAVAILABLE)
            if "faithfulness" in df.columns:
                out["faithfulness"] = float(df["faithfulness"].mean())
            if "answer_relevancy" in df.columns:
                out["answer_relevancy"] = float(df["answer_relevancy"].mean())
            return out
        except Exception as exc:  # pragma: no cover
            logger.warning("RAGAS computation failed: %s", exc)
            return dict(_UNAVAILABLE)
