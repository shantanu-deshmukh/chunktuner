"""Build a deterministic behaviour snapshot used as the regression baseline.

Covers chunk boundaries for every registered strategy x default param grid, retrieval
metrics/scores with ``DummyEmbeddingFunction``, default tuner rankings, and the trivial
dataset. ``agentic`` is excluded because it calls an LLM.
"""

from pathlib import Path
from typing import Any

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import ChunkConfig, Document
from chunktuner.tuner.auto_tuner import AutoTuner

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "golden"
EXCLUDED_STRATEGIES = {"agentic"}
USE_CASES = ["rag_qa", "search", "summarization", "code_assist"]
# Volatile metric fields (timing) are not part of the snapshot.
VOLATILE_METRICS = {"embedding_latency_ms"}


def load_docs() -> list[Document]:
    ingestor = FileIngestor(root=FIXTURES)
    docs = ingestor.ingest_dir(FIXTURES)
    # FileIngestor assigns random uuid4 ids; use filenames so the snapshot is stable.
    docs = [d.model_copy(update={"id": d.metadata["filename"]}) for d in docs]
    return sorted(docs, key=lambda d: d.id)


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    data = metrics.model_dump(mode="json")
    return {k: v for k, v in data.items() if k not in VOLATILE_METRICS}


def build_snapshot() -> dict[str, Any]:
    reg = build_full_registry()
    docs = load_docs()
    strategy_names = sorted(n for n in reg.names() if n not in EXCLUDED_STRATEGIES)
    snap: dict[str, Any] = {
        "registered_strategies": sorted(reg.names()),
        "docs": {d.id: {"content_type": d.content_type, "length": len(d.content)} for d in docs},
        "chunks": {},
        "eval": {},
        "recommend": {},
        "trivial_dataset": trivial_dataset_for_docs(docs).model_dump(mode="json"),
    }

    # Chunk boundaries per (strategy, params, doc).
    for name in strategy_names:
        strat = reg.get(name)
        for i, params in enumerate(strat.default_param_grid()):
            cfg = ChunkConfig(name=name, params=params)
            for doc in docs:
                if doc.content_type not in strat.supported_content_types:
                    continue
                chunks = strat.chunk(doc, cfg)
                key = f"{name}|{i}|{doc.id}"
                snap["chunks"][key] = {
                    "params": params,
                    "spans": [[c.start_offset, c.end_offset] for c in chunks],
                }

    # Metrics + scores per (content_type, strategy, params).
    by_ct: dict[str, list[Document]] = {}
    for d in docs:
        by_ct.setdefault(d.content_type, []).append(d)
    evaluator = Evaluator(DummyEmbeddingFunction())
    for ct, ct_docs in sorted(by_ct.items()):
        ds = trivial_dataset_for_docs(ct_docs)
        for name in strategy_names:
            strat = reg.get(name)
            if ct not in strat.supported_content_types:
                continue
            for i, params in enumerate(strat.default_param_grid()):
                cfg = ChunkConfig(name=name, params=params)
                res = evaluator.evaluate(strat, cfg, ct_docs, ds)
                scores = {uc: ScoreCalculator(uc).score(res.metrics) for uc in USE_CASES}
                snap["eval"][f"{ct}|{name}|{i}"] = {
                    "metrics": _metrics_dict(res.metrics),
                    "scores": scores,
                }

        # Tuner run over every applicable strategy except LLM-backed ones.
        snap.setdefault("registry_names_by_ct", {})[ct] = reg.names(ct)
        names = [n for n in reg.names(ct) if n not in EXCLUDED_STRATEGIES]
        tuner = AutoTuner(reg, Evaluator(DummyEmbeddingFunction()), ScoreCalculator("rag_qa"))
        try:
            rec = tuner.recommend(ct_docs, "rag_qa", content_type=ct, strategies=names, dataset=ds)
        except ValueError as exc:
            snap["recommend"][ct] = {"error": str(exc)}
            continue
        snap["recommend"][ct] = [
            {"strategy": r.strategy_name, "params": r.config.params, "score": r.score}
            for r in rec.ranked
        ]
    return snap
