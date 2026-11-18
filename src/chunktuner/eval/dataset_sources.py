"""Resolve the eval dataset for CLI / MCP runs: user file, cached LLM generation, or trivial."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from chunktuner.eval.dataset_builder import DatasetBuilder
from chunktuner.models import Document, EvalDataset

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PER_DOC = 3
DEFAULT_MAX_QUERIES = 50
_GENERATED_CACHE_VERSION = 1


def trivial_dataset_warning(dataset_flag: str, generate_flag: str) -> str:
    """Warning shown when the placeholder dataset is used (flag names differ CLI vs MCP)."""
    return (
        "Using the built-in trivial eval dataset (one placeholder query per document whose "
        "answer is the document's first 200 characters). Scores are only a rough structural "
        f"signal; pass {dataset_flag} or {generate_flag} for meaningful retrieval metrics."
    )


def _content_sha(doc: Document) -> str:
    return hashlib.sha256(doc.content.encode("utf-8")).hexdigest()


def align_user_dataset(
    ds: EvalDataset,
    docs: list[Document],
    root: Path | None = None,
) -> tuple[EvalDataset, list[str]]:
    """Map each query's ``document_id`` onto the ids of the ingested ``docs``.

    Ingested document ids are random per run, so user files reference documents by any of:
    the document id, the file path relative to ``root``, the absolute path, or the bare file
    name (when unique in the corpus). Returns the remapped dataset plus warnings.

    Raises:
        ValueError: If the dataset has no queries or none match an ingested document.
    """
    if not ds.queries:
        raise ValueError(f"Dataset {ds.name!r} contains no queries")
    lookup: dict[str, str] = {}
    name_counts: dict[str, int] = {}
    for d in docs:
        if d.path:
            name = Path(d.path).name
            name_counts[name] = name_counts.get(name, 0) + 1
    for d in docs:
        lookup[d.id] = d.id
        if not d.path:
            continue
        p = Path(d.path)
        lookup.setdefault(str(p), d.id)
        lookup.setdefault(str(p.resolve()), d.id)
        if root is not None:
            try:
                lookup.setdefault(p.resolve().relative_to(root.resolve()).as_posix(), d.id)
            except ValueError:
                pass
        if name_counts.get(p.name) == 1:
            lookup.setdefault(p.name, d.id)

    queries = []
    unmatched: list[str] = []
    for q in ds.queries:
        doc_id = lookup.get(q.document_id) or lookup.get(q.document_id.removeprefix("./"))
        if doc_id is None:
            unmatched.append(q.document_id)
            continue
        queries.append(q.model_copy(update={"document_id": doc_id}))
    if not queries:
        raise ValueError(
            f"None of the {len(ds.queries)} queries in dataset {ds.name!r} reference an "
            "ingested document. Set document_id to the file path relative to the corpus "
            "path, the absolute path, or the file name."
        )
    warnings: list[str] = []
    if unmatched:
        sample = ", ".join(repr(x) for x in sorted(set(unmatched))[:5])
        warnings.append(
            f"Skipped {len(unmatched)} dataset queries whose document_id matches no ingested "
            f"document (e.g. {sample})."
        )
    if len(queries) < 20:
        warnings.append(
            f"Dataset has only {len(queries)} usable queries (20+ recommended); "
            "MRR / recall will be noisy."
        )
    return ds.model_copy(update={"queries": queries}), warnings


def load_user_dataset(
    path: Path,
    docs: list[Document],
    root: Path | None = None,
) -> tuple[EvalDataset, list[str]]:
    """Parse a JSON/YAML dataset file and align it with ``docs`` (see `align_user_dataset`)."""
    try:
        ds = DatasetBuilder().build_from_user_file(path)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Invalid dataset file {path.name}: each entry under 'queries' needs "
            f"'question' and 'document_id' ({exc!r})"
        ) from exc
    return align_user_dataset(ds, docs, root)


def generated_dataset_cache_path(
    cache_dir: Path,
    docs: list[Document],
    llm_model: str,
    *,
    questions_per_doc: int = DEFAULT_QUESTIONS_PER_DOC,
    max_queries: int = DEFAULT_MAX_QUERIES,
    llm_api_base: str | None = None,
) -> Path:
    """Cache file for a generated dataset, keyed by doc contents + generation settings."""
    key_src = {
        "v": _GENERATED_CACHE_VERSION,
        "docs": sorted(_content_sha(d) for d in docs),
        "llm_model": llm_model,
        "llm_api_base": llm_api_base,
        "questions_per_doc": questions_per_doc,
        "max_queries": max_queries,
    }
    key = hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:32]
    return cache_dir / "datasets" / f"generated_{key}.json"


def load_or_generate_dataset(
    docs: list[Document],
    *,
    llm_model: str,
    cache_dir: Path,
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    questions_per_doc: int = DEFAULT_QUESTIONS_PER_DOC,
    max_queries: int = DEFAULT_MAX_QUERIES,
    before_generate: Callable[[], None] | None = None,
) -> tuple[EvalDataset, Path, bool]:
    """Return ``(dataset, cache_path, cache_hit)``; generate with the LLM only on a cache miss.

    ``before_generate`` runs just before the (paid) LLM calls — e.g. a confirmation prompt
    that raises to abort. Cached query ``document_id`` values are remapped to the current
    ``docs`` by content hash, since ingested ids change between runs.

    Raises:
        ValueError: If the LLM produced no valid queries (nothing is cached in that case).
    """
    path = generated_dataset_cache_path(
        cache_dir,
        docs,
        llm_model,
        questions_per_doc=questions_per_doc,
        max_queries=max_queries,
        llm_api_base=llm_api_base,
    )
    sha_to_id: dict[str, str] = {}
    for d in docs:
        sha_to_id.setdefault(_content_sha(d), d.id)

    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ds = EvalDataset.model_validate(payload["dataset"])
            doc_sha: dict[str, str] = payload["doc_content_sha"]
            queries = [
                q.model_copy(update={"document_id": sha_to_id[doc_sha[q.document_id]]})
                for q in ds.queries
                if doc_sha.get(q.document_id) in sha_to_id
            ]
            if queries:
                return ds.model_copy(update={"queries": queries}), path, True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable generated-dataset cache %s: %s", path, exc)

    if before_generate is not None:
        before_generate()
    builder = DatasetBuilder(llm_model, llm_api_base=llm_api_base, llm_api_key=llm_api_key)
    # build_from_docs shuffles its input in place; pass a copy so evaluation order is untouched.
    ds = builder.build_from_docs(
        list(docs), max_queries=max_queries, questions_per_doc=questions_per_doc
    )
    if not ds.queries:
        raise ValueError(
            f"LLM dataset generation with {llm_model!r} produced no valid queries "
            "(check the model / API settings, or provide a dataset file instead)."
        )
    id_to_sha = {d.id: _content_sha(d) for d in docs}
    payload = {
        "dataset": ds.model_dump(mode="json"),
        "doc_content_sha": {q.document_id: id_to_sha[q.document_id] for q in ds.queries},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ds, path, False
