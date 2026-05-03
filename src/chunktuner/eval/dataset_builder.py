"""Build ``EvalDataset`` from LLM outputs or user JSON/YAML files."""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from pathlib import Path

import tiktoken
import yaml

from chunktuner.models import Document, EvalDataset, EvalQuery

logger = logging.getLogger(__name__)


def _token_f1(enc: tiktoken.Encoding, a: str, b: str) -> float:
    ta = enc.encode(a)
    tb = enc.encode(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    prec = inter / len(sa)
    rec = inter / len(sb)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


class DatasetBuilder:
    """Build `EvalDataset` from user files or LLM-generated Q&A over documents."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm_model = llm_model
        self._enc = tiktoken.get_encoding("cl100k_base")

    def build_from_user_file(self, path: Path) -> EvalDataset:
        """Parse JSON or YAML with a ``queries`` list into an `EvalDataset` (user-provided)."""
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        queries = []
        for row in data.get("queries", []):
            spans = [tuple(x) for x in row.get("answer_spans", [])]
            queries.append(
                EvalQuery(
                    id=row.get("id", str(uuid.uuid4())),
                    question=row["question"],
                    document_id=row["document_id"],
                    answer_spans=spans,
                    reference_answer=row.get("reference_answer"),
                )
            )
        return EvalDataset(
            name=data.get("name", path.stem),
            queries=queries,
            source="user_provided",
        )

    def build_from_docs(
        self,
        docs: list[Document],
        *,
        max_queries: int = 50,
        questions_per_doc: int = 3,
    ) -> EvalDataset:
        """LLM-generated Q&A with span validation (requires ``litellm`` + API keys)."""
        import litellm

        if not docs:
            return EvalDataset(name="empty", queries=[], source="llm_generated")
        random.shuffle(docs)
        budget = max(1, max_queries // max(1, questions_per_doc))
        sample = docs[:budget]
        queries: list[EvalQuery] = []
        for d in sample:
            snippet = d.content[:6000]
            prompt = (
                f"Document id: {d.id}\n"
                "Create evaluation questions with answer spans as CHARACTER offsets into the "
                "EXACT document text below. Return JSON: "
                '{"queries":[{"question":str,"start":int,"end":int,"reference_answer":str}]}\n\n'
                f"DOCUMENT:\n{snippet}"
            )
            try:
                resp = litellm.completion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content or "{}"
                payload = json.loads(raw)
            except Exception as exc:
                logger.warning("Dataset generation failed for doc %s: %s", d.id, exc)
                continue
            rows = payload.get("queries", payload) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                rows = []
            for row in rows[:questions_per_doc]:
                if not isinstance(row, dict):
                    continue
                try:
                    q = str(row.get("question", "")).strip()
                    a = int(row.get("start", row.get("start_offset", 0)))
                    b = int(row.get("end", row.get("end_offset", 0)))
                    ref = str(row.get("reference_answer", ""))
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed query row %r: %s", row, exc)
                    continue
                span_text = d.content[a:b] if 0 <= a < b <= len(d.content) else ""
                if not q or not span_text:
                    continue
                if ref and _token_f1(self._enc, span_text, ref) < 0.85:
                    continue
                queries.append(
                    EvalQuery(
                        id=f"q_{uuid.uuid4()}",
                        question=q,
                        document_id=d.id,
                        answer_spans=[(a, b)],
                        reference_answer=ref or None,
                    )
                )
                if len(queries) >= max_queries:
                    break
            if len(queries) >= max_queries:
                break
        if len(queries) < 20:
            logger.warning(
                "Only %d valid eval queries generated (minimum recommended: 20). "
                "Results may be noisy.",
                len(queries),
            )
        return EvalDataset(name="llm_generated", queries=queries, source="llm_generated")

    def build_code_function_qa(self, docs: list[Document], *, max_queries: int = 40) -> EvalDataset:
        """Heuristic code questions: point at first function body span per file."""
        queries: list[EvalQuery] = []
        for d in docs:
            if d.content_type != "code":
                continue
            m = re.search(r"(?m)^def\s+\w+\s*\(", d.content)
            if not m:
                continue
            start = m.start()
            end = min(len(d.content), start + 400)
            queries.append(
                EvalQuery(
                    id=f"code_{d.id}",
                    question="What does the first top-level function in this file do?",
                    document_id=d.id,
                    answer_spans=[(start, end)],
                )
            )
            if len(queries) >= max_queries:
                break
        return EvalDataset(name="code_heuristic", queries=queries, source="user_provided")
