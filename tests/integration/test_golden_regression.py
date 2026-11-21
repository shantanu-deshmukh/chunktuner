"""Regression gate: current behaviour must match ``tests/golden/baseline.json``.

The baseline was recorded on the pre-change commit with all extras installed. Intentional
behaviour changes are applied to the baseline in ``_apply_known_changes`` and new fields are
stripped from the snapshot in ``_normalize_snapshot``, each tagged with its decision id.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import tiktoken

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.models import ChunkConfig
from tests.golden.snapshot import build_snapshot, load_docs

BASELINE = Path(__file__).resolve().parent.parent / "golden" / "baseline.json"
SCORE_TOL = 1e-9


def _apply_known_changes(base: dict[str, Any]) -> dict[str, Any]:
    """Patch the recorded baseline with intentional, reviewed behaviour changes."""
    # D5: trivial_dataset_for_docs now labels its placeholder dataset source="trivial"
    # (was mislabelled "user_provided").
    base["trivial_dataset"]["source"] = "trivial"
    # D8: semantic default grid renamed `similarity_threshold` -> `overlap_ratio` (same values,
    # same semantics); only the params key changes — spans, metrics and scores are identical.
    for key, entry in base["chunks"].items():
        if key.startswith("semantic|"):
            entry["params"] = _rename_key(entry["params"], "similarity_threshold", "overlap_ratio")
    for ranked in base["recommend"].values():
        if isinstance(ranked, list):
            for row in ranked:
                if row["strategy"] == "semantic":
                    row["params"] = _rename_key(
                        row["params"], "similarity_threshold", "overlap_ratio"
                    )
    # D20: recommend on code used to raise (fixed_tokens baseline unsupported for code), so the
    # recorded value is an error. Drop it here (and from the snapshot in _normalize_snapshot);
    # the new value is checked structurally in test_recommend_code_no_longer_errors.
    base["recommend"].pop("code", None)
    # D21: code_ast now honours merge_small (default grid sets True): Python files are fully
    # covered and small top-level units merged, so sample.py spans and the code eval entries
    # change. Dropped from both sides; checked structurally in test_code_ast_merge_small_covers.
    _drop_d21(base)
    return base


def _is_d21_chunk_key(key: str) -> bool:
    strategy, _, doc = key.split("|")
    return strategy == "code_ast" and doc == D21_DOC


def _drop_d21(data: dict[str, Any]) -> None:
    data["chunks"] = {k: v for k, v in data["chunks"].items() if not _is_d21_chunk_key(k)}
    data["eval"] = {k: v for k, v in data["eval"].items() if k.split("|")[1] != "code_ast"}


# D21: the Python fixture whose code_ast spans changed (sample.js still uses line windows).
D21_DOC = "sample.py"


def _rename_key(d: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    return {(new if k == old else k): v for k, v in d.items()}


# D6: new optional corpus-scope fields on EvalMetrics (not in the recorded baseline).
CORPUS_SCOPE_METRIC_KEYS = ("corpus_recall_at_k", "corpus_mrr", "corpus_ndcg_at_k")
# D17: new optional retrieval-pipeline fields on EvalMetrics (not in the recorded baseline).
RETRIEVAL_PIPELINE_METRIC_KEYS = ("retrieval_mode", "reranker", "reranker_overflow_ratio")


def _normalize_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Pre-compare normalization of the current snapshot (returns a modified deep copy)."""
    snap = copy.deepcopy(snap)
    # D6: new optional corpus-scope fields — strip before comparing; checked separately in
    # test_corpus_scope_metrics_present.
    for entry in snap["eval"].values():
        for key in CORPUS_SCOPE_METRIC_KEYS:
            entry["metrics"].pop(key, None)
        # D17: retrieval pipeline fields — the default (dense, no reranker) path is unchanged,
        # so only these new keys differ; strip them before comparing.
        for key in RETRIEVAL_PIPELINE_METRIC_KEYS:
            entry["metrics"].pop(key, None)
    # D20: code recommend is compared structurally, not exactly (see _apply_known_changes).
    snap["recommend"].pop("code", None)
    # D21: code_ast merge_small — compared structurally (see _apply_known_changes).
    _drop_d21(snap)
    # D11: new strategy paragraph_group (not in the recorded baseline). Drop its entries;
    # scores are per config (no cross-config normalization), so the remaining recommend rows
    # keep their recorded order and scores. Checked separately in test_paragraph_group_present.
    snap["registered_strategies"] = [s for s in snap["registered_strategies"] if s != NEW_D11]
    for ct, names in snap["registry_names_by_ct"].items():
        snap["registry_names_by_ct"][ct] = [n for n in names if n != NEW_D11]
    snap["chunks"] = {k: v for k, v in snap["chunks"].items() if k.split("|")[0] != NEW_D11}
    snap["eval"] = {k: v for k, v in snap["eval"].items() if k.split("|")[1] != NEW_D11}
    for ct, ranked in snap["recommend"].items():
        if isinstance(ranked, list):
            snap["recommend"][ct] = [r for r in ranked if r["strategy"] != NEW_D11]
    return snap


# D11: strategy added after the baseline was recorded.
NEW_D11 = "paragraph_group"


def _assert_close(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) == set(actual), f"{path}: keys {set(expected) ^ set(actual)}"
        for k in expected:
            _assert_close(expected[k], actual[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _assert_close(e, a, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert math.isclose(expected, float(actual), abs_tol=SCORE_TOL), (
            f"{path}: {expected} != {actual}"
        )
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(json.dumps(build_snapshot()))


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return _apply_known_changes(json.loads(BASELINE.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "section",
    [
        "registered_strategies",
        "registry_names_by_ct",
        "docs",
        "chunks",
        "eval",
        "recommend",
        "trivial_dataset",
    ],
)
def test_matches_baseline(section: str, snapshot: dict, baseline: dict) -> None:
    if section == "registered_strategies":
        missing = set(baseline[section]) - set(snapshot[section])
        if missing:
            pytest.fail(
                f"Strategies missing (optional extras not installed?): {sorted(missing)}. "
                "Run `uv sync --all-extras --dev`."
            )
    _assert_close(baseline[section], _normalize_snapshot(snapshot)[section], section)


def test_corpus_scope_metrics_present(snapshot: dict) -> None:
    """D6: every eval entry carries corpus-scope ranking metrics within [0, 1]."""
    assert snapshot["eval"]
    for name, entry in snapshot["eval"].items():
        m = entry["metrics"]
        for key in CORPUS_SCOPE_METRIC_KEYS:
            assert m.get(key) is not None, f"{name}: missing {key}"
        assert 0.0 <= m["corpus_mrr"] <= 1.0, name
        for key in ("corpus_recall_at_k", "corpus_ndcg_at_k"):
            assert set(m[key]) == {"1", "3", "5"}, f"{name}.{key}"
            assert all(0.0 <= v <= 1.0 for v in m[key].values()), f"{name}.{key}"


def test_recommend_code_no_longer_errors(snapshot: dict) -> None:
    """D20: code recommend uses a code_window 512/0 baseline instead of raising."""
    ranked = snapshot["recommend"]["code"]
    assert isinstance(ranked, list) and ranked, ranked
    rows = [{"strategy": r["strategy"], "params": r["params"]} for r in ranked]
    assert {"strategy": "code_window", "params": {"max_tokens": 512, "overlap_lines": 0}} in rows
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_paragraph_group_present(snapshot: dict) -> None:
    """D11: paragraph_group is registered for prose types and in every prose recommend run."""
    assert NEW_D11 in snapshot["registered_strategies"]
    for ct in ("text", "markdown", "html"):
        assert NEW_D11 in snapshot["registry_names_by_ct"][ct], ct
        rows = [r for r in snapshot["recommend"][ct] if r["strategy"] == NEW_D11]
        assert len(rows) == 6, ct
    chunk_keys = [k for k in snapshot["chunks"] if k.startswith(f"{NEW_D11}|")]
    eval_keys = [k for k in snapshot["eval"] if k.split("|")[1] == NEW_D11]
    assert chunk_keys and eval_keys


def test_code_ast_merge_small_covers(snapshot: dict) -> None:
    """D21: code_ast default grid (merge_small=True) covers all of sample.py, capped per chunk."""
    strat = build_full_registry().get("code_ast")
    if getattr(strat, "_parser", None) is None:
        pytest.skip("tree-sitter not installed")
    doc = next(d for d in load_docs() if d.id == D21_DOC)
    enc = tiktoken.get_encoding("cl100k_base")
    for i, params in enumerate(strat.default_param_grid()):
        assert params.get("merge_small") is True
        key = f"code_ast|{i}|{D21_DOC}"
        chunks = strat.chunk(doc, ChunkConfig(name="code_ast", params=params))
        assert [[c.start_offset, c.end_offset] for c in chunks] == snapshot["chunks"][key]["spans"]
        covered = [False] * len(doc.content)
        prev_end = 0
        for c in chunks:
            assert doc.content[c.start_offset : c.end_offset] == c.text, key
            assert c.start_offset >= prev_end, f"{key}: overlapping/unordered chunks"
            prev_end = c.end_offset
            # Every top-level unit of sample.py fits max_tokens, so no code_window fallback
            # pieces occur and the cap holds for every chunk.
            assert len(enc.encode(c.text)) <= params["max_tokens"], key
            for j in range(c.start_offset, c.end_offset):
                covered[j] = True
        missed = "".join(ch for ch, cov in zip(doc.content, covered, strict=True) if not cov)
        assert not missed.strip(), f"{key}: uncovered code {missed.strip()[:60]!r}"
        assert "import math" in chunks[0].text, key
