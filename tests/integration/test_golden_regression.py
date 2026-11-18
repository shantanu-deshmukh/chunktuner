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

from tests.golden.snapshot import build_snapshot

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
    return base


def _rename_key(d: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    return {(new if k == old else k): v for k, v in d.items()}


# D6: new optional corpus-scope fields on EvalMetrics (not in the recorded baseline).
CORPUS_SCOPE_METRIC_KEYS = ("corpus_recall_at_k", "corpus_mrr", "corpus_ndcg_at_k")


def _normalize_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Pre-compare normalization of the current snapshot (returns a modified deep copy)."""
    snap = copy.deepcopy(snap)
    # D6: new optional corpus-scope fields — strip before comparing; checked separately in
    # test_corpus_scope_metrics_present.
    for entry in snap["eval"].values():
        for key in CORPUS_SCOPE_METRIC_KEYS:
            entry["metrics"].pop(key, None)
    # D20: code recommend is compared structurally, not exactly (see _apply_known_changes).
    snap["recommend"].pop("code", None)
    return snap


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
    ["registered_strategies", "docs", "chunks", "eval", "recommend", "trivial_dataset"],
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
