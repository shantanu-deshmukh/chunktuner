"""Picklable worker entry for parallel evaluation."""

from __future__ import annotations

import pytest

from chunktuner.tuner.mp_worker import mp_evaluate_task


def test_mp_evaluate_unknown_strategy_raises_value_error() -> None:
    task = {
        "strategy_name": "totally_unknown_strategy_xyz",
        "config": {"name": "totally_unknown_strategy_xyz", "params": {}},
        "docs": [{"id": "d1", "content": "a" * 200, "content_type": "text"}],
        "dataset": {
            "name": "ds",
            "queries": [
                {
                    "id": "q1",
                    "question": "what?",
                    "document_id": "d1",
                    "answer_spans": [(0, 10)],
                }
            ],
        },
        "use_case": "rag_qa",
    }
    with pytest.raises(ValueError, match="not available in this worker process"):
        mp_evaluate_task(task)
