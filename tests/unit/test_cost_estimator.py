"""CostEstimator dry-run structure."""

from __future__ import annotations

from chunktuner.eval.cost_estimator import CostEstimator
from chunktuner.models import Document


def test_estimate_returns_valid_structure() -> None:
    docs = [Document(id=f"d{i}", content="word " * 200, content_type="text") for i in range(5)]
    est = CostEstimator().estimate(
        docs=docs,
        strategies=["fixed_tokens", "recursive_character"],
        param_grid={"fixed_tokens": [{"max_tokens": 512, "overlap_tokens": 0}]},
        embedding_model="text-embedding-3-small",
        generate_dataset=False,
    )
    assert est.total_tokens > 0
    assert est.strategy_configs >= 1
    assert est.estimated_wall_time_min > 0
    assert est.embedding_cost_usd >= 0.0


def test_more_strategies_means_more_tokens() -> None:
    docs = [Document(id="d1", content="word " * 500, content_type="text")]
    est1 = CostEstimator().estimate(docs, ["fixed_tokens"], {}, "text-embedding-3-small")
    est2 = CostEstimator().estimate(
        docs, ["fixed_tokens", "recursive_character"], {}, "text-embedding-3-small"
    )
    assert est2.total_tokens > est1.total_tokens


def test_embedding_cost_uses_litellm_per_token_price(monkeypatch) -> None:
    import litellm

    from chunktuner.eval import cost_estimator

    monkeypatch.setitem(litellm.model_cost, "test-embed", {"input_cost_per_token": 2e-08})
    assert cost_estimator._embedding_cost_usd("test-embed", 20_160) == 20_160 * 2e-08


def test_embedding_cost_fallback_price_for_unknown_model() -> None:
    from chunktuner.eval import cost_estimator

    assert cost_estimator._embedding_cost_usd("no-such-model-xyz", 1_000_000) == 0.02
