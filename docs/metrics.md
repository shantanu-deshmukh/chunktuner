# Evaluation metrics

Each field on [`EvalMetrics`](api/models.md#chunktuner.models.EvalMetrics) is aggregated (usually averaged) over evaluation queries. Composite **scores** are produced by [`ScoreCalculator`](api/eval/score_calculator.md#chunktuner.eval.score_calculator.ScoreCalculator) from weighted metrics for a given use case.

!!! note "Absolute numbers vs comparisons"

    Metrics are always relative to the embedding profile and corpus. Use them to **compare strategies** in the same run, not as standalone quality guarantees.

## Definitions

Full definitions, ranges, and intuition for every metric field live in the **[metrics glossary](metrics_glossary.md)** (same content as previously lived on this page).

## Use-case weights (defaults)

Default metric weights are returned by `chunktuner.config.score_profile_weights(use_case)` for:

| `use_case`      | Notes |
|-----------------|--------|
| `rag_qa`        | Weights `token_recall`, `mrr`, `token_iou`, optional `faithfulness` (treated as `0.0` when missing), and `duplication_ratio` (negative). |
| `search`        | Weights `recall_at_1` (from `recall_at_k[1]`), `mrr`, and `duplication_ratio` (negative). |
| `summarization` | Weights `token_recall`, `token_iou`, and `duplication_ratio` (negative). |
| `code_assist`   | Weights `token_recall`, `mrr`, and `duplication_ratio` (negative). |

Inspect `src/chunktuner/config.py` for the exact floating-point weights.

## Further reading

- [Strategy guide](strategy_guide.md) — how strategies feed these metrics
- [Python API — Evaluator](api/eval/evaluator.md)
