"""Grid search over strategies and configs."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from chunktuner.chunking.registry import StrategyRegistry
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.models import (
    ChunkConfig,
    Document,
    EvalDataset,
    EvalResult,
    Recommendation,
    UseCase,
)

logger = logging.getLogger(__name__)


def _embed_task_fields(embedding_fn: object) -> dict:
    cls = embedding_fn.__class__.__name__
    if cls == "LiteLLMEmbeddingFunction":
        return {"embed_type": "litellm", "embed_model": getattr(embedding_fn, "model", "")}
    return {"embed_type": "dummy", "profile": getattr(embedding_fn, "profile_name", "dummy/test")}


class AutoTuner:
    """Runs a parameter grid over registered strategies and ranks `EvalResult` scores."""

    def __init__(
        self,
        strategies: StrategyRegistry,
        evaluator: Evaluator,
        scorer: ScoreCalculator,
        cache: object | None = None,
    ):
        self.strategies = strategies
        self.evaluator = evaluator
        self.scorer = scorer
        self.cache = cache

    def recommend(
        self,
        docs: list[Document],
        use_case: UseCase,
        *,
        content_type: str | None = None,
        strategies: list[str] | None = None,
        param_grid: dict[str, list[dict]] | None = None,
        max_docs: int | None = 100,
        embedding_profile: str | None = None,
        dataset: EvalDataset | None = None,
        baseline: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Recommendation:
        """Run full tuning and return the best chunking config.

        Evaluates strategies (optionally filtered) across each strategy's param grid
        or a custom ``param_grid``, scores each run, and returns a ranked
        `Recommendation`.

        Args:
            docs: Ingested documents to tune against.
            use_case: Scoring profile (e.g. ``rag_qa``, ``search``, ``summarization``,
                ``code_assist``).
            content_type: If set, selects strategies for this content type; otherwise
                inferred from ``docs[0]``.
            strategies: Subset of registered strategy names; ``None`` means all
                compatible with the content type.
            param_grid: Optional map ``strategy_name -> list[params dict]`` overriding
                ``default_param_grid()``.
            max_docs: Cap documents used (after sampling order); ``None`` uses all.
            embedding_profile: Override label stored on results; default from evaluator.
            dataset: Optional `EvalDataset`; default is `trivial_dataset_for_docs`.
            baseline: When True and ``fixed_tokens`` is registered, evaluates a simple
                baseline config first.
            parallel: Use a process pool for independent evaluations.
            max_workers: Worker count when ``parallel`` is True.

        Returns:
            `Recommendation` with best config, ranked results, and optional baseline.

        Raises:
            ValueError: If ``docs`` is empty or ``strategies`` contains unknown names.
            RuntimeError: If no evaluation results were produced.
        """
        if not docs:
            raise ValueError("No documents to tune on")

        ct = content_type or docs[0].content_type
        sampled = docs if max_docs is None else docs[:max_docs]
        ds = dataset or trivial_dataset_for_docs(sampled)

        allowed = set(self.strategies.names(ct))
        if strategies is not None:
            all_registered = set(self.strategies.names())
            invalid = [n for n in strategies if n not in all_registered]
            if invalid:
                available = sorted(all_registered)
                raise ValueError(f"Unknown strategies: {invalid}. Available: {available}")
        if strategies is None:
            names = [n for n in self.strategies.names(ct)]
        else:
            names = [n for n in strategies if n in allowed]

        results: list[EvalResult] = []
        baseline_res: EvalResult | None = None

        if baseline and "fixed_tokens" in self.strategies.names():
            ft = self.strategies.get("fixed_tokens")
            bl_cfg = ChunkConfig(
                name="fixed_tokens",
                params={"max_tokens": 512, "overlap_tokens": 0},
            )
            baseline_res = self.evaluator.evaluate(ft, bl_cfg, sampled, ds, scorer=self.scorer)
            results.append(baseline_res)

        grid_in = param_grid or {}
        jobs: list[tuple[str, dict]] = []
        for name in names:
            strat = self.strategies.get(name)
            grid = grid_in.get(name)
            if grid is None:
                grid = strat.default_param_grid()
            for params in grid:
                cfg = ChunkConfig(name=name, params=dict(params))
                if (
                    baseline_res
                    and name == "fixed_tokens"
                    and int(cfg.params.get("max_tokens", -1)) == 512
                    and int(cfg.params.get("overlap_tokens", -1)) == 0
                ):
                    continue
                jobs.append((name, dict(params)))

        if parallel and jobs:
            from chunktuner.tuner.mp_worker import mp_evaluate_task

            docs_payload = [d.model_dump(mode="json") for d in sampled]
            ds_payload = ds.model_dump(mode="json")
            embed_fields = _embed_task_fields(self.evaluator.embedding_fn)
            tasks = []
            for name, params in jobs:
                tasks.append(
                    {
                        "strategy_name": name,
                        "config": ChunkConfig(name=name, params=params).model_dump(mode="json"),
                        "docs": docs_payload,
                        "dataset": ds_payload,
                        "use_case": use_case,
                        "top_k": self.evaluator.top_k,
                        "enable_generation_metrics": self.evaluator.enable_generation_metrics,
                        "encoding": "cl100k_base",
                        **embed_fields,
                    }
                )
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                fut_to_task: dict = {}
                for t in tasks:
                    fut = pool.submit(mp_evaluate_task, t)
                    fut_to_task[fut] = t
                for fut in as_completed(fut_to_task):
                    try:
                        results.append(EvalResult.model_validate(fut.result()))
                    except Exception as exc:
                        t = fut_to_task[fut]
                        logger.error(
                            "Parallel eval failed for strategy=%r params=%r: %s",
                            t["strategy_name"],
                            t["config"],
                            exc,
                        )
        else:
            for name, params in jobs:
                strat = self.strategies.get(name)
                cfg = ChunkConfig(name=name, params=params)
                results.append(self.evaluator.evaluate(strat, cfg, sampled, ds, scorer=self.scorer))

        if not results:
            raise RuntimeError("No evaluation results; check strategy names and filters")

        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        best = ranked[0]
        prof = embedding_profile or self.evaluator.embedding_fn.profile_name

        return Recommendation(
            content_type=ct,
            use_case=use_case,
            embedding_profile=prof,
            best=best,
            ranked=ranked,
            baseline=baseline_res,
        )
