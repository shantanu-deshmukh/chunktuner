"""Grid search over strategies and configs."""

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

# Strategies skipped when ``strategies`` is None; they run only when named explicitly.
# ``late_chunking`` is a fixed-token fallback and would only duplicate ``fixed_tokens``.
_EXPLICIT_ONLY = frozenset({"late_chunking"})


def _embed_task_fields(embedding_fn: object) -> dict:
    cls = embedding_fn.__class__.__name__
    if cls == "LiteLLMEmbeddingFunction":
        return {
            "embed_type": "litellm",
            "embed_model": getattr(embedding_fn, "model", ""),
            "embed_api_base": getattr(embedding_fn, "api_base", None),
            "embed_api_key": getattr(embedding_fn, "api_key", None),
        }
    if cls == "CachedEmbeddingFunction":
        inner = getattr(embedding_fn, "_inner", None)
        cache = getattr(embedding_fn, "_cache", None)
        if inner is None or cache is None or not getattr(cache, "db_path", None):
            return {
                "embed_type": "dummy",
                "profile": getattr(embedding_fn, "profile_name", "dummy/test"),
            }
        out: dict[str, str] = {
            "embed_type": "cached",
            "cache_db_path": str(cache.db_path),
            "cache_model": cache.model,
        }
        if inner.__class__.__name__ == "LiteLLMEmbeddingFunction":
            out["inner_embed_type"] = "litellm"
            out["embed_model"] = getattr(inner, "model", "")
            out["embed_api_base"] = getattr(inner, "api_base", None)
            out["embed_api_key"] = getattr(inner, "api_key", None)
        else:
            out["inner_embed_type"] = "dummy"
            out["profile"] = getattr(inner, "profile_name", "dummy/test")
        return out
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

    def _baseline_config(self, ct: str) -> ChunkConfig | None:
        """Baseline config for ``ct``: fixed_tokens 512/0, else code_window 512/0, else None."""
        registered = set(self.strategies.names())
        candidates = (
            ("fixed_tokens", {"max_tokens": 512, "overlap_tokens": 0}),
            ("code_window", {"max_tokens": 512, "overlap_lines": 0}),
        )
        for name, params in candidates:
            if name in registered and ct in self.strategies.get(name).supported_content_types:
                return ChunkConfig(name=name, params=params)
        logger.warning("No baseline strategy supports content_type=%r; skipping baseline", ct)
        return None

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
        warm_cache: bool = False,
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
                compatible with the content type except explicit-only ones
                (``late_chunking``).
            param_grid: Optional map ``strategy_name -> list[params dict]`` overriding
                ``default_param_grid()``.
            max_docs: Cap documents used (after sampling order); ``None`` uses all.
            embedding_profile: Override label stored on results; default from evaluator.
            dataset: Optional `EvalDataset`; default is `trivial_dataset_for_docs`.
            baseline: When True, evaluates a simple baseline config first:
                ``fixed_tokens`` 512/0, or ``code_window`` 512/0 when ``fixed_tokens`` does not
                support the content type (e.g. code). Skipped with a warning if neither does.
            parallel: Use a process pool for independent evaluations.
            max_workers: Worker count when ``parallel`` is True.
            warm_cache: When True with ``parallel``, pre-chunk and ``embed_documents`` all chunk
                texts in the main process so workers using ``CachedEmbeddingFunction`` can hit the
                same SQLite cache (requires WAL + ``check_same_thread=False`` on the cache DB).

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
            names = [n for n in self.strategies.names(ct) if n not in _EXPLICIT_ONLY]
        else:
            names = [n for n in strategies if n in allowed]

        results: list[EvalResult] = []
        baseline_res: EvalResult | None = None

        bl_cfg = self._baseline_config(ct) if baseline else None
        if bl_cfg is not None:
            bl_strat = self.strategies.get(bl_cfg.name)
            baseline_res = self.evaluator.evaluate(
                bl_strat, bl_cfg, sampled, ds, scorer=self.scorer
            )
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
                if bl_cfg is not None and name == bl_cfg.name and cfg.params == bl_cfg.params:
                    continue
                jobs.append((name, dict(params)))

        if parallel and jobs and warm_cache:
            all_texts: set[str] = set()
            for name, params in jobs:
                strat = self.strategies.get(name)
                cfg = ChunkConfig(name=name, params=params)
                for d in sampled:
                    for c in strat.chunk(d, cfg):
                        all_texts.add(c.text)
            if all_texts:
                self.evaluator.embedding_fn.embed_documents(sorted(all_texts))

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
                        "user_custom_weights": self.scorer.user_custom_weights,
                        "score_retrieval_scope": self.scorer.retrieval_scope,
                        "retrieval_scope": self.evaluator.retrieval_scope,
                        "top_k": self.evaluator.top_k,
                        "enable_generation_metrics": self.evaluator.enable_generation_metrics,
                        "llm_answer_model": self.evaluator.llm_answer_model,
                        "llm_api_base": self.evaluator.llm_api_base,
                        "llm_api_key": self.evaluator.llm_api_key,
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
