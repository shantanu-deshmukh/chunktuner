#!/usr/bin/env python3
"""Tune chunking strategies on S&P 500 earnings transcripts (Hugging Face) or local fixtures.

Uses ``build_full_registry()`` like the CLI: ``AutoTuner`` scores **multiple** strategies
(``fixed_tokens`` + ``recursive_character`` by default, optionally every ``text`` strategy)
and picks the best ``(strategy, params)`` pair.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

from chunktuner.chunking import build_full_registry
from chunktuner.chunking.registry import StrategyRegistry
from chunktuner.config import DEFAULT_LLM_MODEL
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import Document, EvalDataset, EvalQuery, Recommendation, UseCase
from chunktuner.tuner import AutoTuner

logger = logging.getLogger(__name__)
_console = Console(width=120)

TextMode = Literal["raw", "structured", "structured_prefixed"]


def default_recursive_separators() -> list[str]:
    return ["\n\n", "\n", ". ", " ", ""]


def finance_aware_separators() -> list[str]:
    """Sector-style delimiters before generic hierarchy (dataset often mentions Q&A)."""
    return [
        "question-and-answer",
        "Question-and-Answer",
        "Operator Instructions",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]


def finance_recursive_param_grid() -> dict[str, list[dict[str, Any]]]:
    """Three character-window configs for ``recursive_character``: small / medium / finance-aware."""
    base_sep = default_recursive_separators()
    fin_sep = finance_aware_separators()
    return {
        "recursive_character": [
            {
                "chunk_size_chars": 512,
                "chunk_overlap_chars": 51,
                "separators": list(base_sep),
            },
            {
                "chunk_size_chars": 1024,
                "chunk_overlap_chars": 154,
                "separators": list(base_sep),
            },
            {
                "chunk_size_chars": 1024,
                "chunk_overlap_chars": 154,
                "separators": list(fin_sep),
            },
        ],
    }


def finance_fixed_tokens_param_grid() -> dict[str, list[dict[str, Any]]]:
    """Token-window grid: small / medium / large, aligned with recursive demo intent."""
    return {
        "fixed_tokens": [
            {"max_tokens": 256, "overlap_tokens": 0},
            {"max_tokens": 512, "overlap_tokens": 51},
            {"max_tokens": 1024, "overlap_tokens": 154},
        ],
    }


def financial_param_grid_for(
    strategy_names: list[str],
    llm_model: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Demo-sized grids per strategy (smaller than full ``default_param_grid()``).

    To add a new strategy, append its name to ``--strategies`` and add its grid here.
    Check the strategy's ``param_schema()`` for valid keys.
    """
    grid: dict[str, list[dict[str, Any]]] = {}
    if "fixed_tokens" in strategy_names:
        grid.update(finance_fixed_tokens_param_grid())
    if "recursive_character" in strategy_names:
        grid.update(finance_recursive_param_grid())
    if "semantic" in strategy_names:
        grid["semantic"] = [
            {"max_tokens": 512, "similarity_threshold": 0.0},
            {"max_tokens": 512, "similarity_threshold": 0.1},
            {"max_tokens": 1024, "similarity_threshold": 0.1},
        ]
    if "late_chunking" in strategy_names:
        grid["late_chunking"] = [
            {"chunk_size_tokens": 256, "overlap_tokens": 0},
            {"chunk_size_tokens": 512, "overlap_tokens": 51},
            {"chunk_size_tokens": 1024, "overlap_tokens": 154},
        ]
    if "agentic" in strategy_names:
        grid["agentic"] = [
            {"model": llm_model or DEFAULT_LLM_MODEL, "max_propositions": 30},
        ]
    return grid


def resolve_strategy_names(
    registry: StrategyRegistry,
    *,
    strategies_csv: str | None,
    all_text: bool,
    include_agentic: bool,
) -> list[str]:
    """Resolve CLI strategy selection against ``build_full_registry()``."""
    if strategies_csv:
        names = [x.strip() for x in strategies_csv.split(",") if x.strip()]
        registered = set(registry.names())
        invalid = [n for n in names if n not in registered]
        if invalid:
            raise SystemExit(f"Unknown strategies: {invalid}. Registered: {sorted(registered)}")
        return names
    if all_text:
        text_names = list(registry.names("text"))
        if not include_agentic:
            text_names = [n for n in text_names if n != "agentic"]
        try:
            import semchunk  # noqa: F401
        except ImportError:
            text_names = [n for n in text_names if n != "semantic"]
        return text_names
    return ["fixed_tokens", "recursive_character"]


def print_strategy_availability(
    registry: StrategyRegistry,
    evaluated: list[str],
    content_type: str,
) -> None:
    """Print which strategies will run and which are skipped or unavailable."""
    all_for_ct = set(registry.names(content_type))
    skipped = all_for_ct - set(evaluated)
    _console.print(f"Strategies selected : [bold]{', '.join(sorted(evaluated))}[/bold]")
    if skipped:
        _console.print(f"[dim]Skipped (not in run): {', '.join(sorted(skipped))}[/dim]")
    try:
        import semchunk  # noqa: F401
    except ImportError:
        if "semantic" in skipped:
            _console.print(
                "[dim]  semantic needs semchunk: "
                "uv add 'chunktuner\\[semantic]'[/dim]"
            )
    if "agentic" in skipped:
        _console.print("[dim]  agentic needs --include-agentic (LLM API cost)[/dim]")
    _console.print()


def extract_speaker_segments(record: dict[str, Any]) -> list[str]:
    """Flatten ``structured_content`` into ``Speaker: text`` lines (dialogue-aware view)."""
    raw = record.get("structured_content") or []
    segments: list[str] = []
    if not isinstance(raw, list):
        return segments
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        sp = str(turn.get("speaker", "Unknown")).strip()
        tx = str(turn.get("text", "")).strip()
        if tx:
            segments.append(f"{sp}: {tx}")
    return segments


def _metadata_prefix(record: dict[str, Any]) -> str:
    sym = record.get("symbol", "")
    yr = record.get("year", "")
    q = record.get("quarter", "")
    return f"[{sym} FY{yr} Q{q}]"


def transcript_record_to_document(
    record: dict[str, Any],
    *,
    text_mode: TextMode,
    max_chars: int | None,
    index: int,
) -> Document:
    """Build a ``Document`` from one HF row (or synthetic fixture row)."""
    symbol = str(record.get("symbol", "UNK"))
    year = record.get("year", "")
    quarter = record.get("quarter", "")
    date = str(record.get("date", "")).replace(" ", "_").replace(":", "-")
    doc_id = f"{symbol}_{year}_Q{quarter}_{date}_{index}"

    raw_content = str(record.get("content") or "")
    turns = extract_speaker_segments(record)
    structured_body = "\n\n".join(turns) if turns else raw_content

    if text_mode == "raw":
        body = raw_content
    elif text_mode == "structured":
        body = structured_body
    else:
        body = f"{_metadata_prefix(record)}\n\n{structured_body}"

    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars]

    meta = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "date": record.get("date"),
        "company_name": record.get("company_name"),
        "text_mode": text_mode,
    }
    return Document(
        id=doc_id,
        content=body,
        content_type="text",
        path=None,
        metadata=meta,
    )


def synthetic_fixture_records() -> list[dict[str, Any]]:
    """Two realistic earnings call transcripts for offline use (no Hugging Face download)."""
    return [
        {
            "symbol": "DEMO",
            "year": 2024,
            "quarter": 1,
            "date": "2024-01-15 16:00:00",
            "company_name": "Demo Corp",
            "content": (
                "Operator: Welcome to Demo Corp's First Quarter 2024 Earnings Call. "
                "At this time, all participants are in a listen-only mode. "
                "Following management's prepared remarks, we will open the call for questions.\n\n"
                "CFO: Thank you. Revenue grew 10% year over year to $4.2 billion, "
                "driven primarily by strong performance in our cloud segment, which expanded 34%. "
                "Operating margin held at 23%, consistent with our guidance range of 22-24%. "
                "Free cash flow was $820 million, up from $640 million in the prior-year period. "
                "Capital expenditure came in at $310 million as we continue to invest in "
                "data center infrastructure. Total headcount grew 8% to 42,000 employees, "
                "with the majority of additions in engineering and customer success roles.\n\n"
                "CEO: Our earnings per share of $1.87 exceeded the consensus estimate of $1.74. "
                "We are reaffirming full-year guidance of $7.20-$7.40 EPS on revenue of "
                "$17.0-$17.4 billion. Supply chain headwinds from the March disruptions have "
                "been largely resolved and we do not expect further material impact in Q2. "
                "Our international segment now represents 38% of total revenue, up from 33% "
                "a year ago, with particularly strong growth in the Asia-Pacific region at 28% "
                "year over year.\n\n"
                "Analyst (Morgan Stanley): Can you give us more color on the operating margin "
                "trajectory into Q2? Specifically, do you expect the step-up in cloud hosting "
                "costs to persist, and how does that square with the full-year guidance band?\n\n"
                "CFO: Great question. We expect Q2 operating margin to be in the 22-23% range "
                "as we absorb the higher hosting costs. However, by Q3, the new co-location "
                "agreements kick in and we expect margin recovery to the upper end of our "
                "22-24% annual guidance band. Free cash flow conversion should remain above 90% "
                "of net income for the full year. I would also note that stock-based compensation "
                "as a percentage of revenue is declining from 7.2% last year to a target of "
                "6.5% this year, which provides additional operating leverage.\n\n"
                "Analyst (Goldman Sachs): On capital expenditure — you mentioned $310 million "
                "this quarter. Should we model a step-up in the second half given the new "
                "data center commitments? And can you speak to the return on invested capital "
                "you expect from these infrastructure investments?\n\n"
                "CEO: Yes. Full-year capex guidance is $1.4-$1.6 billion, implying "
                "roughly $350-$400 million per quarter in H2. This is consistent with our "
                "long-term target of investing 8-9% of revenue into infrastructure. On ROIC, "
                "our data center investments historically generate a 4-year payback period, "
                "translating to an unlevered IRR of approximately 22-25%, well above our "
                "weighted average cost of capital of 9%.\n\n"
                "Analyst (JP Morgan): You mentioned international revenue growing to 38% of the "
                "mix. Are there specific markets where you are seeing acceleration, and does "
                "currency translation create a meaningful headwind at current spot rates?\n\n"
                "CFO: The strongest growth is coming from Japan, Australia, and Southeast Asia, "
                "where enterprise cloud adoption is still in early stages. On foreign exchange, "
                "the strong dollar creates roughly a 2-point headwind to reported revenue growth "
                "at current rates. Our full-year guidance already assumes a 1.5-point FX drag, "
                "so there is modest incremental risk if the dollar strengthens further. "
                "We do not hedge revenue, but we do hedge operating expenses denominated in "
                "foreign currencies, which provides partial natural offset.\n\n"
                "Analyst (Barclays): One last question on the cloud segment gross margins — "
                "you mentioned 34% growth. Can you give us the gross margin for cloud "
                "specifically versus the legacy on-premise segment?\n\n"
                "CFO: We don't break out gross margin by segment, but I can tell you that "
                "cloud gross margins are meaningfully above our company-wide 60.5% gross margin, "
                "while legacy on-premise is below. As cloud continues to grow as a percentage "
                "of mix, we expect overall gross margin to expand 50-100 basis points annually "
                "over the next three years.\n\n"
                "Operator: This concludes today's conference call. Thank you for participating."
            ),
            "structured_content": [
                {
                    "speaker": "Operator",
                    "text": (
                        "Welcome to Demo Corp's First Quarter 2024 Earnings Call. At this time, "
                        "all participants are in a listen-only mode. Following management's "
                        "prepared remarks, we will open the call for questions."
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "Thank you. Revenue grew 10% year over year to $4.2 billion, driven "
                        "primarily by strong performance in our cloud segment, which expanded 34%. "
                        "Operating margin held at 23%, consistent with our guidance range of "
                        "22-24%. Free cash flow was $820 million, up from $640 million in the "
                        "prior-year period. Capital expenditure came in at $310 million as we "
                        "continue to invest in data center infrastructure. Total headcount grew "
                        "8% to 42,000 employees, with the majority of additions in engineering "
                        "and customer success roles."
                    ),
                },
                {
                    "speaker": "CEO",
                    "text": (
                        "Our earnings per share of $1.87 exceeded the consensus estimate of $1.74. "
                        "We are reaffirming full-year guidance of $7.20-$7.40 EPS on revenue of "
                        "$17.0-$17.4 billion. Supply chain headwinds from the March disruptions "
                        "have been largely resolved and we do not expect further material impact "
                        "in Q2. Our international segment now represents 38% of total revenue, "
                        "up from 33% a year ago, with particularly strong growth in the "
                        "Asia-Pacific region at 28% year over year."
                    ),
                },
                {
                    "speaker": "Analyst (Morgan Stanley)",
                    "text": (
                        "Can you give us more color on the operating margin trajectory into Q2? "
                        "Specifically, do you expect the step-up in cloud hosting costs to "
                        "persist, and how does that square with the full-year guidance band?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "Great question. We expect Q2 operating margin to be in the 22-23% range "
                        "as we absorb the higher hosting costs. However, by Q3, the new "
                        "co-location agreements kick in and we expect margin recovery to the "
                        "upper end of our 22-24% annual guidance band. Free cash flow conversion "
                        "should remain above 90% of net income for the full year. I would also "
                        "note that stock-based compensation as a percentage of revenue is "
                        "declining from 7.2% last year to a target of 6.5% this year, which "
                        "provides additional operating leverage."
                    ),
                },
                {
                    "speaker": "Analyst (Goldman Sachs)",
                    "text": (
                        "On capital expenditure — you mentioned $310 million this quarter. "
                        "Should we model a step-up in the second half given the new data center "
                        "commitments? And can you speak to the return on invested capital you "
                        "expect from these infrastructure investments?"
                    ),
                },
                {
                    "speaker": "CEO",
                    "text": (
                        "Yes. Full-year capex guidance is $1.4-$1.6 billion, implying roughly "
                        "$350-$400 million per quarter in H2. This is consistent with our "
                        "long-term target of investing 8-9% of revenue into infrastructure. "
                        "On ROIC, our data center investments historically generate a 4-year "
                        "payback period, translating to an unlevered IRR of approximately "
                        "22-25%, well above our weighted average cost of capital of 9%."
                    ),
                },
                {
                    "speaker": "Analyst (JP Morgan)",
                    "text": (
                        "You mentioned international revenue growing to 38% of the mix. Are there "
                        "specific markets where you are seeing acceleration, and does currency "
                        "translation create a meaningful headwind at current spot rates?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "The strongest growth is coming from Japan, Australia, and Southeast Asia, "
                        "where enterprise cloud adoption is still in early stages. On foreign "
                        "exchange, the strong dollar creates roughly a 2-point headwind to "
                        "reported revenue growth at current rates. Our full-year guidance already "
                        "assumes a 1.5-point FX drag, so there is modest incremental risk if the "
                        "dollar strengthens further. We do not hedge revenue, but we do hedge "
                        "operating expenses denominated in foreign currencies, which provides "
                        "partial natural offset."
                    ),
                },
                {
                    "speaker": "Analyst (Barclays)",
                    "text": (
                        "One last question on the cloud segment gross margins — you mentioned 34% "
                        "growth. Can you give us the gross margin for cloud specifically versus "
                        "the legacy on-premise segment?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "We don't break out gross margin by segment, but I can tell you that "
                        "cloud gross margins are meaningfully above our company-wide 60.5% gross "
                        "margin, while legacy on-premise is below. As cloud continues to grow as "
                        "a percentage of mix, we expect overall gross margin to expand 50-100 "
                        "basis points annually over the next three years."
                    ),
                },
                {
                    "speaker": "Operator",
                    "text": "This concludes today's conference call. Thank you for participating.",
                },
            ],
        },
        {
            "symbol": "DEMO",
            "year": 2024,
            "quarter": 2,
            "date": "2024-04-20 16:00:00",
            "company_name": "Demo Corp",
            "content": (
                "Operator: Good afternoon, and welcome to Demo Corp's Q2 2024 Earnings Call.\n\n"
                "CFO: Q2 revenue was $4.5 billion, up 12% year over year, ahead of our guidance "
                "midpoint of $4.35 billion. Gross margin expanded 80 basis points to 61.4% "
                "as cloud mix continued to shift favorably. Operating margin came in at 23.5%, "
                "the high end of our guidance band, as the new co-location agreements we "
                "discussed last quarter began to deliver savings ahead of schedule. "
                "Research and development expense was $680 million, or 15.1% of revenue, "
                "as we continue to invest in our next-generation platform capabilities.\n\n"
                "CEO: Earnings per share of $2.01 represent a 20% year-over-year increase. "
                "We are raising full-year EPS guidance to $7.60-$7.80 from the prior range "
                "of $7.20-$7.40, reflecting the operating leverage we are now seeing at scale. "
                "Free cash flow for the quarter was $940 million, putting us well on track "
                "for our full-year free cash flow target of $3.5 billion. We repurchased "
                "$500 million of shares during the quarter under our existing buyback program, "
                "leaving $1.2 billion of remaining authorization.\n\n"
                "Analyst (JP Morgan): The guidance raise is meaningful. Can you walk us through "
                "what changed versus 90 days ago — is this purely the co-location benefit, or "
                "are there also pricing dynamics at play in the cloud segment?\n\n"
                "CFO: It's a combination of factors. The co-location savings are approximately "
                "$40 million of annualized benefit starting Q3. Additionally, average selling "
                "prices in our enterprise cloud tier increased 4% as we rolled out the new "
                "premium support tier in February. Supply chain normalization also reduced "
                "hardware provisioning costs by roughly 200 basis points versus Q1. Together, "
                "these three factors account for the majority of the EPS guidance raise.\n\n"
                "Analyst (Citi): On free cash flow — $940 million is well above consensus of "
                "$780 million. Was there a working capital benefit in the quarter?\n\n"
                "CFO: Yes. We collected approximately $120 million of deferred revenue from "
                "a large government contract that signed in Q4 2023. Stripping that out, "
                "normalized free cash flow was approximately $820 million, still above "
                "consensus. We expect Q3 free cash flow of $850-$900 million. Our days sales "
                "outstanding improved by 3 days to 47 days, reflecting better collections "
                "discipline across our enterprise accounts.\n\n"
                "Analyst (Morgan Stanley): On capital allocation — you mentioned $500 million "
                "in buybacks this quarter. How should we think about the balance between "
                "buybacks and M&A over the next 12 months?\n\n"
                "CEO: Our capital allocation framework prioritizes organic investment first, "
                "then M&A for technology or talent where we see strategic fit, and finally "
                "buybacks when our stock is trading below our intrinsic value estimate. "
                "At current prices, we view buybacks as an attractive use of capital. "
                "We are actively evaluating two bolt-on acquisitions in the security and "
                "observability spaces, each in the $200-400 million range, which would be "
                "funded from existing cash without impacting the buyback program.\n\n"
                "Analyst (Goldman Sachs): Can you comment on customer retention and net "
                "revenue retention in the enterprise segment specifically?\n\n"
                "CFO: We do not disclose NRR as a standalone metric, but I can share that "
                "gross revenue retention in our enterprise cohort remains above 95%, "
                "consistent with last year. Expansion revenue from existing customers "
                "contributed approximately 60% of total new revenue in Q2, up from 55% "
                "a year ago. This shift toward land-and-expand economics is a key driver "
                "of our improving unit economics and lower customer acquisition cost per "
                "dollar of annual recurring revenue.\n\n"
                "Operator: Thank you. This concludes Demo Corp's Q2 2024 Earnings Call."
            ),
            "structured_content": [
                {
                    "speaker": "Operator",
                    "text": "Good afternoon, and welcome to Demo Corp's Q2 2024 Earnings Call.",
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "Q2 revenue was $4.5 billion, up 12% year over year, ahead of our guidance "
                        "midpoint of $4.35 billion. Gross margin expanded 80 basis points to 61.4% "
                        "as cloud mix continued to shift favorably. Operating margin came in at "
                        "23.5%, the high end of our guidance band, as the new co-location "
                        "agreements we discussed last quarter began to deliver savings ahead of "
                        "schedule. Research and development expense was $680 million, or 15.1% "
                        "of revenue, as we continue to invest in our next-generation platform "
                        "capabilities."
                    ),
                },
                {
                    "speaker": "CEO",
                    "text": (
                        "Earnings per share of $2.01 represent a 20% year-over-year increase. "
                        "We are raising full-year EPS guidance to $7.60-$7.80 from the prior range "
                        "of $7.20-$7.40, reflecting the operating leverage we are now seeing at "
                        "scale. Free cash flow for the quarter was $940 million, putting us well "
                        "on track for our full-year free cash flow target of $3.5 billion. "
                        "We repurchased $500 million of shares during the quarter under our "
                        "existing buyback program, leaving $1.2 billion of remaining authorization."
                    ),
                },
                {
                    "speaker": "Analyst (JP Morgan)",
                    "text": (
                        "The guidance raise is meaningful. Can you walk us through what changed "
                        "versus 90 days ago — is this purely the co-location benefit, or are there "
                        "also pricing dynamics at play in the cloud segment?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "It's a combination of factors. The co-location savings are approximately "
                        "$40 million of annualized benefit starting Q3. Additionally, average "
                        "selling prices in our enterprise cloud tier increased 4% as we rolled out "
                        "the new premium support tier in February. Supply chain normalization also "
                        "reduced hardware provisioning costs by roughly 200 basis points versus Q1. "
                        "Together, these three factors account for the majority of the EPS "
                        "guidance raise."
                    ),
                },
                {
                    "speaker": "Analyst (Citi)",
                    "text": (
                        "On free cash flow — $940 million is well above consensus of $780 million. "
                        "Was there a working capital benefit in the quarter?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "Yes. We collected approximately $120 million of deferred revenue from a "
                        "large government contract that signed in Q4 2023. Stripping that out, "
                        "normalized free cash flow was approximately $820 million, still above "
                        "consensus. We expect Q3 free cash flow of $850-$900 million. Our days "
                        "sales outstanding improved by 3 days to 47 days, reflecting better "
                        "collections discipline across our enterprise accounts."
                    ),
                },
                {
                    "speaker": "Analyst (Morgan Stanley)",
                    "text": (
                        "On capital allocation — you mentioned $500 million in buybacks this "
                        "quarter. How should we think about the balance between buybacks and M&A "
                        "over the next 12 months?"
                    ),
                },
                {
                    "speaker": "CEO",
                    "text": (
                        "Our capital allocation framework prioritizes organic investment first, "
                        "then M&A for technology or talent where we see strategic fit, and finally "
                        "buybacks when our stock is trading below our intrinsic value estimate. "
                        "At current prices, we view buybacks as an attractive use of capital. "
                        "We are actively evaluating two bolt-on acquisitions in the security and "
                        "observability spaces, each in the $200-400 million range, which would be "
                        "funded from existing cash without impacting the buyback program."
                    ),
                },
                {
                    "speaker": "Analyst (Goldman Sachs)",
                    "text": (
                        "Can you comment on customer retention and net revenue retention in the "
                        "enterprise segment specifically?"
                    ),
                },
                {
                    "speaker": "CFO",
                    "text": (
                        "We do not disclose NRR as a standalone metric, but I can share that "
                        "gross revenue retention in our enterprise cohort remains above 95%, "
                        "consistent with last year. Expansion revenue from existing customers "
                        "contributed approximately 60% of total new revenue in Q2, up from 55% "
                        "a year ago. This shift toward land-and-expand economics is a key driver "
                        "of our improving unit economics and lower customer acquisition cost per "
                        "dollar of annual recurring revenue."
                    ),
                },
                {
                    "speaker": "Operator",
                    "text": "Thank you. This concludes Demo Corp's Q2 2024 Earnings Call.",
                },
            ],
        },
    ]


def build_financial_eval_dataset(docs: list[Document]) -> EvalDataset:
    """Build a domain-specific eval dataset from financial transcript documents.

    Queries target content that is not at the document start, so chunk boundary
    placement affects retrieval quality. Gold ``answer_spans`` are computed by
    searching for known phrases in the document body.
    """
    queries: list[EvalQuery] = []
    financial_anchors = [
        "operating margin",
        "revenue grew",
        "guidance",
        "year over year",
        "supply chain",
        "earnings per share",
        "free cash flow",
        "capital expenditure",
    ]
    questions: dict[str, str] = {
        "operating margin": "What is the operating margin mentioned in this transcript?",
        "revenue grew": "By how much did revenue grow according to this transcript?",
        "guidance": "What guidance did management provide?",
        "year over year": "What year-over-year change is described?",
        "supply chain": "What supply chain issue is discussed?",
        "earnings per share": "What were the earnings per share?",
        "free cash flow": "What was the free cash flow figure mentioned?",
        "capital expenditure": "What capital expenditure was reported?",
    }

    for doc in docs:
        content_lower = doc.content.lower()
        for phrase in financial_anchors:
            idx = content_lower.find(phrase)
            if idx == -1:
                continue
            span_start = max(0, idx - 50)
            span_end = min(len(doc.content), idx + len(phrase) + 100)
            gold_text = doc.content[span_start:span_end].strip()
            queries.append(
                EvalQuery(
                    id=f"q_{doc.id}_{phrase.replace(' ', '_')}",
                    question=questions[phrase],
                    document_id=doc.id,
                    answer_spans=[(span_start, span_end)],
                    reference_answer=gold_text or None,
                )
            )

    if not queries:
        from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs

        return trivial_dataset_for_docs(docs)

    return EvalDataset(name="financial_qa", queries=queries, source="user_provided")


def build_eval_dataset_for(
    docs: list[Document],
    llm_model: str | None,
    api_base: str | None,
    api_key: str | None,
) -> EvalDataset:
    """Prefer LLM-built eval data when ``llm_model`` is set; fall back to heuristic financial QA."""
    if llm_model:
        from chunktuner.eval.dataset_builder import DatasetBuilder

        builder = DatasetBuilder(
            llm_model=llm_model,
            llm_api_base=api_base,
            llm_api_key=api_key,
        )
        try:
            dataset = builder.build_from_docs(docs, max_queries=200, questions_per_doc=4)
            if len(dataset.queries) >= 10:
                return dataset
        except Exception as exc:
            logger.warning("LLM dataset generation failed (%s); falling back to heuristic.", exc)
    return build_financial_eval_dataset(docs)


def load_hf_transcripts(
    num: int,
    *,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Load up to ``num`` rows from ``kurry/sp500_earnings_transcripts``."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The `datasets` package is required for Hugging Face loading. "
            "Install with: uv pip install datasets\n"
            f"Import error: {e}"
        ) from e

    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train", streaming=streaming)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= num:
            break
        out.append(dict(row))
    return out


FINANCIAL_RAG_SCORE_WEIGHTS: dict[str, float] = {
    "token_recall": 0.40,
    "mrr": 0.30,
    "token_iou": 0.15,
    "duplication_ratio": -0.15,
}


def run_recommendation(
    docs: list[Document],
    *,
    use_case: UseCase,
    top_k: int,
    embedding_model: str | None,
    api_base: str | None,
    api_key: str | None,
    llm_model: str | None,
    no_baseline: bool,
    registry: StrategyRegistry,
    strategy_names: list[str],
    dataset: EvalDataset,
    parallel: bool,
    max_workers: int,
    financial_weights: bool,
) -> Recommendation:
    embed_fn: DummyEmbeddingFunction | LiteLLMEmbeddingFunction
    if embedding_model:
        embed_fn = LiteLLMEmbeddingFunction(
            embedding_model,
            api_base=api_base,
            api_key=api_key,
        )
    else:
        embed_fn = DummyEmbeddingFunction(profile_name="dummy/financial-example")

    has_reference_answers = any(q.reference_answer for q in dataset.queries)
    enable_gen = bool(llm_model and has_reference_answers)
    ev = Evaluator(
        embed_fn,
        top_k=top_k,
        enable_generation_metrics=enable_gen,
        llm_answer_model=llm_model,
        llm_api_base=api_base,
        llm_api_key=api_key,
    )
    if financial_weights:
        if use_case != "rag_qa":
            raise SystemExit("--financial-weights is only supported with --use-case rag_qa")
        scorer = ScoreCalculator(use_case, custom_weights=FINANCIAL_RAG_SCORE_WEIGHTS)
    else:
        scorer = ScoreCalculator(use_case)
    tuner = AutoTuner(registry, ev, scorer)
    param_grid = financial_param_grid_for(strategy_names, llm_model=llm_model)
    return tuner.recommend(
        docs,
        use_case,
        strategies=strategy_names,
        param_grid=param_grid,
        max_docs=len(docs),
        baseline=not no_baseline,
        dataset=dataset,
        parallel=parallel,
        max_workers=max_workers,
    )


def format_summary(rec: Recommendation) -> str:
    best = rec.best
    m = best.metrics
    strat_counts: dict[str, int] = {}
    for r in rec.ranked:
        strat_counts[r.strategy_name] = strat_counts.get(r.strategy_name, 0) + 1
    strat_summary = ", ".join(f"{k}×{v}" for k, v in sorted(strat_counts.items()))
    lines = [
        f"Evaluated configs by strategy: {strat_summary}",
        f"Best strategy: {best.strategy_name}",
        f"Params: {json.dumps(best.config.params, sort_keys=True)}",
        f"Score: {best.score:.4f}",
        f"token_recall={m.token_recall:.4f} token_iou={m.token_iou:.4f} mrr={m.mrr:.4f}",
    ]
    if m.recall_at_k:
        rk = ", ".join(f"@{k}={v:.3f}" for k, v in sorted(m.recall_at_k.items()))
        lines.append(f"recall_at_k: {rk}")
    lines.extend(
        [
            "",
            "Metric interpretation for financial RAG:",
            f"  token_recall={m.token_recall:.4f}  "
            "(fraction of gold answer tokens retrieved — primary signal for RAG QA)",
            f"  mrr={m.mrr:.4f}               "
            "(mean reciprocal rank — 1.0 means the best chunk is always retrieved first)",
            f"  token_iou={m.token_iou:.4f}    "
            "(overlap precision — low IOU means retrieved chunks contain too much noise)",
            f"  duplication_ratio={m.duplication_ratio:.4f}  "
            "(>0.3 disqualifies — heavily overlapping chunks corrupt generation)",
            f"  avg_chunk_length={m.avg_chunk_length:.1f} tokens",
        ]
    )
    return "\n".join(lines)


_LATE_CHUNKING_NOTE = (
    "¹ late_chunking currently uses fixed token windows internally. "
    "Per-token embedding model required for full late-pooling support."
)
_DISQUALIFIED_THRESHOLD = 0.3


def _params_label(result: Any) -> str:
    p = result.config.params
    name = result.strategy_name
    if name == "fixed_tokens":
        return f"{p.get('max_tokens', '?')} tok / {p.get('overlap_tokens', 0)} ov"
    if name == "recursive_character":
        default_seps = default_recursive_separators()
        is_finance = p.get("separators") not in (None, default_seps)
        suffix = " fin-sep" if is_finance else ""
        return f"{p.get('chunk_size_chars', '?')} chr / {p.get('chunk_overlap_chars', 0)} ov{suffix}"
    if name == "late_chunking":
        return f"{p.get('chunk_size_tokens', '?')} tok / {p.get('overlap_tokens', 0)} ov"
    if name == "semantic":
        return f"{p.get('max_tokens', '?')} tok / thr={p.get('similarity_threshold', 0):.2f}"
    if name == "agentic":
        return f"{p.get('model', '?')} / {p.get('max_propositions', '?')} props"
    return str(p)


def print_comparison_table(
    rec: Recommendation,
    docs: list[Document],
    *,
    show_generation_metrics: bool = False,
) -> None:
    """Print a rich ranked table of all evaluated configs."""
    baseline_score = rec.baseline.score if rec.baseline else None
    has_late = any(r.strategy_name == "late_chunking" for r in rec.ranked)
    has_disqualified = any(
        r.metrics.duplication_ratio > _DISQUALIFIED_THRESHOLD for r in rec.ranked
    )

    _console.print()
    _console.rule("[bold]Strategy Comparison — Financial Earnings Transcripts[/bold]")
    _console.print(
        f"  [dim]{len(docs)} doc{'s' if len(docs) != 1 else ''}  ·  "
        f"use-case: {rec.use_case}  ·  "
        f"embedding: {rec.embedding_profile}[/dim]"
    )
    _console.print()

    table = Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1))
    table.add_column("Rank", justify="right", style="dim", no_wrap=True)
    table.add_column("Strategy", style="bold", no_wrap=True)
    table.add_column("Params", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Recall", justify="right", style="dim")
    table.add_column("MRR", justify="right", style="dim")
    table.add_column("IOU", justify="right", style="dim")
    table.add_column("AvgTok", justify="right", style="dim")
    if show_generation_metrics:
        table.add_column("Faith", justify="right", style="dim")
        table.add_column("AnsRel", justify="right", style="dim")

    for rank, result in enumerate(rec.ranked, 1):
        m = result.metrics
        is_winner = rank == 1
        is_disqualified = m.duplication_ratio > _DISQUALIFIED_THRESHOLD
        is_late = result.strategy_name == "late_chunking"
        note = " ¹" if is_late else ""

        if is_disqualified:
            row_style = "red"
            rank_label = f"{rank} ✗"
        elif is_winner:
            row_style = "green"
            rank_label = f"{rank} ★"
        elif baseline_score is not None and result.score >= baseline_score:
            row_style = "green"
            rank_label = str(rank)
        else:
            row_style = ""
            rank_label = str(rank)

        row_cells: list[str] = [
            rank_label,
            result.strategy_name + note,
            _params_label(result),
            f"{result.score:.3f}",
            f"{m.token_recall:.3f}",
            f"{m.mrr:.3f}",
            f"{m.token_iou:.3f}",
            f"{m.avg_chunk_length:.0f}",
        ]
        if show_generation_metrics:
            row_cells.append(
                f"{m.faithfulness:.3f}" if m.faithfulness is not None else "—"
            )
            row_cells.append(
                f"{m.answer_relevancy:.3f}" if m.answer_relevancy is not None else "—"
            )
        table.add_row(*row_cells, style=row_style)

    _console.print(table)

    if rec.baseline:
        b = rec.baseline
        delta = rec.best.score - b.score
        pct = (delta / b.score * 100) if b.score else 0
        sign = "+" if delta >= 0 else ""
        _console.print(
            f"  Baseline  [dim]{b.strategy_name}  {_params_label(b)}[/dim]"
            f"  →  score [bold]{b.score:.3f}[/bold]"
        )
        _console.print(
            f"  Winner beats baseline by "
            f"[{'green' if delta >= 0 else 'red'}]{sign}{delta:.3f}  ({sign}{pct:.1f}%)[/]"
        )

    if has_disqualified:
        _console.print(
            "\n  [red]✗[/red]  Configs marked ✗ have duplication_ratio > 0.3 "
            "and are not recommended for production."
        )
    if has_late:
        _console.print(f"\n  [dim]{_LATE_CHUNKING_NOTE}[/dim]")

    _console.print()


def run_benchmark_cli(argv: list[str] | None = None) -> Recommendation:
    p = argparse.ArgumentParser(description="Financial transcript chunking benchmark (chunktuner).")
    p.add_argument(
        "--fixture",
        action="store_true",
        help="Use built-in synthetic transcripts (no HF).",
    )
    p.add_argument("--num-transcripts", type=int, default=50, metavar="N")
    p.add_argument(
        "--text-mode",
        choices=("raw", "structured", "structured_prefixed"),
        default="structured_prefixed",
        help="raw=verbatim content; structured=speaker turns; structured_prefixed=metadata + turns",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        metavar="C",
        help="Truncate each document body to C characters (default: no truncation).",
    )
    p.add_argument(
        "--use-case",
        default="rag_qa",
        choices=("rag_qa", "search", "summarization", "code_assist"),
        help="Scoring profile. rag_qa matches financial retrieval / QA intent.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help=(
            "If set, call LiteLLM embeddings (API cost). Default: DummyEmbeddingFunction. "
            "For Google Gemini use e.g. text-embedding-004 with GEMINI_API_KEY in a .env file."
        ),
    )
    p.add_argument(
        "--lm-studio",
        action="store_true",
        help="Use LM Studio default base URL http://localhost:1234/v1 (requires --embedding-model and --llm-model).",
    )
    p.add_argument(
        "--api-base",
        "--lm-studio-url",
        default=None,
        dest="lm_studio_url",
        metavar="URL",
        help="Base URL for any OpenAI-compatible server (LM Studio, Ollama, vLLM, Azure). --lm-studio-url is a legacy alias.",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help=(
            "LLM for agentic strategy, optional LLM dataset generation, and generation metrics. "
            "Examples: gemini/gemini-1.5-flash, claude-3-haiku-20240307, openai/llama-3.2-3b-instruct."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key for the provider (LiteLLM). Overrides env vars for custom api_base runs.",
    )
    p.add_argument("--no-baseline", action="store_true", help="Skip fixed_tokens baseline eval.")
    p.add_argument(
        "--strategies",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated strategy names (must be registered). "
            "Default: fixed_tokens,recursive_character (same spirit as chunk-tune recommend)."
        ),
    )
    p.add_argument(
        "--all-text-strategies",
        action="store_true",
        help=(
            "Run every strategy compatible with content_type=text (from build_full_registry). "
            "Excludes agentic unless --include-agentic (agentic calls an LLM)."
        ),
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help=(
            "Evaluate every strategy compatible with content_type=text. "
            "Equivalent to --all-text-strategies. Excludes agentic unless --include-agentic."
        ),
    )
    p.add_argument(
        "--include-agentic",
        action="store_true",
        help=(
            "With --all-text-strategies, also evaluate agentic (LiteLLM completion; costs tokens)."
        ),
    )
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write Recommendation JSON to this path.",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate strategies in parallel (multi-core).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        metavar="N",
        help="Worker count when --parallel is set (default: 4).",
    )
    p.add_argument(
        "--financial-weights",
        action="store_true",
        help=(
            "Use finance-oriented ScoreCalculator weights (requires --use-case rag_qa): "
            "higher token_recall and mrr, stronger duplication penalty."
        ),
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    example_dir = Path(__file__).resolve().parent
    load_dotenv(example_dir / ".env")
    load_dotenv()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    api_base: str | None = args.lm_studio_url
    if args.lm_studio and not api_base:
        api_base = "http://localhost:1234/v1"
    api_key: str | None = args.api_key
    if api_base and not api_key and args.lm_studio:
        api_key = "lm-studio"
    llm_model: str | None = args.llm_model

    if api_base:
        if not args.embedding_model:
            raise SystemExit(
                "--lm-studio / --api-base requires --embedding-model.\n"
                "Example: --embedding-model openai/nomic-embed-text-v1.5"
            )
        if not llm_model:
            raise SystemExit(
                "--lm-studio / --api-base requires --llm-model.\n"
                "Example: --llm-model openai/llama-3.2-3b-instruct"
            )

    if args.fixture:
        rows = synthetic_fixture_records()[: max(1, args.num_transcripts)]
    else:
        rows = load_hf_transcripts(max(1, args.num_transcripts), streaming=True)

    max_chars = args.max_chars
    docs = [
        transcript_record_to_document(r, text_mode=args.text_mode, max_chars=max_chars, index=i)
        for i, r in enumerate(rows)
    ]

    registry = build_full_registry()
    all_text = args.all_text_strategies or args.compare_all
    include_agentic = args.include_agentic or (
        api_base is not None and llm_model is not None
    )
    strategy_names = resolve_strategy_names(
        registry,
        strategies_csv=args.strategies,
        all_text=all_text,
        include_agentic=include_agentic,
    )
    if api_base and llm_model and "agentic" in strategy_names:
        try:
            from chunktuner.chunking.agentic import AgenticStrategy

            registry.register(AgenticStrategy(api_base=api_base, api_key=api_key))
        except ImportError:
            strategy_names = [n for n in strategy_names if n != "agentic"]
    allowed = set(registry.names(docs[0].content_type))
    strategy_names = [n for n in strategy_names if n in allowed]
    if not strategy_names:
        raise SystemExit(
            f"No strategies apply to content_type={docs[0].content_type!r}. "
            f"Requested (after filter) would be empty. Compatible: {sorted(allowed)}"
        )

    dataset = build_eval_dataset_for(docs, llm_model, api_base, api_key)
    rec = run_recommendation(
        docs,
        use_case=cast(UseCase, args.use_case),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        api_base=api_base,
        api_key=api_key,
        llm_model=llm_model,
        no_baseline=args.no_baseline,
        registry=registry,
        strategy_names=strategy_names,
        dataset=dataset,
        parallel=args.parallel,
        max_workers=max(1, args.max_workers),
        financial_weights=args.financial_weights,
    )

    if not args.quiet:
        print_strategy_availability(registry, strategy_names, docs[0].content_type)
        has_gen = any(r.metrics.faithfulness is not None for r in rec.ranked)
        print_comparison_table(rec, docs, show_generation_metrics=has_gen)

    if args.export:
        args.export.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {args.export}", file=sys.stderr)

    return rec


def main() -> None:
    run_benchmark_cli()


if __name__ == "__main__":
    main()
