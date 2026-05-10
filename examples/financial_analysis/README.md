# Financial earnings transcripts — chunking benchmark

This example benchmarks **chunktuner** on financial text — S&P 500 earnings call transcripts from [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) — and picks the best **(strategy, hyperparameters)** pair for RAG retrieval.

Earnings calls are a useful benchmark because they mix structured dialogue (operator introductions, management remarks, analyst Q&A) with dense financial figures — exactly the content where separator choice and chunk size make a measurable difference in retrieval quality.

**What you get from running this:**
- A ranked table showing how each chunking config performs on your corpus
- Retrieval metrics: token recall, MRR, token IOU, duplication ratio
- A recommended `(strategy, params)` pair you can drop into your production pipeline
- Optionally: generation metrics (faithfulness, answer relevancy) when `--llm-model` is set

## Setup

```bash
cd examples/financial_analysis
uv sync
```

`uv sync` creates a local virtual environment and installs `chunktuner` as an editable path dependency from the repo root alongside `datasets`.

---

## Quick start (no API key, no internet)

Uses two built-in synthetic transcripts — good for CI and offline checks:

```bash
uv run python run_benchmark.py --fixture --num-transcripts 2
```

---

## Choosing a run mode

There are two independent axes:

| | **No `--llm-model`** | **With `--llm-model`** |
|---|---|---|
| **What runs** | Chunking + embeddings only | Chunking + embeddings + generation metrics |
| **Speed** | Fast (seconds–minutes) | Slow — 1 LLM call per query × strategy config |
| **Extra metrics** | token_recall, MRR, IOU | + faithfulness, answer_relevancy |
| **Recommended for** | Comparing chunk strategies quickly | Final validation of the winning config |

> **Tip:** Use embeddings-only for exploration. Add `--llm-model` only when you want faithfulness/answer_relevancy on your shortlisted config.

---

## Provider setup

You need at minimum an **embedding model**. An LLM is optional.

### LM Studio (local, no API cost)

Start LM Studio, load your models, then start the server (Developer tab → Start Server, default port 1234).

Use the `openai/` prefix so LiteLLM routes through LM Studio's OpenAI-compatible endpoint. Find the exact model ID in LM Studio's API tab.

`run_benchmark.py` requires **both** `--embedding-model` and `--llm-model` whenever you set a custom OpenAI-compatible base URL (`--lm-studio` is a shortcut for `http://localhost:1234/v1`). You can still run **without** generation metrics by omitting `--llm-model` only when you are **not** using `--lm-studio` / `--lm-studio-url` (for example cloud embeddings with `OPENAI_API_KEY` and no custom base).

```bash
# Local LM Studio (embedding + LLM model ids required by the script):
uv run python run_benchmark.py \
  --fixture --num-transcripts 2 \
  --lm-studio \
  --embedding-model openai/<your-embedding-model-id> \
  --llm-model openai/<your-llm-model-id>

# Same, with explicit URL instead of --lm-studio:
uv run python run_benchmark.py \
  --fixture --num-transcripts 2 \
  --lm-studio-url http://localhost:1234/v1 \
  --embedding-model openai/<your-embedding-model-id> \
  --llm-model openai/<your-llm-model-id>
```

**Recommended local models:**

| Role | Good choices |
|------|-------------|
| Embeddings | `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`, `bge-small-en-v1.5` |
| LLM (fast) | `llama-3.2-3b-instruct`, `qwen2.5-3b-instruct` |
| LLM (quality) | `llama-3.1-8b-instruct`, `qwen2.5-7b-instruct`, `deepseek-r1-0528-qwen3-8b` |

> **Note on thinking models** (DeepSeek R1, QwQ, etc.): each LLM call takes 20–40 seconds on Apple Silicon. With generation metrics enabled, the benchmark makes ~1 LLM call per query × strategy config — plan for 20–30 minutes with 2 docs and 7 configs.

### Google Gemini

Create `examples/financial_analysis/.env`:

```bash
GEMINI_API_KEY=your-key-here
```

```bash
uv run python run_benchmark.py \
  --fixture --num-transcripts 2 \
  --embedding-model gemini/gemini-embedding-001 \
  --llm-model gemini/gemini-2.0-flash
```

> **Free tier limit:** 100 embedding requests/minute. Use `--fixture --num-transcripts 2` to stay within limits; scale up with a paid key.

### OpenAI

```bash
export OPENAI_API_KEY=your-key-here

uv run python run_benchmark.py \
  --fixture --num-transcripts 2 \
  --embedding-model text-embedding-3-small \
  --llm-model gpt-4o-mini
```

### Any OpenAI-compatible server (Ollama, vLLM, Azure, etc.)

```bash
uv run python run_benchmark.py \
  --fixture --num-transcripts 2 \
  --lm-studio-url http://localhost:11434/v1 \
  --embedding-model openai/<model-id> \
  --llm-model openai/<model-id>
```

`--lm-studio` sets the base URL to `http://localhost:1234/v1` and, when no `--api-key` is given, uses `api_key=lm-studio`. It still requires `--embedding-model` and `--llm-model` (see above).

---

## Run on live Hugging Face data

Streams transcripts directly from the ~33,000-row S&P 500 corpus (no full download):

```bash
# 10 real transcripts — LM Studio (script requires a chat model id too):
uv run python run_benchmark.py \
  --num-transcripts 10 \
  --lm-studio \
  --embedding-model openai/text-embedding-nomic-embed-text-v1.5 \
  --llm-model openai/llama-3.2-3b-instruct

# 50 transcripts (default) — Gemini:
uv run python run_benchmark.py \
  --embedding-model gemini/gemini-embedding-001
```

---

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--fixture` | off | Use 2 built-in synthetic transcripts instead of HF |
| `--num-transcripts N` | 50 | Number of HF transcripts to stream |
| `--max-chars C` | none | Truncate each document to C characters |
| `--text-mode` | `structured_prefixed` | `raw` / `structured` / `structured_prefixed` |
| `--use-case` | `rag_qa` | Scoring profile: `rag_qa` / `search` / `summarization` / `code_assist` |
| `--financial-weights` | off | Finance-tuned weights (higher recall/MRR, stronger dup penalty) |
| `--top-k N` | 5 | Chunks retrieved per query |
| `--embedding-model MODEL` | none (dummy) | LiteLLM embedding model |
| `--llm-model MODEL` | none | LiteLLM LLM — enables LLM dataset generation + generation metrics |
| `--lm-studio` | off | LM Studio at `http://localhost:1234/v1`; requires `--embedding-model` and `--llm-model` |
| `--lm-studio-url URL` | none | Custom OpenAI-compatible base URL; same requirements as `--lm-studio` |
| `--api-key KEY` | none | Override API key |
| `--strategies NAMES` | `fixed_tokens,recursive_character` | Comma-separated strategy names |
| `--all-text-strategies` | off | Run every text-compatible strategy |
| `--compare-all` | off | Same as `--all-text-strategies` |
| `--include-agentic` | off | Also run agentic strategy (LLM calls; costly) |
| `--parallel` | off | Evaluate strategy configs in parallel |
| `--max-workers N` | 4 | Worker count when `--parallel` is set |
| `--no-baseline` | off | Skip the `fixed_tokens` baseline run |
| `--export PATH` | none | Write full `Recommendation` JSON to file |
| `--quiet` | off | Suppress rich table output |

---

## Understanding the output

```
  Rank   Strategy              Params                  Score   Recall   MRR    IOU   AvgTok
 ────────────────────────────────────────────────────────────────────────────────────────
   1 ★   recursive_character   1024 chr / 154 ov        0.821    0.950  0.880  0.062      212
     2   fixed_tokens          512 tok / 51 ov           0.764    0.920  0.840  0.059      444
   ...
  Baseline  fixed_tokens  512 tok / 0 ov  →  score 0.682
  Winner beats baseline by +0.139  (+20.4%)
```

- **Score** — weighted composite from `ScoreCalculator(use_case)` (default `rag_qa` matches `chunktuner.config.score_profile_weights`: token_recall 0.45, mrr 0.30, token_iou 0.15, faithfulness 0.10, duplication_ratio −0.10). With `--financial-weights`, the example uses its own `rag_qa` weights (see `FINANCIAL_RAG_SCORE_WEIGHTS` in `run_benchmark.py`).
- **Recall** — fraction of gold answer tokens retrieved; primary signal for RAG QA
- **MRR** — mean reciprocal rank; 1.0 means the right chunk is always retrieved first
- **IOU** — overlap precision; low IOU means retrieved chunks contain too much noise
- **AvgTok** — average chunk length in tokens
- Configs marked **✗** have `duplication_ratio > 0.3` and are not recommended

When `--llm-model` is set, two extra columns appear:
- **Faith** — faithfulness of generated answers to the retrieved context
- **AnsRel** — relevancy of generated answers to the question

---

## Export and reload (production handoff)

```bash
uv run python run_benchmark.py \
    --fixture \
    --lm-studio \
    --embedding-model openai/text-embedding-nomic-embed-text-v1.5 \
    --llm-model openai/llama-3.2-3b-instruct \
    --export best_config.json
```

Reload in your own ingest pipeline:

```python
from pathlib import Path
from chunktuner.chunking import build_full_registry
from chunktuner.models import ChunkConfig, Recommendation

rec = Recommendation.model_validate_json(Path("best_config.json").read_text(encoding="utf-8"))
print(rec.best.strategy_name, rec.best.config.params)

registry = build_full_registry()
strategy = registry.get(rec.best.strategy_name)
chunks = strategy.chunk(doc, ChunkConfig(name=rec.best.strategy_name, params=rec.best.config.params))
```

---

## Extending the benchmark

`build_full_registry()` also registers `semantic` and `late_chunking`. To include them:

```bash
# All text-compatible strategies (excludes agentic by default):
uv run python run_benchmark.py --fixture --lm-studio \
  --embedding-model openai/text-embedding-nomic-embed-text-v1.5 \
  --llm-model openai/llama-3.2-3b-instruct \
  --compare-all

# Add semantic (requires semchunk extra):
uv sync --extra semantic
uv run python run_benchmark.py --fixture --compare-all \
  --lm-studio \
  --embedding-model openai/text-embedding-nomic-embed-text-v1.5 \
  --llm-model openai/llama-3.2-3b-instruct
```

| Strategy | Default run | `--compare-all` | Notes |
|----------|-------------|-----------------|-------|
| `fixed_tokens` | Yes | Yes | Baseline-friendly token windows |
| `recursive_character` | Yes | Yes | Core demo; finance-aware separator variant |
| `late_chunking` | No | Yes | Per-token embedding model needed for full late pooling |
| `semantic` | No | Yes (if semchunk installed) | `uv sync --extra semantic` |
| `agentic` | No | With `--include-agentic` | LLM cost; use sparingly |

---

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Load data → `Document` → domain `EvalDataset` → `AutoTuner.recommend()` → optional JSON export |
| `pyproject.toml` | uv project config; pins `chunktuner` as an editable path dep and `datasets` |

## CI

`tests/integration/test_financial_example.py` exercises `--fixture` mode so releases do not break this workflow without downloading HF data.
