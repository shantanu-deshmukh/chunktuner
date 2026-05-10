# CLI reference

All commands are provided by the `chunk-tune` Typer app. Global patterns:

- **`path`**: file or directory to ingest (must exist).
- **`--config`**: optional path to `.autochunk.yaml` (see [Configuration](configuration.md)).
- **`--output-format`**: `table` (default), `json`, or `yaml` where supported.

---

## `chunk-tune init`

Create `.autochunk.yaml` in the **current working directory** with defaults for provider and embedding model.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--provider` | string | `openai` | Provider label stored in config |
| `-m`, `--model` | string | `text-embedding-3-small` | Default embedding model id |

```bash
chunk-tune init --provider openai -m text-embedding-3-small
```

```text
Wrote .autochunk.yaml
```

Fails with exit code `1` if `.autochunk.yaml` already exists.

---

## `chunk-tune analyze`

Structural scan using tiktoken and heuristics — **no embeddings, no external APIs**.

| Argument / flag | Type | Default | Description |
|-----------------|------|---------|-------------|
| `path` | path | (required) | File or directory |
| `--content-type` | string | auto | Override detected type |
| `--output-format` | choice | `table` | `table`, `json`, or `yaml` |

```bash
chunk-tune analyze ./README.md
```

```text
path: README.md
sample_path: README.md
content_type: markdown
token_count: 412
...
heuristic_starting_strategy: recursive_character (~1600 chars)
```

`heuristic_starting_strategy` is chosen from the sampled text: many Markdown headers → `markdown_semantic (when available)`; several fenced code blocks → `code_ast (when available)`; otherwise `recursive_character (~1600 chars)`.

For directories, the first `.md` / `.txt` file under the tree is sampled.

---

## `chunk-tune estimate`

Dry-run **token, cost, and wall-time** estimates using `CostEstimator` — no API calls.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `path` | path | (required) | File or directory |
| `--strategies` | string | `fixed_tokens,recursive_character` | Comma-separated strategy names |
| `--use-case` | string | `rag_qa` | Scoring profile label (for consistency with other commands) |
| `--max-docs` | int | `100` | Cap documents after ingest |
| `--config` | path | none | `.autochunk.yaml` |
| `--output-format` | choice | `table` | `table`, `json`, or `yaml` |

```bash
chunk-tune estimate ./docs --strategies fixed_tokens,recursive_character
```

```text
Total embedding tokens (est.): ...
Embedding cost (USD est.): ...
LLM dataset cost (USD est.): ...
Wall time (min est.): ...
Strategy-config combinations: ...
```

---

## `chunk-tune evaluate`

Runs the evaluator for each strategy × default param grid. Uses **dummy embeddings** only when both `--embedding-model` is omitted **and** `embedding_model` in the workspace YAML (from `--config` or cwd `.autochunk.yaml`) is empty or unset; otherwise `LiteLLMEmbeddingFunction` is used and you are prompted unless `--yes`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `path` | path | (required) | File or directory |
| `--strategies` | string | `fixed_tokens,recursive_character` | Comma-separated names |
| `--use-case` | string | `rag_qa` | `rag_qa`, `search`, `summarization`, `code_assist` |
| `--max-docs` | int | `20` | Max documents |
| `--top-k` | int | `5` | Retrieval depth (or workspace default) |
| `--config` | path | none | Workspace YAML |
| `--output-format` | choice | `table` | `table`, `json`, or `yaml` |
| `--embedding-model` | string | none | If set, overrides workspace `embedding_model`; a non-empty resolved model uses LiteLLM (prompts unless `--yes`) |
| `--api-base` | string | none | OpenAI-compatible API base (overrides workspace / `CHUNKTUNER_API_BASE`) |
| `--api-key` | string | none | API key for embeddings/LLM (overrides workspace / `CHUNKTUNER_API_KEY`) |
| `--llm-model` | string | none | LiteLLM id for generation paths; default is workspace `llm_model` |
| `--yes` | bool | false | Skip confirmation when using real embeddings |

```bash
chunk-tune evaluate ./samples --output-format json
```

```text
fixed_tokens {'max_tokens': 256, ...} score=0.1234 recall=0.100 iou=0.050
...
```

---

## `chunk-tune recommend`

Full **AutoTuner** grid over strategies and default (or filtered) param grids; prints best config. Embedding resolution matches `evaluate`: **`--embedding-model` else workspace `embedding_model`**; dummy only when the resolved value is empty.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `path` | path | (required) | File or directory |
| `--strategies` | string | `fixed_tokens,recursive_character` | Comma-separated names |
| `--use-case` | string | `rag_qa` | Use case for scoring |
| `--max-docs` | int | `20` | Max documents |
| `--top-k` | int | `5` | Evaluator top-k |
| `--config` | path | none | Workspace YAML |
| `--output-format` | choice | `table` | `table`, `json`, or `yaml` |
| `--embedding-model` | string | none | Overrides workspace `embedding_model` (see `evaluate`) |
| `--api-base` | string | none | Same as `evaluate` |
| `--api-key` | string | none | Same as `evaluate` |
| `--llm-model` | string | none | Same as `evaluate` |
| `--yes` | bool | false | Skip paid-call confirmation |
| `--no-baseline` | bool | false | Skip fixed-token baseline run |

```bash
chunk-tune recommend ./samples --no-baseline
```

---

## `chunk-tune compare`

Side-by-side comparison using the **first** default param set per strategy. Rich table to stdout; optional Markdown report. Embedding resolution matches `evaluate`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `path` | path | (required) | File or directory |
| `--strategies` | string | **required** | Comma-separated names |
| `--use-case` | string | `rag_qa` | Use case |
| `--max-docs` | int | `15` | Max documents |
| `--top-k` | int | `5` | Evaluator top-k |
| `--config` | path | none | Workspace YAML |
| `--report` | path | none | Write Markdown comparison table |
| `--embedding-model` | string | none | Overrides workspace `embedding_model` (see `evaluate`) |
| `--api-base` | string | none | Same as `evaluate` |
| `--api-key` | string | none | Same as `evaluate` |
| `--llm-model` | string | none | Same as `evaluate` |
| `--yes` | bool | false | Skip confirmation |

```bash
chunk-tune compare ./samples --strategies fixed_tokens,recursive_character --report cmp.md
```

---

## `chunk-tune preview`

Chunk a **single** document from a file path or inline string — no embeddings.

| Argument / flag | Type | Default | Description |
|-----------------|------|---------|-------------|
| `target` | string | (required) | File path or raw text |
| `-s`, `--strategy` | string | **required** | Registered strategy name |
| `--params` | JSON string | `{}` | Strategy params, e.g. `'{"max_tokens":256}'` |
| `--output-format` | choice | `table` | `table`, `json`, or `yaml` |

```bash
chunk-tune preview ./README.md -s recursive_character --params '{"max_chunk_chars":1200}'
```

```text
[0:1150] tok=280 '## Overview\n\n...'
```

---

## `chunk-tune cache`

SQLite-backed embedding and chunk caches. Subcommands:

### `chunk-tune cache stats`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | path | none | Workspace YAML (for `cache_dir`) |
| `--model` | string | *(unset)* | Embedding model key; defaults to workspace `embedding_model`, else `text-embedding-3-small` |

```bash
chunk-tune cache stats
```

```text
Database: /Users/you/.cache/chunktuner/embeddings.sqlite
Embeddings: ...
Chunks: ...
```

### `chunk-tune cache clear`

Same flags as `stats`. Clears embedding and chunk rows for the resolved database.

```bash
chunk-tune cache clear --model text-embedding-3-small
```

```text
Caches cleared.
```
