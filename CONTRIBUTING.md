# Contributing to chunktuner

## Dev setup

```bash
git clone https://github.com/shantanu-deshmukh/chunktuner.git
cd chunktuner
uv sync --all-extras --dev
uv run pytest
```

## Adding a new chunking strategy

1. Implement the `ChunkingStrategy` protocol in `src/chunktuner/chunking/<name>.py`.
2. Register the strategy in `src/chunktuner/chunking/bootstrap.py` (and `__init__.py` if exported).
3. Add an offset invariant test (see `tests/unit/test_chunking_offsets.py`).
4. Update `docs/strategy_guide.md`.

## Running tests

```bash
uv run pytest
uv run pytest tests/unit/
uv run pytest -k test_offset
```

## Code style

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Pull request checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check src/ tests/` is clean
- [ ] New or changed strategy: offset invariant test added
- [ ] New public API: type hints present
- [ ] `CHANGELOG.md` updated under `[Unreleased]` when behaviour is user-visible
- [ ] `docs/` updated if behaviour or configuration changed

See `.github/pull_request_template.md` for the same checklist in the PR UI.
