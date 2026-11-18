#!/usr/bin/env bash
# Test chunktuner against the NEWEST dependency versions a user would get today.
#
# uv.lock pins versions for CI and development, but `pip install chunktuner` ignores
# it and resolves the newest versions allowed by pyproject.toml. A breaking upstream
# release (e.g. typer 0.29 removing `click_type`, which broke 0.2.0) is invisible to
# locked tests, so this script installs the package the way users do, into throwaway
# venvs, without the lockfile:
#   1. core   — `chunktuner` with no extras: the CLI must start and run `recommend`.
#   2. extras — `chunktuner[semantic,code,ragas,mcp]`: the full test suite must pass.
# (docling is excluded: it is heavy and caps several dependencies at older versions.)
#
# Usage: ./scripts/check_latest_deps.sh            # uses PYTHON=3.12 by default
#        PYTHON=3.14 ./scripts/check_latest_deps.sh
# Used by CI (job `test-latest-deps`) and the pre-push hook in .pre-commit-config.yaml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-3.12}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/chunktuner-latest-deps.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Never call real LLM/embedding APIs from this check.
unset OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY AZURE_OPENAI_API_KEY \
  CHUNKTUNER_API_KEY CHUNKTUNER_API_BASE || true

key_versions() {
  uv pip list --python "$1" 2>/dev/null \
    | grep -Ei '^(typer|click|pydantic|litellm|mcp|semchunk|tree-sitter|ragas|datasets|numpy) ' \
    | tr -s ' ' | tr '\n' ' ' || true
  echo
}

echo "== [core] chunktuner (no extras), newest deps, Python $PYTHON"
uv venv -q --python "$PYTHON" "$WORK/core"
uv pip install -q --python "$WORK/core" --refresh "."
echo "   resolved: $(key_versions "$WORK/core")"
"$WORK/core/bin/chunk-tune" --help > /dev/null
"$WORK/core/bin/chunk-tune" recommend tests/fixtures/golden/prose.txt \
  --yes --output-format json > /dev/null
echo "   CLI OK"

echo "== [extras] chunktuner[semantic,code,ragas,mcp] + dev group, newest deps, Python $PYTHON"
uv venv -q --python "$PYTHON" "$WORK/extras"
uv pip install -q --python "$WORK/extras" --refresh ".[semantic,code,ragas,mcp]" --group dev
echo "   resolved: $(key_versions "$WORK/extras")"
"$WORK/extras/bin/chunk-tune" --help > /dev/null
CHUNKTUNER_REQUIRE_EXTRAS=semantic,code,ragas,mcp \
  "$WORK/extras/bin/python" -m pytest -q -p no:cacheprovider

echo "== latest-deps check passed"
