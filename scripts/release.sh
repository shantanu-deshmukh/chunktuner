#!/usr/bin/env bash
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.2.0
#
# What this does:
#   1. Validates the version and checks for conflicts
#   2. Updates pyproject.toml version and refreshes uv.lock
#   3. Promotes [Unreleased] → [version] in CHANGELOG.md
#   4. Commits, creates annotated tag, pushes both
#      → pushing the tag triggers .github/workflows/publish.yml (PyPI)
#   5. Creates a GitHub Release with the changelog notes

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Parse & validate ────────────────────────────────────────────────────────

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Error: version argument required"
  echo "Usage: $0 <version>   e.g. $0 0.2.0"
  exit 1
fi

# Strip leading 'v' if supplied
VERSION="${VERSION#v}"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.]+)?$ ]]; then
  echo "Error: '$VERSION' is not a valid semver (e.g. 1.2.3 or 1.2.3-rc1)"
  exit 1
fi

TAG="v$VERSION"

# Ensure working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: working tree has uncommitted changes — commit or stash them first"
  exit 1
fi

# Ensure we're on main (or allow override with ALLOW_BRANCH=1)
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "main" && -z "${ALLOW_BRANCH:-}" ]]; then
  echo "Error: must release from 'main' (current branch: $CURRENT_BRANCH)"
  echo "       set ALLOW_BRANCH=1 to override"
  exit 1
fi

# Check tag doesn't already exist locally or on remote
if git rev-parse "$TAG" &>/dev/null; then
  echo "Error: tag '$TAG' already exists locally"
  exit 1
fi
if git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
  echo "Error: tag '$TAG' already exists on remote"
  exit 1
fi

# Check uv is available (needed to refresh uv.lock after the version bump)
if ! command -v uv &>/dev/null; then
  echo "Error: 'uv' not found — install it to update uv.lock"
  exit 1
fi

# Check gh CLI is available (needed for GitHub Release)
if ! command -v gh &>/dev/null; then
  echo "Error: 'gh' CLI not found — install it to create GitHub Releases"
  exit 1
fi

echo "Releasing $TAG …"

# ── 2. Bump version in pyproject.toml ─────────────────────────────────────────

PYPROJECT="$REPO_ROOT/pyproject.toml"
OLD_VERSION="$(grep -E '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')"

if [[ -z "$OLD_VERSION" ]]; then
  echo "Error: could not detect current version in pyproject.toml"
  exit 1
fi

echo "  pyproject.toml: $OLD_VERSION → $VERSION"
sed -i.bak "s/^version = \"$OLD_VERSION\"/version = \"$VERSION\"/" "$PYPROJECT"
rm -f "$PYPROJECT.bak"

# Keep uv.lock in sync with the new version. Plain `uv lock` (no --upgrade) only
# updates the project's own version entry; it does not move dependency pins.
uv lock
echo "  uv.lock: chunktuner → $VERSION"

# ── 3. Promote [Unreleased] in CHANGELOG.md ───────────────────────────────────

CHANGELOG="$REPO_ROOT/CHANGELOG.md"
TODAY="$(date +%Y-%m-%d)"

if ! grep -q '## \[Unreleased\]' "$CHANGELOG"; then
  echo "Warning: no [Unreleased] section found in CHANGELOG.md — skipping promotion"
  RELEASE_NOTES="No changelog entry for this release."
else
  # Replace the first occurrence of "## [Unreleased]" with both the new version
  # header and a fresh empty [Unreleased] section above it.
  REPLACEMENT="## [Unreleased]\n\n## [$VERSION] - $TODAY"
  # Use perl for reliable multi-line-safe substitution on macOS and Linux
  perl -i -0pe "s/## \[Unreleased\]/## [Unreleased]\n\n## [$VERSION] - $TODAY/" "$CHANGELOG"
  echo "  CHANGELOG.md: promoted [Unreleased] → [$VERSION] - $TODAY"

  # Extract the notes for this version (lines between the new header and the next ## header)
  RELEASE_NOTES="$(awk "/^## \[$VERSION\]/{found=1; next} found && /^## /{exit} found{print}" "$CHANGELOG" | sed '/^[[:space:]]*$/d' | head -100)"
  if [[ -z "$RELEASE_NOTES" ]]; then
    RELEASE_NOTES="See [CHANGELOG.md](CHANGELOG.md) for details."
  fi
fi

# ── 4. Commit, tag, push ───────────────────────────────────────────────────────

git add "$PYPROJECT" "$CHANGELOG" "$REPO_ROOT/uv.lock"
git commit -m "chore: bump version to $VERSION"
echo "  committed version bump"

git tag -a "$TAG" -m "Release $TAG"
echo "  created tag $TAG"

git push origin main
git push origin "$TAG"
echo "  pushed main + $TAG  →  publish.yml will build & publish to PyPI"

# ── 5. GitHub Release ─────────────────────────────────────────────────────────

gh release create "$TAG" \
  --title "Release $TAG" \
  --notes "$RELEASE_NOTES"

echo ""
echo "Done! GitHub Release created for $TAG."
echo "PyPI publish is running in CI — check: https://github.com/shantanu-deshmukh/chunktuner/actions"
