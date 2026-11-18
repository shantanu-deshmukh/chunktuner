"""Regenerate ``baseline.json``. Only run on purpose, after reviewing a behaviour change.

Usage: uv run python -m tests.golden.generate_baseline
"""

import json
from pathlib import Path

from tests.golden.snapshot import build_snapshot

BASELINE = Path(__file__).resolve().parent / "baseline.json"


def main() -> None:
    snap = build_snapshot()
    BASELINE.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE} ({len(snap['chunks'])} chunk sets, {len(snap['eval'])} evals)")


if __name__ == "__main__":
    main()
