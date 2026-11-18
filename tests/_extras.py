"""Optional-extra gating for tests.

``require_extra(name, module)`` behaves like ``pytest.importorskip`` unless
``name`` is listed in the comma-separated ``CHUNKTUNER_REQUIRE_EXTRAS`` env var,
in which case the import is mandatory and an ``ImportError`` fails the test
(or the module collection, when called at module level) instead of skipping.
"""

import importlib
import os
from types import ModuleType

import pytest

REQUIRE_EXTRAS_ENV = "CHUNKTUNER_REQUIRE_EXTRAS"


def required_extras() -> set[str]:
    raw = os.environ.get(REQUIRE_EXTRAS_ENV, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def require_extra(name: str, module: str, reason: str | None = None) -> ModuleType:
    """Import ``module`` from optional extra ``name``; skip or fail if unavailable."""
    if name in required_extras():
        try:
            return importlib.import_module(module)
        except ImportError as exc:
            pytest.fail(
                f"extra {name!r} is required ({REQUIRE_EXTRAS_ENV}) but "
                f"importing {module!r} failed: {exc}",
                pytrace=False,
            )
    return pytest.importorskip(module, reason=reason)
