"""Construct a ``StrategyRegistry`` with all built-in strategies (best-effort optional deps)."""

import logging

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy
from chunktuner.chunking.registry import StrategyRegistry

_log = logging.getLogger(__name__)


def build_full_registry(encoding: str = "cl100k_base") -> StrategyRegistry:
    reg = StrategyRegistry()
    reg.register(FixedTokenStrategy(encoding_name=encoding))
    reg.register(RecursiveCharacterStrategy(encoding_name=encoding))
    try:
        from chunktuner.chunking.semantic import SemanticStrategy

        reg.register(SemanticStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "semantic", e)
    try:
        from chunktuner.chunking.markdown_semantic import MarkdownSemanticStrategy

        reg.register(MarkdownSemanticStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "markdown_semantic", e)
    try:
        from chunktuner.chunking.pdf_structural import PdfStructuralStrategy

        reg.register(PdfStructuralStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "pdf_structural", e)
    try:
        from chunktuner.chunking.structural_semantic import StructuralSemanticStrategy

        reg.register(StructuralSemanticStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "structural_semantic", e)
    try:
        from chunktuner.chunking.late_chunking import LateChunkingStrategy

        reg.register(LateChunkingStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "late_chunking", e)
    try:
        from chunktuner.chunking.code_ast import CodeASTStrategy

        reg.register(CodeASTStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "code_ast", e)
    try:
        from chunktuner.chunking.code_window import CodeWindowStrategy

        reg.register(CodeWindowStrategy(encoding_name=encoding))
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "code_window", e)
    try:
        from chunktuner.chunking.agentic import AgenticStrategy

        reg.register(AgenticStrategy())
    except ImportError as e:
        _log.debug("Strategy %r unavailable (missing dep): %s", "agentic", e)
    return reg
