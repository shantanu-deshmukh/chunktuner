"""Generate virtual ``docs/api/*.md`` and ``SUMMARY.md`` via mkdocs-gen-files."""

from pathlib import Path

import mkdocs_gen_files

NAV_FILE = "SUMMARY.md"

nav = mkdocs_gen_files.Nav()
nav["Home"] = "index.md"
nav["Quickstart"] = "quickstart.md"
nav["Guides", "Strategy guide"] = "strategy_guide.md"
nav["Guides", "Metrics reference"] = "metrics.md"
nav["Guides", "Configuration"] = "configuration.md"
nav["Guides", "MCP setup"] = "mcp_setup.md"
nav["CLI reference"] = "cli_reference.md"
nav["Python API"] = "python_api.md"

# (module_dotted_path, source_file_for_edit_link)
PUBLIC_MODULES = [
    ("models", "src/chunktuner/models.py"),
    ("chunking", "src/chunktuner/chunking/__init__.py"),
    ("chunking.fixed_tokens", "src/chunktuner/chunking/fixed_tokens.py"),
    ("chunking.recursive_character", "src/chunktuner/chunking/recursive_character.py"),
    ("chunking.paragraph_group", "src/chunktuner/chunking/paragraph_group.py"),
    ("chunking.semantic", "src/chunktuner/chunking/semantic.py"),
    ("chunking.markdown_semantic", "src/chunktuner/chunking/markdown_semantic.py"),
    ("chunking.pdf_structural", "src/chunktuner/chunking/pdf_structural.py"),
    ("chunking.structural_semantic", "src/chunktuner/chunking/structural_semantic.py"),
    ("chunking.late_chunking", "src/chunktuner/chunking/late_chunking.py"),
    ("chunking.agentic", "src/chunktuner/chunking/agentic.py"),
    ("chunking.code_ast", "src/chunktuner/chunking/code_ast.py"),
    ("chunking.code_window", "src/chunktuner/chunking/code_window.py"),
    ("eval.evaluator", "src/chunktuner/eval/evaluator.py"),
    ("eval.score_calculator", "src/chunktuner/eval/score_calculator.py"),
    ("eval.dataset_builder", "src/chunktuner/eval/dataset_builder.py"),
    ("eval.cost_estimator", "src/chunktuner/eval/cost_estimator.py"),
    ("eval.embeddings", "src/chunktuner/eval/embeddings.py"),
    ("eval.ragas_bridge", "src/chunktuner/eval/ragas_bridge.py"),
    ("eval.lexical", "src/chunktuner/eval/lexical.py"),
    ("eval.rerankers", "src/chunktuner/eval/rerankers.py"),
    ("ingestion.file_ingestor", "src/chunktuner/ingestion/file_ingestor.py"),
    ("ingestion.url_ingestor", "src/chunktuner/ingestion/url_ingestor.py"),
    ("ingestion.repo_ingestor", "src/chunktuner/ingestion/repo_ingestor.py"),
    ("tuner.auto_tuner", "src/chunktuner/tuner/auto_tuner.py"),
    ("cache.embedding_cache", "src/chunktuner/cache/embedding_cache.py"),
    ("cache.chunk_cache", "src/chunktuner/cache/chunk_cache.py"),
    ("cache.wrapped_embeddings", "src/chunktuner/cache/wrapped_embeddings.py"),
]

overview_path = Path("api/index.md")
with mkdocs_gen_files.open(overview_path, "w") as f:
    f.write("# API reference\n\n")
    f.write(
        "Module-level API generated from docstrings. "
        "For usage patterns, see [Python API](../python_api.md).\n\n"
    )
    for module_path, _src in PUBLIC_MODULES:
        rel = f"{module_path.replace('.', '/')}.md"
        f.write(f"- [`chunktuner.{module_path}`]({rel})\n")
mkdocs_gen_files.set_edit_path(overview_path, "scripts/gen_api_pages.py")
nav["API reference", "Overview"] = "api/index.md"

for module_path, src_path in PUBLIC_MODULES:
    doc_path = Path("api") / f"{module_path.replace('.', '/')}.md"
    parts = ("API reference", *module_path.split("."))
    nav[parts] = str(doc_path)
    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = f"chunktuner.{module_path}"
        f.write(f"# `{module_path}`\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(doc_path, src_path)

with mkdocs_gen_files.open(NAV_FILE, "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
