"""Typer CLI entrypoint: ``chunk-tune``."""

from __future__ import annotations

import sys

import typer

from chunktuner.cli import (
    analyze_cmd,
    cache_cmd,
    compare_cmd,
    estimate_cmd,
    evaluate_cmd,
    init_cmd,
    preview_cmd,
    recommend_cmd,
)

app = typer.Typer(
    name="chunk-tune",
    help="Auto chunking tuner for RAG pipelines",
)


@app.callback(invoke_without_command=True)
def _root_callback(ctx: typer.Context) -> None:
    """Show root help when no subcommand is given; optional interactive tip on stderr."""
    if ctx.invoked_subcommand is not None:
        return
    if sys.stderr.isatty():
        typer.secho(
            "Tip: run chunk-tune estimate ./my_docs for a free token/cost estimate, "
            "or chunk-tune --help for all commands. "
            "Docs: https://shantanu-deshmukh.github.io/chunktuner/",
            dim=True,
            err=True,
        )
    typer.echo(ctx.get_help())
    raise typer.Exit(code=2)


init_cmd.register(app)
analyze_cmd.register(app)
estimate_cmd.register(app)
evaluate_cmd.register(app)
recommend_cmd.register(app)
preview_cmd.register(app)
compare_cmd.register(app)
cache_cmd.register(app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
