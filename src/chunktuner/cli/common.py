"""Shared CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

console = Console()

_VALID_FORMATS = frozenset({"json", "yaml", "table"})


def load_workspace_path(config: Path | None) -> Path | None:
    if config is not None:
        return config
    p = Path.cwd() / ".autochunk.yaml"
    return p if p.is_file() else None


def emit_output(data: Any, fmt: str) -> None:
    if fmt not in _VALID_FORMATS:
        raise typer.BadParameter(
            f"Unknown output format {fmt!r}. Valid choices: {sorted(_VALID_FORMATS)}",
            param_hint="'--output-format'",
        )
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        payload = [x.model_dump(mode="json") for x in data]
    else:
        payload = data

    if fmt == "json":
        console.print(json.dumps(payload, indent=2, default=str))
    elif fmt == "yaml":
        console.print(yaml.safe_dump(payload, sort_keys=False))
    else:
        _emit_table(payload)


def _emit_table(payload: Any) -> None:
    if isinstance(payload, dict) and "best" in payload:
        rec = payload
        console.print("[bold]Best configuration[/bold]")
        console.print_json(json.dumps(rec.get("best"), indent=2))
        return
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys = list(payload[0].keys())
        table = Table(show_header=True)
        for k in keys:
            table.add_column(k)
        for row in payload[:50]:
            table.add_row(*[str(row.get(k, "")) for k in keys])
        console.print(table)
        return
    console.print(str(payload))
