"""Filesystem directory source CARD driver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CARD = {
    "name": "source_folder",
    "role": "source",
    "source_kind": "folder",
    "summary": "Reads self-describing Python CARD bricks from a local directory tree.",
    "capabilities": ["source_folder", "filesystem_catalog"],
}


def list_entries(src_spec: dict[str, Any], _registry: Any) -> list[dict[str, Any]]:
    folder = Path(str(src_spec["path"]))
    if not folder.exists():
        return []
    entries: list[dict[str, Any]] = []
    for py_path in sorted(folder.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        entries.append(
            {
                "name": py_path.stem,
                "locator": py_path.name,
                "source_text": py_path.read_text(encoding="utf-8"),
            }
        )
    return entries


def read_entry(src_spec: dict[str, Any], locator: str, comp_name: str, _registry: Any) -> str | None:
    folder = Path(str(src_spec["path"]))
    if not folder.exists():
        return None
    direct = folder / locator
    if direct.is_file():
        return direct.read_text(encoding="utf-8")
    fallback = folder / f"{comp_name}.py"
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8")
    return None
