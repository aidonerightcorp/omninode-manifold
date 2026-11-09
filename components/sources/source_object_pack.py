"""Portable read-only JSON object pack source CARD driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CARD = {
    "name": "source_object_pack",
    "role": "source",
    "source_kind": "object_pack",
    "summary": "Reads immutable CARD bricks from a portable read-only JSON object pack.",
    "capabilities": ["source_object_pack", "portable_pack"],
}


def build_object_pack(source_folder: Path, pack_path: Path) -> Path:
    """Serialize all .py CARD files in source_folder into an immutable JSON object pack."""
    objects: list[dict[str, str]] = []
    for py_file in sorted(source_folder.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        dig = hashlib.sha256(text.encode("utf-8")).hexdigest()
        objects.append(
            {
                "name": py_file.stem,
                "locator": py_file.name,
                "sha256": dig,
                "source_text": text,
            }
        )
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(json.dumps({"schema": "omninode.object_pack.v1", "objects": objects}, sort_keys=True), encoding="utf-8")
    return pack_path


def list_entries(src_spec: dict[str, Any], _registry: Any) -> list[dict[str, Any]]:
    pack_p = Path(str(src_spec["path"]))
    if not pack_p.exists():
        return []
    data = json.loads(pack_p.read_text(encoding="utf-8"))
    return [
        {
            "name": str(obj["name"]),
            "locator": str(obj["locator"]),
            "source_text": str(obj["source_text"]),
        }
        for obj in list(data.get("objects", []))
    ]


def read_entry(src_spec: dict[str, Any], locator: str, comp_name: str, _registry: Any) -> str | None:
    for item in list_entries(src_spec, _registry):
        if item["locator"] == locator or item["name"] == comp_name:
            return str(item["source_text"])
    return None
