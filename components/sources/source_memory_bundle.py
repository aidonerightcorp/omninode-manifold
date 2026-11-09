"""Portable archive / git-bundle snapshot source CARD driver."""

from __future__ import annotations

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

CARD = {
    "name": "source_memory_bundle",
    "role": "source",
    "source_kind": "bundle",
    "summary": "Reads CARD bricks from a git bundle or deterministic tar archive snapshot.",
    "capabilities": ["source_bundle", "git_bundle_snapshot", "archive_snapshot"],
}


def create_bundle_snapshot(source_folder: Path, bundle_path: Path, use_git_bundle: bool = True) -> Path:
    """Package all .py CARD files from source_folder into a git bundle or tar snapshot."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if use_git_bundle:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            for py_file in sorted(source_folder.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                (repo / py_file.name).write_text(py_file.read_text(encoding="utf-8"), encoding="utf-8")
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "OmniNode Bundle"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "omninode-local"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "snapshot"], cwd=repo, check=True)
            subprocess.run(["git", "bundle", "create", str(bundle_path.resolve()), "main"], cwd=repo, check=True, capture_output=True)
        return bundle_path

    with tarfile.open(bundle_path, "w:gz") as tar_h:
        for py_file in sorted(source_folder.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            data = py_file.read_bytes()
            info = tarfile.TarInfo(name=py_file.name)
            info.size = len(data)
            tar_h.addfile(info, io.BytesIO(data))
    return bundle_path


def _extract_bundle_files(bundle_path: Path, timeout_s: int) -> dict[str, str]:
    if not bundle_path.exists():
        return {}
    if tarfile.is_tarfile(bundle_path):
        out: dict[str, str] = {}
        with tarfile.open(bundle_path, "r:*") as tar_h:
            for member in tar_h.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    extracted = tar_h.extractfile(member)
                    if extracted is not None:
                        out[Path(member.name).name] = extracted.read().decode("utf-8")
        return out

    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_dir = Path(tmp_dir) / "unpacked"
        subprocess.run(
            ["git", "clone", "-q", str(bundle_path.resolve()), str(clone_dir)],
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
        out_git: dict[str, str] = {}
        for py_file in sorted(clone_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                out_git[py_file.name] = py_file.read_text(encoding="utf-8")
        return out_git


def list_entries(src_spec: dict[str, Any], registry: Any) -> list[dict[str, Any]]:
    timeout_s = int(registry.compiler_grid["compiler_limits"]["subprocess_timeout_sec"])
    files = _extract_bundle_files(Path(str(src_spec["path"])), timeout_s)
    return [
        {"name": Path(fname).stem, "locator": fname, "source_text": content}
        for fname, content in sorted(files.items())
    ]


def read_entry(src_spec: dict[str, Any], locator: str, comp_name: str, registry: Any) -> str | None:
    timeout_s = int(registry.compiler_grid["compiler_limits"]["subprocess_timeout_sec"])
    files = _extract_bundle_files(Path(str(src_spec["path"])), timeout_s)
    if locator in files:
        return files[locator]
    alt = f"{comp_name}.py"
    return files.get(alt)
