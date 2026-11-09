"""Mechanical AST and naming conformance gates enforcing zero magic literals and universal abstractions."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ALLOWED_NUMERIC_IDENTITIES: frozenset[int | float] = frozenset({0, 1, -1, 0.0, 1.0})

FORBIDDEN_DOMAIN_PATTERN = re.compile(
    r"(real_estate|healthcare|fintech|mortgage|ecommerce|hospitality|parcel_permit|"
    r"insurance_claim|retail_checkout|clinical_patient|loan_underwriting|hotel_booking)",
    re.IGNORECASE,
)


class ZeroHardcodedValuesAstGate:
    """AST conformance gate rejecting numeric literals outside {0, 1, -1, 0.0, 1.0, True, False}."""

    @classmethod
    def scan_source(cls, source_text: str, label: str = "<memory>") -> list[dict[str, Any]]:
        tree = ast.parse(source_text, filename=label)
        violations: list[dict[str, Any]] = []
        parent_map: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent

        for node in ast.walk(tree):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                if (
                    isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, (int, float))
                    and not isinstance(node.operand.value, bool)
                ):
                    neg_val = -node.operand.value
                    if neg_val not in ALLOWED_NUMERIC_IDENTITIES:
                        violations.append(
                            {
                                "file": label,
                                "line": node.lineno,
                                "value": neg_val,
                            }
                        )
                    continue

            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                parent = parent_map.get(id(node))
                if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
                    continue
                if node.value not in ALLOWED_NUMERIC_IDENTITIES:
                    violations.append(
                        {
                            "file": label,
                            "line": node.lineno,
                            "value": node.value,
                        }
                    )
        return violations

    @classmethod
    def scan_file(cls, py_path: Path) -> list[dict[str, Any]]:
        return cls.scan_source(py_path.read_text(encoding="utf-8"), label=str(py_path))

    @classmethod
    def scan_directories(cls, directories: list[Path]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for directory in directories:
            if not directory.exists():
                continue
            for py_path in sorted(directory.rglob("*.py")):
                violations.extend(cls.scan_file(py_path))
        return violations


class DomainAgnosticNamingGate:
    """Mechanical naming gate rejecting industry-specific file, schema, class, or function names."""

    @classmethod
    def scan_workspace(cls, workspace_root: Path) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        target_dirs = (
            workspace_root / "config",
            workspace_root / "omninode_manifold",
            workspace_root / "components",
            workspace_root / "tests",
        )
        for folder in target_dirs:
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                if "__pycache__" in path.parts:
                    continue
                rel = str(path.relative_to(workspace_root))
                if FORBIDDEN_DOMAIN_PATTERN.search(rel):
                    violations.append({"kind": "path_name", "target": rel})
                if path.is_file() and path.suffix == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    schema_id = str(data.get("schema_id", ""))
                    if FORBIDDEN_DOMAIN_PATTERN.search(schema_id):
                        violations.append({"kind": "schema_id", "target": schema_id})
                if path.is_file() and path.suffix == ".py":
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            if FORBIDDEN_DOMAIN_PATTERN.search(node.name):
                                violations.append({"kind": "ast_symbol", "file": rel, "symbol": node.name})
        return violations
