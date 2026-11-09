"""Command-line interface for OmniNode Manifold."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from omninode_manifold.benchmark_suite import run_full_benchmark_suite
from omninode_manifold.gates import DomainAgnosticNamingGate, ZeroHardcodedValuesAstGate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omninode-manifold", description="OmniNode Manifold CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-gates", help="Run AST magic-literal and domain-agnostic naming gates")
    sub.add_parser("benchmark", help="Run full empirical OmniNode benchmark suite")

    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.command == "check-gates":
        ast_v = ZeroHardcodedValuesAstGate.scan_directories(
            [root / "omninode_manifold", root / "components", root / "tests"]
        )
        nam_v = DomainAgnosticNamingGate.scan_workspace(root)
        print(json.dumps({"ast_violations": ast_v, "naming_violations": nam_v}))
        return 1 if (ast_v or nam_v) else 0

    if args.command == "benchmark":
        rep = run_full_benchmark_suite(root)
        print(json.dumps(rep["conformance_gates"]))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
