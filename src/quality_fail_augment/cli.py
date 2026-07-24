from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .generator import generate, verify_dataset
from .planner import audit_raw, create_plan


def _config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="quality-fail-augment")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-raw", help="Read-only raw-root compatibility audit")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="Scan raw data and freeze a deterministic plan")
    plan.add_argument("--raw-root", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    make = commands.add_parser("generate", help="Generate from an approved plan")
    make.add_argument("--raw-root", type=Path, required=True)
    make.add_argument("--config", type=Path, required=True)
    make.add_argument("--plan", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--limit-per-modality", type=int)
    make.add_argument("--resume", action="store_true")
    make.add_argument(
        "--trust-plan",
        action="store_true",
        help="Skip the full raw re-scan and fingerprint recheck; still verifies each "
        "planned source by SHA-256",
    )
    check = commands.add_parser("verify", help="Verify an existing generated dataset")
    check.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit-raw":
        result = audit_raw(
            args.raw_root.resolve(), _config(args.config), args.output.resolve()
        )
    elif args.command == "plan":
        result = create_plan(args.raw_root.resolve(), _config(args.config), args.output.resolve())
    elif args.command == "generate":
        result = generate(args.raw_root.resolve(), _config(args.config), args.plan.resolve(), args.output.resolve(), args.limit_per_modality, args.resume, args.trust_plan)
    else:
        verify_dataset(args.output.resolve())
        result = {"verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
