from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .augment import CT_CASES, RGB_CASES
from .generator import generate, verify_dataset
from .planner import audit_raw, create_plan
from .rgb_testset import extract_rgb_testset


def _config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _case_set(value: str) -> set[str]:
    cases = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(cases - set(CT_CASES) - set(RGB_CASES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown failure case: {', '.join(unknown)}")
    return cases


_REUSE_SCAN_HELP = (
    "Reuse a previous run's scan_cache.csv (or a directory containing one) instead of "
    "re-validating every raw pair. Pairs whose image and JSON still have the same size "
    "and mtime are taken from the cache; everything else is validated normally"
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="quality-fail-augment")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-raw", help="Read-only raw-root compatibility audit")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--reuse-scan", type=Path, help=_REUSE_SCAN_HELP)
    plan = commands.add_parser("plan", help="Scan raw data and freeze a deterministic plan")
    plan.add_argument("--raw-root", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--reuse-scan", type=Path, help=_REUSE_SCAN_HELP)
    make = commands.add_parser("generate", help="Generate from a frozen deterministic plan")
    make.add_argument("--raw-root", type=Path, required=True)
    make.add_argument("--config", type=Path, required=True)
    make.add_argument("--plan", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--limit-per-modality", type=int)
    make.add_argument("--resume", action="store_true")
    make.add_argument(
        "--fast-resume",
        action="store_true",
        help="With --resume, use one augmentation attempt per missing ID; intended for "
        "recovering a completed run's recorded failures",
    )
    make.add_argument(
        "--trust-plan",
        action="store_true",
        help="Skip the full raw re-scan and fingerprint recheck; still verifies each "
        "planned source by SHA-256",
    )
    make.add_argument(
        "--drop-cases",
        type=_case_set,
        default=set(),
        metavar="CASE[,CASE...]",
        help="With --resume, forget the committed samples of these failure cases so they "
        "are regenerated after their augmentation parameters change",
    )
    rgb_test = commands.add_parser(
        "extract-rgb-test",
        help="Generate a battery-disjoint RGB test set from an approved full plan",
    )
    rgb_test.add_argument("--raw-root", type=Path, required=True)
    rgb_test.add_argument("--config", type=Path, required=True)
    rgb_test.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Approved full generation_plan.csv containing RGB main/test assignments",
    )
    rgb_test.add_argument("--output", type=Path, required=True)
    rgb_test.add_argument("--total", type=int, default=1000)
    rgb_test.add_argument("--augmented", type=int, default=100)
    rgb_test.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed RGB test samples and regenerate only missing/failed IDs",
    )
    rgb_test.add_argument(
        "--trust-plan",
        action="store_true",
        help="Skip a full raw re-scan; selected source image/JSON hashes are still checked",
    )
    check = commands.add_parser("verify", help="Verify an existing generated dataset")
    check.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit-raw":
        result = audit_raw(
            args.raw_root.resolve(),
            _config(args.config),
            args.output.resolve(),
            args.reuse_scan.resolve() if args.reuse_scan else None,
        )
    elif args.command == "plan":
        result = create_plan(
            args.raw_root.resolve(),
            _config(args.config),
            args.output.resolve(),
            args.reuse_scan.resolve() if args.reuse_scan else None,
        )
    elif args.command == "generate":
        result = generate(
            args.raw_root.resolve(),
            _config(args.config),
            args.plan.resolve(),
            args.output.resolve(),
            args.limit_per_modality,
            args.resume,
            args.trust_plan,
            args.drop_cases,
            args.fast_resume,
        )
    elif args.command == "extract-rgb-test":
        result = extract_rgb_testset(
            args.raw_root.resolve(),
            _config(args.config),
            args.plan.resolve(),
            args.output.resolve(),
            args.total,
            args.augmented,
            args.trust_plan,
            args.resume,
        )
    else:
        verify_dataset(args.output.resolve())
        result = {"verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
