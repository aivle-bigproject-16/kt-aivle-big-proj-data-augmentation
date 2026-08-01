from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .augment import CT_CASES, RGB_CASES
from .generator import generate, verify_dataset
from .planner import audit_raw, create_plan
from .settings import (
    AUDIT_DIR_VAR,
    CONFIG_VAR,
    ENV_FILE_VAR,
    OUTPUT_DIR_VAR,
    PLAN_CSV_VAR,
    PLAN_DIR_VAR,
    RAW_ROOT_VAR,
    env_path,
    load_dotenv,
)


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


def _env_help(variable: str) -> str:
    return f"생략하면 `.env` 의 {variable} 를 쓴다"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="quality-fail-augment")
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument(
        "--env-file",
        type=Path,
        help="사용할 .env 경로. 생략하면 현재 폴더부터 위로 올라가며 찾는다",
    )
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-raw", help="Read-only raw-root compatibility audit")
    audit.add_argument("--raw-root", type=Path, help=_env_help(RAW_ROOT_VAR))
    audit.add_argument("--config", type=Path, help=_env_help(CONFIG_VAR))
    audit.add_argument("--output", type=Path, help=_env_help(AUDIT_DIR_VAR))
    audit.add_argument("--reuse-scan", type=Path, help=_REUSE_SCAN_HELP)
    plan = commands.add_parser("plan", help="Scan raw data and freeze a deterministic plan")
    plan.add_argument("--raw-root", type=Path, help=_env_help(RAW_ROOT_VAR))
    plan.add_argument("--config", type=Path, help=_env_help(CONFIG_VAR))
    plan.add_argument("--output", type=Path, help=_env_help(PLAN_DIR_VAR))
    plan.add_argument("--reuse-scan", type=Path, help=_REUSE_SCAN_HELP)
    make = commands.add_parser("generate", help="Generate from a frozen deterministic plan")
    make.add_argument("--raw-root", type=Path, help=_env_help(RAW_ROOT_VAR))
    make.add_argument("--config", type=Path, help=_env_help(CONFIG_VAR))
    make.add_argument(
        "--plan",
        type=Path,
        help=f"생략하면 `.env` 의 {PLAN_CSV_VAR}, 그것도 없으면 "
        f"{PLAN_DIR_VAR}/manifests/generation_plan.csv 를 쓴다",
    )
    make.add_argument("--output", type=Path, help=_env_help(OUTPUT_DIR_VAR))
    make.add_argument("--limit-per-modality", type=int)
    make.add_argument("--resume", action="store_true")
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
    check = commands.add_parser("verify", help="Verify an existing generated dataset")
    check.add_argument("--output", type=Path, help=_env_help(OUTPUT_DIR_VAR))
    return root


def _resolve(value: Path | None, variable: str, flag: str) -> Path:
    """CLI 인자를 우선하고, 없으면 환경변수에서 경로를 가져온다."""
    chosen = value if value is not None else env_path(variable)
    if chosen is None:
        raise SystemExit(f"{flag} 가 없습니다. 인자로 주거나 `.env` 에 {variable} 를 설정하세요.")
    return chosen.expanduser().resolve()


def _resolve_plan_csv(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    explicit = env_path(PLAN_CSV_VAR)
    if explicit is not None:
        return explicit.expanduser().resolve()
    plan_dir = env_path(PLAN_DIR_VAR)
    if plan_dir is not None:
        return (plan_dir.expanduser() / "manifests" / "generation_plan.csv").resolve()
    raise SystemExit(
        f"--plan 이 없습니다. 인자로 주거나 `.env` 에 {PLAN_CSV_VAR} 또는 "
        f"{PLAN_DIR_VAR} 를 설정하세요."
    )


def main() -> None:
    args = parser().parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise SystemExit(f"env 파일이 없습니다: {args.env_file}")
        os.environ[ENV_FILE_VAR] = str(args.env_file.expanduser().resolve())
    load_dotenv()
    if args.command == "audit-raw":
        result = audit_raw(
            _resolve(args.raw_root, RAW_ROOT_VAR, "--raw-root"),
            _config(_resolve(args.config, CONFIG_VAR, "--config")),
            _resolve(args.output, AUDIT_DIR_VAR, "--output"),
            args.reuse_scan.resolve() if args.reuse_scan else None,
        )
    elif args.command == "plan":
        result = create_plan(
            _resolve(args.raw_root, RAW_ROOT_VAR, "--raw-root"),
            _config(_resolve(args.config, CONFIG_VAR, "--config")),
            _resolve(args.output, PLAN_DIR_VAR, "--output"),
            args.reuse_scan.resolve() if args.reuse_scan else None,
        )
    elif args.command == "generate":
        result = generate(
            _resolve(args.raw_root, RAW_ROOT_VAR, "--raw-root"),
            _config(_resolve(args.config, CONFIG_VAR, "--config")),
            _resolve_plan_csv(args.plan),
            _resolve(args.output, OUTPUT_DIR_VAR, "--output"),
            args.limit_per_modality,
            args.resume,
            args.trust_plan,
            args.drop_cases,
        )
    else:
        verify_dataset(_resolve(args.output, OUTPUT_DIR_VAR, "--output"))
        result = {"verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
