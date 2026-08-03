from __future__ import annotations

import copy
import csv
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .augment import CT_CASES, RGB_CASES
from .common import atomic_write, canonical_json_bytes
from .generator import generate
from .planner import config_hash


REQUIRED_PLAN_FIELDS = {
    "synthetic_id", "modality", "partition", "original_battery_id",
    "original_image_id", "new_battery_id", "new_image_id", "raw_image_path",
    "raw_json_path", "source_stem", "form", "axis", "image_sha256",
    "json_sha256", "pixel_hash", "quality_label", "assignment",
    "failure_case", "item_seed", "case_seed", "group_key", "group_seed",
    "raw_fingerprint", "config_sha256",
}
IDENTITY_FIELDS = (
    "original_battery_id", "new_battery_id", "raw_image_path",
    "image_sha256", "pixel_hash",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row[field].casefold() for row in rows if row.get(field)}


def _assert_disjoint(
    main_rows: list[dict[str, str]], test_rows: list[dict[str, str]], field: str
) -> None:
    overlap = _values(main_rows, field) & _values(test_rows, field)
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(
            f"RGB main/test overlap in {field}: {len(overlap)} value(s); "
            f"example={example}"
        )


def _assert_complete_identity(rows: list[dict[str, str]], label: str) -> None:
    for field in IDENTITY_FIELDS:
        missing = [row.get("synthetic_id", "<unknown>") for row in rows if not row.get(field)]
        if missing:
            raise ValueError(
                f"{label} has {len(missing)} row(s) without {field}; example={missing[0]}"
            )


def build_rgb_test_plan(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    total: int = 1000,
    augmented: int = 100,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Build a deterministic RGB-only test plan from an approved full plan."""
    if total <= 0:
        raise ValueError("RGB test total must be positive")
    if augmented < 0 or augmented > total:
        raise ValueError("RGB augmented count must be between 0 and total")

    rgb_rows = [row for row in rows if row.get("modality") == "RGB"]
    main_rows = [row for row in rgb_rows if row.get("partition") == "main"]
    candidates = [row for row in rgb_rows if row.get("partition") == "test"]
    if len(candidates) < total:
        raise ValueError(
            f"RGB test partition contains {len(candidates)} rows; {total} required"
        )

    _assert_complete_identity(main_rows, "RGB main")
    _assert_complete_identity(candidates, "RGB test")

    # Check both group-level leakage and individual source duplication against train.
    for field in IDENTITY_FIELDS:
        _assert_disjoint(main_rows, candidates, field)

    candidates.sort(
        key=lambda row: (
            int(row.get("new_image_id") or 0),
            row.get("synthetic_id", ""),
        )
    )
    fail_candidates = [
        row
        for row in candidates
        if row.get("quality_label") == "fail"
        and row.get("failure_case") in RGB_CASES
    ]
    if len(fail_candidates) < augmented:
        raise ValueError(
            "Approved RGB test rows do not contain enough augmentation assignments: "
            f"{len(fail_candidates)} available, {augmented} required"
        )
    selected_fail = fail_candidates[:augmented]
    fail_ids = {row["synthetic_id"] for row in selected_fail}
    selected_original = [row for row in candidates if row["synthetic_id"] not in fail_ids][
        : total - augmented
    ]
    selected = [copy.deepcopy(row) for row in selected_fail + selected_original]
    selected.sort(
        key=lambda row: (
            int(row.get("new_image_id") or 0),
            row.get("synthetic_id", ""),
        )
    )
    for field in ("raw_image_path", "image_sha256", "pixel_hash"):
        values = [row[field].casefold() for row in selected if row.get(field)]
        if len(values) != len(set(values)):
            raise ValueError(f"RGB test selection contains duplicate {field}")

    for row in selected:
        row["partition"] = "test"
        if row["synthetic_id"] in fail_ids:
            row["assignment"] = "augmented"
            row["quality_label"] = "fail"
            row["quality_class"] = "fail"
        else:
            row["assignment"] = "original"
            row["quality_label"] = "pass"
            row["quality_class"] = "pass"
            row["failure_case"] = ""
            row["case_seed"] = row.get("item_seed", "")
            row["group_key"] = ""
            row["group_seed"] = ""

    case_counts = Counter(
        row["failure_case"] for row in selected if row["quality_label"] == "fail"
    )
    derived_config = copy.deepcopy(config)
    derived_config.update(
        {
            "ct_target": 0,
            "ct_augmented_target": 0,
            "ct_test_target": 0,
            "ct_test_fail_target": 0,
            "rgb_target": total,
            "rgb_augmented_target": augmented,
            "rgb_test_target": total,
            "rgb_test_fail_target": augmented,
        }
    )
    derived_config["ct_failure_case_quotas"] = {case: 0 for case in CT_CASES}
    derived_config["ct_test_failure_case_quotas"] = {case: 0 for case in CT_CASES}
    derived_config["rgb_failure_case_quotas"] = {
        case: case_counts.get(case, 0) for case in RGB_CASES
    }
    derived_config["rgb_test_failure_case_quotas"] = {
        case: case_counts.get(case, 0) for case in RGB_CASES
    }
    derived_config_hash = config_hash(derived_config)
    for row in selected:
        row["config_sha256"] = derived_config_hash

    selected_main_batteries = _values(main_rows, "original_battery_id")
    selected_test_batteries = _values(selected, "original_battery_id")
    audit = {
        "schema_version": "1.1",
        "modality": "RGB",
        "total": total,
        "augmented": augmented,
        "original": total - augmented,
        "main_battery_count": len(selected_main_batteries),
        "test_battery_count": len(selected_test_batteries),
        "main_test_battery_overlap": 0,
        "main_test_source_path_overlap": 0,
        "main_test_image_sha256_overlap": 0,
        "main_test_pixel_hash_overlap": 0,
        "failure_case_counts": dict(sorted(case_counts.items())),
        "derived_config_sha256": derived_config_hash,
    }
    return selected, derived_config, audit


def extract_rgb_testset(
    raw_root: Path,
    config: dict[str, Any],
    plan_path: Path,
    output: Path,
    total: int = 1000,
    augmented: int = 100,
    trust_plan: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    fields, rows = _read_csv(plan_path)
    missing = sorted(REQUIRED_PLAN_FIELDS - set(fields))
    if missing:
        raise ValueError(
            "--plan must be generation_plan.csv; missing columns: " + ", ".join(missing)
        )
    selected, derived_config, audit = build_rgb_test_plan(
        rows, config, total, augmented
    )
    rgb_main = [
        row
        for row in rows
        if row.get("modality") == "RGB" and row.get("partition") == "main"
    ]
    blocked_source_paths = {
        row["raw_image_path"] for row in rgb_main if row.get("raw_image_path")
    }
    blocked_pixel_hashes = {
        row["pixel_hash"] for row in rgb_main if row.get("pixel_hash")
    }

    with tempfile.TemporaryDirectory(prefix="quality-fail-rgb-test-") as temporary:
        temporary_path = Path(temporary)
        derived_plan = temporary_path / "generation_plan.csv"
        _write_csv(derived_plan, fields, selected)
        summary = generate(
            raw_root,
            derived_config,
            derived_plan,
            output,
            resume=resume,
            trust_plan=trust_plan,
            support_manifest_dir=plan_path.parent,
            blocked_source_paths=blocked_source_paths,
            blocked_pixel_hashes=blocked_pixel_hashes,
        )

    manifests = output / "manifests"
    atomic_write(
        manifests / "rgb_test_config.json", canonical_json_bytes(derived_config)
    )
    atomic_write(
        manifests / "rgb_test_selection_audit.json", canonical_json_bytes(audit)
    )
    return {**summary, "rgb_test_selection": audit}
