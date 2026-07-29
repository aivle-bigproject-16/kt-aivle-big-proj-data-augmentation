from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .augment import CT_CASES, RGB_CASES
from .common import (
    atomic_write,
    canonical_json_bytes,
    load_json,
    normalized_relative,
    open_normalized,
    pixel_hash,
    sha256_file,
    stable_seed,
)
from .geometry import extract_ct_roi, point_rings, porosity_bbox_metric
from .models import Candidate, IMAGE_SUFFIXES, ParsedName
from .progress import ProgressReporter, configure_logger


AUDIT_FIELDS = [
    "modality",
    "raw_split",
    "raw_image_path",
    "raw_json_path",
    "image_stem",
    "battery_id",
    "image_id",
    "image_sha256",
    "json_sha256",
    "pixel_hash",
    "validation_status",
    "error_level",
    "exclusion_reason",
    "porosity_bbox_max_ratio",
    "ct_bbox_excluded",
    "extraction_rank",
    "selected",
    "assignment",
    "partition",
    "failure_case",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _raw_split(path: Path, root: Path) -> str | None:
    parts = path.resolve().relative_to(root.resolve()).parts
    for part in parts:
        value = part.casefold()
        if value == "training":
            return "training"
        if value == "validation":
            return "validation"
    return None


def _path_kind(path: Path) -> str | None:
    folded = {part.casefold() for part in path.parts}
    if {"01.원천데이터", "원천데이터"} & folded:
        return "image"
    if {"02.라벨링데이터", "라벨링데이터"} & folded:
        return "json"
    return None


def _safe_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path_traversal_or_symlink_escape: {path}") from exc
    return relative.as_posix()


def _index_raw(raw_root: Path, logger=None) -> tuple[dict[tuple[str, str, str], dict[str, list[Path]]], int]:
    if not raw_root.is_dir():
        raise ValueError(f"raw-root does not exist or is not a directory: {raw_root}")
    groups: dict[tuple[str, str, str], dict[str, list[Path]]] = defaultdict(
        lambda: {"images": [], "json": []}
    )
    discovered = 0
    started = time.monotonic()
    for directory, _, names in os.walk(raw_root):
        base = Path(directory)
        for name in names:
            discovered += 1
            path = base / name
            split = _raw_split(path, raw_root)
            parsed = ParsedName.parse(path.stem)
            if split is None or parsed is None:
                continue
            suffix = path.suffix.casefold()
            kind = _path_kind(path)
            key = (split.casefold(), parsed.modality, path.stem.casefold())
            if suffix in IMAGE_SUFFIXES and kind != "json":
                groups[key]["images"].append(path)
            elif suffix == ".json" and kind != "image":
                groups[key]["json"].append(path)
            if logger and discovered % 50_000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                logger.info(
                    "원본 파일 탐색 | %s개 | %.1f 파일/초",
                    f"{discovered:,}",
                    discovered / elapsed,
                )
    return groups, discovered


def _validate_pair(
    raw_root: Path,
    split: str,
    modality: str,
    stem: str,
    image_path: Path,
    json_path: Path,
    config: dict[str, Any],
) -> tuple[Candidate | None, dict[str, Any]]:
    parsed = ParsedName.parse(stem)
    if parsed is None:
        raise ValueError(f"invalid source stem: {stem}")
    base: dict[str, Any] = {
        "modality": modality,
        "raw_split": split,
        "image_stem": stem,
        "battery_id": parsed.battery_id,
        "image_id": parsed.image_id,
        "raw_image_path": _safe_relative(image_path, raw_root),
        "raw_json_path": _safe_relative(json_path, raw_root),
    }
    try:
        label = load_json(json_path)
        for required in ("data_info", "swelling", "defects", "image_info"):
            if required not in label:
                raise ValueError(f"missing top-level field: {required}")
        declared = label.get("image_info", {}).get("file_name")
        if declared and Path(str(declared)).stem.casefold() != stem.casefold():
            raise ValueError(f"image_info.file_name stem mismatch: {declared}")
        outline = label.get("swelling", {}).get("battery_outline")
        if outline not in (None, []):
            point_rings(outline)
        defects = label.get("defects")
        if defects is not None and not isinstance(defects, list):
            raise ValueError(f"defects must be null or list, got {type(defects).__name__}")
        for defect in defects or []:
            if not isinstance(defect, dict):
                raise ValueError("defect entry must be an object")
            if defect.get("points") not in (None, []):
                point_rings(defect.get("points"))
        image = open_normalized(image_path, modality)
        roi = extract_ct_roi(label, image.width, image.height) if modality == "CT" else None
        ratio, component_count = porosity_bbox_metric(label, roi) if roi else (0.0, 0)
        image_hash = sha256_file(image_path, int(config.get("hash_chunk_bytes", 1_048_576)))
        json_hash = sha256_file(json_path, int(config.get("hash_chunk_bytes", 1_048_576)))
        pixels = pixel_hash(image)
        excluded = modality == "CT" and ratio >= float(config["ct_porosity_threshold"])
        row = {
            **base,
            "image_sha256": image_hash,
            "json_sha256": json_hash,
            "pixel_hash": pixels,
            "validation_status": "excluded_source" if excluded else "valid",
            "error_level": "excluded_source" if excluded else "valid",
            "exclusion_reason": (
                "ct_porosity_bbox_max_ratio_ge_0.25" if excluded else ""
            ),
            "porosity_bbox_max_ratio": f"{ratio:.8f}",
            "ct_bbox_excluded": str(excluded).lower(),
        }
        candidate = Candidate(
            split,
            image_path,
            json_path,
            parsed,
            label,
            image.width,
            image.height,
            image_hash,
            json_hash,
            pixels,
            roi,
            ratio,
            component_count,
        )
        return (None if excluded else candidate), row
    except Exception as exc:
        return None, {
            **base,
            "validation_status": "excluded_source",
            "error_level": "excluded_source",
            "exclusion_reason": f"{type(exc).__name__}: {exc}",
        }


def _scan_jobs(config: dict[str, Any]) -> int:
    requested = max(1, int(config.get("jobs", 1)))
    return min(requested, os.cpu_count() or 1)


def _validate_pair_worker(
    payload: tuple[Path, str, str, str, Path, Path, dict[str, Any]]
) -> tuple[Candidate | None, dict[str, Any]]:
    """1:1 이미지-JSON 쌍을 독립 검증한다(멀티프로세스 워커).

    공유 상태를 만지지 않고 (candidate, row)만 반환한다. 순서 의존 병합
    (픽셀 중복 제거, 계통 오류 누적, 감사 순서)은 메인이 정렬 순서대로
    직렬 수행하므로 병렬/직렬 산출물과 raw fingerprint가 동일하다.
    검증 후 사용되지 않는 label(파싱된 JSON)은 IPC 비용을 줄이려 비운다.
    """
    raw_root, split, modality, stem, image_path, json_path, config = payload
    candidate, row = _validate_pair(
        raw_root, split, modality, stem, image_path, json_path, config
    )
    if candidate is not None:
        candidate.label = {}
    return candidate, row


def _preflight(
    raw_root: Path,
    groups: dict[tuple[str, str, str], dict[str, list[Path]]],
    config: dict[str, Any],
) -> None:
    count = int(config.get("preflight_per_stratum", 25))
    if count <= 0:
        return
    repeated_limit = int(config.get("preflight_repeated_error_limit", 5))
    strata = [(split, modality) for split in ("training", "validation") for modality in ("CT", "RGB")]
    for split, modality in strata:
        pairs = [
            (key, group)
            for key, group in sorted(groups.items())
            if key[0] == split
            and key[1] == modality
            and len(group["images"]) == 1
            and len(group["json"]) == 1
        ][:count]
        if not pairs:
            raise ValueError(f"preflight stratum has 0 valid pair candidates: {split}/{modality}")
        valid = 0
        errors: Counter[str] = Counter()
        for (raw_split, actual_modality, _), group in pairs:
            stem = group["images"][0].stem
            candidate, row = _validate_pair(
                raw_root,
                raw_split,
                actual_modality,
                stem,
                group["images"][0],
                group["json"][0],
                config,
            )
            if candidate is not None:
                valid += 1
            elif row["exclusion_reason"] != "ct_porosity_bbox_max_ratio_ge_0.25":
                reason = str(row["exclusion_reason"]).split(":", 1)[0]
                errors[reason] += 1
                if errors[reason] >= repeated_limit:
                    raise ValueError(
                        f"preflight systemic failure: {split}/{modality} {reason} repeated {errors[reason]}"
                    )
        if valid == 0:
            raise ValueError(f"preflight stratum has 0 valid samples: {split}/{modality}")


def scan(
    raw_root: Path,
    config: dict[str, Any],
    logger=None,
    audit_checkpoint_path: Path | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    groups, discovered = _index_raw(raw_root, logger)
    _preflight(raw_root, groups, config)
    sorted_groups = sorted(groups.items())
    pair_keys: list[tuple[str, str, str]] = []
    pair_payloads: list[tuple[Path, str, str, str, Path, Path, dict[str, Any]]] = []
    for key, group in sorted_groups:
        if len(group["images"]) == 1 and len(group["json"]) == 1:
            split, modality, _ = key
            stem = group["images"][0].stem
            pair_keys.append(key)
            pair_payloads.append(
                (raw_root, split, modality, stem, group["images"][0], group["json"][0], config)
            )
    jobs = _scan_jobs(config)
    if logger:
        logger.info(
            "쌍 검증 시작 | pair %s | worker %s",
            f"{len(pair_payloads):,}",
            jobs,
        )
    results_by_key: dict[tuple[str, str, str], tuple[Candidate | None, dict[str, Any]]] = {}
    if jobs > 1 and pair_payloads:
        chunk = max(1, len(pair_payloads) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for key, result in zip(
                pair_keys,
                executor.map(_validate_pair_worker, pair_payloads, chunksize=chunk),
            ):
                results_by_key[key] = result
    else:
        for key, payload in zip(pair_keys, pair_payloads):
            results_by_key[key] = _validate_pair_worker(payload)
    candidates: list[Candidate] = []
    audit: list[dict[str, Any]] = []
    systemic: list[str] = []
    orphans: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    last_error = ""
    consecutive = 0
    seen_pixels: dict[tuple[str, str], str] = {}
    progress = (
        ProgressReporter(
            logger,
            "이미지·JSON 검증",
            len(groups),
            log_every_items=int(config.get("plan_log_interval", 250)),
        )
        if logger
        else None
    )
    logged = 0
    max_logged = int(config.get("max_logged_error_examples", 20))
    for index, (key, group) in enumerate(sorted_groups, 1):
        split, modality, folded_stem = key
        paths = group["images"] + group["json"]
        stem = paths[0].stem if paths else folded_stem
        parsed = ParsedName.parse(stem)
        base = {
            "modality": modality,
            "raw_split": split,
            "image_stem": stem,
            "battery_id": parsed.battery_id if parsed else "",
            "image_id": parsed.image_id if parsed else "",
        }
        if len(group["images"]) > 1 or len(group["json"]) > 1:
            reason = (
                f"matching_cardinality_ambiguous: images={len(group['images'])}, "
                f"json={len(group['json'])}"
            )
            systemic.append(f"{split}/{stem}: {reason}")
            audit.append(
                {
                    **base,
                    "validation_status": "blocking_systemic",
                    "error_level": "blocking_systemic",
                    "exclusion_reason": reason,
                }
            )
            if logger and logged < max_logged:
                logged += 1
                logger.error("차단 오류 %s/%s | %s | %s", logged, max_logged, stem, reason)
        elif len(group["images"]) != 1 or len(group["json"]) != 1:
            reason = (
                "json_only"
                if group["json"]
                else "image_only"
                if group["images"]
                else "empty_pair"
            )
            row = {
                **base,
                "raw_image_path": (
                    _safe_relative(group["images"][0], raw_root) if group["images"] else ""
                ),
                "raw_json_path": (
                    _safe_relative(group["json"][0], raw_root) if group["json"] else ""
                ),
                "validation_status": "excluded_source",
                "error_level": "excluded_source",
                "exclusion_reason": reason,
            }
            audit.append(row)
            orphans.append(row)
        else:
            candidate, row = results_by_key[key]
            if candidate is not None:
                duplicate = seen_pixels.get((modality, candidate.pixel_hash))
                if duplicate:
                    row["validation_status"] = "excluded_source"
                    row["error_level"] = "excluded_source"
                    row["exclusion_reason"] = f"duplicate_pixel_hash:{duplicate}"
                else:
                    seen_pixels[(modality, candidate.pixel_hash)] = stem
                    candidates.append(candidate)
            reason = str(row["exclusion_reason"]).split(":", 1)[0]
            if row["validation_status"] == "excluded_source" and reason not in {
                "ct_porosity_bbox_max_ratio_ge_0.25",
                "duplicate_pixel_hash",
            }:
                error_counts[reason] += 1
                consecutive = consecutive + 1 if reason == last_error else 1
                last_error = reason
                if (
                    consecutive >= int(config.get("systemic_consecutive_limit", 20))
                    or error_counts[reason]
                    >= int(config.get("systemic_cumulative_limit", 100))
                ):
                    if audit_checkpoint_path:
                        _write_csv(audit_checkpoint_path, audit + [row], AUDIT_FIELDS)
                    raise ValueError(
                        f"systemic schema mismatch: {reason}; consecutive={consecutive}, "
                        f"cumulative={error_counts[reason]}"
                    )
            else:
                consecutive, last_error = 0, ""
            audit.append(row)
        if progress:
            progress.update(
                index,
                detail=f"정상 {len(candidates):,} | 차단 오류 {len(systemic):,}",
            )
        if audit_checkpoint_path and index % int(config.get("plan_checkpoint_interval", 250)) == 0:
            _write_csv(audit_checkpoint_path, audit, AUDIT_FIELDS)
    if audit_checkpoint_path:
        _write_csv(audit_checkpoint_path, audit, AUDIT_FIELDS)
    if logger:
        logger.info(
            "후보 탐색 완료 | 전체 파일 %s | pair key %s | 정상 %s",
            f"{discovered:,}",
            f"{len(groups):,}",
            f"{len(candidates):,}",
        )
    return candidates, audit, systemic, orphans


def _validate_config(config: dict[str, Any]) -> None:
    for modality, allowed in (("ct", CT_CASES), ("rgb", RGB_CASES)):
        target = int(config[f"{modality}_target"])
        fail_target = int(config[f"{modality}_augmented_target"])
        test_target = int(config.get(f"{modality}_test_target", 0))
        test_fail = int(config.get(f"{modality}_test_fail_target", 0))
        quotas = config[f"{modality}_failure_case_quotas"]
        unknown = set(quotas) - set(allowed)
        if unknown:
            raise ValueError(f"{modality.upper()} unknown failure cases: {sorted(unknown)}")
        if sum(int(value) for value in quotas.values()) != fail_target:
            raise ValueError(f"{modality.upper()} failure case quotas do not sum to {fail_target}")
        if not (0 <= test_fail <= fail_target and 0 <= test_target - test_fail <= target - fail_target):
            raise ValueError(f"{modality.upper()} test PASS/FAIL targets are impossible")


def _case_assignments(config: dict[str, Any], modality: str) -> list[str]:
    quotas = config[f"{modality.lower()}_failure_case_quotas"]
    cases = [case for case, count in quotas.items() for _ in range(int(count))]
    random.Random(stable_seed(config["seed"], modality, "failure-cases")).shuffle(cases)
    return cases


def _fingerprint(candidates: list[Candidate], raw_root: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(
        candidates,
        key=lambda c: (c.raw_split, c.parsed.modality, normalized_relative(c.image_path, raw_root)),
    ):
        digest.update(
            (
                f"{candidate.raw_split}|{candidate.parsed.modality}|"
                f"{normalized_relative(candidate.image_path, raw_root)}|{candidate.image_sha256}|"
                f"{normalized_relative(candidate.json_path, raw_root)}|{candidate.json_sha256}\n"
            ).encode()
        )
    return digest.hexdigest()


def audit_raw(
    raw_root: Path, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("quality_fail_augment.audit", output / "logs" / "audit.log")
    candidates, audit, systemic, orphans = scan(
        raw_root, config, logger, output / "manifests" / "extraction_audit.csv"
    )
    _write_csv(output / "manifests" / "orphan_sources.csv", orphans, AUDIT_FIELDS)
    _write_csv(
        output / "manifests" / "ct_bbox_exclusions.csv",
        [row for row in audit if row.get("ct_bbox_excluded") == "true"],
        AUDIT_FIELDS,
    )
    if systemic:
        raise ValueError(f"Raw scan has {len(systemic)} blocking_systemic errors")
    counts = Counter(candidate.parsed.modality for candidate in candidates)
    summary = {
        "schema_version": "1.1",
        "valid_pairs": dict(counts),
        "orphan_count": len(orphans),
        "ambiguous_pair_count": 0,
        "raw_fingerprint": _fingerprint(candidates, raw_root),
        "ct_bbox_policy_version": config["ct_bbox_policy_version"],
    }
    atomic_write(
        output / "raw_schema_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode(),
    )
    return summary


def create_plan(
    raw_root: Path, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    _validate_config(config)
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("quality_fail_augment.plan", output / "logs" / "plan.log")
    candidates, audit, systemic, orphans = scan(
        raw_root, config, logger, manifests / "extraction_audit.csv"
    )
    _write_csv(manifests / "orphan_sources.csv", orphans, AUDIT_FIELDS)
    if systemic:
        raise ValueError(
            f"Raw scan has {len(systemic)} blocking_systemic errors; inspect extraction_audit.csv"
        )
    fingerprint = _fingerprint(candidates, raw_root)
    config_hash = _config_hash(config)
    plan_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    audit_by_path = {
        (row.get("modality"), row.get("raw_image_path")): row
        for row in audit
        if row.get("raw_image_path")
    }
    battery_starts = {
        "CT": int(config["ct_battery_id_start"]),
        "RGB": int(config["rgb_battery_id_start"]),
    }
    image_starts = {
        "CT": int(config["ct_image_id_start"]),
        "RGB": int(config["rgb_image_id_start"]),
    }
    for modality in ("CT", "RGB"):
        target = int(config[f"{modality.lower()}_target"])
        fail_target = int(config[f"{modality.lower()}_augmented_target"])
        reserve = int(config.get("reserve_per_modality", 0))
        pool = [candidate for candidate in candidates if candidate.parsed.modality == modality]
        pool.sort(
            key=lambda candidate: (
                candidate.raw_split,
                normalized_relative(candidate.image_path, raw_root),
            )
        )
        random.Random(stable_seed(config["seed"], modality, "extract")).shuffle(pool)
        if len(pool) < target + reserve:
            raise ValueError(
                f"{modality}: needs {target + reserve} valid unique sources including reserve, "
                f"found {len(pool)}"
            )
        chosen, reserves = pool[:target], pool[target : target + reserve]
        shuffled = list(range(target))
        random.Random(stable_seed(config["seed"], modality, "assignment")).shuffle(shuffled)
        fail_indices = set(shuffled[:fail_target])
        case_values = _case_assignments(config, modality)
        case_by_index = dict(zip(sorted(fail_indices), case_values, strict=True))
        fail_order = list(sorted(fail_indices))
        pass_order = [index for index in range(target) if index not in fail_indices]
        random.Random(stable_seed(config["seed"], modality, "test-fail")).shuffle(fail_order)
        random.Random(stable_seed(config["seed"], modality, "test-pass")).shuffle(pass_order)
        test_indices = set(
            fail_order[: int(config.get(f"{modality.lower()}_test_fail_target", 0))]
            + pass_order[
                : int(config.get(f"{modality.lower()}_test_target", 0))
                - int(config.get(f"{modality.lower()}_test_fail_target", 0))
            ]
        )
        battery_map: dict[tuple[str, str], int] = {}
        for slot, candidate in enumerate(chosen, 1):
            source_key = (candidate.parsed.form, candidate.parsed.battery_id)
            if source_key not in battery_map:
                battery_map[source_key] = battery_starts[modality] + len(battery_map)
            battery_id = battery_map[source_key]
            if battery_id > int(config[f"{modality.lower()}_battery_id_end"]):
                raise ValueError(f"{modality} synthetic battery ID range exhausted")
            index = slot - 1
            fail = index in fail_indices
            partition = "test" if index in test_indices else "main"
            image_id = image_starts[modality] + index
            row = {
                "synthetic_id": f"QF15_{modality}_{slot:08d}",
                "modality": modality,
                "raw_split": candidate.raw_split,
                "raw_image_path": normalized_relative(candidate.image_path, raw_root),
                "raw_json_path": normalized_relative(candidate.json_path, raw_root),
                "source_stem": candidate.image_path.stem,
                "form": candidate.parsed.form,
                "axis": candidate.parsed.axis,
                "original_battery_id": candidate.parsed.battery_id,
                "original_image_id": candidate.parsed.image_id,
                "new_battery_id": battery_id,
                "new_image_id": image_id,
                "assignment": "augmented" if fail else "original",
                "quality_label": "fail" if fail else "pass",
                "partition": partition,
                "failure_case": case_by_index.get(index, ""),
                "selected": "true",
                "extraction_rank": slot,
                "item_seed": stable_seed(config["seed"], modality, slot),
                "image_sha256": candidate.image_sha256,
                "json_sha256": candidate.json_sha256,
                "pixel_hash": candidate.pixel_hash,
                "porosity_bbox_max_ratio": f"{candidate.porosity_bbox_max_ratio:.8f}",
                "roi": json.dumps(candidate.roi),
                "raw_fingerprint": fingerprint,
                "config_sha256": config_hash,
                "ct_bbox_policy_version": config["ct_bbox_policy_version"],
            }
            plan_rows.append(row)
            audit_by_path[(modality, row["raw_image_path"])].update(
                {
                    "extraction_rank": slot,
                    "selected": "true",
                    "assignment": row["assignment"],
                    "partition": partition,
                    "failure_case": row["failure_case"],
                }
            )
        for rank, candidate in enumerate(reserves, 1):
            reserve_rows.append(
                {
                    "modality": modality,
                    "reserve_rank": rank,
                    "raw_split": candidate.raw_split,
                    "raw_image_path": normalized_relative(candidate.image_path, raw_root),
                    "raw_json_path": normalized_relative(candidate.json_path, raw_root),
                    "source_stem": candidate.image_path.stem,
                    "image_sha256": candidate.image_sha256,
                    "json_sha256": candidate.json_sha256,
                    "synthetic_id": "",
                    "new_image_id": "",
                    "failure_case": "",
                    "partition": "",
                }
            )
    _write_csv(manifests / "generation_plan.csv", plan_rows, list(plan_rows[0]))
    _write_csv(manifests / "reserve_sources.csv", reserve_rows, list(reserve_rows[0]) if reserve_rows else ["modality"])
    _write_csv(manifests / "extraction_audit.csv", audit, AUDIT_FIELDS)
    _write_csv(
        manifests / "ct_bbox_exclusions.csv",
        [row for row in audit if row.get("ct_bbox_excluded") == "true"],
        AUDIT_FIELDS,
    )
    metadata = {
        "schema_version": "1.1",
        "package_version": "1.7",
        "raw_fingerprint": fingerprint,
        "config_sha256": config_hash,
        "plan_rows": len(plan_rows),
        "selected_rows": len(plan_rows),
        "reserve_rows": len(reserve_rows),
        "orphan_count": len(orphans),
        "ambiguous_pair_count": 0,
        "ct_bbox_policy_version": config["ct_bbox_policy_version"],
    }
    atomic_write(output / "plan_metadata.json", canonical_json_bytes(metadata))
    return metadata
