from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import platform
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import shapely
from PIL import Image

from . import __version__
from .augment import CASE_NAMES_KO, EXHAUSTED_SEARCH_MARKER, SOURCE_REFERENCES
from .common import (
    atomic_write,
    canonical_json_bytes,
    load_json,
    open_normalized,
    pixel_hash,
    sha256_bytes,
    sha256_file,
)
from .geometry import point_rings
from .models import ParsedName
from .planner import _config_hash, _fingerprint, scan
from .preprocessing import apply_quality_transform, finalize_sample, prepare_source
from .progress import ProgressReporter, configure_logger


DATASET_FIELDS = [
    "synthetic_id",
    "modality",
    "new_battery_id",
    "new_image_id",
    "image_path",
    "label_json_path",
    "augmentation_json_path",
    "quality_label",
    "image_relative_path",
    "label_json_relative_path",
    "augmentation_json_relative_path",
    "quality_class",
    "assignment",
    "is_augmented",
    "partition",
    "failure_case",
    "augmentations",
    "width",
    "height",
    "image_sha256",
    "label_json_sha256",
    "augmentation_json_sha256",
    "replacement_for",
]
LINEAGE_FIELDS = [
    "synthetic_id",
    "raw_image_path",
    "raw_json_path",
    "resolved_raw_root",
    "source_image_relative_path",
    "source_json_relative_path",
    "original_battery_id",
    "original_image_id",
    "new_battery_id",
    "new_image_id",
    "axis",
    "ct_axis",
    "original_roi",
    "ct_roi",
    "item_seed",
    "case_seed",
    "group_key",
    "group_seed",
    "failure_case",
    "affine_matrix",
    "augmentation_parameters",
    "actual_parameters_json",
]
ERROR_FIELDS = [
    "synthetic_id",
    "modality",
    "raw_image_path",
    "attempt",
    "error",
]
RECOVERY_FIELDS = ["path", "action", "reason"]

# How many reserve sources one failed sample may try before it is recorded as a
# failure. Walking the whole queue costs a SHA-256 of both files plus a full
# augmentation per entry, so an unbounded walk let a few hundred failures consume
# hours. This is a safety bound rather than a tuning knob, so it stays out of the
# config and cannot perturb the plan hash.
RESERVE_ATTEMPTS_PER_SAMPLE = 40


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode("utf-8-sig"))


def _write_checkpoint(
    output: Path,
    manifest_rows: list[dict[str, Any]],
    lineage_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    ordered_manifest = sorted(manifest_rows, key=lambda row: row["synthetic_id"])
    ordered_lineage = sorted(lineage_rows, key=lambda row: row["synthetic_id"])
    _write_csv(output / "manifests" / "dataset_manifest.csv", ordered_manifest, DATASET_FIELDS)
    _write_csv(output / "manifests" / "lineage_private.csv", ordered_lineage, LINEAGE_FIELDS)
    _write_csv(output / "manifests" / "generation_errors.csv", errors, ERROR_FIELDS)


def _remove_path_with_retry(path: Path, attempts: int = 5) -> None:
    if not path.exists():
        return
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1 * (2**attempt))
    raise RuntimeError(f"Could not clean output path after {attempts} attempts: {path}") from last_error


def _jpeg_bytes(image: Image.Image, config: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    options: dict[str, Any] = {
        "quality": int(config["jpeg_quality"]),
        "optimize": bool(config.get("jpeg_optimize", False)),
        "progressive": bool(config.get("jpeg_progressive", False)),
        "exif": b"",
    }
    if image.mode == "RGB":
        options["subsampling"] = int(config.get("jpeg_subsampling", 0))
    image.save(buffer, "JPEG", **options)
    return buffer.getvalue()


def _safe_source(raw_root: Path, relative: str) -> Path:
    root = raw_root.resolve()
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path_traversal_or_symlink_escape: {relative}") from exc
    return path


def _validate_plan(
    raw_root: Path,
    config: dict[str, Any],
    rows: list[dict[str, str]],
    trust_plan: bool = False,
) -> None:
    if not rows:
        raise ValueError("Plan is empty")
    if any(row["config_sha256"] != rows[0]["config_sha256"] for row in rows):
        raise ValueError("Plan contains mixed config hashes")
    if _config_hash(config) != rows[0]["config_sha256"]:
        raise ValueError("Config SHA-256 differs from approved plan")
    ids = [row["synthetic_id"] for row in rows]
    image_ids = [(row["modality"], row["new_image_id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(image_ids) != len(set(image_ids)):
        raise ValueError("Plan contains duplicate synthetic/image IDs")
    for row in rows:
        image_path = _safe_source(raw_root, row["raw_image_path"])
        json_path = _safe_source(raw_root, row["raw_json_path"])
        if not image_path.is_file() or not json_path.is_file():
            raise ValueError(f"Plan source missing: {row['raw_image_path']}")
        if (
            sha256_file(image_path, int(config.get("hash_chunk_bytes", 1_048_576)))
            != row["image_sha256"]
            or sha256_file(json_path, int(config.get("hash_chunk_bytes", 1_048_576)))
            != row["json_sha256"]
        ):
            raise ValueError(f"Plan source changed: {row['raw_image_path']}")
    # trust_plan은 전체 raw 재스캔(약 44분)과 fingerprint 재대조를 건너뛴다. plan이 쓰는
    # 40k+reserve 소스는 위에서 sha256으로 이미 pin 검증했으므로, 선택된 소스의 무결성은
    # 보장된다. 건너뛰는 것은 미선택 소스까지 포함한 전체 fingerprint 일치 확인뿐이다.
    if trust_plan:
        return
    candidates, _, systemic, _ = scan(raw_root, config)
    if systemic:
        raise ValueError("Raw dataset now contains blocking_systemic errors")
    if _fingerprint(candidates, raw_root) != rows[0]["raw_fingerprint"]:
        raise ValueError("Raw fingerprint differs from approved plan")


def _output_paths(row: dict[str, str], stem: str) -> tuple[Path, Path, Path | None]:
    modality, partition = row["modality"], row["partition"]
    image = Path(modality) / partition / "images" / f"{stem}.jpg"
    label = Path(modality) / partition / "labels_json" / f"{stem}.json"
    history = (
        Path(modality)
        / partition
        / "augmentation_json"
        / f"{stem}.augmentation.json"
        if row["quality_label"] == "fail"
        else None
    )
    return image, label, history


def _make_one(
    raw_root_text: str,
    output_text: str,
    row: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_root, output = Path(raw_root_text), Path(output_text)
    modality = row["modality"]
    parsed = ParsedName.parse(row["source_stem"])
    if parsed is None:
        raise ValueError("Invalid source stem in plan")
    new_battery, new_image = int(row["new_battery_id"]), int(row["new_image_id"])
    new_stem = parsed.new_stem(new_battery, new_image)
    image_name = f"{new_stem}.jpg"
    quality = row["quality_label"]
    failure_case = row.get("failure_case", "")
    prepared = prepare_source(
        _safe_source(raw_root, row["raw_image_path"]),
        _safe_source(raw_root, row["raw_json_path"]),
        modality,
    )
    transformed = apply_quality_transform(
        prepared,
        modality,
        quality,
        failure_case,
        int(row["item_seed"]),
        int(config.get("max_augmentation_retries", 8)),
        int(row.get("case_seed") or row["item_seed"]),
        int(row["group_seed"]) if row.get("group_seed") else None,
        config,
    )
    finalized = finalize_sample(
        transformed,
        modality,
        quality,
        new_battery,
        new_image,
        image_name,
        int(config["resize_long_side"]),
        bool(config.get("upscale_small_images", False)),
    )
    resized = finalized.image
    label = finalized.label
    transform = finalized.transform
    records = finalized.records
    original_roi = finalized.original_roi
    image_bytes = _jpeg_bytes(resized, config)
    label_bytes = canonical_json_bytes(label)
    image_hash, label_hash = sha256_bytes(image_bytes), sha256_bytes(label_bytes)
    relative_image, relative_label, relative_history = _output_paths(row, new_stem)

    history_bytes: bytes | None = None
    history_hash = ""
    if quality == "fail":
        history = {
            "schema_version": "1.1",
            "synthetic_id": row["synthetic_id"],
            "output_image_file": relative_image.name,
            "label_json_file": relative_label.name,
            "issued_ids": {
                "battery_id": new_battery,
                "image_id": new_image,
            },
            "quality_label": "fail",
            "is_augmented": True,
            "item_seed": int(row["item_seed"]),
            "case_seed": int(row.get("case_seed") or row["item_seed"]),
            "group_key": row.get("group_key", ""),
            "group_seed": int(row["group_seed"]) if row.get("group_seed") else None,
            "failure_case": {
                "id": failure_case,
                "name_ko": CASE_NAMES_KO[failure_case],
                "source_reference": SOURCE_REFERENCES[failure_case],
            },
            "failure_case_count": 1,
            "augmentation_count": len(records),
            "augmentations": records,
            "automatic_checks": {
                "passed": True,
                "quality_gate_version": "v2.0",
                "record_types": [record["type"] for record in records],
                "measurements": {
                    record["type"]: record.get("parameters", {})
                    for record in records
                },
            },
            "affine_matrix": transform.matrix(),
            "output": {
                "width": resized.width,
                "height": resized.height,
                "format": "JPEG",
                "quality": int(config["jpeg_quality"]),
                "jpeg_quality": int(config["jpeg_quality"]),
                "image_sha256": image_hash,
                "label_json_sha256": label_hash,
            },
        }
        history_bytes = canonical_json_bytes(history)
        history_hash = sha256_bytes(history_bytes)

    staging = output / ".sample_staging" / row["synthetic_id"]
    if staging.exists():
        _remove_path_with_retry(staging)
    staging.mkdir(parents=True)
    staged_image = staging / relative_image.name
    staged_label = staging / relative_label.name
    atomic_write(staged_image, image_bytes)
    atomic_write(staged_label, label_bytes)
    staged_history: Path | None = None
    if relative_history is not None and history_bytes is not None:
        staged_history = staging / relative_history.name
        atomic_write(staged_history, history_bytes)
    required = [(staged_image, relative_image, image_hash), (staged_label, relative_label, label_hash)]
    if staged_history is not None and relative_history is not None:
        required.append((staged_history, relative_history, history_hash))
    for staged, _, expected_hash in required:
        if not staged.is_file() or sha256_file(staged) != expected_hash:
            raise ValueError(f"staging validation failed: {staged}")
    moved: list[Path] = []
    try:
        for staged, relative, _ in required:
            destination = output / relative
            if destination.exists():
                raise ValueError(f"final output path already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
            moved.append(destination)
        for destination, (_, _, expected_hash) in zip(moved, required, strict=True):
            if sha256_file(destination) != expected_hash:
                raise ValueError(f"final hash mismatch: {destination}")
    except Exception:
        for path in moved:
            _remove_path_with_retry(path)
        _remove_path_with_retry(staging)
        raise

    manifest = {
        "synthetic_id": row["synthetic_id"],
        "modality": modality,
        "new_battery_id": new_battery,
        "new_image_id": new_image,
        "image_path": relative_image.as_posix(),
        "label_json_path": relative_label.as_posix(),
        "augmentation_json_path": relative_history.as_posix() if relative_history else "",
        "quality_label": quality,
        "image_relative_path": relative_image.as_posix(),
        "label_json_relative_path": relative_label.as_posix(),
        "augmentation_json_relative_path": relative_history.as_posix() if relative_history else "",
        "quality_class": quality,
        "assignment": row["assignment"],
        "is_augmented": str(quality == "fail").lower(),
        "partition": row["partition"],
        "failure_case": failure_case,
        "augmentations": "|".join(record["type"] for record in records),
        "width": resized.width,
        "height": resized.height,
        "image_sha256": image_hash,
        "label_json_sha256": label_hash,
        "augmentation_json_sha256": history_hash,
        "replacement_for": "",
    }
    lineage = {
        "synthetic_id": row["synthetic_id"],
        "raw_image_path": row["raw_image_path"],
        "raw_json_path": row["raw_json_path"],
        "resolved_raw_root": str(raw_root.resolve()),
        "source_image_relative_path": row["raw_image_path"],
        "source_json_relative_path": row["raw_json_path"],
        "original_battery_id": row["original_battery_id"],
        "original_image_id": row["original_image_id"],
        "new_battery_id": new_battery,
        "new_image_id": new_image,
        "axis": row["axis"],
        "ct_axis": row["axis"],
        "original_roi": json.dumps(original_roi),
        "ct_roi": json.dumps(original_roi),
        "item_seed": row["item_seed"],
        "case_seed": row.get("case_seed", row["item_seed"]),
        "group_key": row.get("group_key", ""),
        "group_seed": row.get("group_seed", ""),
        "failure_case": failure_case,
        "affine_matrix": json.dumps(transform.matrix(), separators=(",", ":")),
        "augmentation_parameters": json.dumps(
            records, ensure_ascii=False, separators=(",", ":")
        ),
        "actual_parameters_json": json.dumps(
            records, ensure_ascii=False, separators=(",", ":")
        ),
    }
    return manifest, lineage


def _make_one_safe(
    raw_root_text: str,
    output_text: str,
    row: dict[str, str],
    config: dict[str, Any],
) -> tuple[bool, Any]:
    try:
        return True, _make_one(raw_root_text, output_text, row, config)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _cleanup_uncommitted(output: Path, manifest_rows: list[dict[str, str]]) -> None:
    committed = {
        value
        for row in manifest_rows
        for value in (
            row.get("image_path", ""),
            row.get("label_json_path", ""),
            row.get("augmentation_json_path", ""),
        )
        if value
    }
    recovery: list[dict[str, str]] = []
    for pattern in (
        "*/main/images/*",
        "*/test/images/*",
        "*/main/labels_json/*",
        "*/test/labels_json/*",
        "*/main/augmentation_json/*",
        "*/test/augmentation_json/*",
    ):
        for path in output.glob(pattern):
            if path.is_file() and path.relative_to(output).as_posix() not in committed:
                recovery.append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "action": "deleted",
                        "reason": "manifest-less partial output",
                    }
                )
                _remove_path_with_retry(path)
    staging = output / ".sample_staging"
    if staging.exists():
        for child in staging.iterdir():
            recovery.append(
                {
                    "path": child.relative_to(output).as_posix(),
                    "action": "deleted",
                    "reason": "stale staging",
                }
            )
            _remove_path_with_retry(child)
    _write_csv(output / "manifests" / "recovery_audit.csv", recovery, RECOVERY_FIELDS)


def _drop_failure_cases(
    output: Path,
    manifest_rows: list[dict[str, Any]],
    lineage_rows: list[dict[str, Any]],
    drop_cases: set[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Forget the committed samples of the given failure cases so --resume rebuilds them.

    This is used after a case's augmentation parameters change. Regenerated samples reappear
    under the same synthetic_ids. The files are not deleted here: once the rows are gone,
    _cleanup_uncommitted treats the leftovers as manifest-less output and removes them through
    the existing recovery path.
    """
    dropped = {
        row["synthetic_id"]
        for row in manifest_rows
        if row.get("failure_case", "") in drop_cases
    }
    if not dropped:
        logger.info("삭제 대상 case 없음 | %s", ", ".join(sorted(drop_cases)))
        return manifest_rows, lineage_rows
    kept_manifest = [row for row in manifest_rows if row["synthetic_id"] not in dropped]
    kept_lineage = [
        row for row in lineage_rows if row.get("synthetic_id", "") not in dropped
    ]
    _write_csv(output / "manifests" / "dataset_manifest.csv", kept_manifest, DATASET_FIELDS)
    _write_csv(output / "manifests" / "lineage_private.csv", kept_lineage, LINEAGE_FIELDS)
    logger.info(
        "case 재생성 대상 제외 | case %s | 샘플 %d건 | 잔여 %d건",
        ", ".join(sorted(drop_cases)),
        len(dropped),
        len(kept_manifest),
    )
    return kept_manifest, kept_lineage


def _effective_jobs(config: dict[str, Any], task_count: int) -> int:
    requested = max(1, int(config.get("jobs", 1)))
    measured_worker_bytes = config.get("worker_peak_rss_bytes")
    if requested > 1 and not measured_worker_bytes:
        return 1
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        conservative_worker_bytes = int(measured_worker_bytes or 1)
        memory_jobs = max(
            1,
            int(
                available
                * float(config.get("memory_budget_ratio", 0.70))
                // conservative_worker_bytes
            ),
        )
        return min(requested, memory_jobs, max(1, task_count))
    except Exception:
        return int(config.get("memory_probe_fallback_jobs", 1))


def generate(
    raw_root: Path,
    config: dict[str, Any],
    plan_path: Path,
    output: Path,
    limit_per_modality: int | None = None,
    resume: bool = False,
    trust_plan: bool = False,
    drop_cases: set[str] | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty: {output}")
    if drop_cases and not resume:
        raise ValueError("--drop-cases requires --resume")
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("quality_fail_augment.generate", output / "logs" / "generation.log")
    rows = _read_csv(plan_path)
    _validate_plan(raw_root, config, rows, trust_plan)
    # Publish the frozen planning contract and audit beside the generated dataset.
    output_manifests = output / "manifests"
    output_manifests.mkdir(parents=True, exist_ok=True)
    for source in sorted(plan_path.parent.glob("*")):
        if source.is_file():
            shutil.copy2(source, output_manifests / source.name)
    plan_log = plan_path.parent.parent / "logs" / "plan.log"
    if plan_log.is_file():
        (output / "logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_log, output / "logs" / "plan.log")
    reserve_path = plan_path.parent / "reserve_sources.csv"
    reserve_rows = _read_csv(reserve_path) if reserve_path.exists() else []
    reserve_queues = {
        modality: iter(
            sorted(
                [row for row in reserve_rows if row.get("modality") == modality],
                key=lambda row: int(row.get("reserve_rank", 0)),
            )
        )
        for modality in ("CT", "RGB")
    }
    consumed_source_keys = {
        (row["raw_split"].casefold(), row["source_stem"].casefold()) for row in rows
    }
    # (source_key, failure_case) pairs whose geometry search is known to be
    # unsatisfiable. Kept for the whole run so one dead reserve is paid for once.
    exhausted_reserve_keys: set[tuple[tuple[str, str], str]] = set()
    selected = list(rows)
    if limit_per_modality is not None:
        limited: list[dict[str, str]] = []
        for modality in ("CT", "RGB"):
            modality_rows = [row for row in selected if row["modality"] == modality]
            fail_count = round(
                limit_per_modality
                * int(config[f"{modality.lower()}_augmented_target"])
                / int(config[f"{modality.lower()}_target"])
            )
            limited.extend(
                [row for row in modality_rows if row["quality_label"] == "fail"][:fail_count]
            )
            limited.extend(
                [row for row in modality_rows if row["quality_label"] == "pass"][
                    : limit_per_modality - fail_count
                ]
            )
        selected = limited

    manifest_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    manifest_path = output / "manifests" / "dataset_manifest.csv"
    if resume and manifest_path.exists():
        manifest_rows = _read_csv(manifest_path)
        lineage_path = output / "manifests" / "lineage_private.csv"
        lineage_rows = _read_csv(lineage_path) if lineage_path.exists() else []
        if drop_cases:
            manifest_rows, lineage_rows = _drop_failure_cases(
                output, manifest_rows, lineage_rows, drop_cases, logger
            )
        verify_dataset(output, manifest_rows)
    if resume:
        _cleanup_uncommitted(output, manifest_rows)
    existing = {row["synthetic_id"] for row in manifest_rows}
    tasks = [row for row in selected if row["synthetic_id"] not in existing]
    effective_jobs = _effective_jobs(config, len(tasks))
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 25)))
    chunk_size = max(
        1,
        len(tasks)
        // max(1, effective_jobs * int(config.get("parallel_chunk_multiplier", 8))),
    )
    progress = ProgressReporter(
        logger,
        "데이터 생성",
        len(selected),
        initial=len(existing),
        item_name="sample",
        log_every_items=max(1, int(config.get("generation_log_interval", 25))),
        log_every_seconds=float(config.get("generation_log_seconds", 30)),
    )
    generation_started = time.monotonic()
    observed_peak_rss = 0
    peak_rss_by_pid: dict[int, int] = {}
    monitor_stop = threading.Event()
    monitored_pids: set[int] = {os.getpid()}

    def monitor_memory() -> None:
        nonlocal observed_peak_rss
        try:
            import psutil

            interval = float(config.get("memory_sample_interval_seconds", 0.5))
            while not monitor_stop.wait(interval):
                total = 0
                for pid in list(monitored_pids):
                    try:
                        rss = int(psutil.Process(pid).memory_info().rss)
                        total += rss
                        peak_rss_by_pid[pid] = max(
                            peak_rss_by_pid.get(pid, 0), rss
                        )
                    except (psutil.Error, OSError):
                        continue
                observed_peak_rss = max(observed_peak_rss, total)
        except Exception:
            return

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    def try_reserves(
        failed_row: dict[str, str], initial_error: Exception
    ) -> tuple[Any | None, Exception | None]:
        last_error: Exception = initial_error
        queue = reserve_queues[failed_row["modality"]]
        failure_case = failed_row.get("failure_case", "")
        attempts = 0
        for reserve in queue:
            if attempts >= RESERVE_ATTEMPTS_PER_SAMPLE:
                break
            source_key = (
                reserve["raw_split"].casefold(),
                reserve["source_stem"].casefold(),
            )
            if source_key in consumed_source_keys:
                continue
            # A reserve that exhausted its search for this case will do so again:
            # the search is driven by the source image, not by the seed. Without
            # this memo every later failure re-walked the same dead reserves, which
            # is what turned a few hundred failures into an unfinishable run.
            if (source_key, failure_case) in exhausted_reserve_keys:
                continue
            attempts += 1
            image_path = _safe_source(raw_root, reserve["raw_image_path"])
            json_path = _safe_source(raw_root, reserve["raw_json_path"])
            if (
                sha256_file(image_path) != reserve["image_sha256"]
                or sha256_file(json_path) != reserve["json_sha256"]
            ):
                last_error = ValueError(
                    f"reserve source changed: {reserve['raw_image_path']}"
                )
                continue
            parsed = ParsedName.parse(reserve["source_stem"])
            if parsed is None:
                last_error = ValueError(
                    f"invalid reserve stem: {reserve['source_stem']}"
                )
                continue
            replacement = dict(failed_row)
            replacement.update(
                {
                    "raw_split": reserve["raw_split"],
                    "raw_image_path": reserve["raw_image_path"],
                    "raw_json_path": reserve["raw_json_path"],
                    "source_stem": reserve["source_stem"],
                    "form": parsed.form,
                    "axis": parsed.axis,
                    "original_battery_id": parsed.battery_id,
                    "original_image_id": parsed.image_id,
                    "image_sha256": reserve["image_sha256"],
                    "json_sha256": reserve["json_sha256"],
                    "replacement_for": failed_row["raw_image_path"],
                }
            )
            try:
                manifest, lineage = _make_one(
                    str(raw_root), str(output), replacement, config
                )
                manifest["replacement_for"] = failed_row["raw_image_path"]
                consumed_source_keys.add(source_key)
                return (manifest, lineage), None
            except Exception as exc:
                last_error = exc
                if EXHAUSTED_SEARCH_MARKER in str(exc):
                    exhausted_reserve_keys.add((source_key, failure_case))
        return None, last_error

    def consume(row: dict[str, str], result: Any = None, error: Exception | None = None) -> None:
        if error is not None:
            result, unresolved = try_reserves(row, error)
            if unresolved is not None:
                errors.append(
                    {
                        "synthetic_id": row["synthetic_id"],
                        "modality": row["modality"],
                        "raw_image_path": row["raw_image_path"],
                        "attempt": int(config.get("max_augmentation_retries", 8)),
                        "error": f"{type(unresolved).__name__}: {unresolved}",
                    }
                )
            else:
                manifest, lineage = result
                manifest_rows.append(manifest)
                lineage_rows.append(lineage)
                existing.add(row["synthetic_id"])
        else:
            manifest, lineage = result
            manifest_rows.append(manifest)
            lineage_rows.append(lineage)
            existing.add(row["synthetic_id"])
        # Checkpoint every `checkpoint_interval` consumed samples, not every one:
        # rewriting the full manifest CSVs per sample is O(n^2) over a 40k run and
        # widens the window where a reader/antivirus can trip the atomic replace.
        # The loop flushes a final checkpoint once generation finishes.
        if (len(manifest_rows) + len(errors)) % checkpoint_interval == 0:
            _write_checkpoint(output, manifest_rows, lineage_rows, errors)
        staging = output / ".sample_staging" / row["synthetic_id"]
        _remove_path_with_retry(staging)
        progress.update(len(existing), detail=f"실패 {len(errors):,}")

    if effective_jobs <= 1:
        for row in tasks:
            try:
                consume(row, _make_one(str(raw_root), str(output), row, config))
            except Exception as exc:
                consume(row, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            iterator = executor.map(
                _make_one_safe,
                [str(raw_root)] * len(tasks),
                [str(output)] * len(tasks),
                tasks,
                [config] * len(tasks),
                chunksize=chunk_size,
            )
            monitored_pids.update(
                process.pid
                for process in getattr(executor, "_processes", {}).values()
                if process.pid is not None
            )
            for row, (ok, payload) in zip(tasks, iterator, strict=True):
                if ok:
                    consume(row, payload)
                else:
                    consume(row, error=RuntimeError(payload))
    _write_checkpoint(output, manifest_rows, lineage_rows, errors)
    monitor_stop.set()
    monitor_thread.join(timeout=2.0)
    generation_elapsed = max(time.monotonic() - generation_started, 1e-9)
    if errors:
        raise ValueError(f"{len(errors)} samples failed; inspect generation_errors.csv")

    verify_dataset(output, manifest_rows)
    counts = Counter((row["modality"], row["quality_label"]) for row in manifest_rows)
    if limit_per_modality is None:
        expected = Counter(
            {
                ("CT", "pass"): int(config["ct_target"]) - int(config["ct_augmented_target"]),
                ("CT", "fail"): int(config["ct_augmented_target"]),
                ("RGB", "pass"): int(config["rgb_target"]) - int(config["rgb_augmented_target"]),
                ("RGB", "fail"): int(config["rgb_augmented_target"]),
            }
        )
        if counts != expected:
            raise ValueError(f"Final counts differ: actual={dict(counts)}, expected={dict(expected)}")
        partition_counts = Counter(
            (row["modality"], row["partition"], row["quality_label"])
            for row in manifest_rows
        )
        expected_partitions: Counter[tuple[str, str, str]] = Counter()
        for modality in ("CT", "RGB"):
            prefix = modality.lower()
            target = int(config[f"{prefix}_target"])
            fail_target = int(config[f"{prefix}_augmented_target"])
            test_target = int(config[f"{prefix}_test_target"])
            test_fail = int(config[f"{prefix}_test_fail_target"])
            expected_partitions[(modality, "test", "fail")] = test_fail
            expected_partitions[(modality, "test", "pass")] = test_target - test_fail
            expected_partitions[(modality, "main", "fail")] = fail_target - test_fail
            expected_partitions[(modality, "main", "pass")] = (
                target - fail_target - (test_target - test_fail)
            )
        if partition_counts != expected_partitions:
            raise ValueError(
                f"main/test counts differ: actual={dict(partition_counts)}, "
                f"expected={dict(expected_partitions)}"
            )
    case_counts = Counter(
        (row["modality"], row["failure_case"])
        for row in manifest_rows
        if row["quality_label"] == "fail"
    )
    if limit_per_modality is None:
        expected_cases = Counter(
            {
                (modality, case): int(count)
                for modality in ("CT", "RGB")
                for case, count in config[
                    f"{modality.lower()}_failure_case_quotas"
                ].items()
            }
        )
        if case_counts != expected_cases:
            raise ValueError(
                f"failure case quotas differ: actual={dict(case_counts)}, "
                f"expected={dict(expected_cases)}"
            )
    summary = {
        "schema_version": "1.1",
        "package_version": __version__,
        "raw_fingerprint": rows[0]["raw_fingerprint"],
        "config_sha256": rows[0]["config_sha256"],
        "counts": {f"{modality}_{quality}": count for (modality, quality), count in counts.items()},
        "failure_case_counts": {
            f"{modality}_{case}": count for (modality, case), count in case_counts.items()
        },
        "requested_jobs": int(config.get("jobs", 1)),
        "effective_jobs": effective_jobs,
        "chunk_size": chunk_size,
        "worker_peak_rss_bytes_observed": max(
            (
                value
                for pid, value in peak_rss_by_pid.items()
                if pid != os.getpid()
            ),
            default=peak_rss_by_pid.get(os.getpid(), 0),
        ),
        "aggregate_peak_rss_bytes_observed": observed_peak_rss,
        "process_peak_rss_bytes_by_pid": {
            str(pid): value for pid, value in sorted(peak_rss_by_pid.items())
        },
        "generation_elapsed_seconds": generation_elapsed,
        "average_sample_seconds": generation_elapsed / max(len(tasks), 1),
        "generation_samples_per_second": len(tasks) / generation_elapsed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "Pillow": PIL.__version__,
            "numpy": np.__version__,
            "shapely": shapely.__version__,
        },
        "serialization": {
            "json": "UTF-8, sorted keys, compact separators, LF",
            "jpeg": "Pillow q90, optimize=false, progressive=false",
        },
    }
    atomic_write(
        output / "generation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode(),
    )
    archive_path = output / "augmentation_json_4k_v2.0.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.glob("*/*/augmentation_json/*.augmentation.json")):
            archive.write(path, path.relative_to(output).as_posix())
    return summary


def _coordinate_values(value: Any) -> list[tuple[float, float]]:
    return [point for ring in point_rings(value) for point in ring]


def _verify_row_worker(
    payload: tuple[str, dict[str, Any]]
) -> tuple[str, tuple[str, str], str, str]:
    """한 출력 샘플의 모든 로컬 검증을 독립 수행한다(멀티프로세스 워커).

    이미지 디코드·해시·polygon·라벨·이력 검증 등 전역 상태가 필요 없는 검사를 모두
    처리하고, 중복 판정에 필요한 키(synthetic_id, (modality, stem), image_sha256,
    output_pixel_hash)만 반환한다. 4종 중복 검사는 메인이 row 순서대로 직렬 수행한다.
    유효 데이터셋에서는 병렬/직렬 산출이 동일하다.
    """
    output_text, row = payload
    output = Path(output_text)
    image_path = output / row["image_path"]
    label_path = output / row["label_json_path"]
    history_text = row.get("augmentation_json_path", "")
    history_path = output / history_text if history_text else None
    if not image_path.is_file() or not label_path.is_file():
        raise ValueError(f"Missing image/label output: {row['synthetic_id']}")
    if row["quality_label"] == "pass":
        if history_text or row.get("augmentation_json_sha256"):
            raise ValueError(f"PASS sample references augmentation JSON: {row['synthetic_id']}")
        unexpected = (
            image_path.parent.parent
            / "augmentation_json"
            / f"{image_path.stem}.augmentation.json"
        )
        if unexpected.exists():
            raise ValueError(f"PASS sample has augmentation JSON: {row['synthetic_id']}")
    else:
        if history_path is None or not history_path.is_file():
            raise ValueError(f"FAIL sample missing augmentation JSON: {row['synthetic_id']}")
    label = load_json(label_path)
    image = open_normalized(image_path, row["modality"])
    if label.get("quality_class") != row["quality_label"]:
        raise ValueError(f"quality_class mismatch: {row['synthetic_id']}")
    if Path(label["image_info"]["file_name"]).name != image_path.name:
        raise ValueError(f"label file_name mismatch: {row['synthetic_id']}")
    if (label["image_info"]["width"], label["image_info"]["height"]) != image.size:
        raise ValueError(f"label size mismatch: {row['synthetic_id']}")
    for points in [label.get("swelling", {}).get("battery_outline")] + [
        defect.get("points") for defect in label.get("defects") or []
    ]:
        if points in (None, []):
            continue
        for x, y in _coordinate_values(points):
            if not (0 <= x <= image.width and 0 <= y <= image.height):
                raise ValueError(f"polygon outside output frame: {row['synthetic_id']}")
    if history_path is not None:
        history = load_json(history_path)
        if history.get("schema_version") != "1.1":
            raise ValueError(f"augmentation schema mismatch: {row['synthetic_id']}")
        if history.get("failure_case_count") != 1:
            raise ValueError(f"FAIL case count mismatch: {row['synthetic_id']}")
        if history.get("output_image_file") != image_path.name:
            raise ValueError(f"history image link mismatch: {row['synthetic_id']}")
        if history.get("label_json_file") != label_path.name:
            raise ValueError(f"history label link mismatch: {row['synthetic_id']}")
        if history.get("quality_label") != "fail":
            raise ValueError(f"history quality mismatch: {row['synthetic_id']}")
    if sha256_file(image_path) != row["image_sha256"]:
        raise ValueError(f"image hash mismatch: {row['synthetic_id']}")
    output_pixel_hash = pixel_hash(image)
    if sha256_file(label_path) != row["label_json_sha256"]:
        raise ValueError(f"label hash mismatch: {row['synthetic_id']}")
    if history_path and sha256_file(history_path) != row["augmentation_json_sha256"]:
        raise ValueError(f"history hash mismatch: {row['synthetic_id']}")
    return (
        row["synthetic_id"],
        (row["modality"], image_path.stem),
        row["image_sha256"],
        output_pixel_hash,
    )


def verify_dataset(
    output: Path, rows: list[dict[str, Any]] | None = None, jobs: int | None = None
) -> None:
    rows = rows if rows is not None else _read_csv(output / "manifests" / "dataset_manifest.csv")
    if jobs is None:
        jobs = min(8, os.cpu_count() or 1)
    seen_ids: set[str] = set()
    seen_stems: set[tuple[str, str]] = set()
    seen_image_hashes: set[str] = set()
    seen_pixel_hashes: set[str] = set()

    def merge(result: tuple[str, tuple[str, str], str, str]) -> None:
        synthetic_id, stem_key, image_hash, output_pixel_hash = result
        if synthetic_id in seen_ids:
            raise ValueError(f"Duplicate synthetic ID: {synthetic_id}")
        seen_ids.add(synthetic_id)
        if stem_key in seen_stems:
            raise ValueError(f"Duplicate output stem: {stem_key[1]}")
        seen_stems.add(stem_key)
        if image_hash in seen_image_hashes:
            raise ValueError(f"duplicate output image SHA-256: {synthetic_id}")
        seen_image_hashes.add(image_hash)
        if output_pixel_hash in seen_pixel_hashes:
            raise ValueError(f"duplicate output pixel hash: {synthetic_id}")
        seen_pixel_hashes.add(output_pixel_hash)

    if jobs > 1 and len(rows) > 1:
        chunk = max(1, len(rows) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for result in executor.map(
                _verify_row_worker,
                [(str(output), row) for row in rows],
                chunksize=chunk,
            ):
                merge(result)
    else:
        for row in rows:
            merge(_verify_row_worker((str(output), row)))
