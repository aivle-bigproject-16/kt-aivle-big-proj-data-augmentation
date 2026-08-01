from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from .augment import CT_CASES, RGB_CASES, _largest_connected_component
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
from .geometry import (
    extract_ct_roi,
    point_rings,
    porosity_bbox_metric,
    repaired_polygons,
)
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


# 실행 성능만 바꾸고 산출물 바이트에는 영향을 주지 않는 키다.
#
# 이 키들이 config 해시에 들어가면, smoke 가 측정한 worker_peak_rss_bytes 를 config 에
# 넣는 순간 승인된 plan 이 무효가 되어 전체 재스캔(약 66분)을 다시 해야 한다. 계획
# 산출물이 한 바이트도 달라지지 않는데 치르는 비용이므로 해시 재료에서 뺀다.
#
# 각 키가 무해한 근거:
#   jobs, parallel_chunk_multiplier, worker_peak_rss_bytes, memory_budget_ratio,
#   memory_probe_fallback_jobs, memory_sample_interval_seconds
#       워커 수와 청크 크기만 정한다. 병렬·직렬 산출물과 raw fingerprint 가 동일함은
#       _validate_pair_worker 의 계약이다.
#   plan_log_interval, plan_checkpoint_interval, checkpoint_interval,
#   max_logged_error_examples
#       로그와 체크포인트 주기만 정한다.
#   hash_chunk_bytes
#       sha256_file 이 파일을 몇 바이트씩 읽는지만 정한다. 해시값은 같다.
#
# 목록에 없는 키는 전부 해시에 남는다. 새 키가 추가되면 기본적으로 plan 이 무효화되는
# 쪽으로 동작하므로, 빠뜨렸을 때 안전한 방향으로 실패한다.
PERFORMANCE_ONLY_KEYS = frozenset(
    {
        "checkpoint_interval",
        "hash_chunk_bytes",
        "jobs",
        "max_logged_error_examples",
        "memory_budget_ratio",
        "memory_probe_fallback_jobs",
        "memory_sample_interval_seconds",
        "parallel_chunk_multiplier",
        "plan_checkpoint_interval",
        "plan_log_interval",
        "worker_peak_rss_bytes",
    }
)


def _config_hash(config: dict[str, Any]) -> str:
    material = {key: value for key, value in config.items() if key not in PERFORMANCE_ONLY_KEYS}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pcg64_shuffle(values: list[Any], seed: int) -> None:
    """Shuffle in place with the v2.0 canonical NumPy PCG64 generator."""
    if len(values) < 2:
        return
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(values))
    values[:] = [values[int(index)] for index in order]


# scan 캐시 스키마 버전. _validate_pair 가 쓰는 검증 로직 — open_normalized, pixel_hash,
# point_rings, extract_ct_roi, porosity_bbox_metric, 필수 필드 목록 — 이 바뀌면 반드시
# 올린다. 올리지 않으면 낡은 판정을 그대로 재사용하게 된다.
SCAN_CACHE_VERSION = "3"
EMBEDDED_SCAN_CACHE = Path(__file__).resolve().parents[2] / "cache" / "scan_cache.csv"

SCAN_CACHE_FIELDS = [
    "cache_version",
    "raw_split",
    "modality",
    "image_stem",
    "raw_image_path",
    "raw_json_path",
    "image_size",
    "image_mtime_ns",
    "json_size",
    "json_mtime_ns",
    "status",
    "image_sha256",
    "json_sha256",
    "pixel_hash",
    "width",
    "height",
    "porosity_bbox_max_ratio",
    "porosity_component_count",
    "has_battery_outline",
    "roi",
    "exclusion_reason",
]


def _stat_signature(path: Path) -> tuple[str, str]:
    info = path.stat()
    return str(info.st_size), str(info.st_mtime_ns)


def _load_scan_cache(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """scan 캐시 CSV 를 pair key 로 색인해 읽는다.

    디렉터리를 주면 그 안의 manifests/scan_cache.csv 또는 scan_cache.csv 를 찾는다.
    """
    if path.is_dir():
        for nested in (path / "manifests" / "scan_cache.csv", path / "scan_cache.csv"):
            if nested.is_file():
                path = nested
                break
        else:
            raise ValueError(f"scan cache not found under directory: {path}")
    if not path.is_file():
        raise ValueError(f"scan cache not found: {path}")
    entries: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCAN_CACHE_FIELDS:
            raise ValueError(
                f"scan cache schema mismatch: {path}\n"
                f"  expected: {SCAN_CACHE_FIELDS}\n  found: {reader.fieldnames}"
            )
        for entry in reader:
            if entry["cache_version"] != SCAN_CACHE_VERSION:
                raise ValueError(
                    f"scan cache version {entry['cache_version']!r} != "
                    f"{SCAN_CACHE_VERSION!r}; rebuild the cache: {path}"
                )
            key = (entry["raw_split"], entry["modality"], entry["image_stem"].casefold())
            entries[key] = entry
    return entries


def _cache_fresh(entry: dict[str, str], image_path: Path, json_path: Path) -> bool:
    """캐시 항목이 현재 원본 파일과 같은 크기·수정시각인지 본다.

    sha256 재계산이 이 캐시가 없애려는 비용 자체이므로, 신선도는 stat 로만 판정한다.
    plan 에 실제로 선택된 소스는 generate 가 sha256 으로 다시 pin 검증하므로
    (_validate_plan), 실제 사용되는 40k 장의 무결성은 그대로 보장된다.
    """
    try:
        image_size, image_mtime = _stat_signature(image_path)
        json_size, json_mtime = _stat_signature(json_path)
    except OSError:
        return False
    return (
        entry["image_size"] == image_size
        and entry["image_mtime_ns"] == image_mtime
        and entry["json_size"] == json_size
        and entry["json_mtime_ns"] == json_mtime
    )


def _result_from_cache(
    entry: dict[str, str],
    image_path: Path,
    json_path: Path,
    config: dict[str, Any],
) -> tuple[Candidate | None, dict[str, Any], dict[str, Any]]:
    """캐시 항목에서 _validate_pair 와 동일한 (candidate, audit row, cache entry) 를 만든다.

    ct_porosity_threshold 는 캐시된 비율 원값에 다시 적용한다. 임계값만 바꾼 경우
    캐시를 버리지 않아도 되도록 하기 위해서다.
    """
    stem = entry["image_stem"]
    modality = entry["modality"]
    parsed = ParsedName.parse(stem)
    if parsed is None:
        raise ValueError(f"invalid source stem in scan cache: {stem}")
    base: dict[str, Any] = {
        "modality": modality,
        "raw_split": entry["raw_split"],
        "image_stem": stem,
        "battery_id": parsed.battery_id,
        "image_id": parsed.image_id,
        "raw_image_path": entry["raw_image_path"],
        "raw_json_path": entry["raw_json_path"],
    }
    if entry["status"] != "valid":
        return (
            None,
            {
                **base,
                "validation_status": "excluded_source",
                "error_level": "excluded_source",
                "exclusion_reason": entry["exclusion_reason"],
            },
            entry,
        )
    ratio = float(entry["porosity_bbox_max_ratio"])
    excluded = modality == "CT" and ratio >= float(config["ct_porosity_threshold"])
    row = {
        **base,
        "image_sha256": entry["image_sha256"],
        "json_sha256": entry["json_sha256"],
        "pixel_hash": entry["pixel_hash"],
        "validation_status": "excluded_source" if excluded else "valid",
        "error_level": "excluded_source" if excluded else "valid",
        "exclusion_reason": ("ct_porosity_bbox_max_ratio_ge_0.25" if excluded else ""),
        "porosity_bbox_max_ratio": f"{ratio:.8f}",
        "ct_bbox_excluded": str(excluded).lower(),
    }
    candidate = Candidate(
        entry["raw_split"],
        image_path,
        json_path,
        parsed,
        {},
        int(entry["width"]),
        int(entry["height"]),
        entry["image_sha256"],
        entry["json_sha256"],
        entry["pixel_hash"],
        tuple(json.loads(entry["roi"])) if entry["roi"] else None,
        ratio,
        int(entry["porosity_component_count"]),
        entry["has_battery_outline"].casefold() == "true",
    )
    return (None if excluded else candidate), row, entry


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
) -> tuple[Candidate | None, dict[str, Any], dict[str, Any]]:
    parsed = ParsedName.parse(stem)
    if parsed is None:
        raise ValueError(f"invalid source stem: {stem}")
    try:
        image_size, image_mtime = _stat_signature(image_path)
        json_size, json_mtime = _stat_signature(json_path)
    except OSError:
        image_size = image_mtime = json_size = json_mtime = ""
    cache_entry: dict[str, Any] = {
        "cache_version": SCAN_CACHE_VERSION,
        "raw_split": split,
        "modality": modality,
        "image_stem": stem,
        "raw_image_path": _safe_relative(image_path, raw_root),
        "raw_json_path": _safe_relative(json_path, raw_root),
        "image_size": image_size,
        "image_mtime_ns": image_mtime,
        "json_size": json_size,
        "json_mtime_ns": json_mtime,
        "status": "error",
        "image_sha256": "",
        "json_sha256": "",
        "pixel_hash": "",
        "width": "",
        "height": "",
        "porosity_bbox_max_ratio": "",
        "porosity_component_count": "",
        "has_battery_outline": "",
        "roi": "",
        "exclusion_reason": "",
    }
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
        has_battery_outline = False
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
        outline_frame = roi if roi is not None else (0.0, 0.0, image.width, image.height)
        has_battery_outline = bool(repaired_polygons(outline, outline_frame))
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
            has_battery_outline,
        )
        # 캐시에는 제외 판정 이전의 사실만 담는다. excluded 는 ct_porosity_threshold 를
        # 다시 적용해 복원하므로, 임계값만 바뀐 경우 캐시가 그대로 유효하다.
        cache_entry.update(
            {
                "status": "valid",
                "image_sha256": image_hash,
                "json_sha256": json_hash,
                "pixel_hash": pixels,
                "width": image.width,
                "height": image.height,
                "porosity_bbox_max_ratio": f"{ratio:.8f}",
                "porosity_component_count": component_count,
                "has_battery_outline": str(has_battery_outline).lower(),
                "roi": json.dumps(roi) if roi is not None else "",
            }
        )
        return (None if excluded else candidate), row, cache_entry
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        cache_entry["exclusion_reason"] = reason
        return (
            None,
            {
                **base,
                "validation_status": "excluded_source",
                "error_level": "excluded_source",
                "exclusion_reason": reason,
            },
            cache_entry,
        )


def _scan_jobs(config: dict[str, Any]) -> int:
    requested = max(1, int(config.get("jobs", 1)))
    return min(requested, os.cpu_count() or 1)


def _validate_pair_worker(
    payload: tuple[Path, str, str, str, Path, Path, dict[str, Any]]
) -> tuple[Candidate | None, dict[str, Any], dict[str, Any]]:
    """1:1 이미지-JSON 쌍을 독립 검증한다(멀티프로세스 워커).

    공유 상태를 만지지 않고 (candidate, row, cache entry)만 반환한다. 순서 의존 병합
    (픽셀 중복 제거, 계통 오류 누적, 감사 순서)은 메인이 정렬 순서대로
    직렬 수행하므로 병렬/직렬 산출물과 raw fingerprint가 동일하다.
    검증 후 사용되지 않는 label(파싱된 JSON)은 IPC 비용을 줄이려 비운다.
    """
    raw_root, split, modality, stem, image_path, json_path, config = payload
    candidate, row, cache_entry = _validate_pair(
        raw_root, split, modality, stem, image_path, json_path, config
    )
    if candidate is not None:
        candidate.label = {}
    return candidate, row, cache_entry


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
            candidate, row, _ = _validate_pair(
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
    reuse_scan: Path | None = None,
    cache_path: Path | None = None,
    cache_only: bool = False,
) -> tuple[list[Candidate], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    cache = _load_scan_cache(reuse_scan) if reuse_scan else {}
    if cache_only:
        if not cache:
            raise ValueError("cache-only planning requires a non-empty scan cache")
        groups: dict[tuple[str, str, str], dict[str, list[Path]]] = {}
        for key, entry in cache.items():
            groups[key] = {
                "images": [raw_root / Path(entry["raw_image_path"])],
                "json": [raw_root / Path(entry["raw_json_path"])],
            }
        discovered = len(groups) * 2
        if logger:
            logger.info(
                "내장 scan cache 전용 모드 | 폴더 탐색·preflight·전체 stat 생략 | pair %s",
                f"{len(groups):,}",
            )
    else:
        groups, discovered = _index_raw(raw_root, logger)
        _preflight(raw_root, groups, config)
    sorted_groups = sorted(groups.items())
    results_by_key: dict[
        tuple[str, str, str], tuple[Candidate | None, dict[str, Any], dict[str, Any]]
    ] = {}
    pair_keys: list[tuple[str, str, str]] = []
    pair_payloads: list[tuple[Path, str, str, str, Path, Path, dict[str, Any]]] = []
    for key, group in sorted_groups:
        if len(group["images"]) == 1 and len(group["json"]) == 1:
            split, modality, _ = key
            image_path, json_path = group["images"][0], group["json"][0]
            stem = image_path.stem
            entry = cache.get(key)
            if entry is not None and (
                cache_only or _cache_fresh(entry, image_path, json_path)
            ):
                results_by_key[key] = _result_from_cache(entry, image_path, json_path, config)
                continue
            pair_keys.append(key)
            pair_payloads.append(
                (raw_root, split, modality, stem, image_path, json_path, config)
            )
    jobs = _scan_jobs(config)
    if logger:
        if cache:
            logger.info(
                "scan 캐시 적중 | 재사용 %s | 재검증 %s",
                f"{len(results_by_key):,}",
                f"{len(pair_payloads):,}",
            )
        logger.info(
            "쌍 검증 시작 | pair %s | worker %s",
            f"{len(pair_payloads):,}",
            jobs,
        )
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
            candidate, row, _ = results_by_key[key]
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
    if cache_path is not None:
        # 정렬 순서대로 써서 같은 원본이면 캐시 파일도 바이트 동일하게 나온다.
        _write_csv(
            cache_path,
            [results_by_key[key][2] for key, _ in sorted_groups if key in results_by_key],
            SCAN_CACHE_FIELDS,
        )
    if logger:
        logger.info(
            "후보 탐색 완료 | 전체 파일 %s | pair key %s | 정상 %s",
            f"{discovered:,}",
            f"{len(groups):,}",
            f"{len(candidates):,}",
        )
    return candidates, audit, systemic, orphans


def _validate_config(config: dict[str, Any]) -> None:
    if str(config.get("rng_algorithm", "PCG64")).upper() != "PCG64":
        raise ValueError("rng_algorithm must be PCG64")
    if str(config.get("conveyor_axis", "horizontal")).casefold() not in {
        "horizontal",
        "vertical",
    }:
        raise ValueError("conveyor_axis must be horizontal or vertical")
    if str(config.get("forward_direction", "positive")).casefold() not in {
        "positive",
        "negative",
    }:
        raise ValueError("forward_direction must be positive or negative")
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
        test_quotas = config.get(f"{modality}_test_failure_case_quotas")
        if test_quotas is not None:
            if set(test_quotas) != set(quotas):
                raise ValueError(
                    f"{modality.upper()} test failure-case quota keys must match "
                    "the full failure-case quota keys"
                )
            if any(
                not 0 <= int(test_quotas[case]) <= int(quotas[case])
                for case in quotas
            ):
                raise ValueError(
                    f"{modality.upper()} test failure-case quotas must be between "
                    "zero and their full quotas"
                )
            if sum(int(value) for value in test_quotas.values()) != test_fail:
                raise ValueError(
                    f"{modality.upper()} test failure-case quotas do not sum to "
                    f"{test_fail}"
                )
        if not (0 <= test_fail <= fail_target and 0 <= test_target - test_fail <= target - fail_target):
            raise ValueError(f"{modality.upper()} test PASS/FAIL targets are impossible")


def _case_assignments(config: dict[str, Any], modality: str) -> list[str]:
    quotas = config[f"{modality.lower()}_failure_case_quotas"]
    cases = [case for case, count in quotas.items() for _ in range(int(count))]
    _pcg64_shuffle(cases, stable_seed(config["seed"], modality, "failure-cases"))
    return cases


def _select_battery_group_subset(
    groups: list[tuple[str, int, tuple[int, ...]]],
    *,
    target: int,
    test_minimums: tuple[int, ...],
    main_minimums: tuple[int, ...],
    modality: str,
) -> set[str]:
    """Select whole battery groups while preserving case capacity on both sides.

    Each capability tuple contains the number of eligible sources in the group.
    Unlike the old v2.0 protection step, this DP never pins a battery merely
    because one of its images is useful to main.  It instead constrains the
    selected test capacity from below and from above (the latter leaves the
    requested residual capacity for main).
    """
    if len(test_minimums) != len(main_minimums):
        raise ValueError("battery-group capability dimensions do not match")
    dimensions = len(test_minimums)
    if any(len(capabilities) != dimensions for _, _, capabilities in groups):
        raise ValueError("battery-group capability dimensions do not match")
    totals = tuple(
        sum(capabilities[index] for _, _, capabilities in groups)
        for index in range(dimensions)
    )
    test_maximums = tuple(
        totals[index] - main_minimums[index] for index in range(dimensions)
    )
    if any(
        maximum < minimum
        for minimum, maximum in zip(test_minimums, test_maximums, strict=True)
    ):
        raise ValueError(
            f"{modality} does not have enough eligible sources to satisfy "
            "test and main failure-case quotas"
        )

    zero_capabilities = (0,) * dimensions
    Record = tuple[tuple[str, ...], tuple[int, ...]]

    def add_to_frontier(frontier: list[Record], candidate: Record) -> None:
        candidate_ids, candidate_capabilities = candidate
        for existing_ids, existing_capabilities in frontier:
            if all(
                existing_capabilities[index] <= candidate_capabilities[index]
                for index in range(dimensions)
            ):
                if (
                    existing_capabilities != candidate_capabilities
                    or existing_ids <= candidate_ids
                ):
                    return
        frontier[:] = [
            (existing_ids, existing_capabilities)
            for existing_ids, existing_capabilities in frontier
            if not all(
                candidate_capabilities[index] <= existing_capabilities[index]
                for index in range(dimensions)
            )
            or (
                candidate_capabilities == existing_capabilities
                and existing_ids < candidate_ids
            )
        ]
        frontier.append(candidate)

    states: dict[tuple[int, tuple[int, ...]], list[Record]] = {
        (0, zero_capabilities): [((), zero_capabilities)]
    }
    for battery_id, size, capabilities in groups:
        additions: dict[tuple[int, tuple[int, ...]], list[Record]] = {}
        for (count, _), records in list(states.items()):
            for selected_ids, selected_capabilities in records:
                next_count = count + size
                if next_count > target:
                    continue
                next_capabilities = tuple(
                    selected_capabilities[index] + capabilities[index]
                    for index in range(dimensions)
                )
                if any(
                    value > test_maximums[index]
                    for index, value in enumerate(next_capabilities)
                ):
                    continue
                capped = tuple(
                    min(next_capabilities[index], test_minimums[index])
                    for index in range(dimensions)
                )
                key = (next_count, capped)
                add_to_frontier(
                    additions.setdefault(key, []),
                    (selected_ids + (battery_id,), next_capabilities),
                )
        for key, candidates in additions.items():
            frontier = states.setdefault(key, [])
            for candidate in candidates:
                add_to_frontier(frontier, candidate)
        feasible = [
            selected_ids
            for (count, _), records in states.items()
            for selected_ids, capabilities in records
            if count == target
            and all(capabilities[index] >= test_minimums[index] for index in range(dimensions))
        ]
        if feasible:
            return set(min(feasible))

    raise ValueError(
        f"{modality} battery-group split has no exact {target}-image subset "
        "that preserves failure-case capacity in both test and main"
    )


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


def _has_dense_ct_anchor(candidate: Candidate) -> bool:
    image = open_normalized(candidate.image_path, "CT")
    if candidate.roi is not None:
        image = image.crop(tuple(int(round(value)) for value in candidate.roi))
    array = np.asarray(image.convert("L"), dtype=np.float32)
    threshold = float(np.quantile(array, 0.99))
    component = _largest_connected_component(array >= threshold)
    return float(component.mean()) >= 0.001


def audit_raw(
    raw_root: Path, config: dict[str, Any], output: Path, reuse_scan: Path | None = None
) -> dict[str, Any]:
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("quality_fail_augment.audit", output / "logs" / "audit.log")
    candidates, audit, systemic, orphans = scan(
        raw_root,
        config,
        logger,
        output / "manifests" / "extraction_audit.csv",
        reuse_scan=reuse_scan,
        cache_path=output / "manifests" / "scan_cache.csv",
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
    raw_root: Path, config: dict[str, Any], output: Path, reuse_scan: Path | None = None
) -> dict[str, Any]:
    _validate_config(config)
    cache_only = False
    if reuse_scan is None and bool(config.get("use_embedded_scan_cache", False)):
        if not EMBEDDED_SCAN_CACHE.is_file():
            raise ValueError(
                f"embedded scan cache is missing: {EMBEDDED_SCAN_CACHE}"
            )
        reuse_scan = EMBEDDED_SCAN_CACHE
        cache_only = True
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("quality_fail_augment.plan", output / "logs" / "plan.log")
    candidates, audit, systemic, orphans = scan(
        raw_root,
        config,
        logger,
        manifests / "extraction_audit.csv",
        reuse_scan=reuse_scan,
        cache_path=manifests / "scan_cache.csv",
        cache_only=cache_only,
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
    dense_anchor_cache: dict[Path, bool] = {}

    def has_dense_anchor(candidate: Candidate) -> bool:
        if candidate.image_path not in dense_anchor_cache:
            dense_anchor_cache[candidate.image_path] = _has_dense_ct_anchor(candidate)
        return dense_anchor_cache[candidate.image_path]
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
        _pcg64_shuffle(pool, stable_seed(config["seed"], modality, "extract"))
        if len(pool) < target + reserve:
            raise ValueError(
                f"{modality}: needs {target + reserve} valid unique sources including reserve, "
                f"found {len(pool)}"
            )
        shuffled = list(range(target))
        _pcg64_shuffle(shuffled, stable_seed(config["seed"], modality, "assignment"))
        fail_indices = set(shuffled[:fail_target])
        case_values = _case_assignments(config, modality)
        case_by_index = dict(zip(sorted(fail_indices), case_values, strict=True))
        available = list(pool[: target + reserve])
        chosen: list[Candidate] = []
        for index in range(target):
            failure_case = case_by_index.get(index, "")
            eligible_index = 0
            if failure_case == "ct_cell_alignment_failure":
                eligible_index = next(
                    (
                        position
                        for position, candidate in enumerate(available)
                        if candidate.has_battery_outline
                    ),
                    -1,
                )
                if eligible_index < 0:
                    raise ValueError(
                        "CT alignment quota cannot be filled from outline sources"
                    )
            elif failure_case == "ct_beam_hardening_metal_streak":
                eligible_index = next(
                    (
                        position
                        for position, candidate in enumerate(available)
                        if has_dense_anchor(candidate)
                    ),
                    -1,
                )
                if eligible_index < 0:
                    raise ValueError(
                        "CT metal-streak quota cannot be filled from dense-anchor sources"
                    )
            elif failure_case == "rgb_reflection_glare":
                eligible_index = next(
                    (
                        position
                        for position, candidate in enumerate(available)
                        if candidate.has_battery_outline
                    ),
                    -1,
                )
                if eligible_index < 0:
                    raise ValueError(
                        "RGB glare quota cannot be filled from outline sources"
                    )
            chosen.append(available.pop(eligible_index))
        reserves = available
        test_target = int(config.get(f"{modality.lower()}_test_target", 0))
        test_fail_target = int(
            config.get(f"{modality.lower()}_test_fail_target", 0)
        )
        test_pass_target = test_target - test_fail_target
        if modality == "CT":
            test_case_quotas = config.get("ct_test_failure_case_quotas")
            if test_case_quotas is None:
                if test_fail_target % len(config["ct_failure_case_quotas"]) != 0:
                    raise ValueError(
                        "ct_test_failure_case_quotas is required when CT test FAIL "
                        "cannot be divided equally across failure cases"
                    )
                per_case = test_fail_target // len(config["ct_failure_case_quotas"])
                test_case_quotas = {
                    case: per_case for case in config["ct_failure_case_quotas"]
                }
            main_case_quotas = {
                case: int(total_count) - int(test_case_quotas.get(case, 0))
                for case, total_count in config["ct_failure_case_quotas"].items()
            }
            # These slots were filled through has_dense_anchor() during source
            # selection above. Reuse that proof instead of decoding all 20,000
            # selected CT images a second time for partition capacity checks.
            dense_eligible_indices = {
                index
                for index, case in case_by_index.items()
                if case == "ct_beam_hardening_metal_streak"
            }
            logger.info(
                "CT source selection complete | selected %s | guaranteed dense %s",
                f"{len(chosen):,}",
                f"{len(dense_eligible_indices):,}",
            )
            # A battery is the leakage boundary for CT. Select complete battery
            # groups totaling the requested test size, then assign FAIL cases
            # inside each partition so both leakage and per-case quotas hold.
            groups: dict[str, list[int]] = {}
            for index, candidate in enumerate(chosen):
                groups.setdefault(candidate.parsed.battery_id, []).append(index)
            group_items = sorted(groups.items())
            _pcg64_shuffle(
                group_items, stable_seed(config["seed"], modality, "test-battery-groups")
            )
            outline_needed = int(
                test_case_quotas.get("ct_cell_alignment_failure", 0)
            )
            dense_needed = int(
                test_case_quotas.get("ct_beam_hardening_metal_streak", 0)
            )
            ct_group_capacities = []
            for battery_id, indices in group_items:
                outline = [
                    bool(chosen[index].has_battery_outline) for index in indices
                ]
                dense = [index in dense_eligible_indices for index in indices]
                ct_group_capacities.append(
                    (
                        battery_id,
                        len(indices),
                        (
                            sum(outline),
                            sum(dense),
                            sum(a or d for a, d in zip(outline, dense, strict=True)),
                        ),
                    )
                )
            test_battery_ids = _select_battery_group_subset(
                ct_group_capacities,
                target=test_target,
                test_minimums=(
                    outline_needed,
                    dense_needed,
                    outline_needed + dense_needed,
                ),
                main_minimums=(
                    int(main_case_quotas.get("ct_cell_alignment_failure", 0)),
                    int(main_case_quotas.get("ct_beam_hardening_metal_streak", 0)),
                    int(main_case_quotas.get("ct_cell_alignment_failure", 0))
                    + int(main_case_quotas.get("ct_beam_hardening_metal_streak", 0)),
                ),
                modality="CT",
            )
            logger.info(
                "CT battery-group split complete | test images %s | test batteries %s",
                f"{test_target:,}",
                f"{len(test_battery_ids):,}",
            )
            test_indices = {
                index
                for index, candidate in enumerate(chosen)
                if candidate.parsed.battery_id in test_battery_ids
            }

            if sum(map(int, test_case_quotas.values())) != test_fail_target:
                raise ValueError("CT test failure-case quotas must sum to ct_test_fail_target")
            if sum(main_case_quotas.values()) != fail_target - test_fail_target:
                raise ValueError("CT main failure-case quotas do not match the main FAIL target")

            def assign_ct_cases(
                indices: set[int], quotas: dict[str, int], scope: str
            ) -> dict[int, str]:
                order = sorted(indices)
                _pcg64_shuffle(
                    order, stable_seed(config["seed"], "CT", scope, "failure-cases")
                )
                remaining = list(order)
                assigned: dict[int, str] = {}
                alignment_needed = int(
                    quotas.get("ct_cell_alignment_failure", 0)
                )
                alignment_order = [
                    index
                    for index in remaining
                    if chosen[index].has_battery_outline
                    and index not in dense_eligible_indices
                ] + [
                    index
                    for index in remaining
                    if chosen[index].has_battery_outline
                    and index in dense_eligible_indices
                ]
                if len(alignment_order) < alignment_needed:
                    raise ValueError(
                        f"CT {scope} cannot fill ct_cell_alignment_failure quota "
                        f"({alignment_needed}) with eligible sources"
                    )
                for index in alignment_order[:alignment_needed]:
                    assigned[index] = "ct_cell_alignment_failure"
                    remaining.remove(index)

                dense_needed_here = int(
                    quotas.get("ct_beam_hardening_metal_streak", 0)
                )
                dense_order = [
                    index for index in remaining if index in dense_eligible_indices
                ]
                if len(dense_order) < dense_needed_here:
                    raise ValueError(
                        f"CT {scope} cannot fill ct_beam_hardening_metal_streak "
                        f"quota ({dense_needed_here}) with eligible sources"
                    )
                for index in dense_order[:dense_needed_here]:
                    assigned[index] = "ct_beam_hardening_metal_streak"
                    remaining.remove(index)

                cases_in_order = [
                    case
                    for case in quotas
                    if case
                    not in {
                        "ct_cell_alignment_failure",
                        "ct_beam_hardening_metal_streak",
                    }
                ]
                for case in cases_in_order:
                    needed = int(quotas.get(case, 0))
                    for _ in range(needed):
                        if not remaining:
                            raise ValueError(
                                f"CT {scope} cannot fill {case} quota ({needed}) "
                                "with eligible sources"
                            )
                        assigned[remaining.pop(0)] = case
                return assigned

            main_indices = set(range(target)) - test_indices
            case_by_index = {
                **assign_ct_cases(
                    test_indices,
                    {case: int(count) for case, count in test_case_quotas.items()},
                    "test",
                ),
                **assign_ct_cases(main_indices, main_case_quotas, "main"),
            }
            fail_indices = set(case_by_index)
        else:
            # RGB uses the same physical-battery leakage boundary as CT.
            # Derive deterministic per-case test quotas proportionally when the
            # config only supplies the total RGB case quotas.
            total_case_quotas = {
                case: int(count)
                for case, count in config["rgb_failure_case_quotas"].items()
            }
            configured_test_quotas = config.get("rgb_test_failure_case_quotas")
            if configured_test_quotas is None:
                exact = {
                    case: count * test_fail_target / fail_target
                    for case, count in total_case_quotas.items()
                }
                test_case_quotas = {
                    case: int(value) for case, value in exact.items()
                }
                remainder = test_fail_target - sum(test_case_quotas.values())
                ranked = sorted(
                    total_case_quotas,
                    key=lambda case: (
                        -(exact[case] - test_case_quotas[case]),
                        case,
                    ),
                )
                for case in ranked[:remainder]:
                    test_case_quotas[case] += 1
            else:
                test_case_quotas = {
                    case: int(count)
                    for case, count in configured_test_quotas.items()
                }
            main_case_quotas = {
                case: total_case_quotas[case] - test_case_quotas.get(case, 0)
                for case in total_case_quotas
            }

            groups: dict[str, list[int]] = {}
            for index, candidate in enumerate(chosen):
                groups.setdefault(candidate.parsed.battery_id, []).append(index)
            group_items = sorted(groups.items())
            _pcg64_shuffle(
                group_items, stable_seed(config["seed"], "RGB", "test-battery-groups")
            )
            glare_needed = int(test_case_quotas.get("rgb_reflection_glare", 0))
            rgb_group_capacities = [
                (
                    battery_id,
                    len(indices),
                    (
                        sum(
                            bool(chosen[index].has_battery_outline)
                            for index in indices
                        ),
                    ),
                )
                for battery_id, indices in group_items
            ]
            test_battery_ids = _select_battery_group_subset(
                rgb_group_capacities,
                target=test_target,
                test_minimums=(glare_needed,),
                main_minimums=(
                    int(main_case_quotas.get("rgb_reflection_glare", 0)),
                ),
                modality="RGB",
            )
            test_indices = {
                index
                for index, candidate in enumerate(chosen)
                if candidate.parsed.battery_id in test_battery_ids
            }

            def assign_rgb_cases(
                indices: set[int], quotas: dict[str, int], scope: str
            ) -> dict[int, str]:
                order = sorted(indices)
                _pcg64_shuffle(
                    order, stable_seed(config["seed"], "RGB", scope, "failure-cases")
                )
                remaining = list(order)
                assigned: dict[int, str] = {}
                cases_in_order = ["rgb_reflection_glare"] + [
                    case for case in quotas if case != "rgb_reflection_glare"
                ]
                for case in cases_in_order:
                    for _ in range(int(quotas.get(case, 0))):
                        position = next(
                            (
                                position
                                for position, index in enumerate(remaining)
                                if case != "rgb_reflection_glare"
                                or chosen[index].has_battery_outline
                            ),
                            -1,
                        )
                        if position < 0:
                            raise ValueError(
                                f"RGB {scope} cannot fill {case} quota with "
                                "eligible sources"
                            )
                        assigned[remaining.pop(position)] = case
                return assigned

            main_indices = set(range(target)) - test_indices
            case_by_index = {
                **assign_rgb_cases(test_indices, test_case_quotas, "test"),
                **assign_rgb_cases(main_indices, main_case_quotas, "main"),
            }
            fail_indices = set(case_by_index)
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
            synthetic_id = f"QF20_{modality}_{slot:08d}"
            failure_case = case_by_index.get(index, "")
            item_seed = stable_seed(config["seed"], synthetic_id)
            case_seed = stable_seed(item_seed, failure_case) if failure_case else item_seed
            group_key = (
                f"{candidate.parsed.battery_id}|{candidate.parsed.axis}"
                if modality == "CT"
                else ""
            )
            group_seed = (
                stable_seed(
                    config["seed"],
                    candidate.parsed.battery_id,
                    candidate.parsed.axis,
                    failure_case,
                )
                if modality == "CT" and failure_case
                else ""
            )
            row = {
                "synthetic_id": synthetic_id,
                "modality": modality,
                "raw_split": candidate.raw_split,
                "raw_image_path": normalized_relative(candidate.image_path, raw_root),
                "raw_json_path": normalized_relative(candidate.json_path, raw_root),
                "source_image_relative_path": normalized_relative(candidate.image_path, raw_root),
                "source_json_relative_path": normalized_relative(candidate.json_path, raw_root),
                "source_stem": candidate.image_path.stem,
                "form": candidate.parsed.form,
                "axis": candidate.parsed.axis,
                "original_battery_id": candidate.parsed.battery_id,
                "original_image_id": candidate.parsed.image_id,
                "new_battery_id": battery_id,
                "new_image_id": image_id,
                "assignment": "augmented" if fail else "original",
                "quality_label": "fail" if fail else "pass",
                "quality_class": "fail" if fail else "pass",
                "partition": partition,
                "failure_case": failure_case,
                "selected": "true",
                "extraction_rank": slot,
                "item_seed": item_seed,
                "case_seed": case_seed,
                "group_key": group_key,
                "group_seed": group_seed,
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
    overlap_counts: dict[str, int] = {}
    for modality in ("CT", "RGB"):
        main_ids = {
            row["original_battery_id"]
            for row in plan_rows
            if row["modality"] == modality and row["partition"] == "main"
        }
        test_ids = {
            row["original_battery_id"]
            for row in plan_rows
            if row["modality"] == modality and row["partition"] == "test"
        }
        overlap = main_ids & test_ids
        overlap_counts[modality] = len(overlap)
        if overlap:
            raise ValueError(
                f"{modality} original battery ID leakage detected between main/test: "
                + ", ".join(sorted(overlap)[:10])
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
        "package_version": "2.0",
        "ct_main_test_original_battery_overlap": overlap_counts["CT"],
        "rgb_main_test_original_battery_overlap": overlap_counts["RGB"],
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
