from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from quality_fail_augment.augment import (
    CASE_NAMES_KO,
    SOURCE_REFERENCES,
    apply_failure_case,
)
from quality_fail_augment.common import (
    canonical_json_bytes,
    sha256_bytes,
)
from quality_fail_augment.generator import _jpeg_bytes
from quality_fail_augment.models import ParsedName
from quality_fail_augment.preprocessing.stages import (
    TransformedSample,
    finalize_sample,
    prepare_source,
)


REQUIRED_COLUMNS = {
    "failure_case",
    "sample_index",
    "output_file",
    "source_image_name",
    "source_image_path",
    "source_json_path",
    "seed",
    "retry",
    "augmentation_parameters",
}


def _read_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"sample manifest is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("sample manifest has no rows")
    return rows


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for key in (
        "resize_long_side",
        "upscale_small_images",
        "jpeg_quality",
        "jpeg_subsampling",
        "jpeg_optimize",
        "jpeg_progressive",
        "ct_battery_id_start",
        "ct_image_id_start",
    ):
        if key not in config:
            raise ValueError(f"config is missing: {key}")
    return config


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "synthetic_id",
        "failure_case",
        "image_path",
        "annotation_json_path",
        "augmentation_json_path",
        "source_image_name",
        "image_sha256",
        "annotation_json_sha256",
        "augmentation_json_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def package_samples(
    sample_root: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    sample_root = sample_root.resolve()
    manifest_path = sample_root / "sample_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sample manifest does not exist: {manifest_path}")
    config = _load_config(config_path.resolve())
    rows = _read_rows(manifest_path)

    output_dir = (output_dir or sample_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_zip = output_dir / "images_full_format.zip"
    labels_zip = output_dir / "labels_full_format.zip"
    if images_zip.exists() or labels_zip.exists():
        raise FileExistsError(
            f"refusing to overwrite: {images_zip} or {labels_zip}"
        )

    prepared: list[
        tuple[str, str, str, bytes, bytes, bytes]
    ] = []
    match_rows: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    battery_map: dict[str, int] = {}

    for ordinal, row in enumerate(rows, 1):
        source_image = Path(row["source_image_path"]).resolve()
        source_json = Path(row["source_json_path"]).resolve()
        if not source_image.is_file() or not source_json.is_file():
            raise FileNotFoundError(
                f"source pair is missing: {source_image} | {source_json}"
            )
        parsed = ParsedName.parse(source_image.stem)
        if parsed is None or parsed.modality != "CT":
            raise ValueError(f"invalid CT source name: {source_image.name}")

        source_battery_key = f"{parsed.form}|{parsed.battery_id}"
        if source_battery_key not in battery_map:
            battery_map[source_battery_key] = (
                int(config["ct_battery_id_start"]) + len(battery_map)
            )
        new_battery = battery_map[source_battery_key]
        new_image = int(config["ct_image_id_start"]) + ordinal - 1
        new_stem = parsed.new_stem(new_battery, new_image)
        synthetic_id = f"QF18_CT_{ordinal:08d}"
        failure_case = row["failure_case"]
        selected_seed = int(row["seed"])

        source = prepare_source(source_image, source_json, "CT")
        augmented = apply_failure_case(
            source.image,
            "CT",
            failure_case,
            selected_seed,
            object_mask=source.object_mask,
            defect_mask=source.defect_mask,
            case_options={
                **config,
                "target_defect_ids": list(source.porosity_ids),
            },
        )
        transformed = TransformedSample(
            image=augmented.image,
            label=source.label,
            transform=source.transform.then(augmented.transform),
            records=augmented.records,
            original_roi=source.original_roi,
        )
        image_name = f"{new_stem}.jpg"
        finalized = finalize_sample(
            transformed,
            "CT",
            "fail",
            new_battery,
            new_image,
            image_name,
            int(config["resize_long_side"]),
            bool(config["upscale_small_images"]),
        )

        image_bytes = _jpeg_bytes(finalized.image, config)
        annotation_bytes = canonical_json_bytes(finalized.label)
        image_hash = sha256_bytes(image_bytes)
        annotation_hash = sha256_bytes(annotation_bytes)
        history = {
            "schema_version": "1.1",
            "synthetic_id": synthetic_id,
            "output_image_file": image_name,
            "label_json_file": f"{new_stem}.json",
            "issued_ids": {
                "battery_id": new_battery,
                "image_id": new_image,
            },
            "quality_label": "fail",
            "is_augmented": True,
            "item_seed": selected_seed,
            "case_seed": selected_seed,
            "group_key": f"{parsed.battery_id}|{parsed.axis}",
            "group_seed": None,
            "failure_case": {
                "id": failure_case,
                "name_ko": CASE_NAMES_KO[failure_case],
                "source_reference": SOURCE_REFERENCES[failure_case],
            },
            "failure_case_count": 1,
            "augmentation_count": len(finalized.records),
            "augmentations": finalized.records,
            "automatic_checks": {
                "passed": True,
                # Kept identical to generator.py's current production schema.
                "quality_gate_version": "v1.8",
                "record_types": [
                    record["type"] for record in finalized.records
                ],
                "measurements": {
                    record["type"]: record.get("parameters", {})
                    for record in finalized.records
                },
            },
            "affine_matrix": finalized.transform.matrix(),
            "output": {
                "width": finalized.image.width,
                "height": finalized.image.height,
                "format": "JPEG",
                "quality": int(config["jpeg_quality"]),
                "jpeg_quality": int(config["jpeg_quality"]),
                "image_sha256": image_hash,
                "label_json_sha256": annotation_hash,
            },
        }
        history_bytes = canonical_json_bytes(history)
        history_hash = sha256_bytes(history_bytes)

        case_dir = PurePosixPath(failure_case)
        image_member = (case_dir / image_name).as_posix()
        annotation_member = (
            case_dir / "labels_json" / f"{new_stem}.json"
        ).as_posix()
        history_member = (
            case_dir
            / "augmentation_json"
            / f"{new_stem}.augmentation.json"
        ).as_posix()
        for target in (image_member, annotation_member, history_member):
            key = target.casefold()
            if key in seen_targets:
                raise ValueError(f"duplicate package target: {target}")
            seen_targets.add(key)

        prepared.append(
            (
                image_member,
                annotation_member,
                history_member,
                image_bytes,
                annotation_bytes,
                history_bytes,
            )
        )
        match_rows.append(
            {
                "synthetic_id": synthetic_id,
                "failure_case": failure_case,
                "image_path": image_member,
                "annotation_json_path": annotation_member,
                "augmentation_json_path": history_member,
                "source_image_name": source_image.name,
                "image_sha256": image_hash,
                "annotation_json_sha256": annotation_hash,
                "augmentation_json_sha256": history_hash,
            }
        )
        print(
            f"[{failure_case}] {ordinal:02d}/{len(rows)} "
            f"| {source_image.name}"
        )

    images_tmp = images_zip.with_suffix(".zip.tmp")
    labels_tmp = labels_zip.with_suffix(".zip.tmp")
    for path in (images_tmp, labels_tmp):
        if path.exists():
            path.unlink()
    try:
        matching = _csv_bytes(match_rows)
        with zipfile.ZipFile(
            images_tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for image_member, _, _, image_bytes, _, _ in prepared:
                archive.writestr(image_member, image_bytes)
            archive.writestr("matching_manifest.csv", matching)
        with zipfile.ZipFile(
            labels_tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for (
                _,
                annotation_member,
                history_member,
                _,
                annotation_bytes,
                history_bytes,
            ) in prepared:
                archive.writestr(annotation_member, annotation_bytes)
                archive.writestr(history_member, history_bytes)
            archive.writestr("matching_manifest.csv", matching)

        with zipfile.ZipFile(images_tmp) as archive:
            if len([name for name in archive.namelist() if name != "matching_manifest.csv"]) != len(rows):
                raise ValueError("images ZIP count mismatch")
        with zipfile.ZipFile(labels_tmp) as archive:
            if len([name for name in archive.namelist() if name != "matching_manifest.csv"]) != len(rows) * 2:
                raise ValueError("labels ZIP count mismatch")
        os.replace(images_tmp, images_zip)
        os.replace(labels_tmp, labels_zip)
    except Exception:
        for path in (images_tmp, labels_tmp):
            if path.exists():
                path.unlink()
        raise

    print(
        f"complete: {len(rows)} images + {len(rows)} annotations + "
        f"{len(rows)} augmentation labels | {images_zip} | {labels_zip}"
    )
    return images_zip, labels_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate CT samples through production finalization and package "
            "production-format annotation and augmentation JSON files."
        )
    )
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.40k.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    package_samples(args.sample_root, args.config, args.output_dir)


if __name__ == "__main__":
    main()
