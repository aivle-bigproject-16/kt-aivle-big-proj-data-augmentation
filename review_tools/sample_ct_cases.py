from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from quality_fail_augment.augment import CT_CASES, apply_failure_case
from quality_fail_augment.common import stable_seed
from quality_fail_augment.preprocessing.stages import prepare_source


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourcePair:
    image: Path
    label: Path


def _dataset_directories(raw_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in ("Training", "Validation"):
        split_dirs = [path for path in raw_root.rglob(split) if path.is_dir()]
        for split_dir in split_dirs:
            image_dirs = sorted(
                path
                for path in split_dir.rglob("TS_CT_Datasets_images*")
                if path.is_dir()
            )
            label_dirs = sorted(
                path
                for path in split_dir.rglob("TL_CT_Datasets_label*")
                if path.is_dir()
            )
            for image_dir in image_dirs:
                for label_dir in label_dirs:
                    pairs.append((image_dir, label_dir))
    if not pairs:
        raise FileNotFoundError(
            f"CT image/label directories were not found below: {raw_root}"
        )
    return pairs


def _discover_pairs(
    raw_root: Path,
    rng: np.random.Generator,
    candidate_pool: int,
    max_skip: int,
) -> list[SourcePair]:
    """Read only a bounded random window; planner.scan and cache are not used."""
    directories = _dataset_directories(raw_root)
    pairs: list[SourcePair] = []
    order = rng.permutation(len(directories))
    per_directory = max(
        1,
        (candidate_pool + len(directories) - 1) // len(directories),
    )
    for directory_index in order:
        image_dir, label_dir = directories[int(directory_index)]
        skip = int(rng.integers(0, max_skip + 1)) if max_skip else 0
        entries = list(
            itertools.islice(
                image_dir.iterdir(),
                skip,
                skip + per_directory * 2,
            )
        )
        if not entries:
            entries = list(
                itertools.islice(image_dir.iterdir(), per_directory * 2)
            )
        if entries:
            entry_order = rng.permutation(len(entries))
            entries = [entries[int(index)] for index in entry_order]
        for image_path in entries:
            if (
                not image_path.is_file()
                or image_path.suffix.casefold() not in IMAGE_SUFFIXES
            ):
                continue
            label_path = label_dir / f"{image_path.stem}.json"
            if label_path.is_file():
                pairs.append(SourcePair(image_path, label_path))
            if len(pairs) >= candidate_pool:
                return pairs
    if not pairs:
        raise ValueError("no CT image/JSON pairs were found")
    return pairs


def _resize_long_side(image: Image.Image, long_side: int = 512) -> Image.Image:
    current = max(image.size)
    if current <= long_side:
        return image.copy()
    scale = long_side / current
    return image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def _contact_sheet(case: str, files: list[Path], output: Path) -> None:
    columns, cell_width, cell_height = 5, 250, 300
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"{case} | {len(files)} samples", fill="black")
    for index, path in enumerate(files):
        with Image.open(path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((cell_width - 12, cell_height - 34))
        x = (index % columns) * cell_width
        y = 40 + (index // columns) * cell_height
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y))
        draw.text((x + 6, y + cell_height - 26), f"{index + 1:02d}", fill="black")
    sheet.save(output, format="JPEG", quality=90, subsampling=0)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "failure_case",
        "sample_index",
        "output_file",
        "source_image_name",
        "source_image_path",
        "source_json_path",
        "seed",
        "retry",
        "augmentation_parameters",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_samples(
    raw_root: Path,
    output: Path,
    per_case: int = 10,
    global_seed: int = 20260731,
    max_retries: int = 12,
    candidate_pool: int = 500,
    max_skip: int = 5000,
) -> list[dict[str, Any]]:
    raw_root = raw_root.resolve()
    output = output.resolve()
    if output == raw_root or output.is_relative_to(raw_root):
        raise ValueError("output must be outside raw-root")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if per_case < 1 or max_retries < 1:
        raise ValueError("per-case and max-retries must be at least 1")
    needed = per_case * len(CT_CASES)
    if candidate_pool < needed:
        raise ValueError(f"candidate-pool must be at least {needed}")
    if max_skip < 0:
        raise ValueError("max-skip must not be negative")

    rng = np.random.Generator(np.random.PCG64(global_seed))
    pairs = _discover_pairs(raw_root, rng, candidate_pool, max_skip)
    order = rng.permutation(len(pairs))
    shuffled = [pairs[int(index)] for index in order]

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    print(f"staging: {staging}")

    rows: list[dict[str, Any]] = []
    cursor = 0
    for case in CT_CASES:
        case_dir = staging / case
        case_dir.mkdir()
        case_files: list[Path] = []
        while len(case_files) < per_case and cursor < len(shuffled):
            pair = shuffled[cursor]
            cursor += 1
            try:
                source = prepare_source(pair.image, pair.label, "CT")
            except Exception as exc:
                print(f"[skip] {pair.image.name} | prepare failed: {exc}")
                continue

            result = None
            last_error: Exception | None = None
            selected_seed = 0
            selected_retry = -1
            for retry in range(max_retries):
                selected_seed = stable_seed(
                    global_seed, case, pair.image.stem, retry
                )
                try:
                    result = apply_failure_case(
                        source.image,
                        "CT",
                        case,
                        selected_seed,
                        object_mask=source.object_mask,
                        defect_mask=source.defect_mask,
                    )
                    selected_retry = retry
                    break
                except Exception as exc:
                    last_error = exc
            if result is None:
                print(f"[skip] {pair.image.name} | {case}: {last_error}")
                continue

            sample_index = len(case_files) + 1
            output_name = f"{sample_index:02d}_{pair.image.stem}.png"
            output_path = case_dir / output_name
            _resize_long_side(result.image).save(output_path, format="PNG")
            rows.append(
                {
                    "failure_case": case,
                    "sample_index": sample_index,
                    "output_file": output_path.relative_to(staging).as_posix(),
                    "source_image_name": pair.image.name,
                    "source_image_path": str(pair.image),
                    "source_json_path": str(pair.label),
                    "seed": selected_seed,
                    "retry": selected_retry,
                    "augmentation_parameters": json.dumps(
                        result.records,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            case_files.append(output_path)
            print(
                f"[{case}] {sample_index:02d}/{per_case} "
                f"| {pair.image.name} | retry {selected_retry}"
            )

        if len(case_files) != per_case:
            raise RuntimeError(
                f"{case}: generated {len(case_files)}/{per_case}; "
                f"last candidate index {cursor}/{len(shuffled)}"
            )
        _contact_sheet(case, case_files, staging / f"{case}_contact_sheet.jpg")

    _write_manifest(staging / "sample_manifest.csv", rows)
    summary = {
        "raw_root": str(raw_root),
        "global_seed": global_seed,
        "per_case": per_case,
        "case_counts": {
            case: sum(row["failure_case"] == case for row in rows)
            for case in CT_CASES
        },
        "source_reuse": False,
        "planner_scan_used": False,
    }
    (staging / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(staging, output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create samples for all five CT cases without planner.scan."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-case", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--max-skip", type=int, default=5000)
    args = parser.parse_args()
    rows = generate_samples(
        args.raw_root,
        args.output,
        args.per_case,
        args.seed,
        args.max_retries,
        args.candidate_pool,
        args.max_skip,
    )
    print(f"complete: {args.output.resolve()} | {len(rows)} images")


if __name__ == "__main__":
    main()
