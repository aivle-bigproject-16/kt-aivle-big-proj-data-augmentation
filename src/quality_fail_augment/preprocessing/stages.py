"""The image/label preprocessing pipeline, split into explicit stages.

Order:
    1. prepare_source: load image and JSON, then crop CT ROI and build masks.
    2. apply_quality_transform: keep PASS pixels or create one FAIL case.
    3. finalize_sample: resize once and transform every polygon with the same matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..augment import (
    EXHAUSTED_SEARCH_MARKER,
    ExhaustedSearchError,
    apply_failure_case,
)
from ..common import load_json, open_normalized, stable_seed
from ..geometry import Affine, extract_ct_roi, point_rings, transform_label


@dataclass
class PreparedSource:
    image: Image.Image
    label: dict[str, Any]
    transform: Affine
    object_mask: Image.Image
    defect_mask: Image.Image
    original_roi: list[float] | None
    porosity_mask: Image.Image | None = None
    porosity_ids: tuple[str, ...] = ()


@dataclass
class TransformedSample:
    image: Image.Image
    label: dict[str, Any]
    transform: Affine
    records: list[dict[str, Any]]
    original_roi: list[float] | None


# A finalized sample has the same state shape; only its coordinate space differs.
FinalizedSample = TransformedSample


def _draw_polygon_mask(
    values: Any, transform: Affine, size: tuple[int, int]
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if values in (None, []):
        return mask
    for ring in point_rings(values):
        points = [transform.apply_point(x, y) for x, y in ring]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return mask


def _object_mask(
    label: dict[str, Any],
    transform: Affine,
    size: tuple[int, int],
    image: Image.Image,
) -> Image.Image:
    mask = _draw_polygon_mask(
        label.get("swelling", {}).get("battery_outline"), transform, size
    )
    if mask.getbbox() is not None:
        return mask

    luminance = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate(
        (luminance[0], luminance[-1], luminance[:, 0], luminance[:, -1])
    )
    difference = np.abs(luminance - float(np.median(border)))
    threshold = max(float(np.quantile(difference, 0.65)), 8.0)
    estimated = Image.fromarray(
        (difference >= threshold).astype(np.uint8) * 255, mode="L"
    )
    if estimated.getbbox() is None:
        raise ValueError("object_mask_confidence_too_low")
    return estimated


def _defect_mask(
    label: dict[str, Any], transform: Affine, size: tuple[int, int]
) -> Image.Image:
    combined = Image.new("L", size, 0)
    for defect in label.get("defects") or []:
        mask = _draw_polygon_mask(defect.get("points"), transform, size)
        combined = Image.fromarray(
            np.maximum(np.asarray(combined), np.asarray(mask)).astype(np.uint8),
            mode="L",
        )
    return combined


def _porosity_mask(
    label: dict[str, Any], transform: Affine, size: tuple[int, int]
) -> Image.Image:
    combined = Image.new("L", size, 0)
    for defect in label.get("defects") or []:
        name = str(
            defect.get("name", defect.get("class", defect.get("label", "")))
        ).strip().casefold()
        if name != "porosity":
            continue
        mask = _draw_polygon_mask(defect.get("points"), transform, size)
        combined = Image.fromarray(
            np.maximum(np.asarray(combined), np.asarray(mask)).astype(np.uint8),
            mode="L",
        )
    return combined


def _porosity_ids(label: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for index, defect in enumerate(label.get("defects") or []):
        name = str(
            defect.get("name", defect.get("class", defect.get("label", "")))
        ).strip().casefold()
        if name == "porosity":
            values.append(str(defect.get("id", index)))
    return tuple(values)


def prepare_source(
    image_path: Path, json_path: Path, modality: str
) -> PreparedSource:
    """Load one source pair and normalize its coordinate origin."""
    image = open_normalized(image_path, modality)
    label = load_json(json_path)
    transform = Affine()
    original_roi: list[float] | None = None

    if modality == "CT":
        roi = extract_ct_roi(label, image.width, image.height)
        crop = tuple(int(round(value)) for value in roi)
        if not (crop[0] < crop[2] and crop[1] < crop[3]):
            raise ValueError(f"Rounded CT ROI is empty: {crop}")
        original_roi = list(roi)
        image = image.crop(crop)
        transform = transform.then(Affine(xoff=-crop[0], yoff=-crop[1]))

    return PreparedSource(
        image=image,
        label=label,
        transform=transform,
        object_mask=_object_mask(label, transform, image.size, image),
        defect_mask=_defect_mask(label, transform, image.size),
        porosity_mask=_porosity_mask(label, transform, image.size),
        porosity_ids=_porosity_ids(label),
        original_roi=original_roi,
    )


def apply_quality_transform(
    source: PreparedSource,
    modality: str,
    quality: str,
    failure_case: str,
    item_seed: int,
    max_retries: int,
    case_seed: int | None = None,
    group_seed: int | None = None,
    config: dict[str, Any] | None = None,
) -> TransformedSample:
    """Apply no FAIL transform for PASS, or exactly one assigned FAIL case."""
    if quality == "pass":
        if failure_case:
            raise ValueError("PASS slot must not have failure_case")
        return TransformedSample(
            image=source.image,
            label=source.label,
            transform=source.transform,
            records=[],
            original_roi=source.original_roi,
        )
    if quality != "fail":
        raise ValueError(f"Unknown quality label: {quality}")
    if not failure_case:
        raise ValueError("FAIL slot has no failure_case")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = apply_failure_case(
                source.image,
                modality,
                failure_case,
                stable_seed(case_seed if case_seed is not None else item_seed, "retry", attempt),
                object_mask=source.object_mask,
                defect_mask=source.defect_mask,
                group_seed=group_seed,
                case_options={
                    **(config or {}),
                    "target_defect_ids": list(source.porosity_ids),
                },
            )
            return TransformedSample(
                image=result.image,
                label=source.label,
                transform=source.transform.then(result.transform),
                records=result.records,
                original_roi=source.original_roi,
            )
        except ExhaustedSearchError as exc:
            # The geometry search enumerated its whole grid and found nothing. It
            # never reads the seed this loop varies, so the remaining attempts would
            # repeat the same work and fail identically. Give up now and let the
            # caller substitute a different source.
            raise ValueError(f"{EXHAUSTED_SEARCH_MARKER}: {exc}") from exc
        except Exception as exc:
            last_error = exc
    raise ValueError(f"augmentation retries exhausted: {last_error}")


def finalize_sample(
    sample: TransformedSample,
    modality: str,
    quality: str,
    new_battery: int,
    new_image: int,
    image_name: str,
    resize_long_side: int,
    allow_upscale: bool,
) -> FinalizedSample:
    """Resize pixels and update all label polygons with the identical matrix."""
    current = max(sample.image.size)
    if current <= resize_long_side and not allow_upscale:
        image = sample.image.copy()
        resize_transform = Affine()
    else:
        scale = resize_long_side / current
        size = (
            max(1, round(sample.image.width * scale)),
            max(1, round(sample.image.height * scale)),
        )
        image = sample.image.resize(size, Image.Resampling.LANCZOS)
        resize_transform = Affine(
            a=size[0] / sample.image.width,
            e=size[1] / sample.image.height,
        )

    transform = sample.transform.then(resize_transform)
    label = transform_label(
        sample.label,
        transform,
        image.size,
        modality,
        quality,
        new_battery,
        new_image,
        image_name,
    )
    return FinalizedSample(
        image=image,
        label=label,
        transform=transform,
        records=sample.records,
        original_roi=sample.original_roi,
    )
