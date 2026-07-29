from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from .geometry import Affine


CT_CASES = (
    "ct_cell_alignment_failure",
    "ct_acquisition_motion",
    "ct_insufficient_projection_sampling",
    "ct_low_signal_noise",
    "ct_beam_hardening_metal_streak",
)
RGB_CASES = (
    "rgb_trigger_timing_failure",
    "rgb_uneven_lighting",
    "rgb_reflection_glare",
    "rgb_focus_failure",
    "rgb_underexposure",
    "rgb_overexposure",
    "rgb_surface_dust",
    "rgb_hair_contamination",
)
CASE_NAMES_KO = {
    "ct_cell_alignment_failure": "셀 장착·정렬 실패",
    "ct_acquisition_motion": "촬영 중 움직임·진동",
    "ct_insufficient_projection_sampling": "투영 영상 부족",
    "ct_low_signal_noise": "저신호 촬영",
    "ct_beam_hardening_metal_streak": "고밀도 부품 투과·보정 실패",
    "rgb_trigger_timing_failure": "배터리 감지·트리거 실패",
    "rgb_uneven_lighting": "조명 점등 실패",
    "rgb_reflection_glare": "반사 억제 실패",
    "rgb_focus_failure": "카메라 초점 설정 실패",
    "rgb_underexposure": "노출 부족",
    "rgb_overexposure": "노출 과다",
    "rgb_surface_dust": "렌즈·보호유리 먼지 오염",
    "rgb_hair_contamination": "렌즈·보호유리 섬유 오염",
}
SOURCE_REFERENCES = {case: f"v1.7:{case}" for case in (*CT_CASES, *RGB_CASES)}


class _PCG64Random:
    """Small compatibility facade backed exclusively by NumPy PCG64."""

    def __init__(self, generator: np.random.Generator):
        self.generator = generator

    def random(self) -> float:
        return float(self.generator.random())

    def uniform(self, low: float, high: float) -> float:
        return float(self.generator.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        return int(self.generator.integers(low, high + 1))

    def randrange(self, start: int, stop: int, step: int = 1) -> int:
        values = np.arange(start, stop, step)
        if not len(values):
            raise ValueError("empty randrange")
        return int(values[int(self.generator.integers(0, len(values)))])

    def choice(self, values: Any) -> Any:
        return values[int(self.generator.integers(0, len(values)))]


@dataclass
class AugmentResult:
    image: Image.Image
    transform: Affine
    failure_case: str
    records: list[dict[str, Any]]
    object_mask: Image.Image | None = None


def _record(order: int, name: str, severity: float, **parameters: Any) -> dict[str, Any]:
    return {
        "order": order,
        "type": name,
        "severity": round(severity, 8),
        "parameters": parameters,
    }


def background(image: Image.Image, modality: str) -> tuple[int, ...] | int:
    array = np.asarray(image)
    if modality == "CT":
        channels = [array] if array.ndim == 2 else [array[..., c] for c in range(array.shape[2])]
        values: list[int] = []
        for channel in channels:
            border = np.concatenate(
                (channel[0], channel[-1], channel[:, 0], channel[:, -1])
            )
            air = border[border <= np.quantile(border, 0.20)]
            values.append(int(np.median(air if air.size else border)))
        return values[0] if array.ndim == 2 else tuple(values)
    rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))


def _array_image(array: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=mode)


def _luminance_field(image: Image.Image, field: np.ndarray) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 2:
        array += field
    else:
        array += field[..., None]
    return _array_image(array, image.mode)


def _motion_blur(image: Image.Image, kernel: int, angle: float) -> Image.Image:
    kernel = max(3, kernel | 1)
    radius = kernel // 2
    radians = math.radians(angle)
    source = np.asarray(image).astype(np.float32)
    if source.ndim == 2:
        source = source[..., None]
    padded = np.pad(source, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    accumulator = np.zeros_like(source)
    for step in range(-radius, radius + 1):
        dx = round(math.cos(radians) * step)
        dy = round(math.sin(radians) * step)
        top, left = radius + dy, radius + dx
        accumulator += padded[top : top + source.shape[0], left : left + source.shape[1]]
    accumulator /= kernel
    if image.mode == "L":
        accumulator = accumulator[..., 0]
    return _array_image(accumulator, image.mode)


def _translate(image: Image.Image, dx: int, dy: int, fill: Any) -> Image.Image:
    output = Image.new(image.mode, image.size, fill)
    output.paste(image, (dx, dy))
    return output


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component without adding a SciPy dependency."""
    source = mask.astype(bool)
    seen = np.zeros(source.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = source.shape
    for start_y, start_x in zip(*np.where(source & ~seen)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and source[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    output = np.zeros(source.shape, dtype=bool)
    if best:
        ys, xs = zip(*best)
        output[np.asarray(ys), np.asarray(xs)] = True
    return output


def _mask_points(mask: Image.Image | None) -> list[tuple[int, int]]:
    if mask is None:
        return []
    ys, xs = np.where(np.asarray(mask.convert("L")) > 0)
    return list(zip(xs.tolist(), ys.tolist()))


def _gamma(image: Image.Image, gamma: float) -> Image.Image:
    table = [round(((value / 255.0) ** gamma) * 255) for value in range(256)]
    return image.point(table * (3 if image.mode == "RGB" else 1))


def _noise(image: Image.Image, rng: np.random.Generator, sigma: float, monochrome: bool) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    shape = array.shape[:2] if monochrome else array.shape
    noise = rng.normal(0.0, sigma, shape).astype(np.float32)
    if array.ndim == 3 and noise.ndim == 2:
        noise = noise[..., None]
    return _array_image(array + noise, image.mode)


def _srgb_to_linear(array: np.ndarray) -> np.ndarray:
    normalized = np.clip(array.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * clipped ** (1.0 / 2.4) - 0.055,
    )
    return np.clip(encoded * 255.0, 0, 255).astype(np.uint8)


def _signed_lines(
    image: Image.Image,
    rng: _PCG64Random,
    count: int,
    width_range: tuple[int, int],
    delta_range: tuple[int, int],
    radial: bool = False,
) -> Image.Image:
    field = Image.new("F", image.size, 0.0)
    draw = ImageDraw.Draw(field)
    center = (image.width / 2, image.height / 2)
    diagonal = math.hypot(image.width, image.height)
    for _ in range(count):
        angle = math.radians(rng.uniform(0, 179))
        delta = rng.choice((-1, 1)) * rng.randint(*delta_range)
        width = rng.randint(*width_range)
        if radial:
            x2 = center[0] + math.cos(angle) * diagonal
            y2 = center[1] + math.sin(angle) * diagonal
            coordinates = (center[0], center[1], x2, y2)
        else:
            perpendicular = rng.uniform(-diagonal / 2, diagonal / 2)
            px, py = -math.sin(angle), math.cos(angle)
            cx = center[0] + px * perpendicular
            cy = center[1] + py * perpendicular
            coordinates = (
                cx - math.cos(angle) * diagonal,
                cy - math.sin(angle) * diagonal,
                cx + math.cos(angle) * diagonal,
                cy + math.sin(angle) * diagonal,
            )
        draw.line(coordinates, fill=float(delta), width=width)
    return _luminance_field(image, np.asarray(field))


def _ct_case(
    image: Image.Image,
    case: str,
    rng: _PCG64Random,
    np_rng: np.random.Generator,
    severity: float,
    object_mask: Image.Image | None,
    defect_mask: Image.Image | None,
    group_rng: _PCG64Random | None = None,
    case_options: dict[str, Any] | None = None,
) -> tuple[Image.Image, Affine, list[dict[str, Any]], Image.Image | None]:
    result, transform, records = image.copy(), Affine(), []
    mask = object_mask.copy() if object_mask is not None else None
    if case == "ct_cell_alignment_failure":
        target_bbox = defect_mask.getbbox() if defect_mask is not None else None
        if target_bbox is None:
            array = np.asarray(result.convert("L"), dtype=np.float32)
            threshold = float(np.quantile(array, 0.97))
            estimated = Image.fromarray(
                (array >= threshold).astype(np.uint8) * 255, mode="L"
            )
            target_bbox = estimated.getbbox()
        if target_bbox is None:
            raise ValueError("porosity_target_mask_is_empty")
        left, top, right, bottom = target_bbox
        # The full shift needed to push the porosity bbox out of the frame. Using it as-is
        # made the case unsatisfiable: when the porosity sits mid-cell the same shift carries
        # most of the battery out with it and outline retention falls under the 0.10 floor,
        # and when it sits near an edge almost nothing is cropped and retention exceeds the
        # 0.95 ceiling. Both ends are rejected by the gate at the bottom of this module.
        candidates = {
            "left": -(right + 1),
            "right": result.width - left + 1,
            "top": -(bottom + 1),
            "bottom": result.height - top + 1,
        }
        direction = (
            (group_rng or rng).choice(("left", "right", "top", "bottom"))
            if group_rng is not None
            else min(candidates, key=lambda name: (abs(candidates[name]), name))
        )
        # Search the shift instead of fixing it. Retention falls monotonically as the shift
        # grows, so walking the scale down from the full exit shift finds the largest crop
        # that still leaves enough of the cell behind. The preferred direction is tried
        # first and the others are used as fallbacks, because for an off-centre porosity one
        # axis can be unsatisfiable while another is fine. The window is tighter than the
        # gate's so that resampling and the later resize keep some slack.
        target_low, target_high = 0.15, 0.90
        scales = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.34, 0.28, 0.22, 0.16, 0.10)
        order = [direction] + [name for name in ("left", "right", "top", "bottom") if name != direction]
        chosen_direction, chosen_shift = direction, candidates[direction]
        if mask is not None:
            original_area = max(float((np.asarray(mask.convert("L")) > 0).sum()), 1.0)
            best: tuple[float, str, int] | None = None
            found = False
            for name in order:
                horizontal = name in {"left", "right"}
                for scale in scales:
                    shift = int(round(candidates[name] * scale))
                    if shift == 0:
                        continue
                    probe = _translate(
                        mask, shift if horizontal else 0, 0 if horizontal else shift, 0
                    )
                    retained = (
                        float((np.asarray(probe.convert("L")) > 0).sum()) / original_area
                    )
                    if target_low <= retained <= target_high:
                        chosen_direction, chosen_shift = name, shift
                        found = True
                        break
                    gap = (
                        target_low - retained
                        if retained < target_low
                        else retained - target_high
                    )
                    if best is None or gap < best[0]:
                        best = (gap, name, shift)
                if found:
                    break
            if not found and best is not None:
                chosen_direction, chosen_shift = best[1], best[2]
        direction = chosen_direction
        horizontal = direction in {"left", "right"}
        full_shift = candidates[direction]
        dx = chosen_shift if horizontal else 0
        dy = 0 if horizontal else chosen_shift
        fill = background(result, "CT")
        # The shift can be smaller than the one that clears the bbox entirely, so report
        # what actually left the frame instead of assuming the target is always gone.
        target_ids = list((case_options or {}).get("target_defect_ids", []))
        fully_removed = abs(dx if horizontal else dy) >= abs(full_shift)
        result = _translate(result, dx, dy, fill)
        if mask is not None:
            mask = _translate(mask, dx, dy, 0)
        transform = transform.then(Affine(xoff=dx, yoff=dy))
        records.append(
            _record(
                len(records) + 1,
                "porosity_targeted_fov_crop",
                severity,
                direction=direction,
                dx_px=dx,
                dy_px=dy,
                offset_source_space=[dx, dy],
                exit_shift_px=full_shift,
                target_fully_outside_frame=fully_removed,
                background_value=fill,
                target_bbox=list(target_bbox),
                target_defect_ids=target_ids,
                removed_defect_ids=target_ids if fully_removed else [],
                retained_outline_ratio=(
                    float((np.asarray(mask) > 0).sum())
                    / max(float((np.asarray(object_mask) > 0).sum()), 1.0)
                    if mask is not None and object_mask is not None
                    else None
                ),
                output_frame=[0, 0, result.width, result.height],
            )
        )
    elif case == "ct_acquisition_motion":
        direction_rng = group_rng or rng
        angle = direction_rng.uniform(0, 359)
        normalized_offset = rng.uniform(18.0, 28.0)
        scale = max(result.size) / 512.0
        offset = max(1, round(normalized_offset * scale))
        dx = round(math.cos(math.radians(angle)) * offset)
        dy = round(math.sin(math.radians(angle)) * offset)
        blur_half_range = rng.uniform(2.0, 5.0)
        kernel = max(9, (round((blur_half_range * 2 + 1) * scale) | 1))
        blurred = _motion_blur(result, kernel, angle)
        shifted = _translate(result, dx, dy, background(result, "CT"))
        shifted_weight = rng.uniform(0.45, 0.52)
        result = Image.blend(blurred, shifted, shifted_weight)
        records.append(
            _record(
                1,
                "directional_motion_blur",
                severity,
                kernel=kernel,
                angle_deg=angle,
                displacement_range_final_512_px=blur_half_range,
            )
        )
        records.append(
            _record(
                2,
                "double_edge_ghosting",
                severity,
                offset_px=offset,
                offset_final_512_px=normalized_offset,
                dx_px=dx,
                dy_px=dy,
                shifted_weight=shifted_weight,
                blurred_weight=1.0 - shifted_weight,
            )
        )
    elif case == "ct_low_signal_noise":
        factor = rng.uniform(0.30, 0.55)
        result = ImageEnhance.Brightness(result).enhance(factor)
        records.append(_record(1, "signal_to_transmission", severity, signal_factor=factor))
        source = np.asarray(result.convert("L"), dtype=np.float32)
        photons = rng.uniform(10.0, 40.0)
        sampled = np_rng.poisson(np.clip(source / 255.0, 0, 1) * photons) / photons * 255.0
        result = Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8)).convert(
            result.mode
        )
        records.append(_record(2, "poisson_sampling", severity, photon_scale=photons))
        sigma = rng.uniform(1.275, 6.375)
        result = _noise(result, np_rng, sigma, monochrome=True)
        records.append(_record(3, "read_noise", severity, sigma=sigma))
        contrast_factor = rng.uniform(0.45, 0.80)
        result = ImageEnhance.Contrast(result).enhance(contrast_factor)
        records.append(
            _record(
                4,
                "low_contrast_attenuation",
                severity,
                contrast_factor=contrast_factor,
            )
        )
    elif case == "ct_insufficient_projection_sampling":
        # Sparse-view Radon reconstruction: build a sinogram from rotated projections, apply a
        # ramp filter, then back-project only the retained angles. This creates reconstruction
        # streaks from missing acquisition views instead of drawing lines on the final slice.
        full_view_count = 40
        full_angles = np.linspace(0.0, 180.0, full_view_count, endpoint=False)
        subtype = rng.choice(("sparse_view", "limited_angle"))
        if subtype == "sparse_view":
            retained_ratio = rng.uniform(0.20, 0.45)
            view_count = max(8, round(full_view_count * retained_ratio))
            indices = np.linspace(0, full_view_count - 1, view_count).round().astype(int)
            angles = full_angles[indices]
            removed_angle = None
        else:
            removed_width = rng.uniform(60.0, 120.0)
            removed_start = rng.uniform(0.0, 180.0)
            distance = (full_angles - removed_start) % 180.0
            angles = full_angles[distance >= removed_width]
            if len(angles) < 8:
                angles = full_angles[distance >= 60.0]
                removed_width = 60.0
            view_count = len(angles)
            retained_ratio = view_count / full_view_count
            removed_angle = [removed_start, (removed_start + removed_width) % 180.0]
        accumulator = np.zeros((result.height, result.width), dtype=np.float32)
        gray = result.convert("L")
        for angle in angles:
            view = gray.rotate(float(angle), Image.Resampling.BILINEAR, expand=False)
            projection_1d = np.asarray(view, dtype=np.float32).sum(axis=0)
            frequencies = np.fft.rfftfreq(len(projection_1d))
            filtered = np.fft.irfft(
                np.fft.rfft(projection_1d) * np.abs(frequencies),
                n=len(projection_1d),
            ).real
            filtered -= filtered.min()
            filtered *= 255.0 / max(float(np.ptp(filtered)), 1.0)
            projection = np.repeat(filtered[None, :], result.height, axis=0)
            back = Image.fromarray(np.clip(projection, 0, 255).astype(np.uint8)).rotate(
                float(-angle), Image.Resampling.BILINEAR, expand=False
            )
            accumulator += np.asarray(back, dtype=np.float32)
        reconstructed = accumulator / view_count
        low, high = np.percentile(reconstructed, (1.0, 99.0))
        reconstructed = np.clip(
            (reconstructed - low) * 255.0 / max(high - low, 1e-6),
            0.0,
            255.0,
        )
        baseline = np.asarray(gray, dtype=np.float32)
        reconstruction_weight = rng.uniform(0.75, 0.90)
        result = Image.fromarray(
            np.clip(
                (1.0 - reconstruction_weight) * baseline
                + reconstruction_weight * reconstructed,
                0,
                255,
            ).astype(np.uint8)
        ).convert(result.mode)
        records.append(
            _record(
                1,
                "radon_projection_drop",
                severity,
                subtype=subtype,
                full_views=full_view_count,
                retained_views=view_count,
                retained_ratio=retained_ratio,
                removed_angle_deg=removed_angle,
                simulation_domain="reconstructed_slice_approximation",
            )
        )
        records.append(
            _record(
                2,
                "filtered_back_projection",
                severity,
                view_angles_deg=angles.tolist(),
                percentile_normalization=[1, 99],
                reconstruction_weight=reconstruction_weight,
            )
        )
    elif case == "ct_beam_hardening_metal_streak":
        array = np.asarray(result.convert("L"), dtype=np.float32)
        # Widen the slice until a usable dense blob appears instead of failing outright. The
        # draw decides where the search starts, so the streak still varies between samples,
        # but a source whose densest pixels are scattered no longer costs the whole sample.
        # planner._has_dense_ct_anchor screens at the strictest ratio, so this loop is a
        # safety net for plans built before that screen existed.
        top_ratio = rng.uniform(0.003, 0.010)
        dense_region_mask = None
        for widened in (top_ratio, top_ratio * 2.0, top_ratio * 4.0, 0.05):
            threshold = float(np.quantile(array, 1.0 - min(widened, 0.20)))
            component = _largest_connected_component(array >= threshold)
            if float(component.mean()) >= 0.001:
                dense_region_mask = component
                top_ratio = min(widened, 0.20)
                break
        if dense_region_mask is None:
            raise ValueError("dense_region_mask_too_small")
        attenuation = rng.uniform(0.20, 0.70)
        target = np.asarray(result).astype(np.float32)
        target[dense_region_mask] *= 1.0 - attenuation
        result = _array_image(target, result.mode)
        records.append(_record(1, "dense_material_mask", severity, threshold=threshold, area_ratio=float(dense_region_mask.mean()), attenuation=attenuation))
        yy, xx = np.mgrid[0 : result.height, 0 : result.width]
        dense_y, dense_x = np.where(dense_region_mask)
        dense_weights = np.maximum(array[dense_region_mask] - threshold + 1.0, 1.0)
        center_x = float(np.average(dense_x, weights=dense_weights))
        center_y = float(np.average(dense_y, weights=dense_weights))
        object_bbox = object_mask.getbbox() if object_mask is not None else None
        if object_bbox is None:
            object_bbox = (0, 0, result.width, result.height)
        object_width = max(1.0, object_bbox[2] - object_bbox[0])
        object_height = max(1.0, object_bbox[3] - object_bbox[1])
        if object_width <= object_height:
            radius_x = object_width * rng.uniform(0.35, 0.70)
            radius_y = object_height * rng.uniform(0.12, 0.30)
        else:
            radius_x = object_width * rng.uniform(0.12, 0.30)
            radius_y = object_height * rng.uniform(0.35, 0.70)
        elliptical = np.sqrt(
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
        )
        asymmetry = 1.0 + rng.uniform(-0.35, 0.35) * np.clip(
            (xx - center_x) / radius_x, -1.0, 1.0
        )
        cupping_delta = (
            rng.uniform(35.0, 80.0)
            if rng.random() < 0.5
            else rng.uniform(-25.0, -10.0)
        )
        cupping_profile = np.clip(1.0 - elliptical, 0.0, 1.0)
        feather_ratio = rng.uniform(0.08, 0.18)
        feather_t = np.clip(cupping_profile / feather_ratio, 0.0, 1.0)
        feathered_edge = feather_ratio * feather_t**2 * (3.0 - 2.0 * feather_t)
        cupping_profile = np.where(
            cupping_profile < feather_ratio, feathered_edge, cupping_profile
        )
        cupping = cupping_delta * cupping_profile * asymmetry
        result = _luminance_field(result, cupping.astype(np.float32))
        records.append(_record(2, "cupping_field", severity, center=[center_x, center_y], radii=[radius_x, radius_y], asymmetric=True, feather_ratio=feather_ratio))
        count = rng.randint(24, 72)
        field = Image.new("F", result.size, 0.0)
        draw = ImageDraw.Draw(field)
        diagonal = math.hypot(result.width, result.height)
        cx, cy = int(round(center_x)), int(round(center_y))
        start_angle = rng.uniform(0, 360)
        jitter_limit = min(3.0, 120.0 / count)
        angles: list[float] = []
        for index in range(count):
            angle_deg = (
                start_angle
                + index * (360.0 / count)
                + rng.uniform(-jitter_limit, jitter_limit)
            ) % 360.0
            angle = math.radians(angle_deg)
            normalized_width = rng.uniform(1.0, 4.0)
            source_scale = max(result.size) / 512.0
            width = max(1, round(normalized_width * source_scale))
            alpha = rng.uniform(0.10, 0.35)
            delta = rng.choice((-1, 1)) * alpha * 255.0
            draw.line(
                (
                    cx,
                    cy,
                    cx + math.cos(angle) * diagonal,
                    cy + math.sin(angle) * diagonal,
                ),
                fill=float(delta),
                width=width,
            )
            angles.append(angle_deg)
        streak_field = np.asarray(field).copy()
        distance = np.hypot(xx - cx, yy - cy)
        decay_scale = max(result.size) * rng.uniform(0.35, 0.70)
        streak_field *= np.exp(-distance / max(decay_scale, 1.0))
        if object_mask is not None:
            object_array = np.asarray(object_mask.convert("L")) > 0
            bbox = object_mask.getbbox() or (0, 0, result.width, result.height)
            outside_dx = np.maximum.reduce(
                [
                    np.asarray(bbox[0] - xx, dtype=np.float32),
                    np.asarray(xx - (bbox[2] - 1), dtype=np.float32),
                    np.zeros(xx.shape, dtype=np.float32),
                ]
            )
            outside_dy = np.maximum.reduce(
                [
                    np.asarray(bbox[1] - yy, dtype=np.float32),
                    np.asarray(yy - (bbox[3] - 1), dtype=np.float32),
                    np.zeros(yy.shape, dtype=np.float32),
                ]
            )
            outside_distance = np.hypot(outside_dx, outside_dy)
            fade_distance = max(result.size) * rng.uniform(0.10, 0.25)
            air_fade = 0.08 * np.clip(
                1.0 - outside_distance / max(fade_distance, 1.0), 0.0, 1.0
            )
            air_factor = np.where(object_array, 1.0, air_fade)
            streak_field *= air_factor
        streak_mask = streak_field != 0
        dense_intersections = int((streak_mask & dense_region_mask).sum())
        result = _luminance_field(result, streak_field)
        ordered = sorted(angles)
        gaps = [
            (ordered[(index + 1) % len(ordered)] - ordered[index]) % 360.0
            for index in range(len(ordered))
        ]
        records.append(
            _record(
                3,
                "metal_anchored_streaks",
                severity,
                streak_count=count,
                dense_region_center=[cx, cy],
                ray_angles_deg=angles,
                max_angular_gap_deg=max(gaps),
                quadrant_counts=[
                    sum(1 for value in angles if start <= value < start + 90)
                    for start in (0, 90, 180, 270)
                ],
                distance_decay_scale_px=decay_scale,
                air_alpha_cap=0.08,
                dense_intersection_pixels=dense_intersections,
            )
        )
    else:
        raise ValueError(f"Unknown CT failure case: {case}")
    return result, transform, records, mask


def _rgb_case(
    image: Image.Image,
    case: str,
    rng: _PCG64Random,
    np_rng: np.random.Generator,
    severity: float,
    object_mask: Image.Image | None,
    defect_mask: Image.Image | None,
    case_options: dict[str, Any] | None = None,
) -> tuple[Image.Image, Affine, list[dict[str, Any]], Image.Image | None]:
    result, transform, records = image.copy(), Affine(), []
    mask = object_mask.copy() if object_mask is not None else None
    case_options = case_options or {}
    if case == "rgb_trigger_timing_failure":
        conveyor_axis = str(case_options.get("conveyor_axis", "horizontal")).casefold()
        if conveyor_axis not in {"horizontal", "vertical"}:
            raise ValueError("conveyor_axis must be horizontal or vertical")
        forward_direction = str(
            case_options.get("forward_direction", "positive")
        ).casefold()
        if forward_direction not in {"positive", "negative"}:
            raise ValueError("forward_direction must be positive or negative")
        timing_event = rng.choice(("early", "late"))
        negative_side, positive_side = (
            ("left", "right")
            if conveyor_axis == "horizontal"
            else ("top", "bottom")
        )
        leading_side, trailing_side = (
            (positive_side, negative_side)
            if forward_direction == "positive"
            else (negative_side, positive_side)
        )
        side = leading_side if timing_event == "early" else trailing_side
        # The crop has to clip the battery, not merely trim background. Sizing the cut against
        # the frame let a centred cylindrical cell survive intact - it covers about 11% of the
        # frame width, so a left/right cut of 10..38% often removed only white background and
        # the result was indistinguishable from a normal photo (7 of 30 visual QA samples were
        # rejected for this). Put the cut plane inside the outline bounding box instead, so it
        # always removes 15%..45% of the box along the chosen axis.
        axis_size = result.width if side in {"left", "right"} else result.height
        aspect = result.width / result.height
        def crop_for(amount: int) -> list[int]:
            candidate = [0, 0, result.width, result.height]
            if side == "left":
                candidate[0] += amount
            elif side == "right":
                candidate[2] -= amount
            elif side == "top":
                candidate[1] += amount
            else:
                candidate[3] -= amount
            current_width = candidate[2] - candidate[0]
            current_height = candidate[3] - candidate[1]
            if side in {"left", "right"}:
                target_height = min(current_height, max(1, round(current_width / aspect)))
                trim = current_height - target_height
                candidate[1] += trim // 2
                candidate[3] -= trim - trim // 2
            else:
                target_width = min(current_width, max(1, round(current_height * aspect)))
                trim = current_width - target_width
                candidate[0] += trim // 2
                candidate[2] -= trim - trim // 2
            return candidate

        target_retained_ratio = rng.uniform(0.55, 0.85)
        if mask is not None and mask.getbbox() is not None:
            original_area = max(int((np.asarray(mask) > 0).sum()), 1)
            low_amount, high_amount = 1, axis_size - 1
            crop = crop_for(low_amount)
            for _ in range(18):
                amount = (low_amount + high_amount) // 2
                candidate = crop_for(amount)
                retained = int(
                    (np.asarray(mask.crop(tuple(candidate))) > 0).sum()
                ) / original_area
                crop = candidate
                if retained > target_retained_ratio:
                    low_amount = min(amount + 1, high_amount)
                else:
                    high_amount = max(amount - 1, low_amount)
        else:
            crop = crop_for(
                min(max(1, round(axis_size * rng.uniform(0.10, 0.38))), axis_size - 1)
            )
        result = result.crop(tuple(crop))
        if mask is not None:
            mask = mask.crop(tuple(crop))
        transform = transform.then(Affine(xoff=-crop[0], yoff=-crop[1]))
        records.append(
            _record(
                1,
                "timing_edge_crop",
                severity,
                side=side,
                crop_box=crop,
                source_aspect_ratio=aspect,
                output_aspect_ratio=result.width / result.height,
                target_outline_retained_ratio=(
                    target_retained_ratio
                    if object_mask is not None and object_mask.getbbox() is not None
                    else None
                ),
                conveyor_axis=conveyor_axis,
                forward_direction=forward_direction,
                timing_event=timing_event,
            )
        )
        if rng.random() < 0.35:
            kernel = rng.randrange(5, 18, 2)
            result = _motion_blur(result, kernel, 0 if side in {"left", "right"} else 90)
            records.append(_record(2, "conveyor_motion_blur", severity, kernel=kernel))
    elif case == "rgb_uneven_lighting":
        angle_deg = rng.choice((0.0, 90.0, 180.0, 270.0))
        angle = math.radians(angle_deg)
        yy, xx = np.mgrid[0 : result.height, 0 : result.width]
        projection = xx * math.cos(angle) + yy * math.sin(angle)
        if mask is not None and mask.getbbox() is not None:
            left, top, right, bottom = mask.getbbox()
            corners = np.asarray(
                [[left, top], [right, top], [left, bottom], [right, bottom]],
                dtype=np.float64,
            )
            bounds = corners[:, 0] * math.cos(angle) + corners[:, 1] * math.sin(angle)
            low, high = float(bounds.min()), float(bounds.max())
        else:
            low, high = float(projection.min()), float(projection.max())
        projection = np.clip((projection - low) / max(high - low, 1.0), 0.0, 1.0)
        # The gate below wants the asymmetry across the battery between 0.25 and 0.60. The
        # earlier 0.35..0.65 dark gain came from whole-frame visual review and overshot that
        # ceiling badly: measured over 16 sources it produced an asymmetry of 0.82..1.62. A
        # fixed replacement range does not work either, because the achieved asymmetry
        # depends on how the cell's own brightness runs along the gradient. Aim for a target
        # inside the window and walk the dark gain toward it, the same way the trigger
        # timing case searches its crop.
        smooth_projection = projection * projection * (3.0 - 2.0 * projection)
        bright = rng.uniform(1.08, 1.26)
        target_asymmetry = rng.uniform(0.32, 0.52)
        dark = rng.uniform(0.74, 0.90)
        source_array = np.asarray(result).astype(np.float32)
        if mask is not None and mask.getbbox() is not None:
            region = np.asarray(mask.convert("L")) > 0
            axis_values = projection[region]
            dark_select = axis_values <= float(np.quantile(axis_values, 0.20))
            bright_select = axis_values >= float(np.quantile(axis_values, 0.80))

            def asymmetry_for(dark_gain: float) -> float:
                candidate_gain = dark_gain + (bright - dark_gain) * smooth_projection
                shaded = np.asarray(
                    _array_image(source_array * candidate_gain[..., None], "RGB").convert("L"),
                    dtype=np.float32,
                )[region]
                spread = abs(
                    float(shaded[bright_select].mean()) - float(shaded[dark_select].mean())
                )
                return spread / max(float(shaded.mean()), 1.0)

            low_gain, high_gain = 0.45, 0.99
            for _ in range(8):
                middle = (low_gain + high_gain) / 2.0
                dark = middle
                if asymmetry_for(middle) > target_asymmetry:
                    low_gain = middle
                else:
                    high_gain = middle
        gain = dark + (bright - dark) * smooth_projection
        array = source_array * gain[..., None]
        result = _array_image(array, "RGB")
        records.append(_record(1, "lighting_gradient", severity, angle_deg=angle_deg, dark_gain=dark, bright_gain=bright, target_asymmetry=target_asymmetry, transition="smoothstep"))
        if mask is not None and mask.getbbox() is not None and rng.random() < 0.50:
            zone_count = rng.randint(1, 3)
            zone_field = np.ones((result.height, result.width), dtype=np.float32)
            points = _mask_points(mask)
            object_area = max(int((np.asarray(mask) > 0).sum()), 1)
            zones = []
            for _ in range(zone_count):
                cx, cy = rng.choice(points)
                target_area = object_area * rng.uniform(0.04, 0.18)
                radius = math.sqrt(target_area / math.pi)
                zone_gain = rng.uniform(0.20, 0.65)
                distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
                gaussian = np.exp(-distance2 / max(2.0 * radius**2, 1.0))
                zone_field *= 1.0 - (1.0 - zone_gain) * gaussian
                zones.append([cx, cy, radius, zone_gain])
            result = _array_image(
                np.asarray(result, dtype=np.float32) * zone_field[..., None], "RGB"
            )
            records.append(
                _record(2, "led_dead_zone", severity, zone_count=zone_count, zones=zones)
            )
    elif case == "rgb_reflection_glare":
        count = rng.randint(1, 2)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        patches = []
        eligible = _mask_points(mask)
        # The glare must overlap the outline by >=70%. Sizing patches to the frame (up to a
        # quarter of the width/height) and centring them on any outline point let large patches
        # spill off the small cylindrical outline, so 65% of sources failed. Cap each patch to a
        # fraction of the mask's own extent and pull its centre toward the mask centroid so the
        # patch stays inside the outline.
        if eligible:
            eligible_array = np.asarray(eligible, dtype=np.float64)
            centroid_x, centroid_y = eligible_array[:, 0].mean(), eligible_array[:, 1].mean()
            half_w = max(2.0, (eligible_array[:, 0].max() - eligible_array[:, 0].min()) / 2.0)
            half_h = max(2.0, (eligible_array[:, 1].max() - eligible_array[:, 1].min()) / 2.0)
            centred = eligible_array - np.array([centroid_x, centroid_y])
            covariance = np.cov(centred.T) if len(eligible_array) > 1 else np.eye(2)
            _, eigenvectors = np.linalg.eigh(covariance)
            major_axis = eigenvectors[:, -1]
            major_extent = max(float(np.ptp(centred @ major_axis)), 4.0)
            object_luminance = np.asarray(result.convert("L"), dtype=np.float32)
            threshold = float(
                np.quantile(
                    object_luminance[np.asarray(mask.convert("L")) > 0], 0.82
                )
            )
            highlight_y, highlight_x = np.where(
                (np.asarray(mask.convert("L")) > 0) & (object_luminance >= threshold)
            )
            highlight_points = list(zip(highlight_x.tolist(), highlight_y.tolist()))
        else:
            centroid_x, centroid_y = result.width / 2.0, result.height / 2.0
            half_w, half_h = result.width / 4.0, result.height / 4.0
            major_axis = np.array([0.0, 1.0])
            major_extent = result.height * 0.6
            highlight_points = []
        defect_points = _mask_points(defect_mask)
        defect_scale = (
            float(np.sqrt(float((np.asarray(defect_mask.convert("L")) > 0).sum())))
            if defect_mask is not None
            else 0.0
        )
        for patch_index in range(count):
            # Elongate the highlight along the object's principal form instead of emitting a
            # round white blob. The wide Gaussian falloff below removes the hard ellipse edge.
            if half_h >= half_w:
                rx = rng.randint(max(2, int(half_w * 0.08)), max(3, int(half_w * 0.20)))
                ry = rng.randint(max(3, int(half_h * 0.32)), max(4, int(half_h * 0.70)))
            else:
                rx = rng.randint(max(3, int(half_w * 0.32)), max(4, int(half_w * 0.70)))
                ry = rng.randint(max(2, int(half_h * 0.08)), max(3, int(half_h * 0.20)))
            base = (
                rng.choice(defect_points)
                if patch_index == 0 and defect_points
                else rng.choice(highlight_points)
                if highlight_points
                else rng.choice(eligible)
                if eligible
                else (int(centroid_x), int(centroid_y))
            )
            # The defect patch has to sit on the defect, so barely pull it toward the
            # centroid. The decorative patches still drift inward to stay on the cell.
            pull = 0.15 if patch_index == 0 and defect_points else 0.6
            cx = int(round(base[0] * (1.0 - pull) + centroid_x * pull))
            cy = int(round(base[1] * (1.0 - pull) + centroid_y * pull))
            alpha = round(255 * rng.uniform(0.45, 0.75))
            # Two gate conditions pull against each other: the glare must cover 30%..70% of
            # the defect mask but only 1%..12% of the battery. Measured over 30 sources the
            # defects are small (median 1.2% of the cell), so both are satisfiable, but the
            # window is narrow enough that a fixed size lands outside it on one side or the
            # other depending on the defect's shape. Size the defect patch by searching, the
            # same way the trigger timing crop and the lighting gradient do. The bounding box
            # is no use as a scale here because scattered defects give a box as large as the
            # whole battery, so the characteristic size is the square root of the area.
            def geometry_for(half_length: float, width: int):
                return (
                    (
                        cx - major_axis[0] * half_length,
                        cy - major_axis[1] * half_length,
                    ),
                    (
                        cx + major_axis[0] * half_length,
                        cy + major_axis[1] * half_length,
                    ),
                    width,
                )

            if patch_index == 0 and defect_scale > 0.0 and defect_mask is not None:
                defect_array = np.asarray(defect_mask.convert("L")) > 0
                defect_area = max(int(defect_array.sum()), 1)
                target_coverage = rng.uniform(0.38, 0.58)
                chosen = None
                fallback = None
                for scale in (0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45):
                    half_length = max(defect_scale * scale, 3.0)
                    width = max(3, round(defect_scale * scale * 0.85))
                    probe = Image.new("L", result.size, 0)
                    start, end, line_width = geometry_for(half_length, width)
                    ImageDraw.Draw(probe).line((start, end), fill=255, width=line_width)
                    probe_array = np.asarray(probe) > 0
                    coverage = float((probe_array & defect_array).sum()) / defect_area
                    # The same patch also has to stay inside the object-area ceiling, and
                    # only the part of it that lands on the cell counts, because the overlay
                    # is clipped to the outline further down.
                    on_object = (
                        probe_array & (np.asarray(mask.convert("L")) > 0)
                        if mask is not None
                        else probe_array
                    )
                    object_ratio = float(on_object.sum()) / max(
                        float((np.asarray(mask.convert("L")) > 0).sum())
                        if mask is not None
                        else float(probe_array.size),
                        1.0,
                    )
                    if object_ratio > 0.10:
                        break
                    if fallback is None or abs(coverage - target_coverage) < fallback[0]:
                        fallback = (abs(coverage - target_coverage), half_length, width)
                    if coverage >= target_coverage:
                        chosen = (half_length, width)
                        break
                if chosen is None and fallback is not None:
                    chosen = (fallback[1], fallback[2])
                half_length, width = chosen if chosen else (max(defect_scale, 3.0), 3)
            else:
                half_length = major_extent * rng.uniform(0.10, 0.22)
                width = max(2, round(min(rx, ry) * 0.9))
            start, end, width = geometry_for(half_length, width)
            draw.line((start, end), fill=(255, 246, 224, alpha), width=width)
            patches.append({"center": [cx, cy], "axis": major_axis.tolist(), "half_length": half_length, "width": width, "alpha": alpha / 255})
        # A specular highlight lives on the cell surface, so clip the overlay to the outline
        # instead of letting a patch that starts near the edge hang over the background.
        # This is what the "outline overlap is below 90%" rejections were reporting, and
        # clipping also keeps the covered area inside the gate's ceiling.
        if mask is not None:
            outline_alpha = Image.fromarray(
                (np.asarray(mask.convert("L")) > 0).astype(np.uint8) * 255, mode="L"
            )
            clipped_alpha = ImageChops.multiply(overlay.getchannel("A"), outline_alpha)
            overlay.putalpha(clipped_alpha)
        radius = rng.uniform(5, 14)
        result = Image.alpha_composite(result.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(radius))).convert("RGB")
        glare_mask = np.asarray(overlay.getchannel("A")) > 0
        object_array = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None
            else np.ones(glare_mask.shape, dtype=bool)
        )
        defect_array = (
            np.asarray(defect_mask.convert("L")) > 0
            if defect_mask is not None
            else np.zeros(glare_mask.shape, dtype=bool)
        )
        outline_overlap = float(
            (glare_mask & object_array).sum() / max(glare_mask.sum(), 1)
        )
        core_object_ratio = float(
            (glare_mask & object_array).sum() / max(object_array.sum(), 1)
        )
        defect_coverage = float(
            (glare_mask & defect_array).sum() / max(defect_array.sum(), 1)
        )
        saturation_ratio = float(
            ((np.asarray(result.convert("L")) >= 250) & object_array).sum()
            / max(object_array.sum(), 1)
        )
        records.append(_record(1, "surface_aware_specular_reflection", severity, seed="defect" if defect_points else "existing_highlight" if highlight_points else "outline_axis", patches=patches, bloom_radius_final_space=radius, outline_overlap_ratio=outline_overlap, core_object_area_ratio=core_object_ratio, defect_present=bool(defect_points), defect_coverage_ratio=defect_coverage, object_saturation_ratio=saturation_ratio))
        records.append(_record(2, "highlight_bloom", severity, radius_final_space=radius))
    elif case == "rgb_focus_failure":
        # The gate below keeps edge energy between 25% and 75% of the source. Measured over
        # 16 sources, a 2.5 px blur already lands at a median ratio of 0.246 and a 10 px blur
        # at 0.159, so the old 2.5..10 range sat almost entirely under the floor. The window
        # corresponds to roughly 0.8..2.5 px, and the extra motion blur has to stay small
        # enough that the pair still clears the floor.
        radius = rng.uniform(0.8, 2.4)
        result = result.filter(ImageFilter.GaussianBlur(radius))
        records.append(_record(1, "defocus_blur", severity, radius_final_space=radius))
        if rng.random() < 0.25:
            kernel = rng.randrange(3, 8, 2)
            angle = rng.uniform(0, 179)
            result = _motion_blur(result, kernel, angle)
            records.append(_record(2, "mild_motion_blur", severity, kernel=kernel, angle_deg=angle))
    elif case == "rgb_underexposure":
        linear = _srgb_to_linear(np.asarray(result.convert("RGB")))
        # The factor is applied in linear light but the gate below measures the sRGB mean,
        # where a linear factor f shows up as roughly f**(1/2.4). The old upper bound of
        # 0.55 became 0.75 in sRGB and could never satisfy the "<= 0.72 of baseline" check,
        # so the top of the range was dead. 0.45 maps to about 0.70.
        factor = rng.uniform(0.20, 0.45)
        # Shot noise is what breaks the spatial-order check: at a capacity of 80 the
        # correlation with the source dropped to 0.65..0.79 over 16 sources. Raising the
        # capacity keeps the low-signal look while leaving the bright/dark ordering intact.
        photon_capacity = rng.uniform(400.0, 1500.0)
        exposed = linear * factor
        sampled = (
            np_rng.poisson(np.clip(exposed, 0.0, 1.0) * photon_capacity)
            / photon_capacity
        )
        read_sigma = rng.uniform(0.005, 0.015)
        shared = np_rng.normal(0.0, read_sigma, sampled.shape[:2])[..., None]
        chroma = np_rng.normal(
            0.0, read_sigma * 0.25, sampled.shape
        )
        black_level = rng.uniform(0.003, 0.012)
        sampled = np.clip(sampled + shared + chroma - black_level, 0.0, 1.0)
        result = Image.fromarray(_linear_to_srgb(sampled), mode="RGB")
        records.append(
            _record(
                1,
                "linear_exposure_reduction",
                severity,
                exposure_factor=factor,
            )
        )
        records.append(
            _record(
                2,
                "signal_dependent_shot_noise",
                severity,
                photon_capacity=photon_capacity,
            )
        )
        records.append(
            _record(
                3,
                "sensor_read_noise",
                severity,
                sigma=read_sigma,
                chroma_sigma=read_sigma * 0.25,
                black_level=black_level,
            )
        )
    elif case == "rgb_overexposure":
        # The battery occupies the dark part of the frame (measured object-region mean
        # luminance 53..115 over 24 sources), so a 1.45..2.60 gain with a 185..245 clip
        # saturated only 3..7% of it and the gate below never saw its floor. Raising the
        # gain and lowering the clip puts the median object saturation near 10%.
        factor = rng.uniform(2.40, 5.00)
        linear = _srgb_to_linear(np.asarray(result.convert("RGB")))
        result = Image.fromarray(_linear_to_srgb(np.clip(linear * factor, 0.0, 1.0)), mode="RGB")
        threshold = rng.randint(140, 200)
        array = np.asarray(result)
        array = np.where(array >= threshold, 255, array)
        result = _array_image(array, "RGB")
        records.append(_record(1, "overexposure", severity, exposure_factor=factor, color_space="linear_light", clip_threshold=threshold))
        if rng.random() < 0.50:
            radius = rng.uniform(3.0, 18.0) * max(result.size) / 512.0
            bright = result.filter(ImageFilter.GaussianBlur(radius))
            result = Image.blend(result, bright, 0.20)
            records.append(_record(2, "highlight_bloom", severity, radius_final_512_px=radius * 512.0 / max(result.size)))
    elif case == "rgb_surface_dust":
        count = rng.randint(1, 4)
        core = Image.new("RGBA", result.size, (0, 0, 0, 0))
        halo = Image.new("RGBA", result.size, (0, 0, 0, 0))
        core_draw = ImageDraw.Draw(core)
        halo_draw = ImageDraw.Draw(halo)
        particles = []
        # Lens contamination belongs to camera-frame coordinates and must not follow the
        # battery outline when the object moves.
        eligible = _mask_points(mask)
        long_side = max(result.size)
        for index in range(count):
            # Plan specifies core diameter, so radius is half of 1%..6%.
            radius = long_side * rng.uniform(0.005, 0.03)
            cx, cy = (
                rng.choice(eligible)
                if index == 0 and eligible
                else (rng.uniform(0, result.width), rng.uniform(0, result.height))
            )
            color = rng.choice(((65, 62, 58), (105, 101, 94), (145, 140, 130)))
            core_alpha = round(255 * rng.uniform(0.05, 0.20))
            halo_alpha = round(255 * rng.uniform(0.03, 0.10))
            halo_scale = rng.uniform(1.5, 3.5)
            halo_radius = radius * halo_scale
            halo_draw.ellipse(
                (
                    cx - halo_radius,
                    cy - halo_radius,
                    cx + halo_radius,
                    cy + halo_radius,
                ),
                fill=(*color, halo_alpha),
            )
            core_draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=(*color, core_alpha),
            )
            particles.append(
                [
                    round(cx, 2),
                    round(cy, 2),
                    round(radius, 2),
                    core_alpha,
                    round(halo_scale, 3),
                    halo_alpha,
                ]
            )
        blur_radius = long_side * rng.uniform(0.008, 0.025)
        composite = Image.alpha_composite(
            result.convert("RGBA"),
            halo.filter(ImageFilter.GaussianBlur(blur_radius * 1.5)),
        )
        result = Image.alpha_composite(
            composite, core.filter(ImageFilter.GaussianBlur(blur_radius))
        ).convert("RGB")
        combined_alpha = np.maximum(
            np.asarray(core.getchannel("A")), np.asarray(halo.getchannel("A"))
        )
        dust_mask = combined_alpha > 0
        object_array = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None
            else np.ones(dust_mask.shape, dtype=bool)
        )
        defect_array = (
            np.asarray(defect_mask.convert("L")) > 0
            if defect_mask is not None
            else np.zeros(dust_mask.shape, dtype=bool)
        )
        coverage = float((dust_mask & object_array).sum() / max(object_array.sum(), 1))
        defect_coverage = float(
            (dust_mask & defect_array).sum() / max(defect_array.sum(), 1)
        )
        records.append(_record(1, "lens_dust_shadow", severity, shadow_count=count, shadows=particles, frame_affected_ratio=float(dust_mask.mean()), blur_radius_final_space=blur_radius, coordinate_space="camera_frame"))
    elif case == "rgb_hair_contamination":
        count = rng.randint(1, 2)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        curves = []
        long_side = max(result.size)
        # A lens/protective-window fibre is fixed to the camera frame, not to the object.
        eligible = _mask_points(mask)
        # Orient each hair along the battery outline's principal axis and centre it on the
        # mask, so most of its length stays inside the outline. Cylindrical cells occupy only
        # ~11% of the frame (mask diagonal ~0.35-0.6 of the long side), so the previous random
        # outward hair almost never met the >=50% intersection gate — 87% of hair sources
        # exhausted their retries. Capping length to <=1.5x the inside extent keeps the
        # intersection above 0.5 by construction while staying inside the 0.35-0.90 length gate.
        if eligible:
            points_array = np.asarray(eligible, dtype=np.float64)
            centroid = points_array.mean(axis=0)
            centred = points_array - centroid
            covariance = np.cov(centred.T) if len(points_array) > 1 else np.eye(2)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis_major = eigenvectors[:, int(np.argmax(eigenvalues))]
            projections = centred @ axis_major
            inside_len = float(projections.max() - projections.min())
        else:
            centroid = np.array([result.width / 2.0, result.height / 2.0])
            axis_major = np.array([1.0, 0.0])
            inside_len = 0.6 * long_side
        for _ in range(count):
            jitter = rng.uniform(-0.12, 0.12)
            cos_j, sin_j = math.cos(jitter), math.sin(jitter)
            direction = np.array(
                [
                    axis_major[0] * cos_j - axis_major[1] * sin_j,
                    axis_major[0] * sin_j + axis_major[1] * cos_j,
                ]
            )
            desired = rng.uniform(0.15, 0.60) * long_side
            length = min(desired, 0.60 * long_side)
            if rng.random() < 0.70:
                edge = rng.choice(("left", "right", "top", "bottom"))
                if edge == "left":
                    start = np.array([0.0, rng.uniform(0, result.height)])
                elif edge == "right":
                    start = np.array([result.width - 1.0, rng.uniform(0, result.height)])
                elif edge == "top":
                    start = np.array([rng.uniform(0, result.width), 0.0])
                else:
                    start = np.array([rng.uniform(0, result.width), result.height - 1.0])
                toward = centroid - start
                norm = max(float(np.linalg.norm(toward)), 1e-6)
                direction = toward / norm
                end = start + direction * length
            else:
                centre = centroid + axis_major * (rng.uniform(-0.10, 0.10) * inside_len)
                start = centre - direction * (length / 2.0)
                end = centre + direction * (length / 2.0)
            normal = np.array([-direction[1], direction[0]])
            bend1, bend2 = rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)
            points = [
                (float(start[0]), float(start[1])),
                (
                    float(start[0] + (end[0] - start[0]) / 3 + normal[0] * length * bend1),
                    float(start[1] + (end[1] - start[1]) / 3 + normal[1] * length * bend1),
                ),
                (
                    float(start[0] + 2 * (end[0] - start[0]) / 3 + normal[0] * length * bend2),
                    float(start[1] + 2 * (end[1] - start[1]) / 3 + normal[1] * length * bend2),
                ),
                (float(end[0]), float(end[1])),
            ]
            width = max(1, round(long_side * rng.uniform(0.0015, 0.006)))
            alpha = round(255 * rng.uniform(0.05, 0.18))
            samples = []
            for index in range(41):
                t = index / 40
                x = (1 - t) ** 3 * points[0][0] + 3 * (1 - t) ** 2 * t * points[1][0] + 3 * (1 - t) * t**2 * points[2][0] + t**3 * points[3][0]
                y = (1 - t) ** 3 * points[0][1] + 3 * (1 - t) ** 2 * t * points[1][1] + 3 * (1 - t) * t**2 * points[2][1] + t**3 * points[3][1]
                samples.append((x, y))
            widths: list[int] = []
            for segment in range(1, len(samples)):
                phase = segment / (len(samples) - 1)
                tapered = max(1, round(width * (0.45 + 0.55 * math.sin(math.pi * phase))))
                widths.append(tapered)
                draw.line(
                    (samples[segment - 1], samples[segment]),
                    fill=(35, 24, 18, alpha),
                    width=tapered,
                )
            actual_length = sum(
                math.dist(samples[index - 1], samples[index])
                for index in range(1, len(samples))
            )
            if mask is not None:
                mask_array = np.asarray(mask.convert("L")) > 0
                inside = sum(
                    1
                    for x, y in samples
                    if 0 <= round(x) < result.width
                    and 0 <= round(y) < result.height
                    and mask_array[round(y), round(x)]
                )
                intersection_ratio = inside / len(samples)
            else:
                intersection_ratio = 1.0
            curves.append({"control_points": points, "thickness_px": width, "thickness_range_px": [min(widths), max(widths)], "alpha": alpha / 255, "length_px": actual_length, "outline_intersection_ratio": intersection_ratio})
        blur_radius = rng.uniform(1.2, 3.8)
        halo_multiplier = rng.uniform(2.0, 5.0)
        halo_layer = overlay.copy()
        halo_layer.putalpha(
            halo_layer.getchannel("A").point(lambda value: round(value * 0.45))
        )
        composite = Image.alpha_composite(
            result.convert("RGBA"),
            halo_layer.filter(ImageFilter.GaussianBlur(blur_radius * halo_multiplier)),
        )
        result = Image.alpha_composite(
            composite, overlay.filter(ImageFilter.GaussianBlur(blur_radius))
        ).convert("RGB")
        records.append(_record(1, "lens_fiber_shadow", severity, curve_count=count, curves=curves, blur_radius_final_space=blur_radius, halo_multiplier=halo_multiplier, coordinate_space="camera_frame"))
    else:
        raise ValueError(f"Unknown RGB failure case: {case}")
    return result, transform, records, mask


def apply_failure_case(
    image: Image.Image,
    modality: str,
    failure_case: str,
    seed: int,
    object_mask: Image.Image | None = None,
    defect_mask: Image.Image | None = None,
    group_seed: int | None = None,
    case_options: dict[str, Any] | None = None,
) -> AugmentResult:
    if modality == "CT" and failure_case not in CT_CASES:
        raise ValueError(f"Case {failure_case!r} is not valid for CT")
    if modality == "RGB" and failure_case not in RGB_CASES:
        raise ValueError(f"Case {failure_case!r} is not valid for RGB")
    np_rng = np.random.Generator(np.random.PCG64(seed))
    rng = _PCG64Random(np_rng)
    group_rng = (
        _PCG64Random(np.random.Generator(np.random.PCG64(group_seed)))
        if group_seed is not None
        else None
    )
    severity = rng.uniform(0.62, 1.0)
    if modality == "CT":
        result, transform, records, transformed_mask = _ct_case(
            image,
            failure_case,
            rng,
            np_rng,
            severity,
            object_mask,
            defect_mask,
            group_rng,
            case_options,
        )
    else:
        result, transform, records, transformed_mask = _rgb_case(
            image.convert("RGB"),
            failure_case,
            rng,
            np_rng,
            severity,
            object_mask,
            defect_mask,
            case_options,
        )
    validate_augmented(
        result,
        modality,
        records,
        original=image,
        original_object_mask=object_mask,
        output_object_mask=transformed_mask,
    )
    return AugmentResult(result, transform, failure_case, records, transformed_mask)


def validate_augmented(
    image: Image.Image,
    modality: str,
    records: list[dict[str, Any]],
    original: Image.Image | None = None,
    original_object_mask: Image.Image | None = None,
    output_object_mask: Image.Image | None = None,
) -> None:
    if image.width < 1 or image.height < 1:
        raise ValueError("augmentation produced an empty image")
    array = np.asarray(image)
    if not np.isfinite(array).all():
        raise ValueError("augmentation produced NaN/Inf pixels")
    if float(array.std()) < 0.25:
        raise ValueError("augmentation produced a near-constant image")
    if not records:
        raise ValueError("FAIL sample has no augmentation records")
    if modality == "CT" and any(record["type"] == "occlusion_box" for record in records):
        raise ValueError("occlusion_box is forbidden for CT")
    record_types = {record["type"] for record in records}
    luminance = np.asarray(image.convert("L"), dtype=np.float32)
    if float((luminance <= 5).mean()) >= 0.98:
        raise ValueError("quality_gate: 98% or more pixels are near black")
    if float((luminance >= 250).mean()) >= 0.98:
        raise ValueError("quality_gate: 98% or more pixels are saturated")
    if original is None:
        return
    baseline = np.asarray(
        original.convert("L").resize(image.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    baseline_mean = max(float(baseline.mean()), 1.0)
    output_mean = float(luminance.mean())

    if {"signal_to_transmission", "linear_exposure_reduction"} & record_types:
        if output_mean > baseline_mean * 0.72:
            raise ValueError("quality_gate: underexposure is not strong enough")
    if "linear_exposure_reduction" in record_types:
        region = (
            np.asarray(output_object_mask.convert("L")) > 0
            if output_object_mask is not None
            else np.ones(luminance.shape, dtype=bool)
        )
        mean_ratio = float(luminance[region].mean()) / max(
            float(baseline[region].mean()), 1.0
        )
        if not 0.40 <= mean_ratio <= 0.70:
            raise ValueError(
                "quality_gate: underexposure outline luminance reduction is outside 30%..60%"
            )
        if float(baseline[region].std()) >= 1.0:
            correlation = float(
                np.corrcoef(baseline[region].ravel(), luminance[region].ravel())[0, 1]
            )
            # 0.80, not 0.95. The correlation is taken between two sRGB-encoded images, and
            # the encoding is non-linear, so even a noiseless exposure reduction lands near
            # 0.89. A 0.95 floor therefore rejected the transform itself rather than the
            # noise it was meant to police; 0.80 still catches a scrambled lighting order.
            if not np.isfinite(correlation) or correlation < 0.80:
                raise ValueError(
                    "quality_gate: underexposure changed the spatial lighting order"
                )
    if "overexposure" in record_types:
        if output_mean <= baseline_mean:
            raise ValueError("quality_gate: overexposure did not increase luminance")
        if output_object_mask is not None:
            region = np.asarray(output_object_mask.convert("L")) > 0
            saturation = float((luminance[region] >= 250).mean())
            # The floor is 5%, not 15%. A sweep over 24 sources showed that even a gain of
            # 5.0 with a clip at 140 leaves the weakest source at 6.6% object saturation,
            # so a 15% floor was unreachable for part of the corpus no matter how the
            # exposure was drawn. The upper bound still rejects a fully blown frame.
            if not 0.05 <= saturation <= 0.75:
                raise ValueError(
                    "quality_gate: overexposure object saturation is outside 5%..75%"
                )

    def edge_energy(array: np.ndarray) -> float:
        horizontal = np.abs(np.diff(array, axis=1)).mean()
        vertical = np.abs(np.diff(array, axis=0)).mean()
        return float(horizontal + vertical)

    if {"directional_motion_blur", "defocus_blur"} & record_types:
        baseline_edge = max(edge_energy(baseline), 1e-6)
        ratio = edge_energy(luminance) / baseline_edge
        maximum = 0.75 if "defocus_blur" in record_types else 0.85
        minimum = 0.25 if "defocus_blur" in record_types else 0.0
        if ratio > maximum or ratio < minimum:
            raise ValueError(
                f"quality_gate: edge-energy ratio {ratio:.3f} is outside "
                f"{minimum:.2f}..{maximum:.2f}"
            )

    if "lighting_gradient" in record_types:
        # Comparing fixed left/right and top/bottom halves under-reads a diagonal gradient and
        # measures mostly background, so the old 0.12 threshold did not separate the samples
        # visual QA approved from the ones it rejected (both spanned 0.12..0.33). Measure along
        # the gradient axis instead, and inside the battery outline when one is available,
        # because that is what a reviewer actually looks at.
        gradient = next(
            record for record in records if record["type"] == "lighting_gradient"
        )
        angle = math.radians(float(gradient["parameters"].get("angle_deg", 0.0)))
        yy, xx = np.mgrid[0 : image.height, 0 : image.width]
        projection = xx * math.cos(angle) + yy * math.sin(angle)
        region = (
            np.asarray(output_object_mask.convert("L")) > 0
            if output_object_mask is not None
            else np.zeros(projection.shape, dtype=bool)
        )
        if not region.any():
            region = np.ones(projection.shape, dtype=bool)
        sampled = luminance[region]
        axis = projection[region]
        dark_side = sampled[axis <= float(np.quantile(axis, 0.20))]
        bright_side = sampled[axis >= float(np.quantile(axis, 0.80))]
        contrast = abs(float(bright_side.mean()) - float(dark_side.mean()))
        asymmetry = contrast / max(float(sampled.mean()), 1.0)
        baseline_sampled = baseline[region]
        baseline_dark = baseline_sampled[axis <= float(np.quantile(axis, 0.20))]
        baseline_bright = baseline_sampled[axis >= float(np.quantile(axis, 0.80))]
        baseline_asymmetry = abs(
            float(baseline_bright.mean()) - float(baseline_dark.mean())
        ) / max(float(baseline_sampled.mean()), 1.0)
        if not 0.25 <= asymmetry <= 0.60:
            raise ValueError("quality_gate: uneven lighting contrast is too small")
        if asymmetry - baseline_asymmetry < 0.15:
            raise ValueError("quality_gate: uneven lighting did not add enough asymmetry")

    if "lens_dust_shadow" in record_types:
        parameters = next(
            record["parameters"] for record in records if record["type"] == "lens_dust_shadow"
        )
        count = int(parameters.get("shadow_count", 0))
        if not 1 <= count <= 4:
            raise ValueError("quality_gate: lens dust shadow count is outside 1..4")
        coverage = float(parameters.get("frame_affected_ratio", 0))
        if not 0.001 <= coverage <= 0.35:
            raise ValueError("quality_gate: lens dust frame coverage is outside range")
    if "surface_aware_specular_reflection" in record_types:
        parameters = records[0]["parameters"]
        if float(parameters.get("outline_overlap_ratio", 0)) < 0.90:
            raise ValueError("quality_gate: glare outline overlap is below 90%")
        core_ratio = float(parameters.get("core_object_area_ratio", 0))
        if not 0.01 <= core_ratio <= 0.12:
            raise ValueError(
                f"quality_gate: glare core area {core_ratio:.4f} is outside 1%..12% of object"
            )
        if float(parameters.get("defect_coverage_ratio", 0)) > 0.70:
            raise ValueError("quality_gate: glare covers more than 70% of defect mask")
        if bool(parameters.get("defect_present")) and float(
            parameters.get("defect_coverage_ratio", 0)
        ) < 0.30:
            raise ValueError("quality_gate: glare covers less than 30% of defect mask")
        saturation = float(parameters.get("object_saturation_ratio", 0))
        if saturation > 0.12:
            raise ValueError("quality_gate: glare saturates more than 12% of object")
    if "metal_anchored_streaks" in record_types:
        streak_record = next(
            record for record in records if record["type"] == "metal_anchored_streaks"
        )
        if int(streak_record["parameters"].get("dense_intersection_pixels", 0)) <= 0:
            raise ValueError(
                "quality_gate: photon-starvation streak does not cross dense region"
            )
        quadrants = streak_record["parameters"].get("quadrant_counts", [])
        if len(quadrants) != 4 or any(int(count) < 4 for count in quadrants):
            raise ValueError(
                "quality_gate: metal streak rays do not cover all four quadrants"
            )
        if float(streak_record["parameters"].get("max_angular_gap_deg", 360)) > 35:
            raise ValueError(
                "quality_gate: metal streak maximum angular gap exceeds 35 degrees"
            )
    if "lens_fiber_shadow" in record_types:
        curves = records[0]["parameters"].get("curves", [])
        if not 1 <= len(curves) <= 2:
            raise ValueError("quality_gate: hair curve count is outside 1..2")
        maximum_width = max(1, math.ceil(max(image.size) * 0.006))
        if any(
            not 1 <= int(curve["thickness_px"]) <= maximum_width
            for curve in curves
        ):
            raise ValueError("quality_gate: hair thickness is outside v1.7 range")
        long_side = max(image.size)
        if any(
            not 0.15 * long_side <= float(curve["length_px"]) <= 0.60 * long_side
            for curve in curves
        ):
            raise ValueError("quality_gate: hair length is outside 15%..60% of long side")
    if (
        original_object_mask is not None
        and output_object_mask is not None
        and {"porosity_targeted_fov_crop", "timing_edge_crop"}
        & record_types
    ):
        original_area = max(
            int((np.asarray(original_object_mask.convert("L")) > 0).sum()), 1
        )
        output_area = int((np.asarray(output_object_mask.convert("L")) > 0).sum())
        retained = output_area / original_area
        minimum = (
            0.10
            if "porosity_targeted_fov_crop" in record_types
            else 0.50
            if "timing_edge_crop" in record_types
            else 0.60
        )
        if retained < minimum:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is below {minimum:.2f}"
            )
        # The lower bound alone let retention 1.000 through, meaning the crop missed the
        # battery entirely and the output looked like a normal photo. A trigger timing failure
        # has to leave the cell clipped by the frame.
        if "timing_edge_crop" in record_types and retained > 0.90:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is above 0.90; "
                "the crop did not clip the battery"
            )
        if "porosity_targeted_fov_crop" in record_types and retained > 0.95:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is above 0.95; "
                "alignment failure did not remove enough required structure"
            )
