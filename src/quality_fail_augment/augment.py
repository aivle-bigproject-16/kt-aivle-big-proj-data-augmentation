from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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
SOURCE_REFERENCES = {case: f"v2.0:{case}" for case in (*CT_CASES, *RGB_CASES)}
UNEVEN_TAIL_QUANTILE = 0.20
UNEVEN_MIN_ASYMMETRY = 0.45
UNEVEN_MAX_ASYMMETRY = 0.60
UNEVEN_MIN_ADDED_ASYMMETRY = 0.25


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


def _rms_gradient_energy(values: np.ndarray) -> float:
    horizontal = np.diff(values, axis=1)
    vertical = np.diff(values, axis=0)
    return float(
        math.sqrt(
            float(np.mean(horizontal * horizontal))
            + float(np.mean(vertical * vertical))
        )
    )


def _axis_asymmetry(
    luminance: np.ndarray, axis_values: np.ndarray, region: np.ndarray
) -> float:
    sampled = luminance[region]
    axis = axis_values[region]
    low = float(np.quantile(axis, UNEVEN_TAIL_QUANTILE))
    high = float(np.quantile(axis, 1.0 - UNEVEN_TAIL_QUANTILE))
    low_side = sampled[axis <= low]
    high_side = sampled[axis >= high]
    return abs(float(high_side.mean()) - float(low_side.mean())) / max(
        float(sampled.mean()), 1.0
    )


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
        source_size = result.size
        aspect = result.width / result.height
        directions = ("left", "right", "top", "bottom")
        direction_rng = group_rng or rng
        direction_start = direction_rng.randrange(0, len(directions))
        ordered_directions = (
            directions[direction_start:] + directions[:direction_start]
        )

        def crop_for(side: str, amount: int) -> list[int]:
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
                target_height = min(
                    current_height, max(1, round(current_width / aspect))
                )
                trim = current_height - target_height
                candidate[1] += trim // 2
                candidate[3] -= trim - trim // 2
            else:
                target_width = min(
                    current_width, max(1, round(current_height * aspect))
                )
                trim = current_width - target_width
                candidate[0] += trim // 2
                candidate[2] -= trim - trim // 2
            return candidate

        target_retained_ratio = rng.uniform(0.92, 0.96)
        selected: tuple[str, list[int], float | None] | None = None
        if mask is None or mask.getbbox() is None:
            raise ValueError("alignment_crop_requires_object_mask")
        else:
            original_area = max(int((np.asarray(mask) > 0).sum()), 1)
            for direction in ordered_directions:
                axis_size = (
                    result.width
                    if direction in {"left", "right"}
                    else result.height
                )
                low_amount, high_amount = 1, axis_size - 1
                best_crop = crop_for(direction, low_amount)
                best_retained = (
                    int((np.asarray(mask.crop(tuple(best_crop))) > 0).sum())
                    / original_area
                )
                best_distance = abs(best_retained - target_retained_ratio)
                for _ in range(20):
                    amount = (low_amount + high_amount) // 2
                    candidate = crop_for(direction, amount)
                    retained = (
                        int((np.asarray(mask.crop(tuple(candidate))) > 0).sum())
                        / original_area
                    )
                    distance = abs(retained - target_retained_ratio)
                    if distance < best_distance:
                        best_crop = candidate
                        best_retained = retained
                        best_distance = distance
                    if retained > target_retained_ratio:
                        low_amount = min(amount + 1, high_amount)
                    else:
                        high_amount = max(amount - 1, low_amount)
                if 0.90 <= best_retained <= 0.98:
                    selected = (direction, best_crop, best_retained)
                    break
            if selected is None:
                raise ValueError("alignment_crop_no_gate_safe_window")

        direction, crop, retained_outline_ratio = selected
        cropped_width = crop[2] - crop[0]
        cropped_height = crop[3] - crop[1]
        result = result.crop(tuple(crop)).resize(
            source_size, Image.Resampling.LANCZOS
        )
        if mask is not None:
            mask = mask.crop(tuple(crop)).resize(
                source_size, Image.Resampling.NEAREST
            )
        transform = transform.then(
            Affine(xoff=-crop[0], yoff=-crop[1])
        ).then(
            Affine(
                a=source_size[0] / cropped_width,
                e=source_size[1] / cropped_height,
            )
        )
        records.append(
            _record(
                len(records) + 1,
                "alignment_edge_crop",
                severity,
                direction=direction,
                crop_box=crop,
                source_size=list(source_size),
                output_size=list(result.size),
                source_aspect_ratio=aspect,
                output_aspect_ratio=result.width / result.height,
                resize_to_source_size=True,
                target_outline_retained_ratio=target_retained_ratio,
                retained_outline_ratio=retained_outline_ratio,
            )
        )
    elif case == "ct_acquisition_motion":
        direction_rng = group_rng or rng
        angle = direction_rng.uniform(0, 359)
        normalized_offset = rng.uniform(6.0, 9.0)
        scale = max(result.size) / 512.0
        offset = max(1, round(normalized_offset * scale))
        dx = round(math.cos(math.radians(angle)) * offset)
        dy = round(math.sin(math.radians(angle)) * offset)
        blur_half_range = rng.uniform(0.5, 1.2)
        kernel = max(3, (round((blur_half_range * 2 + 1) * scale) | 1))
        blurred = _motion_blur(result, kernel, angle)
        shifted = _translate(result, dx, dy, background(result, "CT"))
        shifted_weight = rng.uniform(0.15, 0.22)
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
        factor = rng.uniform(0.78, 0.86)
        result = ImageEnhance.Brightness(result).enhance(factor)
        records.append(_record(1, "signal_to_transmission", severity, signal_factor=factor))
        source = np.asarray(result.convert("L"), dtype=np.float32)
        photons = rng.uniform(90.0, 130.0)
        sampled = np_rng.poisson(np.clip(source / 255.0, 0, 1) * photons) / photons * 255.0
        result = Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8)).convert(
            result.mode
        )
        records.append(_record(2, "poisson_sampling", severity, photon_scale=photons))
        sigma_normalized = rng.uniform(0.001, 0.002)
        sigma_8bit = sigma_normalized * 255.0
        result = _noise(result, np_rng, sigma_8bit, monochrome=True)
        records.append(
            _record(
                3,
                "read_noise",
                severity,
                sigma_normalized=sigma_normalized,
                sigma_8bit=sigma_8bit,
            )
        )
        contrast_factor = rng.uniform(0.95, 1.00)
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
            retained_ratio = rng.uniform(0.72, 0.82)
            view_count = max(8, round(full_view_count * retained_ratio))
            indices = np.linspace(0, full_view_count - 1, view_count).round().astype(int)
            angles = full_angles[indices]
            removed_angle = None
        else:
            removed_width = rng.uniform(15.0, 25.0)
            removed_start = rng.uniform(0.0, 180.0)
            distance = (full_angles - removed_start) % 180.0
            angles = full_angles[distance >= removed_width]
            if len(angles) < 8:
                raise ValueError("limited_angle_retained_view_count_below_8")
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
        reconstruction_weight = rng.uniform(0.25, 0.40)
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
                removed_width_deg=(
                    removed_width if subtype == "limited_angle" else None
                ),
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
        top_ratio = rng.uniform(0.003, 0.010)
        threshold = float(np.quantile(array, 1.0 - top_ratio))
        dense_region_mask = _largest_connected_component(array >= threshold)
        if float(dense_region_mask.mean()) < 0.001:
            raise ValueError("dense_region_mask_too_small")
        attenuation = rng.uniform(0.05, 0.10)
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
            rng.uniform(8.0, 15.0)
            if rng.random() < 0.5
            else rng.uniform(-4.0, -2.0)
        )
        cupping_profile = np.clip(1.0 - elliptical, 0.0, 1.0)
        feather_ratio = rng.uniform(0.08, 0.14)
        feather_t = np.clip(cupping_profile / feather_ratio, 0.0, 1.0)
        feathered_edge = feather_ratio * feather_t**2 * (3.0 - 2.0 * feather_t)
        cupping_profile = np.where(
            cupping_profile < feather_ratio, feathered_edge, cupping_profile
        )
        cupping = cupping_delta * cupping_profile * asymmetry
        result = _luminance_field(result, cupping.astype(np.float32))
        records.append(
            _record(
                2,
                "cupping_field",
                severity,
                center=[center_x, center_y],
                radii=[radius_x, radius_y],
                luminance_delta=cupping_delta,
                asymmetric=True,
                feather_ratio=feather_ratio,
            )
        )
        count = rng.randint(8, 12)
        field = Image.new("F", result.size, 0.0)
        draw = ImageDraw.Draw(field)
        diagonal = math.hypot(result.width, result.height)
        cx, cy = int(round(center_x)), int(round(center_y))
        start_angle = rng.uniform(0, 360)
        jitter_limit = min(3.0, 120.0 / count)
        angles: list[float] = []
        ray_widths_final: list[float] = []
        ray_alphas: list[float] = []
        for index in range(count):
            angle_deg = (
                start_angle
                + index * (360.0 / count)
                + rng.uniform(-jitter_limit, jitter_limit)
            ) % 360.0
            angle = math.radians(angle_deg)
            normalized_width = rng.uniform(0.4, 0.8)
            source_scale = max(result.size) / 512.0
            width = max(1, round(normalized_width * source_scale))
            alpha = rng.uniform(0.02, 0.05)
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
            ray_widths_final.append(normalized_width)
            ray_alphas.append(alpha)
        streak_field = np.asarray(field).copy()
        distance = np.hypot(xx - cx, yy - cy)
        frame_distance = max(
            math.hypot(center_x - x, center_y - y)
            for x, y in (
                (0.0, 0.0),
                (result.width - 1.0, 0.0),
                (0.0, result.height - 1.0),
                (result.width - 1.0, result.height - 1.0),
            )
        )
        decay_distance_ratio = rng.uniform(0.15, 0.22)
        decay_scale = frame_distance * decay_distance_ratio
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
                ray_widths_final_512_px=ray_widths_final,
                ray_start_alphas=ray_alphas,
                max_angular_gap_deg=max(gaps),
                quadrant_counts=[
                    sum(1 for value in angles if start <= value < start + 90)
                    for start in (0, 90, 180, 270)
                ],
                distance_decay_scale_px=decay_scale,
                decay_distance_ratio=decay_distance_ratio,
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

        if mask is None or mask.getbbox() is None:
            raise ValueError("timing_crop_requires_object_mask")
        target_retained_ratio = rng.uniform(0.55, 0.68)
        retained_outline_ratio: float | None = None
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
        retained_outline_ratio = int(
            (np.asarray(mask.crop(tuple(crop))) > 0).sum()
        ) / original_area
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
                retained_outline_ratio=retained_outline_ratio,
                conveyor_axis=conveyor_axis,
                forward_direction=forward_direction,
                timing_event=timing_event,
            )
        )
        if rng.random() < 0.70:
            kernel = rng.randrange(11, 18, 2)
            result = _motion_blur(result, kernel, 0 if side in {"left", "right"} else 90)
            records.append(_record(2, "conveyor_motion_blur", severity, kernel=kernel))
    elif case == "rgb_uneven_lighting":
        yy, xx = np.mgrid[0 : result.height, 0 : result.width]
        original_array = np.asarray(result).astype(np.float32)
        original_luminance = np.asarray(result.convert("L"), dtype=np.float32)
        object_region = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None and mask.getbbox() is not None
            else np.ones((result.height, result.width), dtype=bool)
        )

        angles = (0.0, 90.0, 180.0, 270.0)
        start = rng.randrange(0, len(angles))
        ordered_angles = angles[start:] + angles[:start]
        selected_gradient: tuple[
            float, float, float, np.ndarray, np.ndarray, float, float, float
        ] | None = None
        gain_pairs = (
            (0.25, 1.65),
            (0.18, 1.85),
            (0.12, 2.05),
        )
        for angle_deg in ordered_angles:
            angle = math.radians(angle_deg)
            projection = xx * math.cos(angle) + yy * math.sin(angle)
            if mask is not None and mask.getbbox() is not None:
                left, top, right, bottom = mask.getbbox()
                corners = np.asarray(
                    [[left, top], [right, top], [left, bottom], [right, bottom]],
                    dtype=np.float64,
                )
                bounds = (
                    corners[:, 0] * math.cos(angle)
                    + corners[:, 1] * math.sin(angle)
                )
                low, high = float(bounds.min()), float(bounds.max())
            else:
                low, high = float(projection.min()), float(projection.max())
            normalized = np.clip(
                (projection - low) / max(high - low, 1.0), 0.0, 1.0
            )
            smooth = normalized * normalized * (3.0 - 2.0 * normalized)
            baseline_asymmetry = _axis_asymmetry(
                original_luminance, projection, object_region
            )
            for dark, bright in gain_pairs:
                gain = dark + (bright - dark) * smooth
                full_field_array = original_array * gain[..., None]
                full_field = _array_image(full_field_array, "RGB")
                full_asymmetry = _axis_asymmetry(
                    np.asarray(full_field.convert("L"), dtype=np.float32),
                    projection,
                    object_region,
                )
                if (
                    full_asymmetry < UNEVEN_MIN_ASYMMETRY
                    or full_asymmetry - baseline_asymmetry
                    < UNEVEN_MIN_ADDED_ASYMMETRY
                ):
                    continue
                field_blend = 1.0
                candidate_array = full_field_array
                candidate_asymmetry = full_asymmetry
                if full_asymmetry > UNEVEN_MAX_ASYMMETRY:
                    low_blend, high_blend = 0.0, 1.0
                    target_asymmetry = 0.55
                    for _ in range(24):
                        trial_blend = (low_blend + high_blend) / 2.0
                        trial_array = (
                            original_array * (1.0 - trial_blend)
                            + full_field_array * trial_blend
                        )
                        trial = _array_image(trial_array, "RGB")
                        trial_asymmetry = _axis_asymmetry(
                            np.asarray(trial.convert("L"), dtype=np.float32),
                            projection,
                            object_region,
                        )
                        if trial_asymmetry < target_asymmetry:
                            low_blend = trial_blend
                        else:
                            high_blend = trial_blend
                    field_blend = (low_blend + high_blend) / 2.0
                    candidate_array = (
                        original_array * (1.0 - field_blend)
                        + full_field_array * field_blend
                    )
                    candidate_asymmetry = _axis_asymmetry(
                        np.asarray(
                            _array_image(candidate_array, "RGB").convert("L"),
                            dtype=np.float32,
                        ),
                        projection,
                        object_region,
                    )
                if (
                    UNEVEN_MIN_ASYMMETRY
                    <= candidate_asymmetry
                    <= UNEVEN_MAX_ASYMMETRY
                    and candidate_asymmetry - baseline_asymmetry
                    >= UNEVEN_MIN_ADDED_ASYMMETRY
                ):
                    selected_gradient = (
                        angle_deg,
                        dark,
                        bright,
                        projection,
                        candidate_array,
                        baseline_asymmetry,
                        candidate_asymmetry,
                        field_blend,
                    )
                    break
            if selected_gradient is not None:
                break
        if selected_gradient is None:
            raise ValueError("uneven_lighting_no_gate_safe_gradient")
        (
            angle_deg,
            dark,
            bright,
            projection,
            selected_array,
            baseline_asymmetry,
            selected_asymmetry,
            field_blend,
        ) = selected_gradient
        result = _array_image(selected_array, "RGB")
        records.append(
            _record(
                1,
                "lighting_gradient",
                severity,
                angle_deg=angle_deg,
                dark_gain=dark,
                bright_gain=bright,
                transition="smoothstep",
                field_blend=field_blend,
                baseline_asymmetry=baseline_asymmetry,
                output_asymmetry=selected_asymmetry,
            )
        )
        if mask is not None and mask.getbbox() is not None and rng.random() < 0.80:
            zone_count = rng.randint(2, 3)
            zone_field = np.ones((result.height, result.width), dtype=np.float32)
            points = _mask_points(mask)
            object_area = max(int((np.asarray(mask) > 0).sum()), 1)
            zones = []
            for _ in range(zone_count):
                cx, cy = rng.choice(points)
                target_area = object_area * rng.uniform(0.10, 0.18)
                radius = math.sqrt(target_area / math.pi)
                zone_gain = rng.uniform(0.20, 0.45)
                distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
                gaussian = np.exp(-distance2 / max(2.0 * radius**2, 1.0))
                zone_field *= 1.0 - (1.0 - zone_gain) * gaussian
                zones.append([cx, cy, radius, zone_gain])
            zone_candidate = _array_image(
                np.asarray(result, dtype=np.float32) * zone_field[..., None], "RGB"
            )
            zone_asymmetry = _axis_asymmetry(
                np.asarray(zone_candidate.convert("L"), dtype=np.float32),
                projection,
                object_region,
            )
            if (
                UNEVEN_MIN_ASYMMETRY
                <= zone_asymmetry
                <= UNEVEN_MAX_ASYMMETRY
                and zone_asymmetry - baseline_asymmetry
                >= UNEVEN_MIN_ADDED_ASYMMETRY
            ):
                result = zone_candidate
                records.append(
                    _record(
                        2,
                        "led_dead_zone",
                        severity,
                        zone_count=zone_count,
                        zones=zones,
                    )
                )
    elif case == "rgb_reflection_glare":
        count = 2
        patches: list[dict[str, Any]] = []
        eligible = _mask_points(mask)
        object_array = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None and mask.getbbox() is not None
            else np.ones((result.height, result.width), dtype=bool)
        )
        defect_array = (
            np.asarray(defect_mask.convert("L")) > 0
            if defect_mask is not None and defect_mask.getbbox() is not None
            else np.zeros(object_array.shape, dtype=bool)
        )
        defect_array &= object_array
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
        defect_y, defect_x = np.where(defect_array)
        defect_points = list(zip(defect_x.tolist(), defect_y.tolist()))
        object_area = max(int(object_array.sum()), 1)
        defect_area = int(defect_array.sum())
        defect_object_ratio = defect_area / object_area
        feasible_defect_coverage = (
            0.085 / max(defect_object_ratio, 1e-6)
            if defect_points
            else 0.45
        )
        minimum_defect_coverage = min(0.45, feasible_defect_coverage)
        if defect_points:
            defect_coordinates = np.asarray(defect_points, dtype=np.float64)
            base_x, base_y = defect_coordinates[:, 0].mean(), defect_coordinates[:, 1].mean()
        elif highlight_points:
            base_x, base_y = rng.choice(highlight_points)
        else:
            base_x, base_y = centroid_x, centroid_y

        # Build a thin, elongated highlight and choose its dimensions from the battery area.
        # The previous frame-relative width frequently produced either <1% object coverage or
        # spilled off a narrow cylindrical cell. Candidate selection makes the generated core
        # satisfy the same geometric contract later enforced by the quality gate.
        target_area = max(0.06 * object_area, 0.55 * defect_area)
        target_area = min(target_area, 0.09 * object_area)
        best_candidate: tuple[
            float, Image.Image, list[dict[str, Any]], float, float
        ] | None = None
        length_factors = (0.50, 0.65, 0.80)
        width_factors = (1.0, 1.25, 1.50)
        axes = [major_axis, np.asarray([-major_axis[1], major_axis[0]])]
        if len(defect_points) > 1:
            defect_centred = defect_coordinates - np.asarray([base_x, base_y])
            defect_covariance = np.cov(defect_centred.T)
            _, defect_vectors = np.linalg.eigh(defect_covariance)
            defect_axis = defect_vectors[:, -1]
            axes.extend(
                [defect_axis, np.asarray([-defect_axis[1], defect_axis[0]])]
            )
        for axis in axes:
            for length_factor in length_factors:
                half_length = max(2.0, major_extent * length_factor / 2.0)
                estimated_width = max(
                    1.0,
                    target_area / max(count * 2.0 * half_length, 1.0),
                )
                for width_factor in width_factors:
                    width = max(1, round(estimated_width * width_factor))
                    if (2.0 * half_length) / width < 5.0:
                        continue
                    candidate_alpha = Image.new("L", result.size, 0)
                    candidate_draw = ImageDraw.Draw(candidate_alpha)
                    candidate_patches: list[dict[str, Any]] = []
                    for patch_index in range(count):
                        offset = (
                            (patch_index - (count - 1) / 2.0)
                            * max(width * 1.8, 2.0)
                        )
                        cx = float(base_x - axis[1] * offset)
                        cy = float(base_y + axis[0] * offset)
                        start = (
                            cx - axis[0] * half_length,
                            cy - axis[1] * half_length,
                        )
                        end = (
                            cx + axis[0] * half_length,
                            cy + axis[1] * half_length,
                        )
                        alpha = min(
                            math.floor(255 * 0.78),
                            max(
                                math.ceil(255 * 0.70),
                                round(255 * rng.uniform(0.70, 0.78)),
                            ),
                        )
                        candidate_draw.line((start, end), fill=alpha, width=width)
                        candidate_patches.append(
                            {
                                "center": [round(cx, 3), round(cy, 3)],
                                "axis": axis.tolist(),
                                "half_length": half_length,
                                "width": width,
                                "alpha": alpha / 255,
                            }
                        )
                    candidate_array = np.asarray(candidate_alpha).copy()
                    candidate_array[~object_array] = 0
                    core = candidate_array > 0
                    core_ratio = float(core.sum() / object_area)
                    defect_coverage = float(
                        (core & defect_array).sum() / max(defect_area, 1)
                    )
                    valid_defect = (
                        not defect_points
                        or minimum_defect_coverage <= defect_coverage <= 0.70
                    )
                    if 0.045 <= core_ratio <= 0.12 and valid_defect:
                        score = abs(core_ratio - 0.075)
                        if defect_points:
                            score += abs(defect_coverage - 0.55)
                        clipped_alpha = Image.fromarray(candidate_array, mode="L")
                        candidate = (
                            score,
                            clipped_alpha,
                            candidate_patches,
                            core_ratio,
                            defect_coverage,
                        )
                        if best_candidate is None or score < best_candidate[0]:
                            best_candidate = candidate
        if best_candidate is None:
            # Narrow/irregular outlines can clip every rasterized width candidate even though
            # the v1.9 contract itself is feasible.  Build the same two elongated highlights
            # directly inside the object mask, targeting 9% core coverage and the existing
            # defect-overlap bounds.  Alpha, bloom and all quality-gate ranges stay unchanged.
            target_count = max(1, min(object_area, round(0.09 * object_area)))
            selected = np.zeros(object_array.shape, dtype=bool)
            axis = np.asarray(major_axis, dtype=np.float32)
            perpendicular = np.asarray([-axis[1], axis[0]])
            yy_all, xx_all = np.ogrid[0 : result.height, 0 : result.width]
            centred_x = xx_all.astype(np.float32) - np.float32(base_x)
            centred_y = yy_all.astype(np.float32) - np.float32(base_y)
            along = centred_x * axis[0] + centred_y * axis[1]
            across = centred_x * perpendicular[0] + centred_y * perpendicular[1]
            band_offset = max(2.0, math.sqrt(target_count) * 0.18)
            band_score = np.minimum(
                np.abs(across - band_offset), np.abs(across + band_offset)
            ) + np.maximum(np.abs(along) - major_extent * 0.45, 0.0)

            if defect_area:
                minimum_pixels = math.ceil(minimum_defect_coverage * defect_area)
                desired_pixels = min(
                    math.floor(0.55 * defect_area),
                    math.floor(0.70 * defect_area),
                    target_count,
                )
                desired_pixels = min(
                    target_count, max(minimum_pixels, desired_pixels)
                )
                defect_indices = np.flatnonzero(defect_array)
                defect_order = defect_indices[
                    np.argsort(band_score.ravel()[defect_indices], kind="stable")
                ]
                selected.ravel()[defect_order[:desired_pixels]] = True

            remaining_count = target_count - int(selected.sum())
            if remaining_count > 0:
                available = object_array & ~selected
                available_indices = np.flatnonzero(available)
                available_order = available_indices[
                    np.argsort(band_score.ravel()[available_indices], kind="stable")
                ]
                selected.ravel()[available_order[:remaining_count]] = True

            alpha = min(
                math.floor(255 * 0.78),
                max(math.ceil(255 * 0.70), round(255 * rng.uniform(0.70, 0.78))),
            )
            candidate_array = np.zeros(object_array.shape, dtype=np.uint8)
            candidate_array[selected] = alpha
            core_ratio = float(selected.sum() / object_area)
            defect_coverage = float(
                (selected & defect_array).sum() / max(defect_area, 1)
            )
            estimated_width = max(
                1, round(target_count / max(2.0 * 2.0 * major_extent * 0.45, 1.0))
            )
            fallback_patches = [
                {
                    "center": [
                        round(float(base_x + perpendicular[0] * offset), 3),
                        round(float(base_y + perpendicular[1] * offset), 3),
                    ],
                    "axis": axis.tolist(),
                    "half_length": major_extent * 0.45,
                    "width": estimated_width,
                    "alpha": alpha / 255,
                }
                for offset in (-band_offset, band_offset)
            ]
            best_candidate = (
                abs(core_ratio - 0.075),
                Image.fromarray(candidate_array, mode="L"),
                fallback_patches,
                core_ratio,
                defect_coverage,
            )
        _, glare_alpha, patches, core_object_ratio, defect_coverage = best_candidate
        radius_final = rng.uniform(10, 14)
        radius = radius_final * max(result.size) / 512.0
        core_alpha = np.asarray(glare_alpha, dtype=np.float32)
        bloom_alpha = np.asarray(
            glare_alpha.filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32,
        )
        visible_alpha = np.maximum(core_alpha, bloom_alpha * 0.72)
        visible_alpha = np.clip(visible_alpha, 0, 255).astype(np.uint8)
        visible_alpha[~object_array] = 0
        overlay = Image.new("RGBA", result.size, (255, 246, 224, 0))
        overlay.putalpha(Image.fromarray(visible_alpha, mode="L"))
        result = Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")
        glare_mask = np.asarray(glare_alpha) > 0
        outline_overlap = float((glare_mask & object_array).sum() / max(glare_mask.sum(), 1))
        saturation_ratio = float(
            ((np.asarray(result.convert("L")) >= 250) & object_array).sum()
            / max(object_array.sum(), 1)
        )
        records.append(
            _record(
                1,
                "surface_aware_specular_reflection",
                severity,
                seed=(
                    "defect"
                    if defect_points
                    else "existing_highlight"
                    if highlight_points
                    else "outline_axis"
                ),
                patches=patches,
                bloom_radius_final_space=radius_final,
                bloom_radius_source_space=radius,
                outline_overlap_ratio=outline_overlap,
                core_object_area_ratio=core_object_ratio,
                defect_present=bool(defect_points),
                defect_object_area_ratio=defect_object_ratio,
                minimum_defect_coverage_ratio=minimum_defect_coverage,
                defect_coverage_gate_relaxed_for_feasibility=(
                    bool(defect_points) and minimum_defect_coverage < 0.45
                ),
                defect_coverage_ratio=defect_coverage,
                object_saturation_ratio=saturation_ratio,
            )
        )
        records.append(_record(2, "highlight_bloom", severity, radius_final_space=radius_final, radius_source_space=radius))
    elif case == "rgb_focus_failure":
        source_scale = max(result.size) / 512.0
        radius_final = rng.uniform(7.5, 10.0)
        radius = radius_final * source_scale
        result = result.filter(ImageFilter.GaussianBlur(radius))
        records.append(
            _record(
                1,
                "defocus_blur",
                severity,
                radius_final_space=radius_final,
                radius_source_space=radius,
            )
        )
    elif case == "rgb_underexposure":
        baseline_luminance = np.asarray(result.convert("L"), dtype=np.float32)
        object_region = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None and mask.getbbox() is not None
            else np.ones(baseline_luminance.shape, dtype=bool)
        )
        linear = _srgb_to_linear(np.asarray(result.convert("RGB")))
        factor = rng.uniform(0.18, 0.28)
        exposed = linear * factor
        photon_capacity = rng.uniform(80.0, 120.0)
        sampled = (
            np_rng.poisson(np.clip(exposed, 0.0, 1.0) * photon_capacity)
            / photon_capacity
        )
        read_sigma = rng.uniform(0.010, 0.015)
        shared = np_rng.normal(0.0, read_sigma, sampled.shape[:2])[..., None]
        chroma = np_rng.normal(
            0.0, read_sigma * 0.25, sampled.shape
        )
        black_level = rng.uniform(0.008, 0.012)
        sampled = np.clip(sampled + shared + chroma - black_level, 0.0, 1.0)
        target_mean_ratio = rng.uniform(0.44, 0.50)
        correction_product = 1.0
        baseline_object_mean = max(
            float(baseline_luminance[object_region].mean()), 1.0
        )
        baseline_frame_mean = max(float(baseline_luminance.mean()), 1.0)
        for _ in range(3):
            candidate_array = _linear_to_srgb(sampled)
            candidate_luminance = np.asarray(
                Image.fromarray(candidate_array, mode="RGB").convert("L"),
                dtype=np.float32,
            )
            current_ratio = (
                float(candidate_luminance[object_region].mean())
                / baseline_object_mean
            )
            current_frame_ratio = (
                float(candidate_luminance.mean()) / baseline_frame_mean
            )
            desired_srgb_scale = min(
                target_mean_ratio / max(current_ratio, 1e-6),
                0.60 / max(current_frame_ratio, 1e-6),
            )
            correction = float(
                np.clip(desired_srgb_scale**2.2, 0.30, 3.0)
            )
            sampled = np.clip(sampled * correction, 0.0, 1.0)
            correction_product *= correction
        result = Image.fromarray(_linear_to_srgb(sampled), mode="RGB")
        actual_mean_ratio = float(
            np.asarray(result.convert("L"), dtype=np.float32)[object_region].mean()
            / baseline_object_mean
        )
        actual_frame_mean_ratio = float(
            np.asarray(result.convert("L"), dtype=np.float32).mean()
            / baseline_frame_mean
        )
        records.append(
            _record(
                1,
                "linear_exposure_reduction",
                severity,
                exposure_factor=factor,
                target_outline_mean_ratio=target_mean_ratio,
                actual_outline_mean_ratio=actual_mean_ratio,
                actual_frame_mean_ratio=actual_frame_mean_ratio,
                linear_correction=correction_product,
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
        factor = rng.uniform(2.10, 2.60)
        linear = _srgb_to_linear(np.asarray(result.convert("RGB")))
        result = Image.fromarray(_linear_to_srgb(np.clip(linear * factor, 0.0, 1.0)), mode="RGB")
        array = np.asarray(result).copy()
        luminance = np.asarray(result.convert("L"), dtype=np.float32)
        object_region = (
            np.asarray(mask.convert("L")) > 0
            if mask is not None and mask.getbbox() is not None
            else np.ones(luminance.shape, dtype=bool)
        )
        object_indices = np.flatnonzero(object_region.ravel())
        target_saturation = rng.uniform(0.50, 0.60)
        saturated_count = min(
            max(1, round(len(object_indices) * target_saturation)),
            len(object_indices),
        )
        object_luminance = luminance.ravel()[object_indices]
        selected_local = np.argpartition(
            object_luminance, len(object_luminance) - saturated_count
        )[-saturated_count:]
        saturated_flat = object_indices[selected_local]
        object_pixels = array[object_region]
        array[object_region] = np.minimum(object_pixels, 249)
        flat = array.reshape(-1, 3)
        flat[saturated_flat] = 255
        threshold = float(object_luminance[selected_local].min())
        result = _array_image(array, "RGB")
        records.append(
            _record(
                1,
                "overexposure",
                severity,
                exposure_factor=factor,
                color_space="linear_light",
                clip_threshold=threshold,
                target_object_saturation_ratio=target_saturation,
            )
        )
        if rng.random() < 0.70:
            radius = rng.uniform(10.0, 18.0) * max(result.size) / 512.0
            bright = result.filter(ImageFilter.GaussianBlur(radius))
            bloom_blend = rng.uniform(0.25, 0.30)
            result = Image.blend(result, bright, bloom_blend)
            # Bloom softens the clipped core. Reapply the selected saturated region so the
            # optional secondary effect cannot invalidate the primary overexposure contract.
            bloomed = np.asarray(result).copy()
            bloomed[object_region] = np.minimum(bloomed[object_region], 249)
            bloomed.reshape(-1, 3)[saturated_flat] = 255
            result = _array_image(bloomed, "RGB")
            records.append(
                _record(
                    2,
                    "highlight_bloom",
                    severity,
                    radius_final_512_px=radius * 512.0 / max(result.size),
                    blend=bloom_blend,
                )
            )
    elif case == "rgb_surface_dust":
        count = rng.randint(3, 4)
        object_overlap_count = rng.randint(1, 2)
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
            # v1.9 uses a small number of large, out-of-focus lens shadows.
            # Radius is half of the planned 4%..6% core diameter.
            radius = long_side * rng.uniform(0.02, 0.03)
            # Keep one or two lens shadows over the projected battery area so the
            # contamination can actually interfere with inspection. The effect is
            # still rendered in camera-frame coordinates and does not follow the object.
            cx, cy = (
                rng.choice(eligible)
                if index < object_overlap_count and eligible
                else (rng.uniform(0, result.width), rng.uniform(0, result.height))
            )
            color = rng.choice(((65, 62, 58), (105, 101, 94), (145, 140, 130)))
            # Calibrated against the approved 0731_188 sample: visible enough to
            # obstruct inspection without turning into an opaque painted patch.
            core_alpha_ratio = rng.uniform(0.22, 0.25)
            halo_alpha_ratio = rng.uniform(0.10, 0.125)
            core_alpha = round(255 * core_alpha_ratio)
            halo_alpha = round(255 * halo_alpha_ratio)
            halo_scale = rng.uniform(2.6, 3.3)
            ellipse_ratio = rng.uniform(0.70, 1.30)
            radial_jitter = [rng.uniform(0.78, 1.22) for _ in range(16)]

            def blob_points(scale: float) -> list[tuple[float, float]]:
                return [
                    (
                        cx
                        + math.cos(2.0 * math.pi * point / 16.0)
                        * radius
                        * scale
                        * radial_jitter[point],
                        cy
                        + math.sin(2.0 * math.pi * point / 16.0)
                        * radius
                        * scale
                        * ellipse_ratio
                        * radial_jitter[point],
                    )
                    for point in range(16)
                ]

            halo_draw.polygon(
                blob_points(halo_scale),
                fill=(*color, halo_alpha),
            )
            core_draw.polygon(blob_points(1.0), fill=(*color, core_alpha))
            particles.append(
                {
                    "center": [round(cx, 2), round(cy, 2)],
                    "radius_px": round(radius, 2),
                    "diameter_long_side_ratio": round(
                        2.0 * radius / long_side, 8
                    ),
                    "core_alpha": core_alpha_ratio,
                    "halo_scale": halo_scale,
                    "halo_alpha": halo_alpha_ratio,
                    "ellipse_ratio": ellipse_ratio,
                }
            )
        blur_radius = long_side * rng.uniform(0.009, 0.014)
        blur_radius_final = blur_radius * 512.0 / long_side
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
        records.append(
            _record(
                1,
                "lens_dust_shadow",
                severity,
                shadow_count=count,
                object_overlap_shadow_count=object_overlap_count,
                shadows=particles,
                frame_affected_ratio=float(dust_mask.mean()),
                blur_radius_final_space=blur_radius_final,
                blur_radius_source_space=blur_radius,
                coordinate_space="camera_frame",
            )
        )
    elif case == "rgb_hair_contamination":
        count = 2
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
            desired = rng.uniform(0.45, 0.60) * long_side
            length = min(desired, 0.60 * long_side)
            if rng.random() < 0.80:
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
            width = max(1, round(long_side * rng.uniform(0.0045, 0.006)))
            alpha = min(
                math.floor(255 * 0.23),
                max(
                    math.ceil(255 * 0.18),
                    round(255 * rng.uniform(0.18, 0.23)),
                ),
            )
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
        blur_radius_final = rng.uniform(1.5, 2.5)
        blur_radius = blur_radius_final * long_side / 512.0
        halo_multiplier = rng.uniform(4.0, 5.0)
        halo_layer = overlay.copy()
        halo_layer.putalpha(
            halo_layer.getchannel("A").point(
                lambda value: min(round(value * 0.45), round(255 * 0.10))
            )
        )
        halo_alphas = [
            min(float(curve["alpha"]) * 0.45, 0.10) for curve in curves
        ]
        blurred_halo = halo_layer.filter(
            ImageFilter.GaussianBlur(blur_radius * halo_multiplier)
        )
        blurred_core = overlay.filter(ImageFilter.GaussianBlur(blur_radius))
        affected_alpha = np.maximum(
            np.asarray(blurred_halo.getchannel("A")),
            np.asarray(blurred_core.getchannel("A")),
        )
        frame_affected_ratio = float((affected_alpha >= 2).mean())
        composite = Image.alpha_composite(result.convert("RGBA"), blurred_halo)
        result = Image.alpha_composite(
            composite, blurred_core
        ).convert("RGB")
        records.append(
            _record(
                1,
                "lens_fiber_shadow",
                severity,
                curve_count=count,
                curves=curves,
                blur_radius_final_space=blur_radius_final,
                blur_radius_source_space=blur_radius,
                halo_multiplier=halo_multiplier,
                halo_alphas=halo_alphas,
                frame_affected_ratio=frame_affected_ratio,
                coordinate_space="camera_frame",
            )
        )
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

    def record_parameters(record_type: str) -> dict[str, Any]:
        return next(
            record["parameters"]
            for record in records
            if record["type"] == record_type
        )

    if "radon_projection_drop" in record_types:
        projection = record_parameters("radon_projection_drop")
        subtype = str(projection.get("subtype"))
        if subtype == "sparse_view":
            if not 0.72 <= float(projection.get("retained_ratio", 0)) <= 0.82:
                raise ValueError(
                    "quality_gate: sparse-view retained ratio is outside 0.72..0.82"
                )
        elif subtype == "limited_angle":
            if not 15.0 <= float(
                projection.get("removed_width_deg", 0)
            ) <= 25.0:
                raise ValueError(
                    "quality_gate: limited-angle removal is outside 15..25 degrees"
                )
        else:
            raise ValueError("quality_gate: unknown projection-drop subtype")
        reconstruction = record_parameters("filtered_back_projection")
        if not 0.25 <= float(
            reconstruction.get("reconstruction_weight", 0)
        ) <= 0.40:
            raise ValueError(
                "quality_gate: reconstruction weight is outside 0.25..0.40"
            )

    if "signal_to_transmission" in record_types:
        signal = record_parameters("signal_to_transmission")
        photons = record_parameters("poisson_sampling")
        read_noise = record_parameters("read_noise")
        contrast = record_parameters("low_contrast_attenuation")
        if (
            not 0.78 <= float(signal.get("signal_factor", 0)) <= 0.86
            or not 90.0 <= float(photons.get("photon_scale", 0)) <= 130.0
            or not 0.001
            <= float(read_noise.get("sigma_normalized", 0))
            <= 0.002
            or not 0.95 <= float(contrast.get("contrast_factor", 0)) <= 1.00
        ):
            raise ValueError(
                "quality_gate: CT low-signal parameters are outside v1.9 range"
            )

    if "dense_material_mask" in record_types:
        dense = record_parameters("dense_material_mask")
        cupping = record_parameters("cupping_field")
        attenuation = float(dense.get("attenuation", 0))
        luminance_delta = float(cupping.get("luminance_delta", 0))
        feather_ratio = float(cupping.get("feather_ratio", 0))
        if (
            not 0.05 <= attenuation <= 0.10
            or not (
                8.0 <= luminance_delta <= 15.0
                or -4.0 <= luminance_delta <= -2.0
            )
            or not 0.08 <= feather_ratio <= 0.14
        ):
            raise ValueError(
                "quality_gate: beam-hardening field parameters are outside v1.9 range"
            )

    if "timing_edge_crop" in record_types:
        timing = record_parameters("timing_edge_crop")
        target = timing.get("target_outline_retained_ratio")
        if target is not None and not 0.55 <= float(target) <= 0.68:
            raise ValueError(
                "quality_gate: trigger crop target retention is outside 0.55..0.68"
            )
        if "conveyor_motion_blur" in record_types:
            kernel = int(record_parameters("conveyor_motion_blur").get("kernel", 0))
            if kernel not in {11, 13, 15, 17}:
                raise ValueError(
                    "quality_gate: conveyor blur kernel is outside v1.9 range"
                )

    if "lighting_gradient" in record_types:
        lighting = record_parameters("lighting_gradient")
        gains = (
            float(lighting.get("dark_gain", 0)),
            float(lighting.get("bright_gain", 0)),
        )
        if gains not in {(0.25, 1.65), (0.18, 1.85), (0.12, 2.05)}:
            raise ValueError(
                "quality_gate: uneven-lighting gains are outside v1.9 range"
            )

    if "surface_aware_specular_reflection" in record_types:
        glare = record_parameters("surface_aware_specular_reflection")
        patches = glare.get("patches", [])
        bloom = record_parameters("highlight_bloom")
        if (
            len(patches) != 2
            or any(
                not 0.70 <= float(patch.get("alpha", 0)) <= 0.78
                for patch in patches
            )
            or not 10.0 <= float(bloom.get("radius_final_space", 0)) <= 14.0
        ):
            raise ValueError(
                "quality_gate: glare parameters are outside v1.9 range"
            )

    if "defocus_blur" in record_types:
        focus = record_parameters("defocus_blur")
        if not 7.5 <= float(focus.get("radius_final_space", 0)) <= 10.0:
            raise ValueError(
                "quality_gate: defocus radius is outside 7.5..10px"
            )

    if "linear_exposure_reduction" in record_types:
        exposure = record_parameters("linear_exposure_reduction")
        shot = record_parameters("signal_dependent_shot_noise")
        sensor = record_parameters("sensor_read_noise")
        if (
            not 0.18 <= float(exposure.get("exposure_factor", 0)) <= 0.28
            or not 0.44
            <= float(exposure.get("target_outline_mean_ratio", 0))
            <= 0.50
            or not 80.0 <= float(shot.get("photon_capacity", 0)) <= 120.0
            or not 0.010 <= float(sensor.get("sigma", 0)) <= 0.015
            or not 0.008 <= float(sensor.get("black_level", 0)) <= 0.012
        ):
            raise ValueError(
                "quality_gate: underexposure parameters are outside v1.9 range"
            )

    if "overexposure" in record_types:
        exposure = record_parameters("overexposure")
        if (
            not 2.10 <= float(exposure.get("exposure_factor", 0)) <= 2.60
            or not 0.50
            <= float(exposure.get("target_object_saturation_ratio", 0))
            <= 0.60
        ):
            raise ValueError(
                "quality_gate: overexposure parameters are outside v1.9 range"
            )
        if "highlight_bloom" in record_types:
            bloom = record_parameters("highlight_bloom")
            if (
                not 10.0
                <= float(bloom.get("radius_final_512_px", 0))
                <= 18.0
                or not 0.25 <= float(bloom.get("blend", 0)) <= 0.30
            ):
                raise ValueError(
                    "quality_gate: overexposure bloom is outside v1.9 range"
                )

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

    if "signal_to_transmission" in record_types:
        if output_mean > baseline_mean * 0.96:
            raise ValueError("quality_gate: CT low-signal effect is not strong enough")
    if "linear_exposure_reduction" in record_types:
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
        if not 0.40 <= mean_ratio <= 0.52:
            raise ValueError(
                "quality_gate: underexposure outline luminance ratio is outside 0.40..0.52"
            )
        frame_ratio = output_mean / baseline_mean
        if frame_ratio > 0.62:
            raise ValueError(
                "quality_gate: underexposure frame luminance ratio exceeds 0.62"
            )
        # Lighting order is a low-frequency property. Measuring it per pixel (or at 32 px)
        # incorrectly treats the deliberately added shot/read noise as a lighting reversal.
        low_size = (min(8, image.width), min(8, image.height))
        baseline_low = np.asarray(
            Image.fromarray(np.clip(baseline, 0, 255).astype(np.uint8)).resize(
                low_size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        output_low = np.asarray(
            image.convert("L").resize(low_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        # Use the whole low-frequency frame. Restricting this to a nearly uniform battery
        # interior leaves only shot noise and makes the correlation unstable.
        region_low = np.ones(low_size[::-1], dtype=bool)
        if float(baseline_low[region_low].std()) >= 1.0:
            correlation = float(
                np.corrcoef(
                    baseline_low[region_low].ravel(),
                    output_low[region_low].ravel(),
                )[0, 1]
            )
            if not np.isfinite(correlation) or correlation < 0.95:
                raise ValueError(
                    "quality_gate: underexposure changed the spatial lighting order"
                )
    if "overexposure" in record_types:
        if output_mean <= baseline_mean:
            raise ValueError("quality_gate: overexposure did not increase luminance")
        if output_object_mask is not None:
            region = np.asarray(output_object_mask.convert("L")) > 0
            saturation = float((luminance[region] >= 250).mean())
            if not 0.45 <= saturation <= 0.70:
                raise ValueError(
                    "quality_gate: overexposure object saturation is outside 45%..70%"
                )

    def edge_energy(array: np.ndarray) -> float:
        horizontal = np.abs(np.diff(array, axis=1)).mean()
        vertical = np.abs(np.diff(array, axis=0)).mean()
        return float(horizontal + vertical)

    if "defocus_blur" in record_types:
        # Defocus is specified in final 512-pixel space. Measuring raw 1920-pixel
        # gradients makes sensor/JPEG texture dominate and rejects a blur that is
        # correctly visible after final resize. Evaluate both images in that same space.
        scale = min(1.0, 512.0 / max(image.size))
        metric_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        baseline_metric = np.asarray(
            Image.fromarray(np.clip(baseline, 0, 255).astype(np.uint8)).resize(
                metric_size, Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        output_metric = np.asarray(
            image.convert("L").resize(metric_size, Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        baseline_edge = max(_rms_gradient_energy(baseline_metric), 1e-6)
        ratio = _rms_gradient_energy(output_metric) / baseline_edge
        maximum = 0.50
        minimum = 0.15
        if ratio > maximum or ratio < minimum:
            raise ValueError("quality_gate: required edge-energy reduction was not reached")

    if "double_edge_ghosting" in record_types:
        ghost = next(
            record["parameters"]
            for record in records
            if record["type"] == "double_edge_ghosting"
        )
        motion = next(
            record["parameters"]
            for record in records
            if record["type"] == "directional_motion_blur"
        )
        if not 6.0 <= float(ghost["offset_final_512_px"]) <= 9.0:
            raise ValueError("quality_gate: CT ghost offset is outside 6..9px")
        if not 0.15 <= float(ghost["shifted_weight"]) <= 0.22:
            raise ValueError("quality_gate: CT ghost blend is outside 0.15..0.22")
        if not 0.5 <= float(
            motion["displacement_range_final_512_px"]
        ) <= 1.2:
            raise ValueError("quality_gate: CT motion blur range is outside 0.5..1.2px")

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
        asymmetry = _axis_asymmetry(luminance, projection, region)
        baseline_asymmetry = _axis_asymmetry(baseline, projection, region)
        if not UNEVEN_MIN_ASYMMETRY <= asymmetry <= UNEVEN_MAX_ASYMMETRY:
            raise ValueError("quality_gate: uneven lighting contrast is too small")
        if asymmetry - baseline_asymmetry < UNEVEN_MIN_ADDED_ASYMMETRY:
            raise ValueError("quality_gate: uneven lighting did not add enough asymmetry")

    if "lens_dust_shadow" in record_types:
        parameters = next(
            record["parameters"] for record in records if record["type"] == "lens_dust_shadow"
        )
        count = int(parameters.get("shadow_count", 0))
        if not 3 <= count <= 4:
            raise ValueError("quality_gate: lens dust shadow count is outside 3..4")
        object_overlap_count = int(
            parameters.get("object_overlap_shadow_count", 0)
        )
        if not 1 <= object_overlap_count <= 2:
            raise ValueError(
                "quality_gate: lens dust object overlap count is outside 1..2"
            )
        coverage = float(parameters.get("frame_affected_ratio", 0))
        if not 0.001 <= coverage <= 0.35:
            raise ValueError("quality_gate: lens dust frame coverage is outside range")
        shadows = parameters.get("shadows", [])
        if len(shadows) != count or any(
            not 0.04
            <= float(shadow.get("diameter_long_side_ratio", 0))
            <= 0.06
            or not 0.22 <= float(shadow.get("core_alpha", 0)) <= 0.25
            or not 2.6 <= float(shadow.get("halo_scale", 0)) <= 3.3
            or not 0.10 <= float(shadow.get("halo_alpha", 0)) <= 0.125
            for shadow in shadows
        ):
            raise ValueError("quality_gate: lens dust parameters are outside v1.9 range")
    if "surface_aware_specular_reflection" in record_types:
        parameters = records[0]["parameters"]
        if float(parameters.get("outline_overlap_ratio", 0)) < 0.90:
            raise ValueError("quality_gate: glare outline overlap is below 90%")
        core_ratio = float(parameters.get("core_object_area_ratio", 0))
        if not 0.045 <= core_ratio <= 0.12:
            raise ValueError(
                "quality_gate: glare core area is outside 4.5%..12% of object"
            )
        if float(parameters.get("defect_coverage_ratio", 0)) > 0.70:
            raise ValueError("quality_gate: glare covers more than 70% of defect mask")
        minimum_defect_coverage = float(
            parameters.get("minimum_defect_coverage_ratio", 0.45)
        )
        if bool(parameters.get("defect_present")) and float(
            parameters.get("defect_coverage_ratio", 0)
        ) < minimum_defect_coverage:
            raise ValueError(
                "quality_gate: glare covers less than the feasible minimum of "
                f"{minimum_defect_coverage:.3f} of the defect mask"
            )
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
        if len(quadrants) != 4 or any(int(count) < 1 for count in quadrants):
            raise ValueError(
                "quality_gate: metal streak rays do not cover all four quadrants"
            )
        if float(streak_record["parameters"].get("max_angular_gap_deg", 360)) > 55:
            raise ValueError(
                "quality_gate: metal streak maximum angular gap exceeds 55 degrees"
            )
        widths = streak_record["parameters"].get(
            "ray_widths_final_512_px", []
        )
        alphas = streak_record["parameters"].get("ray_start_alphas", [])
        if (
            not 8
            <= int(streak_record["parameters"].get("streak_count", 0))
            <= 12
            or len(widths) != len(alphas)
            or any(not 0.4 <= float(width) <= 0.8 for width in widths)
            or any(not 0.02 <= float(alpha) <= 0.05 for alpha in alphas)
            or not 0.15
            <= float(
                streak_record["parameters"].get("decay_distance_ratio", 0)
            )
            <= 0.22
        ):
            raise ValueError(
                "quality_gate: metal streak parameters are outside v1.9 range"
            )
    if "lens_fiber_shadow" in record_types:
        curves = records[0]["parameters"].get("curves", [])
        if len(curves) != 2:
            raise ValueError("quality_gate: hair curve count is not 2")
        minimum_width = max(1, math.floor(max(image.size) * 0.0045))
        maximum_width = max(1, math.ceil(max(image.size) * 0.006))
        if any(
            not minimum_width <= int(curve["thickness_px"]) <= maximum_width
            for curve in curves
        ):
            raise ValueError("quality_gate: hair thickness is outside v1.9 range")
        long_side = max(image.size)
        if any(
            not 0.44 * long_side <= float(curve["length_px"]) <= 0.62 * long_side
            for curve in curves
        ):
            raise ValueError("quality_gate: hair length is outside v1.9 range")
        if any(not 0.18 <= float(curve["alpha"]) <= 0.23 for curve in curves):
            raise ValueError("quality_gate: hair alpha is outside 0.18..0.23")
        fiber = next(
            record["parameters"]
            for record in records
            if record["type"] == "lens_fiber_shadow"
        )
        if not 4.0 <= float(fiber.get("halo_multiplier", 0)) <= 5.0:
            raise ValueError("quality_gate: hair halo multiplier is outside 4..5")
        halo_alphas = fiber.get("halo_alphas", [])
        if len(halo_alphas) != 2 or any(
            not 0.08 <= float(alpha) <= 0.10 for alpha in halo_alphas
        ):
            raise ValueError("quality_gate: hair halo alpha is outside 0.08..0.10")
        if float(fiber.get("frame_affected_ratio", 1.0)) > 0.35:
            raise ValueError("quality_gate: hair affects more than 35% of frame")
    if "alignment_edge_crop" in record_types:
        alignment_record = next(
            record for record in records if record["type"] == "alignment_edge_crop"
        )
        retained = alignment_record["parameters"].get("retained_outline_ratio")
        if retained is not None and not 0.90 <= float(retained) <= 0.98:
            raise ValueError(
                f"quality_gate: alignment outline retention {float(retained):.3f} "
                "is outside 0.90..0.98"
            )
    if (
        original_object_mask is not None
        and output_object_mask is not None
        and "timing_edge_crop" in record_types
    ):
        original_area = max(
            int((np.asarray(original_object_mask.convert("L")) > 0).sum()), 1
        )
        output_area = int((np.asarray(output_object_mask.convert("L")) > 0).sum())
        retained = output_area / original_area
        minimum = 0.52
        if retained < minimum:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is below {minimum:.2f}"
            )
        # The lower bound alone let retention 1.000 through, meaning the crop missed the
        # battery entirely and the output looked like a normal photo. A trigger timing failure
        # has to leave the cell clipped by the frame.
        if retained > 0.72:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is above 0.72"
            )
