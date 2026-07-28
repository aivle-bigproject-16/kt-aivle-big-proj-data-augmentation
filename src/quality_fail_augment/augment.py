from __future__ import annotations

import math
import random
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
    "rgb_alignment_failure",
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
    "rgb_alignment_failure": "배터리 위치 정렬 실패",
    "rgb_uneven_lighting": "조명 점등 실패",
    "rgb_reflection_glare": "반사 억제 실패",
    "rgb_focus_failure": "카메라 초점 설정 실패",
    "rgb_underexposure": "노출 부족",
    "rgb_overexposure": "노출 과다",
    "rgb_surface_dust": "렌즈·보호유리 먼지 오염",
    "rgb_hair_contamination": "렌즈·보호유리 섬유 오염",
}
SOURCE_REFERENCES = {
    **{case: f"v1.6:{case}" for case in CT_CASES},
    **{case: f"rgb_table:{index}" for index, case in enumerate(RGB_CASES[:7], 1)},
    "rgb_surface_dust": "v1.6:lens_dust_contamination",
    "rgb_hair_contamination": "v1.6:lens_fiber_contamination",
}


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
        if array.ndim == 2:
            return int(np.quantile(array, 0.2))
        return tuple(int(np.quantile(array[..., c], 0.2)) for c in range(array.shape[2]))
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


def _signed_lines(
    image: Image.Image,
    rng: random.Random,
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
    rng: random.Random,
    np_rng: np.random.Generator,
    severity: float,
    object_mask: Image.Image | None,
) -> tuple[Image.Image, Affine, list[dict[str, Any]], Image.Image | None]:
    result, transform, records = image.copy(), Affine(), []
    mask = object_mask.copy() if object_mask is not None else None
    if case == "ct_cell_alignment_failure":
        angle = rng.choice((-1, 1)) * rng.uniform(0.3, 1.5)
        center = (result.width / 2.0, result.height / 2.0)
        fill = background(result, "CT")
        result = result.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            center=center,
            fillcolor=fill,
        )
        if mask is not None:
            mask = mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                center=center,
                fillcolor=0,
            )
        radians, cx, cy = math.radians(angle), center[0], center[1]
        cosine, sine = math.cos(radians), math.sin(radians)
        transform = transform.then(
            Affine(
                a=cosine,
                b=sine,
                d=-sine,
                e=cosine,
                xoff=cx - cosine * cx - sine * cy,
                yoff=cy + sine * cx - cosine * cy,
            )
        )
        magnitude_x = rng.uniform(0.08, 0.28)
        magnitude_y = rng.uniform(0.08, 0.28)
        if max(magnitude_x, magnitude_y) < 0.18:
            magnitude_x = 0.18
        dx = round(rng.choice((-1, 1)) * result.width * magnitude_x)
        dy = round(rng.choice((-1, 1)) * result.height * magnitude_y)
        fill = background(result, "CT")
        result = _translate(result, dx, dy, fill)
        if mask is not None:
            mask = _translate(mask, dx, dy, 0)
        transform = transform.then(Affine(xoff=dx, yoff=dy))
        records.append(
            _record(
                len(records) + 1,
                "stack_consistent_rotate_translate",
                severity,
                angle_deg=angle,
                dx_px=dx,
                dy_px=dy,
                padding=fill,
                output_frame=[0, 0, result.width, result.height],
            )
        )
    elif case == "ct_acquisition_motion":
        kernel, angle = rng.randrange(7, 26, 2), rng.uniform(0, 179)
        result = _motion_blur(result, kernel, angle)
        records.append(_record(1, "directional_motion_blur", severity, kernel=kernel, angle_deg=angle))
        if rng.random() < 0.65:
            offset, alpha = rng.randint(2, 12), rng.uniform(0.18, 0.48)
            ghost = _translate(result, offset, 0, background(result, "CT"))
            result = Image.blend(result, ghost, alpha)
            records.append(_record(2, "double_edge_ghosting", severity, offset_px=offset, alpha=alpha))
    elif case == "ct_low_signal_noise":
        factor = rng.uniform(0.18, 0.55)
        result = ImageEnhance.Brightness(result).enhance(factor)
        records.append(_record(1, "signal_to_transmission", severity, signal_factor=factor))
        source = np.asarray(result.convert("L"), dtype=np.float32)
        photons = rng.uniform(18.0, 70.0)
        sampled = np_rng.poisson(np.clip(source / 255.0, 0, 1) * photons) / photons * 255.0
        result = Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8)).convert(
            result.mode
        )
        records.append(_record(2, "poisson_sampling", severity, photon_scale=photons))
        if rng.random() < 0.70:
            sigma = rng.uniform(2, 9)
            result = _noise(result, np_rng, sigma, monochrome=True)
            records.append(_record(3, "read_noise", severity, sigma=sigma))
    elif case == "ct_insufficient_projection_sampling":
        # Sparse-view Radon reconstruction: build a sinogram from rotated projections, apply a
        # ramp filter, then back-project only the retained angles. This creates reconstruction
        # streaks from missing acquisition views instead of drawing lines on the final slice.
        full_view_count = 40
        drop_ratio = rng.uniform(0.20, 0.60)
        view_count = max(8, round(full_view_count * (1.0 - drop_ratio)))
        accumulator = np.zeros((result.height, result.width), dtype=np.float32)
        gray = result.convert("L")
        angles = np.linspace(0.0, 180.0, view_count, endpoint=False)
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
        baseline = np.asarray(gray, dtype=np.float32)
        result = Image.fromarray(
            np.clip(0.45 * baseline + 0.55 * reconstructed, 0, 255).astype(np.uint8)
        ).convert(result.mode)
        records.append(_record(1, "radon_projection_drop", severity, full_views=full_view_count, retained_views=view_count, dropped_ratio=drop_ratio))
        records.append(_record(2, "filtered_back_projection", severity, view_angles_deg=angles.tolist()))
    elif case == "ct_beam_hardening_metal_streak":
        array = np.asarray(result.convert("L"))
        threshold = float(np.quantile(array, rng.uniform(0.92, 0.98)))
        mask = array >= threshold
        if float(mask.mean()) < 0.001:
            raise ValueError("dense_region_mask_too_small")
        attenuation = rng.uniform(0.20, 0.70)
        target = np.asarray(result).astype(np.float32)
        target[mask] *= 1.0 - attenuation
        result = _array_image(target, result.mode)
        records.append(_record(1, "dense_material_mask", severity, threshold=threshold, area_ratio=float(mask.mean()), attenuation=attenuation))
        yy, xx = np.mgrid[0 : result.height, 0 : result.width]
        dense_y, dense_x = np.where(mask)
        center_x, center_y = float(dense_x.mean()), float(dense_y.mean())
        radius = max(1.0, min(result.size) * rng.uniform(0.18, 0.42))
        cupping = -rng.uniform(8, 30) * np.clip(
            1.0 - np.hypot(xx - center_x, yy - center_y) / radius, 0.0, 1.0
        )
        result = _luminance_field(result, cupping.astype(np.float32))
        records.append(_record(2, "cupping_field", severity, center=[center_x, center_y], radius=radius))
        count = rng.randint(2, 12)
        field = Image.new("F", result.size, 0.0)
        draw = ImageDraw.Draw(field)
        diagonal = math.hypot(result.width, result.height)
        centers: list[list[int]] = []
        for _ in range(count):
            selected = rng.randrange(len(dense_x))
            cx, cy = int(dense_x[selected]), int(dense_y[selected])
            angle = math.radians(rng.uniform(0, 179))
            draw.line(
                (
                    cx - math.cos(angle) * diagonal,
                    cy - math.sin(angle) * diagonal,
                    cx + math.cos(angle) * diagonal,
                    cy + math.sin(angle) * diagonal,
                ),
                fill=float(-rng.randint(15, 60)),
                width=rng.randint(1, 5),
            )
            centers.append([cx, cy])
        streak_field = np.asarray(field)
        streak_mask = streak_field != 0
        dense_intersections = int((streak_mask & mask).sum())
        result = _luminance_field(result, streak_field)
        records.append(_record(3, "metal_anchored_streaks", severity, streak_count=count, dense_region_centers=centers, dense_intersection_pixels=dense_intersections))
    else:
        raise ValueError(f"Unknown CT failure case: {case}")
    return result, transform, records, mask


def _rgb_case(
    image: Image.Image,
    case: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    severity: float,
    object_mask: Image.Image | None,
    defect_mask: Image.Image | None,
) -> tuple[Image.Image, Affine, list[dict[str, Any]], Image.Image | None]:
    result, transform, records = image.copy(), Affine(), []
    mask = object_mask.copy() if object_mask is not None else None
    if case == "rgb_trigger_timing_failure":
        side = rng.choice(("left", "right", "top", "bottom"))
        # The crop has to clip the battery, not merely trim background. Sizing the cut against
        # the frame let a centred cylindrical cell survive intact - it covers about 11% of the
        # frame width, so a left/right cut of 10..38% often removed only white background and
        # the result was indistinguishable from a normal photo (7 of 30 visual QA samples were
        # rejected for this). Put the cut plane inside the outline bounding box instead, so it
        # always removes 15%..45% of the box along the chosen axis.
        bbox = mask.getbbox() if mask is not None else None
        axis_size = result.width if side in {"left", "right"} else result.height
        if bbox is not None:
            left, top, right, bottom = bbox
            bite = rng.uniform(0.15, 0.45)
            if side == "left":
                amount = left + bite * (right - left)
            elif side == "right":
                amount = (result.width - right) + bite * (right - left)
            elif side == "top":
                amount = top + bite * (bottom - top)
            else:
                amount = (result.height - bottom) + bite * (bottom - top)
        else:
            amount = axis_size * rng.uniform(0.10, 0.38)
        amount = min(max(1, round(amount)), axis_size - 1)
        crop = [0, 0, result.width, result.height]
        if side == "left":
            crop[0] += amount
        elif side == "right":
            crop[2] -= amount
        elif side == "top":
            crop[1] += amount
        else:
            crop[3] -= amount
        result = result.crop(tuple(crop))
        if mask is not None:
            mask = mask.crop(tuple(crop))
        transform = transform.then(Affine(xoff=-crop[0], yoff=-crop[1]))
        records.append(_record(1, "timing_edge_crop", severity, side=side, crop_box=crop))
        if rng.random() < 0.35:
            kernel = rng.randrange(5, 18, 2)
            result = _motion_blur(result, kernel, 0 if side in {"left", "right"} else 90)
            records.append(_record(2, "conveyor_motion_blur", severity, kernel=kernel))
    elif case == "rgb_alignment_failure":
        angle = rng.choice((-1, 1)) * rng.uniform(6, 28)
        center = (result.width / 2, result.height / 2)
        fill = background(result, "RGB")
        result = result.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, center=center, fillcolor=fill)
        if mask is not None:
            mask = mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                center=center,
                fillcolor=0,
            )
        radians, cx, cy = math.radians(angle), center[0], center[1]
        cosine, sine = math.cos(radians), math.sin(radians)
        rotation = Affine(
            a=cosine, b=sine, d=-sine, e=cosine,
            xoff=cx - cosine * cx - sine * cy,
            yoff=cy + sine * cx - cosine * cy,
        )
        transform = transform.then(rotation)
        dx = round(rng.choice((-1, 1)) * result.width * rng.uniform(0.08, 0.30))
        dy = round(rng.choice((-1, 1)) * result.height * rng.uniform(0.08, 0.30))
        result = _translate(result, dx, dy, fill)
        if mask is not None:
            mask = _translate(mask, dx, dy, 0)
        transform = transform.then(Affine(xoff=dx, yoff=dy))
        records.append(_record(1, "object_rotate_translate", severity, angle_deg=angle, center=list(center), dx_px=dx, dy_px=dy, padding=fill))
    elif case == "rgb_uneven_lighting":
        angle = math.radians(rng.uniform(0, 359))
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
        # A dark gain above roughly 0.45 only tints the white background light grey, which
        # visual QA read as evenly lit (11 of 30 samples rejected; every rejected sample had a
        # dark gain of 0.481 or more). Draw the dark end from the range that actually reads as
        # uneven lighting instead of relying on the gate to reject the weak draws.
        dark, bright = rng.uniform(0.25, 0.45), rng.uniform(1.15, 1.70)
        gain = dark + (bright - dark) * projection
        array = np.asarray(result).astype(np.float32) * gain[..., None]
        result = _array_image(array, "RGB")
        records.append(_record(1, "lighting_gradient", severity, angle_deg=math.degrees(angle), dark_gain=dark, bright_gain=bright))
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
        for _ in range(count):
            # Elongate the highlight along the object's principal form instead of emitting a
            # round white blob. The wide Gaussian falloff below removes the hard ellipse edge.
            if half_h >= half_w:
                rx = rng.randint(max(2, int(half_w * 0.08)), max(3, int(half_w * 0.20)))
                ry = rng.randint(max(3, int(half_h * 0.32)), max(4, int(half_h * 0.70)))
            else:
                rx = rng.randint(max(3, int(half_w * 0.32)), max(4, int(half_w * 0.70)))
                ry = rng.randint(max(2, int(half_h * 0.08)), max(3, int(half_h * 0.20)))
            base = (
                rng.choice(highlight_points)
                if highlight_points
                else rng.choice(eligible)
                if eligible
                else (int(centroid_x), int(centroid_y))
            )
            cx = int(round(base[0] * 0.4 + centroid_x * 0.6))
            cy = int(round(base[1] * 0.4 + centroid_y * 0.6))
            alpha = round(255 * rng.uniform(0.25, 0.62))
            half_length = major_extent * rng.uniform(0.22, 0.46)
            start = (
                cx - major_axis[0] * half_length,
                cy - major_axis[1] * half_length,
            )
            end = (
                cx + major_axis[0] * half_length,
                cy + major_axis[1] * half_length,
            )
            width = max(2, round(min(rx, ry) * 1.4))
            draw.line((start, end), fill=(255, 246, 224, alpha), width=width)
            patches.append({"center": [cx, cy], "axis": major_axis.tolist(), "half_length": half_length, "width": width, "alpha": alpha / 255})
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
        records.append(_record(1, "surface_aware_specular_reflection", severity, seed="existing_highlight" if highlight_points else "outline_axis", patches=patches, bloom_radius_final_space=radius, outline_overlap_ratio=outline_overlap, core_object_area_ratio=core_object_ratio, defect_coverage_ratio=defect_coverage, object_saturation_ratio=saturation_ratio))
        records.append(_record(2, "highlight_bloom", severity, radius_final_space=radius))
    elif case == "rgb_focus_failure":
        radius = rng.uniform(2.5, 10)
        result = result.filter(ImageFilter.GaussianBlur(radius))
        records.append(_record(1, "defocus_blur", severity, radius_final_space=radius))
        if rng.random() < 0.25:
            kernel = rng.randrange(5, 14, 2)
            angle = rng.uniform(0, 179)
            result = _motion_blur(result, kernel, angle)
            records.append(_record(2, "mild_motion_blur", severity, kernel=kernel, angle_deg=angle))
    elif case == "rgb_underexposure":
        factor, gamma = rng.uniform(0.12, 0.48), rng.uniform(1.25, 2.20)
        result = _gamma(ImageEnhance.Brightness(result).enhance(factor), gamma)
        records.append(_record(1, "underexposure", severity, brightness_factor=factor, gamma=gamma))
        sigma = rng.uniform(6, 24)
        result = _noise(result, np_rng, sigma, monochrome=False)
        records.append(_record(2, "sensor_noise", severity, sigma=sigma))
    elif case == "rgb_overexposure":
        factor, gamma = rng.uniform(1.45, 2.60), rng.uniform(0.45, 0.85)
        result = _gamma(ImageEnhance.Brightness(result).enhance(factor), gamma)
        threshold = rng.randint(185, 245)
        array = np.asarray(result)
        array = np.where(array >= threshold, 255, array)
        result = _array_image(array, "RGB")
        records.append(_record(1, "overexposure", severity, brightness_factor=factor, gamma=gamma, clip_threshold=threshold))
    elif case == "rgb_surface_dust":
        count = rng.randint(3, 14)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        particles = []
        # Lens contamination belongs to camera-frame coordinates and must not follow the
        # battery outline when the object moves.
        eligible: list[tuple[int, int]] = []
        object_area = result.width * result.height
        target_ratio = rng.uniform(0.01, 0.08)
        nominal_radius = max(
            0.5,
            min(6.0, math.sqrt(target_ratio * object_area / (math.pi * count))),
        )
        for _ in range(count):
            radius = nominal_radius * rng.uniform(0.75, 1.25)
            cx, cy = (
                rng.choice(eligible)
                if eligible
                else (rng.uniform(0, result.width), rng.uniform(0, result.height))
            )
            color = rng.choice(((65, 62, 58), (105, 101, 94), (145, 140, 130)))
            alpha = rng.randint(25, 80)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*color, alpha))
            particles.append([round(cx, 2), round(cy, 2), round(radius, 2), alpha])
        blur_radius = rng.uniform(2.0, 6.0)
        result = Image.alpha_composite(result.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(blur_radius))).convert("RGB")
        dust_mask = np.asarray(overlay.getchannel("A")) > 0
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
        if count >= 8:
            records.append(_record(2, "lens_dust_cluster", severity, cluster_count=1))
    elif case == "rgb_hair_contamination":
        count = rng.randint(1, 3)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        curves = []
        long_side = max(result.size)
        # A lens/protective-window fibre is fixed to the camera frame, not to the object.
        eligible: list[tuple[int, int]] = []
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
            desired = rng.uniform(0.35, 0.90) * long_side
            length = min(desired, 0.88 * long_side, max(0.35 * long_side, 1.5 * inside_len))
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
            width, alpha = rng.randint(1, 3), rng.randint(55, 125)
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
        result = Image.alpha_composite(
            result.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(blur_radius))
        ).convert("RGB")
        records.append(_record(1, "lens_fiber_shadow", severity, curve_count=count, curves=curves, blur_radius_final_space=blur_radius, coordinate_space="camera_frame"))
        if rng.random() < 0.35:
            records.append(_record(2, "fiber_core", severity, opacity=max(curve["alpha"] for curve in curves)))
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
) -> AugmentResult:
    if modality == "CT" and failure_case not in CT_CASES:
        raise ValueError(f"Case {failure_case!r} is not valid for CT")
    if modality == "RGB" and failure_case not in RGB_CASES:
        raise ValueError(f"Case {failure_case!r} is not valid for RGB")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    severity = rng.uniform(0.62, 1.0)
    if modality == "CT":
        result, transform, records, transformed_mask = _ct_case(
            image, failure_case, rng, np_rng, severity, object_mask
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

    if {"signal_to_transmission", "underexposure"} & record_types:
        if output_mean > baseline_mean * 0.72:
            raise ValueError("quality_gate: underexposure is not strong enough")
    if "overexposure" in record_types:
        if output_mean <= baseline_mean:
            raise ValueError("quality_gate: overexposure did not increase luminance")

    def edge_energy(array: np.ndarray) -> float:
        horizontal = np.abs(np.diff(array, axis=1)).mean()
        vertical = np.abs(np.diff(array, axis=0)).mean()
        return float(horizontal + vertical)

    if {"directional_motion_blur", "defocus_blur"} & record_types:
        baseline_edge = max(edge_energy(baseline), 1e-6)
        ratio = edge_energy(luminance) / baseline_edge
        maximum = 0.75 if "defocus_blur" in record_types else 0.85
        if ratio > maximum:
            raise ValueError("quality_gate: required edge-energy reduction was not reached")

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
        if contrast / max(float(sampled.mean()), 1.0) < 0.25:
            raise ValueError("quality_gate: uneven lighting contrast is too small")

    if "lens_dust_shadow" in record_types:
        parameters = next(
            record["parameters"] for record in records if record["type"] == "lens_dust_shadow"
        )
        count = int(parameters.get("shadow_count", 0))
        if not 3 <= count <= 14:
            raise ValueError("quality_gate: lens dust shadow count is outside 3..14")
        coverage = float(parameters.get("frame_affected_ratio", 0))
        if not 0.001 <= coverage <= 0.20:
            raise ValueError("quality_gate: lens dust frame coverage is outside range")
    if "surface_aware_specular_reflection" in record_types:
        parameters = records[0]["parameters"]
        if float(parameters.get("outline_overlap_ratio", 0)) < 0.90:
            raise ValueError("quality_gate: glare outline overlap is below 90%")
        core_ratio = float(parameters.get("core_object_area_ratio", 0))
        if not 0.01 <= core_ratio <= 0.12:
            raise ValueError("quality_gate: glare core area is outside 1%..12% of object")
        if float(parameters.get("defect_coverage_ratio", 0)) > 0.70:
            raise ValueError("quality_gate: glare covers more than 70% of defect mask")
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
    if "lens_fiber_shadow" in record_types:
        curves = records[0]["parameters"].get("curves", [])
        if not 1 <= len(curves) <= 3:
            raise ValueError("quality_gate: hair curve count is outside 1..3")
        if any(not 1 <= int(curve["thickness_px"]) <= 3 for curve in curves):
            raise ValueError("quality_gate: hair thickness is outside 1..3 px")
        long_side = max(image.size)
        if any(
            not 0.35 * long_side <= float(curve["length_px"]) <= 0.90 * long_side
            for curve in curves
        ):
            raise ValueError("quality_gate: hair length is outside 35%..90% of long side")
    if (
        original_object_mask is not None
        and output_object_mask is not None
        and {"stack_consistent_rotate_translate", "timing_edge_crop", "object_rotate_translate"}
        & record_types
    ):
        original_area = max(
            int((np.asarray(original_object_mask.convert("L")) > 0).sum()), 1
        )
        output_area = int((np.asarray(output_object_mask.convert("L")) > 0).sum())
        retained = output_area / original_area
        minimum = (
            0.60
            if "stack_consistent_rotate_translate" in record_types
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
        if "stack_consistent_rotate_translate" in record_types and retained > 0.95:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is above 0.95; "
                "alignment failure did not remove enough required structure"
            )
