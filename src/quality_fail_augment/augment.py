from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .geometry import Affine


CT_CASES = (
    "ct_positioning_failure",
    "ct_fixation_motion",
    "ct_detector_calibration",
    "ct_low_xray_output",
    "ct_high_xray_output",
    "ct_low_resolution",
    "ct_sparse_projection_aliasing",
    "ct_photon_starvation",
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
    "ct_positioning_failure": "배터리 위치 설정 실패",
    "ct_fixation_motion": "배터리 고정 실패",
    "ct_detector_calibration": "장비·검출기 교정 실패",
    "ct_low_xray_output": "X-ray 출력 부족",
    "ct_high_xray_output": "X-ray 출력·노출 과다",
    "ct_low_resolution": "배율·해상도 설정 실패",
    "ct_sparse_projection_aliasing": "회전 투영 촬영 실패",
    "ct_photon_starvation": "투과 영상 획득 실패",
    "rgb_trigger_timing_failure": "배터리 감지·트리거 실패",
    "rgb_alignment_failure": "배터리 위치 정렬 실패",
    "rgb_uneven_lighting": "조명 점등 실패",
    "rgb_reflection_glare": "반사 억제 실패",
    "rgb_focus_failure": "카메라 초점 설정 실패",
    "rgb_underexposure": "노출 부족",
    "rgb_overexposure": "노출 과다",
    "rgb_surface_dust": "표면 먼지 부착",
    "rgb_hair_contamination": "머리카락 오염",
}
SOURCE_REFERENCES = {
    **{case: f"ct_table:{index}" for index, case in enumerate(CT_CASES, 2)},
    **{case: f"rgb_table:{index}" for index, case in enumerate(RGB_CASES[:7], 1)},
    "rgb_surface_dust": "user_added:surface_dust",
    "rgb_hair_contamination": "user_added:hair_contamination",
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
    if case == "ct_positioning_failure":
        flip = rng.random() < 0.20
        if flip:
            result = result.transpose(Image.Transpose.ROTATE_180)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.ROTATE_180)
            transform = transform.then(Affine(a=-1, e=-1, xoff=result.width, yoff=result.height))
            records.append(_record(1, "position_flip_180", severity, enabled=True))
        magnitude_x = rng.uniform(0.12, 0.38)
        magnitude_y = rng.uniform(0.12, 0.38)
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
                "position_translate_crop",
                severity,
                dx_px=dx,
                dy_px=dy,
                padding=fill,
                output_frame=[0, 0, result.width, result.height],
            )
        )
    elif case == "ct_fixation_motion":
        kernel, angle = rng.randrange(7, 26, 2), rng.uniform(0, 179)
        result = _motion_blur(result, kernel, angle)
        records.append(_record(1, "ct_motion_blur", severity, kernel=kernel, angle_deg=angle))
        if rng.random() < 0.65:
            offset, alpha = rng.randint(2, 12), rng.uniform(0.18, 0.48)
            ghost = _translate(result, offset, 0, background(result, "CT"))
            result = Image.blend(result, ghost, alpha)
            records.append(_record(2, "double_edge_ghosting", severity, offset_px=offset, alpha=alpha))
    elif case == "ct_detector_calibration":
        if rng.random() < 0.55:
            overlay = Image.new("L", result.size, 0)
            draw = ImageDraw.Draw(overlay)
            count = rng.randint(3, 18)
            for index in range(count):
                radius = round(min(result.size) * rng.uniform(0.08, 0.48))
                delta = rng.choice((-1, 1)) * rng.randint(10, 45)
                center = (
                    result.width / 2 + rng.uniform(-0.08, 0.08) * result.width,
                    result.height / 2 + rng.uniform(-0.08, 0.08) * result.height,
                )
                draw.ellipse(
                    (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                    outline=128 + delta,
                    width=rng.randint(1, 4),
                )
            field = np.asarray(overlay).astype(np.float32) - 128
            result = _luminance_field(result, np.where(np.asarray(overlay) > 0, field, 0))
            records.append(_record(1, "ring_artifact", severity, ring_count=count))
        else:
            count = rng.randint(2, 20)
            result = _signed_lines(result, rng, count, (1, 8), (13, 56))
            records.append(_record(1, "detector_stripe_artifact", severity, stripe_count=count))
    elif case == "ct_low_xray_output":
        factor = rng.uniform(0.18, 0.55)
        result = ImageEnhance.Brightness(result).enhance(factor)
        records.append(_record(1, "low_exposure", severity, brightness_factor=factor))
        sigma = rng.uniform(8, 28)
        result = _noise(result, np_rng, sigma, monochrome=True)
        records.append(_record(2, "quantum_noise", severity, gaussian_sigma_final_space=sigma))
        if rng.random() < 0.55:
            count = rng.randint(2, 10)
            result = _signed_lines(result, rng, count, (1, 6), (10, 45))
            records.append(_record(3, "global_metal_streak", severity, streak_count=count))
    elif case == "ct_high_xray_output":
        factor, contrast, gamma = rng.uniform(1.25, 1.90), rng.uniform(0.18, 0.58), rng.uniform(0.55, 0.90)
        result = _gamma(ImageEnhance.Brightness(result).enhance(factor), gamma)
        records.append(_record(1, "high_exposure_compression", severity, brightness_factor=factor, gamma=gamma))
        result = ImageEnhance.Contrast(result).enhance(contrast)
        records.append(_record(2, "low_material_contrast", severity, contrast_factor=contrast))
    elif case == "ct_low_resolution":
        scale = rng.uniform(0.12, 0.42)
        small_size = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
        method = rng.choice((Image.Resampling.BOX, Image.Resampling.BILINEAR))
        result = result.resize(small_size, method).resize(image.size, Image.Resampling.BILINEAR)
        records.append(_record(1, "resolution_downsample_restore", severity, scale=scale, downsample=str(method)))
        if rng.random() < 0.50:
            radius = rng.uniform(0.8, 2.5)
            result = result.filter(ImageFilter.GaussianBlur(radius))
            records.append(_record(2, "mild_blur", severity, radius_final_space=radius))
    elif case == "ct_sparse_projection_aliasing":
        count = rng.randint(12, 64)
        result = _signed_lines(result, rng, count, (1, 3), (10, 40), radial=True)
        records.append(_record(1, "radial_streak_aliasing", severity, radial_line_count=count))
        if rng.random() < 0.35:
            angle, alpha = rng.uniform(-2.5, 2.5), rng.uniform(0.08, 0.28)
            ghost = result.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=background(result, "CT"))
            result = Image.blend(result, ghost, alpha)
            records.append(_record(2, "angular_ghosting", severity, angle_deg=angle, alpha=alpha))
    elif case == "ct_photon_starvation":
        array = np.asarray(result.convert("L"))
        threshold = float(np.quantile(array, rng.uniform(0.92, 0.98)))
        mask = array >= threshold
        if float(mask.mean()) < 0.001:
            raise ValueError("dense_region_mask_too_small")
        attenuation = rng.uniform(0.20, 0.70)
        target = np.asarray(result).astype(np.float32)
        target[mask] *= 1.0 - attenuation
        result = _array_image(target, result.mode)
        records.append(_record(1, "dense_region_mask", severity, threshold=threshold, area_ratio=float(mask.mean()), attenuation=attenuation))
        count = rng.randint(2, 12)
        field = Image.new("F", result.size, 0.0)
        draw = ImageDraw.Draw(field)
        dense_y, dense_x = np.where(mask)
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
        records.append(_record(2, "local_dark_streaks", severity, streak_count=count, dense_region_centers=centers, dense_intersection_pixels=dense_intersections))
        if rng.random() < 0.70:
            sigma = rng.uniform(5, 18)
            result = _noise(result, np_rng, sigma, monochrome=True)
            records.append(_record(3, "local_quantum_noise", severity, sigma=sigma))
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
        axis_size = result.width if side in {"left", "right"} else result.height
        amount = max(1, round(axis_size * rng.uniform(0.10, 0.38)))
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
        projection = (projection - projection.min()) / max(float(np.ptp(projection)), 1.0)
        dark, bright = rng.uniform(0.25, 0.75), rng.uniform(1.15, 1.70)
        gain = dark + (bright - dark) * projection
        array = np.asarray(result).astype(np.float32) * gain[..., None]
        result = _array_image(array, "RGB")
        records.append(_record(1, "lighting_gradient", severity, angle_deg=math.degrees(angle), dark_gain=dark, bright_gain=bright))
    elif case == "rgb_reflection_glare":
        count = rng.randint(1, 4)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        patches = []
        eligible = _mask_points(mask)
        for _ in range(count):
            rx, ry = rng.randint(max(2, result.width // 20), max(3, result.width // 4)), rng.randint(max(2, result.height // 20), max(3, result.height // 4))
            cx, cy = (
                rng.choice(eligible)
                if eligible
                else (rng.randint(0, result.width - 1), rng.randint(0, result.height - 1))
            )
            alpha = round(255 * rng.uniform(0.35, 0.92))
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(255, 250, 240, alpha))
            patches.append({"center": [cx, cy], "radius": [rx, ry], "alpha": alpha / 255})
        radius = rng.uniform(3, 24)
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
        defect_coverage = float(
            (glare_mask & defect_array).sum() / max(defect_array.sum(), 1)
        )
        records.append(_record(1, "specular_glare_patch", severity, patches=patches, bloom_radius_final_space=radius, outline_overlap_ratio=outline_overlap, defect_coverage_ratio=defect_coverage))
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
        count = rng.randint(8, 80)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        particles = []
        eligible = _mask_points(mask)
        object_area = len(eligible) if eligible else result.width * result.height
        target_ratio = rng.uniform(0.002, 0.04)
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
            color = rng.choice(((60, 55, 45), (120, 110, 90), (210, 205, 190)))
            alpha = rng.randint(55, 180)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*color, alpha))
            particles.append([round(cx, 2), round(cy, 2), round(radius, 2), alpha])
        result = Image.alpha_composite(result.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(0.35))).convert("RGB")
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
        records.append(_record(1, "dust_particles", severity, particle_count=count, particles=particles, outline_mask_area_ratio=coverage, defect_coverage_ratio=defect_coverage))
    elif case == "rgb_hair_contamination":
        count = rng.randint(1, 3)
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        curves = []
        long_side = max(result.size)
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
            width, alpha = rng.randint(1, 3), rng.randint(115, 230)
            samples = []
            for index in range(41):
                t = index / 40
                x = (1 - t) ** 3 * points[0][0] + 3 * (1 - t) ** 2 * t * points[1][0] + 3 * (1 - t) * t**2 * points[2][0] + t**3 * points[3][0]
                y = (1 - t) ** 3 * points[0][1] + 3 * (1 - t) ** 2 * t * points[1][1] + 3 * (1 - t) * t**2 * points[2][1] + t**3 * points[3][1]
                samples.append((x, y))
            draw.line(samples, fill=(35, 24, 18, alpha), width=width)
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
            curves.append({"control_points": points, "thickness_px": width, "alpha": alpha / 255, "length_px": actual_length, "outline_intersection_ratio": intersection_ratio})
        result = Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")
        records.append(_record(1, "hair_curve", severity, curve_count=count, curves=curves))
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

    if {"low_exposure", "underexposure"} & record_types:
        if output_mean > baseline_mean * 0.72:
            raise ValueError("quality_gate: underexposure is not strong enough")
    if {"high_exposure_compression", "overexposure"} & record_types:
        if output_mean <= baseline_mean:
            raise ValueError("quality_gate: overexposure did not increase luminance")

    def edge_energy(array: np.ndarray) -> float:
        horizontal = np.abs(np.diff(array, axis=1)).mean()
        vertical = np.abs(np.diff(array, axis=0)).mean()
        return float(horizontal + vertical)

    if {"ct_motion_blur", "defocus_blur", "resolution_downsample_restore"} & record_types:
        baseline_edge = max(edge_energy(baseline), 1e-6)
        ratio = edge_energy(luminance) / baseline_edge
        maximum = 0.75 if "defocus_blur" in record_types else 0.85
        if "resolution_downsample_restore" in record_types:
            maximum = 0.80
        if ratio > maximum:
            raise ValueError("quality_gate: required edge-energy reduction was not reached")
        if "resolution_downsample_restore" in record_types and ratio < 0.30:
            raise ValueError("quality_gate: low-resolution loss exceeded 70%")

    if "lighting_gradient" in record_types:
        horizontal = abs(
            float(luminance[:, : image.width // 2].mean())
            - float(luminance[:, image.width // 2 :].mean())
        )
        vertical = abs(
            float(luminance[: image.height // 2].mean())
            - float(luminance[image.height // 2 :].mean())
        )
        if max(horizontal, vertical) / max(output_mean, 1.0) < 0.12:
            raise ValueError("quality_gate: uneven lighting contrast is too small")

    if "dust_particles" in record_types:
        count = int(records[0]["parameters"].get("particle_count", 0))
        if not 8 <= count <= 80:
            raise ValueError("quality_gate: dust particle count is outside 8..80")
        coverage = float(records[0]["parameters"].get("outline_mask_area_ratio", 0))
        if not 0.002 <= coverage <= 0.04:
            raise ValueError("quality_gate: dust outline-mask area is outside 0.2%..4%")
        if float(records[0]["parameters"].get("defect_coverage_ratio", 0)) > 0.35:
            raise ValueError("quality_gate: dust covers more than 35% of defect mask")
    if "specular_glare_patch" in record_types:
        parameters = records[0]["parameters"]
        if float(parameters.get("outline_overlap_ratio", 0)) < 0.70:
            raise ValueError("quality_gate: glare outline overlap is below 70%")
        if float(parameters.get("defect_coverage_ratio", 0)) > 0.70:
            raise ValueError("quality_gate: glare covers more than 70% of defect mask")
    if "local_dark_streaks" in record_types:
        streak_record = next(
            record for record in records if record["type"] == "local_dark_streaks"
        )
        if int(streak_record["parameters"].get("dense_intersection_pixels", 0)) <= 0:
            raise ValueError(
                "quality_gate: photon-starvation streak does not cross dense region"
            )
    if "hair_curve" in record_types:
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
        if any(
            float(curve["outline_intersection_ratio"]) < 0.50 for curve in curves
        ):
            raise ValueError("quality_gate: hair outline intersection is below 50%")
    if (
        original_object_mask is not None
        and output_object_mask is not None
        and {"position_translate_crop", "timing_edge_crop", "object_rotate_translate"}
        & record_types
    ):
        original_area = max(
            int((np.asarray(original_object_mask.convert("L")) > 0).sum()), 1
        )
        output_area = int((np.asarray(output_object_mask.convert("L")) > 0).sum())
        retained = output_area / original_area
        minimum = (
            0.40
            if "position_translate_crop" in record_types
            else 0.50
            if "timing_edge_crop" in record_types
            else 0.60
        )
        if retained < minimum:
            raise ValueError(
                f"quality_gate: outline retention {retained:.3f} is below {minimum:.2f}"
            )
