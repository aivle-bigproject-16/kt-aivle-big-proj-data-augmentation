from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box


@dataclass(frozen=True)
class Affine:
    """Continuous-coordinate affine matrix.

    x' = a*x + b*y + xoff
    y' = d*x + e*y + yoff
    """

    a: float = 1.0
    b: float = 0.0
    d: float = 0.0
    e: float = 1.0
    xoff: float = 0.0
    yoff: float = 0.0

    def then(self, other: "Affine") -> "Affine":
        return Affine(
            other.a * self.a + other.b * self.d,
            other.a * self.b + other.b * self.e,
            other.d * self.a + other.e * self.d,
            other.d * self.b + other.e * self.e,
            other.a * self.xoff + other.b * self.yoff + other.xoff,
            other.d * self.xoff + other.e * self.yoff + other.yoff,
        )

    def shapely(self) -> list[float]:
        return [self.a, self.b, self.d, self.e, self.xoff, self.yoff]

    def matrix(self) -> list[list[float]]:
        return [
            [self.a, self.b, self.xoff],
            [self.d, self.e, self.yoff],
            [0.0, 0.0, 1.0],
        ]

    @classmethod
    def rotation(cls, degrees: float, center: tuple[float, float]) -> "Affine":
        radians = math.radians(degrees)
        cosine, sine = math.cos(radians), math.sin(radians)
        cx, cy = center
        return cls(
            a=cosine,
            b=-sine,
            d=sine,
            e=cosine,
            xoff=cx - cosine * cx + sine * cy,
            yoff=cy - sine * cx - cosine * cy,
        )


def parse_roi(value: Any, width: int, height: int) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            left, top, right, bottom = 0.0, 0.0, float(value[0]), float(value[1])
        elif len(value) == 4:
            left, top, right, bottom = map(float, value)
        else:
            raise ValueError(f"Unsupported ROI list: {value!r}")
    elif isinstance(value, dict):
        if {"x", "y", "width", "height"} <= value.keys():
            left, top = float(value["x"]), float(value["y"])
            right, bottom = left + float(value["width"]), top + float(value["height"])
        elif {"width", "height"} <= value.keys():
            left, top, right, bottom = 0.0, 0.0, float(value["width"]), float(value["height"])
        else:
            raise ValueError(f"Unsupported ROI object: {value!r}")
    else:
        raise ValueError("CT ROI is missing in data_info.roi and image_info.roi")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"ROI out of bounds: {(left, top, right, bottom)} for {(width, height)}")
    return left, top, right, bottom


def extract_ct_roi(label: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    data = label.get("data_info")
    info = label.get("image_info")
    primary = data.get("roi") if isinstance(data, dict) else None
    legacy = info.get("roi") if isinstance(info, dict) else None
    if primary is not None and legacy is not None:
        parsed_primary = parse_roi(primary, width, height)
        parsed_legacy = parse_roi(legacy, width, height)
        if parsed_primary != parsed_legacy:
            raise ValueError(
                f"Conflicting CT ROI: data_info.roi={primary!r}, image_info.roi={legacy!r}"
            )
        return parsed_primary
    return parse_roi(primary if primary is not None else legacy, width, height)


def point_rings(value: Any) -> list[list[tuple[float, float]]]:
    if value is None or value == []:
        return []
    if not isinstance(value, list):
        raise ValueError(f"unsupported_points_schema: expected list, got {type(value).__name__}")
    if isinstance(value[0], (int, float)):
        if len(value) < 6 or len(value) % 2:
            raise ValueError("unsupported_points_schema: flat coordinates need >= 3 x/y pairs")
        return [[(float(value[i]), float(value[i + 1])) for i in range(0, len(value), 2)]]
    if isinstance(value[0], dict):
        if not all(isinstance(p, dict) and {"x", "y"} <= p.keys() for p in value):
            raise ValueError("unsupported_points_schema: malformed x/y objects")
        return [[(float(p["x"]), float(p["y"])) for p in value]]
    if (
        isinstance(value[0], (list, tuple))
        and len(value[0]) >= 2
        and isinstance(value[0][0], (int, float))
    ):
        if not all(
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
            for point in value
        ):
            raise ValueError("unsupported_points_schema: malformed point pairs")
        return [[(float(p[0]), float(p[1])) for p in value]]
    rings: list[list[tuple[float, float]]] = []
    for component in value:
        rings.extend(point_rings(component))
    if not rings:
        raise ValueError("unsupported_points_schema: non-empty value produced no polygon")
    return rings


def polygon_components(geometry: Any) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        result: list[Polygon] = []
        for child in geometry.geoms:
            result.extend(polygon_components(child))
        return result
    return []


def repaired_polygons(points: Any, frame: tuple[float, float, float, float]) -> list[Polygon]:
    result: list[Polygon] = []
    clip = box(*frame)
    for ring in point_rings(points):
        polygon: Any = Polygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        result.extend(
            component
            for component in polygon_components(polygon.intersection(clip))
            if component.area >= 1e-9
        )
    return result


def porosity_bbox_metric(
    label: dict[str, Any], roi: tuple[float, float, float, float]
) -> tuple[float, int]:
    left, top, right, bottom = roi
    roi_width, roi_height = right - left, bottom - top
    ratios: list[float] = []
    count = 0
    for defect in label.get("defects") or []:
        if str(defect.get("name", "")).strip().lower() != "porosity":
            continue
        for polygon in repaired_polygons(defect.get("points"), roi):
            min_x, min_y, max_x, max_y = polygon.bounds
            ratios.append(((max_x - min_x) / roi_width) * ((max_y - min_y) / roi_height))
            count += 1
    return max(ratios, default=0.0), count


def _points_from_polygon(polygon: Polygon, template: Any) -> list[Any]:
    coordinates = [
        (round(float(x), 8), round(float(y), 8))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]
    if isinstance(template, list) and template and isinstance(template[0], (int, float)):
        return [coordinate for point in coordinates for coordinate in point]
    if isinstance(template, list) and template and isinstance(template[0], dict):
        return [{"x": x, "y": y} for x, y in coordinates]
    return [[x, y] for x, y in coordinates]


def _transform_polygons(
    points: Any, transform: Affine, frame: tuple[float, float, float, float]
) -> list[Polygon]:
    transformed: list[Polygon] = []
    for polygon in repaired_polygons(points, (-1e12, -1e12, 1e12, 1e12)):
        geometry = affine_transform(polygon, transform.shapely()).intersection(box(*frame))
        transformed.extend(p for p in polygon_components(geometry) if p.area >= 1.0)
    return sorted(transformed, key=lambda p: p.area, reverse=True)


def transform_label(
    source: dict[str, Any],
    transform: Affine,
    output_size: tuple[int, int],
    modality: str,
    quality_class: str,
    battery_id: int,
    image_id: int,
    file_name: str,
) -> dict[str, Any]:
    existing = source.get("quality_class")
    if existing is not None and (existing not in {"pass", "fail"} or existing != quality_class):
        raise ValueError(
            f"quality_class conflict: source={existing!r}, assigned={quality_class!r}"
        )
    label = deepcopy(source)
    label["quality_class"] = quality_class
    label.setdefault("data_info", {})["battery_ids"] = battery_id
    info = label.setdefault("image_info", {})
    info.update(
        {
            "id": image_id,
            "file_name": file_name,
            "width": output_size[0],
            "height": output_size[1],
        }
    )
    if modality == "CT":
        data_info = label.setdefault("data_info", {})
        data_info["roi"] = [0, 0, output_size[0], output_size[1]]
        info.pop("roi", None)
    frame = (0.0, 0.0, float(output_size[0]), float(output_size[1]))

    swelling = label.get("swelling")
    if isinstance(swelling, dict) and swelling.get("battery_outline") not in (None, []):
        template = swelling["battery_outline"]
        polygons = _transform_polygons(template, transform, frame)
        swelling["battery_outline"] = (
            _points_from_polygon(polygons[0], template) if polygons else []
        )

    defects: list[dict[str, Any]] = []
    for defect in label.get("defects") or []:
        template = defect.get("points")
        for polygon in _transform_polygons(template, transform, frame):
            copied = deepcopy(defect)
            copied["points"] = _points_from_polygon(polygon, template)
            defects.append(copied)
    label["defects"] = defects
    return label
