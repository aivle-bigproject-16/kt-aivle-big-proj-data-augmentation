from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CT_PATTERN = re.compile(r"^CT_cell_(?P<form>.+)_(?P<battery>\d+)_(?P<axis>[^_]+)_(?P<image>\d+)$")
RGB_PATTERN = re.compile(r"^RGB_cell_(?P<form>.+)_(?P<battery>\d+)_(?P<image>\d+)$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ParsedName:
    modality: str
    form: str
    battery_id: str
    image_id: str
    axis: str = ""

    @classmethod
    def parse(cls, stem: str) -> "ParsedName | None":
        match = CT_PATTERN.fullmatch(stem)
        if match:
            return cls("CT", match["form"], match["battery"], match["image"], match["axis"])
        match = RGB_PATTERN.fullmatch(stem)
        if match:
            return cls("RGB", match["form"], match["battery"], match["image"])
        return None

    def new_stem(self, battery_id: int, image_id: int) -> str:
        if self.modality == "CT":
            return f"CT_cell_{self.form}_{battery_id}_{self.axis}_{image_id}"
        return f"RGB_cell_{self.form}_{battery_id}_{image_id}"


@dataclass
class Candidate:
    raw_split: str
    image_path: Path
    json_path: Path
    parsed: ParsedName
    label: dict[str, Any]
    width: int
    height: int
    image_sha256: str
    json_sha256: str
    pixel_hash: str
    roi: tuple[float, float, float, float] | None
    porosity_bbox_max_ratio: float = 0.0
    porosity_component_count: int = 0
