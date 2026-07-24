from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalized_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON top level must be an object")
    return value


def open_normalized(path: Path, modality: str) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
        if modality == "RGB":
            return image.convert("RGB")
        if image.mode not in {"L", "RGB"}:
            return image.convert("RGB")
        return image.copy()


def pixel_hash(image: Image.Image) -> str:
    header = f"{image.mode}|{image.width}|{image.height}|".encode("ascii")
    return sha256_bytes(header + image.tobytes())


def raw_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda x: (x["modality"], x["raw_image_path"], x["raw_json_path"])):
        digest.update(
            f'{row["modality"]}|{row["raw_image_path"]}|{row["image_sha256"]}|'
            f'{row["raw_json_path"]}|{row["json_sha256"]}\n'.encode("utf-8")
        )
    return digest.hexdigest()


def ensure_empty_output(path: Path, resume: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, data: bytes, replace_attempts: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows raises PermissionError (WinError 5/32) when the destination is momentarily
    # held open by another process — a reader inspecting the file, an antivirus, or the
    # search indexer. os.replace is otherwise atomic, so retry with backoff instead of
    # failing the whole run on a transient sharing violation.
    last_error: OSError | None = None
    for attempt in range(replace_attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (2**attempt))
    try:
        temporary.unlink()
    except OSError:
        pass
    raise RuntimeError(f"atomic replace failed after {replace_attempts} attempts: {path}") from last_error

