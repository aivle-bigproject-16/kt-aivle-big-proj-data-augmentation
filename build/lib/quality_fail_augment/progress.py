from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


class ClosingFileHandler(logging.FileHandler):
    """Open for each record and close immediately, avoiding Windows file locks."""

    def __init__(self, filename: Path, encoding: str = "utf-8") -> None:
        super().__init__(filename, mode="a", encoding=encoding, delay=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        finally:
            self.close()


def configure_logger(name: str, log_path: Path) -> logging.Logger:
    """Create a UTF-8 console + file logger without duplicating handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = ClosingFileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "계산 중"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class ProgressReporter:
    logger: logging.Logger
    label: str
    total: int
    initial: int = 0
    item_name: str = "건"
    log_every_items: int = 100
    log_every_seconds: float = 30.0
    started_at: float = field(default_factory=time.monotonic)
    last_logged_at: float = field(default_factory=time.monotonic)
    last_logged_count: int = 0

    def __post_init__(self) -> None:
        self.last_logged_count = self.initial
        self.logger.info("%s 시작 | 완료 %s/%s %s", self.label, f"{self.initial:,}", f"{self.total:,}", self.item_name)

    def update(self, completed: int, *, detail: str = "", force: bool = False) -> None:
        now = time.monotonic()
        should_log = (
            force
            or completed >= self.total
            or completed - self.last_logged_count >= self.log_every_items
            or now - self.last_logged_at >= self.log_every_seconds
        )
        if not should_log:
            return
        elapsed = max(now - self.started_at, 1e-9)
        newly_completed = max(completed - self.initial, 0)
        rate = newly_completed / elapsed
        remaining = max(self.total - completed, 0)
        eta = remaining / rate if rate > 0 else None
        percent = completed / self.total * 100 if self.total else 100.0
        suffix = f" | {detail}" if detail else ""
        self.logger.info(
            "%s | %s/%s %s (%.2f%%) | %.2f %s/초 | 남은 시간 %s%s",
            self.label,
            f"{completed:,}",
            f"{self.total:,}",
            self.item_name,
            percent,
            rate,
            self.item_name,
            format_duration(eta),
            suffix,
        )
        self.last_logged_at = now
        self.last_logged_count = completed
