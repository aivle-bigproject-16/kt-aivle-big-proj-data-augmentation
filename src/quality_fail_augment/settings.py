"""`.env` 기반 기본 경로 설정.

원본 데이터 위치와 출력 폴더는 실행하는 사람마다 다르다. 매번 CLI 인자로 전체
경로를 적는 대신 저장소 루트의 `.env` 에 한 번 적어두고 그걸 기본값으로 쓴다.
CLI 인자를 직접 준 경우에는 그쪽이 항상 이긴다.

의존성을 늘리지 않으려고 python-dotenv 대신 최소 파서를 직접 둔다. 값에 대한
이스케이프 처리는 하지 않는다. Windows 경로(`C:\\Users\\new`)의 백슬래시가
개행으로 바뀌는 사고를 막기 위해서다.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_VAR = "QFA_ENV_FILE"
DEFAULT_ENV_FILENAME = ".env"

RAW_ROOT_VAR = "QFA_RAW_ROOT"
CONFIG_VAR = "QFA_CONFIG"
PLAN_DIR_VAR = "QFA_PLAN_DIR"
PLAN_CSV_VAR = "QFA_PLAN_CSV"
OUTPUT_DIR_VAR = "QFA_OUTPUT_DIR"
SMOKE_DIR_VAR = "QFA_SMOKE_DIR"
AUDIT_DIR_VAR = "QFA_AUDIT_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def find_env_file(start: Path | None = None) -> Path | None:
    """사용할 `.env` 파일을 찾는다.

    우선순위는 `QFA_ENV_FILE` 환경변수, 현재 폴더에서 위로 올라가며 만나는 첫
    `.env`, 마지막으로 저장소 루트의 `.env` 순이다.
    """
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    origin = (start or Path.cwd()).resolve()
    for directory in (origin, *origin.parents):
        candidate = directory / DEFAULT_ENV_FILENAME
        if candidate.is_file():
            return candidate
    fallback = _REPO_ROOT / DEFAULT_ENV_FILENAME
    return fallback if fallback.is_file() else None


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"{path}:{number}: '=' 가 없는 줄입니다: {raw_line.rstrip()}")
            key = key.strip()
            if not key:
                raise ValueError(f"{path}:{number}: 이름이 빈 항목입니다")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


_loaded_env_dir: Path | None = None


def load_dotenv(start: Path | None = None) -> Path | None:
    """`.env` 를 읽어 `os.environ` 에 채운다.

    이미 셸에 설정된 환경변수는 덮어쓰지 않는다. 일회성으로 다른 경로를 쓰고
    싶을 때 셸에서 변수 하나만 바꿔 실행할 수 있게 하기 위해서다.
    """
    global _loaded_env_dir
    path = find_env_file(start)
    if path is None:
        return None
    for key, value in parse_env_file(path).items():
        os.environ.setdefault(key, value)
    _loaded_env_dir = path.resolve().parent
    return path


def env_path(name: str) -> Path | None:
    """환경변수를 경로로 읽는다.

    `./config.40k.json` 같은 상대경로는 현재 폴더가 아니라 `.env` 가 있는 폴더를
    기준으로 푼다. 어느 폴더에서 실행하든 같은 파일을 가리키게 하기 위해서다.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and _loaded_env_dir is not None:
        return _loaded_env_dir / path
    return path
