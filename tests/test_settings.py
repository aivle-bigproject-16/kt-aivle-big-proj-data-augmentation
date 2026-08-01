from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from quality_fail_augment.settings import (
    ENV_FILE_VAR,
    env_path,
    find_env_file,
    load_dotenv,
    parse_env_file,
)

_MANAGED_VARS = (
    ENV_FILE_VAR,
    "QFA_RAW_ROOT",
    "QFA_CONFIG",
    "QFA_PLAN_DIR",
    "QFA_PLAN_CSV",
    "QFA_OUTPUT_DIR",
)


@contextmanager
def clean_environment():
    saved = {name: os.environ.get(name) for name in _MANAGED_VARS}
    for name in _MANAGED_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class EnvFileTests(unittest.TestCase):
    def test_windows_path_backslashes_survive_unescaped(self):
        """`C:\\Users\\new` 의 `\\n` 이 개행으로 해석되면 안 된다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("QFA_RAW_ROOT=C:\\Users\\new\\tab\\raw\n", encoding="utf-8")
            self.assertEqual(parse_env_file(path)["QFA_RAW_ROOT"], "C:\\Users\\new\\tab\\raw")

    def test_korean_path_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            value = "C:\\Users\\rudtn\\Downloads\\103.배터리 불량 이미지 데이터"
            path.write_text(f"QFA_RAW_ROOT={value}\n", encoding="utf-8")
            self.assertEqual(parse_env_file(path)["QFA_RAW_ROOT"], value)

    def test_comments_blank_lines_quotes_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# 주석\n"
                "\n"
                'QFA_CONFIG="./config.40k.json"\n'
                "export QFA_PLAN_DIR='D:\\qf_plan'\n",
                encoding="utf-8",
            )
            values = parse_env_file(path)
            self.assertEqual(values["QFA_CONFIG"], "./config.40k.json")
            self.assertEqual(values["QFA_PLAN_DIR"], "D:\\qf_plan")

    def test_line_without_equals_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("QFA_RAW_ROOT\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_env_file(path)

    def test_utf8_bom_is_stripped_from_first_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("QFA_RAW_ROOT=D:\\raw\n", encoding="utf-8-sig")
            self.assertEqual(parse_env_file(path)["QFA_RAW_ROOT"], "D:\\raw")


class LoadDotenvTests(unittest.TestCase):
    def test_existing_environment_wins_over_env_file(self):
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory)
            (root / ".env").write_text("QFA_RAW_ROOT=D:\\from_file\n", encoding="utf-8")
            os.environ["QFA_RAW_ROOT"] = "D:\\from_shell"
            load_dotenv(root)
            self.assertEqual(os.environ["QFA_RAW_ROOT"], "D:\\from_shell")

    def test_env_file_fills_unset_variable(self):
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory)
            (root / ".env").write_text("QFA_RAW_ROOT=D:\\from_file\n", encoding="utf-8")
            load_dotenv(root)
            self.assertEqual(env_path("QFA_RAW_ROOT"), Path("D:\\from_file"))

    def test_search_walks_up_to_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory)
            (root / ".env").write_text("QFA_RAW_ROOT=D:\\parent\n", encoding="utf-8")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(find_env_file(nested), root / ".env")

    def test_env_file_var_overrides_search(self):
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory)
            (root / ".env").write_text("QFA_RAW_ROOT=D:\\ignored\n", encoding="utf-8")
            chosen = root / "other.env"
            chosen.write_text("QFA_RAW_ROOT=D:\\chosen\n", encoding="utf-8")
            os.environ[ENV_FILE_VAR] = str(chosen)
            load_dotenv(root)
            self.assertEqual(env_path("QFA_RAW_ROOT"), Path("D:\\chosen"))

    def test_empty_value_is_treated_as_unset(self):
        with clean_environment():
            os.environ["QFA_RAW_ROOT"] = "   "
            self.assertIsNone(env_path("QFA_RAW_ROOT"))

    def test_relative_value_resolves_against_the_env_file_directory(self):
        """`./config.40k.json` 은 실행한 폴더가 아니라 .env 폴더 기준이어야 한다."""
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory).resolve()
            (root / ".env").write_text("QFA_CONFIG=./config.40k.json\n", encoding="utf-8")
            load_dotenv(root)
            self.assertEqual(env_path("QFA_CONFIG"), root / "config.40k.json")

    def test_absolute_value_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory, clean_environment():
            root = Path(directory).resolve()
            (root / ".env").write_text("QFA_RAW_ROOT=D:\\raw\n", encoding="utf-8")
            load_dotenv(root)
            self.assertEqual(env_path("QFA_RAW_ROOT"), Path("D:\\raw"))


if __name__ == "__main__":
    unittest.main()
