from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from quality_fail_augment.planner import (
    PERFORMANCE_ONLY_KEYS,
    SCAN_CACHE_FIELDS,
    SCAN_CACHE_VERSION,
    _config_hash,
    _index_raw,
    _load_scan_cache,
    _preflight,
    create_plan,
)

from test_contract import _config, _write_raw


def _cache_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_stem"]: row for row in csv.DictReader(handle)}


class ConfigHashTests(unittest.TestCase):
    def test_performance_keys_do_not_change_the_plan_hash(self) -> None:
        """smoke 가 넣는 측정값이 승인된 plan 을 무효화하면 안 된다."""
        config = _config()
        baseline = _config_hash(config)
        for key in PERFORMANCE_ONLY_KEYS:
            with self.subTest(key=key):
                self.assertEqual(baseline, _config_hash({**config, key: 999_999}))

    def test_measured_config_matches_the_unmeasured_plan(self) -> None:
        config = _config()
        measured = {**config, "worker_peak_rss_bytes": 929_652_736, "jobs": 8}
        self.assertEqual(_config_hash(config), _config_hash(measured))

    def test_output_affecting_keys_still_change_the_hash(self) -> None:
        config = _config()
        baseline = _config_hash(config)
        for key, value in (
            ("seed", 1),
            ("ct_target", 8),
            ("resize_long_side", 128),
            ("jpeg_quality", 70),
            ("ct_porosity_threshold", 0.9),
            ("visual_qa_min_approval_rate", 0.5),
            ("a_new_key_nobody_thought_of", 1),
        ):
            with self.subTest(key=key):
                self.assertNotEqual(baseline, _config_hash({**config, key: value}))


class PreflightTests(unittest.TestCase):
    """_preflight 는 _validate_pair 를 직접 부르는 두 번째 호출부다.

    tests/test_contract.py 의 픽스처는 preflight_per_stratum 을 0 으로 두어 이 경로를
    한 번도 밟지 않는다. 그래서 _validate_pair 의 반환 형태가 바뀌었을 때 전체 테스트가
    통과하는데도 실제 실행이 4분 뒤 죽었다. 여기서 기본값 경로를 실제로 태운다.
    """

    def test_preflight_validates_pairs_in_every_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            shutil.copytree(raw / "Training", raw / "Validation")
            groups, _ = _index_raw(raw)
            _preflight(raw, groups, {**_config(), "preflight_per_stratum": 3})

    def test_preflight_rejects_an_empty_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            groups, _ = _index_raw(raw)
            with self.assertRaisesRegex(ValueError, "preflight stratum has 0 valid pair"):
                _preflight(raw, groups, {**_config(), "preflight_per_stratum": 3})


class ScanCacheTests(unittest.TestCase):
    def test_reuse_scan_reproduces_a_byte_identical_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            config = _config()

            first = create_plan(raw, config, root / "plan1")
            cache = root / "plan1" / "manifests" / "scan_cache.csv"
            self.assertTrue(cache.is_file())

            # 측정 config 로 다시 계획한다. 실제 파이프라인에서 재스캔을 강요하던 조합이다.
            measured = {**config, "worker_peak_rss_bytes": 929_652_736}
            second = create_plan(raw, measured, root / "plan2", reuse_scan=cache)

            self.assertEqual(first["raw_fingerprint"], second["raw_fingerprint"])
            self.assertEqual(first["config_sha256"], second["config_sha256"])
            self.assertEqual(
                (root / "plan1" / "manifests" / "generation_plan.csv").read_bytes(),
                (root / "plan2" / "manifests" / "generation_plan.csv").read_bytes(),
            )

    def test_reuse_scan_accepts_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            config = _config()
            first = create_plan(raw, config, root / "plan1")
            second = create_plan(raw, config, root / "plan2", reuse_scan=root / "plan1")
            self.assertEqual(first["raw_fingerprint"], second["raw_fingerprint"])

    def test_changed_source_is_revalidated_instead_of_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            config = _config()
            create_plan(raw, config, root / "plan1")
            cache = root / "plan1" / "manifests" / "scan_cache.csv"
            before = _cache_rows(cache)

            target = raw / "Training" / "01.원천데이터" / "RGB" / "images"
            victim = sorted(target.glob("*.png"))[0]
            Image.new("RGB", (100, 100), (7, 7, 7)).save(victim)
            stat = victim.stat()
            os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

            create_plan(raw, config, root / "plan2", reuse_scan=cache)
            after = _cache_rows(root / "plan2" / "manifests" / "scan_cache.csv")

            stem = victim.stem
            self.assertNotEqual(before[stem]["image_sha256"], after[stem]["image_sha256"])
            self.assertNotEqual(before[stem]["pixel_hash"], after[stem]["pixel_hash"])
            untouched = [key for key in before if key != stem]
            for key in untouched:
                self.assertEqual(before[key]["image_sha256"], after[key]["image_sha256"])

    def test_stale_cache_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            create_plan(raw, _config(), root / "plan1")
            cache = root / "plan1" / "manifests" / "scan_cache.csv"

            with cache.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["cache_version"] = "0"
            with cache.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCAN_CACHE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "scan cache version"):
                _load_scan_cache(cache)

    def test_missing_cache_is_an_error_not_a_silent_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "scan cache not found"):
                _load_scan_cache(root / "nope.csv")

    def test_cache_records_every_pair_including_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            broken = raw / "Training" / "02.라벨링데이터" / "CT" / "labels"
            victim = sorted(broken.glob("*.json"))[-1]
            victim.write_text(json.dumps({"data_info": {}}), encoding="utf-8")

            create_plan(raw, _config(), root / "plan1")
            rows = _cache_rows(root / "plan1" / "manifests" / "scan_cache.csv")

            self.assertEqual(rows[victim.stem]["status"], "error")
            self.assertTrue(rows[victim.stem]["exclusion_reason"])
            self.assertTrue(
                all(row["cache_version"] == SCAN_CACHE_VERSION for row in rows.values())
            )


if __name__ == "__main__":
    unittest.main()
