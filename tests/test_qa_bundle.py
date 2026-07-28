from __future__ import annotations

import csv
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_BUILDER = Path(__file__).resolve().parent.parent / "review_tools" / "build_qa_bundle.py"
_spec = importlib.util.spec_from_file_location("build_qa_bundle", _BUILDER)
build_qa_bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_qa_bundle)

QA_FIELDS = build_qa_bundle.QA_FIELDS


def _fake_output(root: Path, cases: dict[str, list[str]], per_case: int) -> Path:
    """generate 가 QA 게이트에서 멈춘 상태의 출력을 흉내낸다."""
    output = root / "full"
    rows = []
    for modality, names in cases.items():
        for case in names:
            for index in range(per_case):
                # 원본 stem 은 케이스 간 겹치게 둔다. 사본 이름이 유일한지 보기 위해서다.
                stem = f"{modality}_cell_pouch_190000000{index}_x_200000{index:04d}"
                relative = f"{modality}/main/images/{stem}.jpg"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), (index * 20 % 255, 90, 140)).save(path)
                rows.append(
                    {
                        "modality": modality,
                        "failure_case": case,
                        "augmentation_subtype": "ring" if index % 2 else "",
                        "synthetic_id": f"QF16_{modality}_{len(rows):08d}",
                        "image_path": relative,
                        "source_filename": f"{stem}.png",
                        "source_image_path": f"Training/01.원천데이터/{stem}.png",
                        "original_battery_id": str(1000 + index),
                        "reviewer": "",
                        "approved": "",
                        "reason": "",
                    }
                )
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    with (manifests / "fail_visual_qa.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _config(root: Path, min_rate: float = 0.90, per_case: int = 10) -> Path:
    path = root / "config.json"
    path.write_text(
        json.dumps(
            {"visual_qa_min_approval_rate": min_rate, "visual_qa_samples_per_case": per_case}
        ),
        encoding="utf-8",
    )
    return path


def _build(output: Path, config: Path, bundle: Path) -> None:
    build_qa_bundle.main(
        ["--output", str(output), "--config", str(config), "--bundle", str(bundle)]
    )


class QaBundleTests(unittest.TestCase):
    CASES = {
        "CT": ["ct_cell_alignment_failure", "ct_low_signal_noise"],
        "RGB": ["rgb_surface_dust"],
    }

    def test_bundle_is_self_contained_and_matches_the_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _fake_output(root, self.CASES, per_case=10)
            bundle = root / "bundle"
            _build(output, _config(root), bundle)

            html = (bundle / "review_tool.html").read_text(encoding="utf-8")
            self.assertEqual([], re.findall(r"__[A-Z_]+__", html), "치환되지 않은 마커")

            fields = json.loads(re.search(r"const FIELDS = (\[.*?\]);", html).group(1))
            self.assertEqual(QA_FIELDS, fields)

            data = json.loads(re.search(r"const DATA = (\[.*?\]);\n", html, re.S).group(1))
            self.assertEqual(30, len(data))
            self.assertTrue(all(item["img"].startswith("data:image/") for item in data))
            self.assertTrue(all(item["srcFile"] for item in data))

            for name in ("fail_visual_qa.csv", "README.txt", "review_tool.html"):
                self.assertTrue((bundle / name).is_file(), name)

    def test_sample_copies_never_collide(self) -> None:
        """원본 stem 이 여러 케이스에 걸쳐 같아도 사본은 유일해야 한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _fake_output(root, self.CASES, per_case=10)
            bundle = root / "bundle"
            _build(output, _config(root), bundle)

            copies = list((bundle / "images").glob("*.jpg"))
            self.assertEqual(30, len(copies))
            self.assertEqual(30, len({path.name for path in copies}))

    def test_threshold_comes_from_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _fake_output(root, self.CASES, per_case=10)
            bundle = root / "bundle"
            _build(output, _config(root, min_rate=0.75, per_case=10), bundle)

            html = (bundle / "review_tool.html").read_text(encoding="utf-8")
            self.assertIn("const MIN_RATE = 0.75;", html)
            self.assertIn("75% 미달 케이스", html)
            self.assertIn("75% 이상", (bundle / "README.txt").read_text(encoding="utf-8"))

    def test_wrong_csv_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _fake_output(root, self.CASES, per_case=10)
            qa = output / "manifests" / "fail_visual_qa.csv"
            with qa.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            # v1.5 의 8칸 스키마로 되돌린다.
            legacy = [field for field in QA_FIELDS if not field.startswith("source_")]
            legacy = [field for field in legacy if field != "original_battery_id"]
            with qa.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=legacy, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(SystemExit):
                _build(output, _config(root), root / "bundle")

    def test_missing_qa_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _fake_output(root, self.CASES, per_case=10)
            (output / "manifests" / "fail_visual_qa.csv").unlink()
            with self.assertRaises(SystemExit):
                _build(output, _config(root), root / "bundle")


if __name__ == "__main__":
    unittest.main()
