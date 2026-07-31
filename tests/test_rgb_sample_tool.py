from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from review_tools.sample_rgb_dust_glare import CASES, generate_samples


class RgbSampleToolTests(unittest.TestCase):
    def test_generates_two_cases_without_planner_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            image_dir = (
                raw
                / "3.개방데이터"
                / "1.데이터"
                / "Training"
                / "01.원천데이터"
                / "TS_Exterior_Img_Datasets_images_1"
            )
            label_dir = (
                raw
                / "3.개방데이터"
                / "1.데이터"
                / "Training"
                / "02.라벨링데이터"
                / "TL_Exterior_Img_Datasets_label"
            )
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index in range(1, 9):
                stem = f"RGB_cell_cylindrical_0001_{index:03d}"
                image = Image.new("RGB", (160, 90), (235, 235, 230))
                for x in range(55, 106):
                    for y in range(8, 83):
                        image.putpixel((x, y), (45 + index, 125, 65))
                image.save(image_dir / f"{stem}.png")
                label = {
                    "swelling": {
                        "battery_outline": [
                            [55, 8],
                            [105, 8],
                            [105, 82],
                            [55, 82],
                        ]
                    },
                    "defects": [],
                }
                (label_dir / f"{stem}.json").write_text(
                    json.dumps(label),
                    encoding="utf-8",
                )

            output = root / "samples"
            before = {
                path.relative_to(raw): path.read_bytes()
                for path in raw.rglob("*")
                if path.is_file()
            }
            rows = generate_samples(raw, output, 2, 20260731, 8, 8, 0)

            self.assertEqual(len(rows), 4)
            self.assertEqual({row["failure_case"] for row in rows}, set(CASES))
            self.assertEqual(
                len({row["source_image_path"] for row in rows}),
                4,
            )
            self.assertEqual(len(list(output.glob("*/*.jpg"))), 4)
            self.assertTrue((output / "sample_manifest.csv").is_file())
            self.assertTrue((output / "sample_summary.json").is_file())
            summary = json.loads(
                (output / "sample_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["planner_scan_used"])
            after = {
                path.relative_to(raw): path.read_bytes()
                for path in raw.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)

            with self.assertRaisesRegex(ValueError, "outside raw-root"):
                generate_samples(raw, raw / "bad-output", 1, 1, 8, 2, 0)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                generate_samples(raw, output, 1, 1, 8, 2, 0)
            with self.assertRaisesRegex(ValueError, "max-retries"):
                generate_samples(raw, root / "bad-retries", 1, 1, 0, 2, 0)


if __name__ == "__main__":
    unittest.main()
