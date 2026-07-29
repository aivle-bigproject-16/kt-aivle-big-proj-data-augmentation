from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from quality_fail_augment import __version__
from quality_fail_augment.augment import CT_CASES, RGB_CASES, apply_failure_case
from quality_fail_augment.geometry import extract_ct_roi, point_rings
from quality_fail_augment.planner import create_plan

from test_contract import _config, _label, _write_raw


class SchemaGateTests(unittest.TestCase):
    def test_version_is_exactly_plan_version(self) -> None:
        self.assertEqual(__version__, "1.7")

    def test_v16_ct_failure_cases_are_the_only_supported_ct_cases(self) -> None:
        self.assertEqual(
            CT_CASES,
            (
                "ct_cell_alignment_failure",
                "ct_acquisition_motion",
                "ct_insufficient_projection_sampling",
                "ct_low_signal_noise",
                "ct_beam_hardening_metal_streak",
            ),
        )

    def test_conflicting_roi_locations_are_rejected(self) -> None:
        payload = _label("CT_cell_pouch_1_x_1.png", 1, 1, "CT")
        payload["image_info"]["roi"] = [80, 80]
        with self.assertRaisesRegex(ValueError, "Conflicting CT ROI"):
            extract_ct_roi(payload, 100, 100)

    def test_missing_roi_mentions_both_supported_locations(self) -> None:
        payload = _label("CT_cell_pouch_1_x_1.png", 1, 1, "RGB")
        with self.assertRaisesRegex(
            ValueError, r"data_info\.roi and image_info\.roi"
        ):
            extract_ct_roi(payload, 100, 100)

    def test_nonempty_unknown_points_schema_is_not_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported_points_schema"):
            point_rings([{"left": 1, "top": 2}])

    def test_orphan_is_audited_but_does_not_block_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            orphan = (
                raw
                / "Training"
                / "02.라벨링데이터"
                / "CT"
                / "labels"
                / "CT_cell_pouch_999_x_999.json"
            )
            orphan.write_text(
                json.dumps(_label("CT_cell_pouch_999_x_999.png", 999, 999, "CT")),
                encoding="utf-8",
            )
            metadata = create_plan(raw, _config(), root / "plan")
            self.assertEqual(metadata["orphan_count"], 1)
            audit = (root / "plan" / "manifests" / "orphan_sources.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("json_only", audit)

    def test_ambiguous_cardinality_blocks_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            original = (
                raw
                / "Training"
                / "01.원천데이터"
                / "CT"
                / "images"
                / "CT_cell_pouch_1_x_1.png"
            )
            duplicate_dir = (
                raw / "Training" / "01.원천데이터" / "CT" / "duplicate"
            )
            duplicate_dir.mkdir()
            Image.open(original).save(duplicate_dir / original.name)
            with self.assertRaisesRegex(ValueError, "blocking_systemic"):
                create_plan(raw, _config(), root / "plan")


class FailureCaseTests(unittest.TestCase):
    def test_all_failure_cases_execute_and_record_only_their_case(self) -> None:
        image = Image.new("RGB", (96, 96), (80, 100, 120))
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 84, 84), outline=(230, 230, 230), width=4)
        draw.rectangle((40, 40, 55, 55), fill=(250, 250, 250))
        for index, case in enumerate(CT_CASES + RGB_CASES):
            modality = "CT" if case.startswith("ct_") else "RGB"
            last_error = None
            for retry in range(16):
                try:
                    result = apply_failure_case(
                        image, modality, case, 1000 + index * 100 + retry
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                self.fail(f"{case} could not pass its quality gate: {last_error}")
            self.assertEqual(result.failure_case, case)
            self.assertGreaterEqual(len(result.records), 1)
            self.assertTrue(np.isfinite(np.asarray(result.image)).all())

    def test_v17_case_set_and_record_types(self) -> None:
        self.assertNotIn("rgb_alignment_failure", RGB_CASES)
        image = Image.new("RGB", (128, 96), (90, 120, 160))
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((36, 8, 92, 88), fill=255)
        defect_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(defect_mask).rectangle((55, 38, 70, 58), fill=255)
        expected = {
            "rgb_surface_dust": "lens_dust_shadow",
            "rgb_hair_contamination": "lens_fiber_shadow",
            "rgb_reflection_glare": "surface_aware_specular_reflection",
            "rgb_underexposure": "linear_exposure_reduction",
        }
        for index, (case, record_type) in enumerate(expected.items()):
            result = apply_failure_case(
                image,
                "RGB",
                case,
                6000 + index,
                object_mask=object_mask,
                defect_mask=defect_mask,
            )
            self.assertIn(record_type, {record["type"] for record in result.records})

    def test_trigger_crop_preserves_source_aspect_ratio(self) -> None:
        image = Image.new("RGB", (160, 90), (120, 140, 160))
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((55, 10, 105, 80), fill=255)
        result = apply_failure_case(
            image,
            "RGB",
            "rgb_trigger_timing_failure",
            20260729,
            object_mask=object_mask,
        )
        self.assertAlmostEqual(
            result.image.width / result.image.height,
            image.width / image.height,
            delta=0.02,
        )


if __name__ == "__main__":
    unittest.main()
