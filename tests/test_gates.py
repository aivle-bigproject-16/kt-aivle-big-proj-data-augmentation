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
        self.assertEqual(__version__, "2.0")

    def test_v19_ct_failure_cases_are_the_only_supported_ct_cases(self) -> None:
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
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((12, 12, 84, 84), fill=255)
        for index, case in enumerate(CT_CASES + RGB_CASES):
            modality = "CT" if case.startswith("ct_") else "RGB"
            last_error = None
            for retry in range(16):
                try:
                    result = apply_failure_case(
                        image,
                        modality,
                        case,
                        1000 + index * 100 + retry,
                        object_mask=object_mask,
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                self.fail(f"{case} could not pass its quality gate: {last_error}")
            self.assertEqual(result.failure_case, case)
            self.assertGreaterEqual(len(result.records), 1)
            self.assertTrue(np.isfinite(np.asarray(result.image)).all())

    def test_v19_case_set_and_record_types(self) -> None:
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

    def test_v19_draws_only_the_severe_parameter_ranges(self) -> None:
        image = Image.new("RGB", (160, 90), (45, 70, 95))
        ImageDraw.Draw(image).rectangle((35, 8, 125, 82), fill=(145, 175, 205))
        ImageDraw.Draw(image).ellipse((70, 35, 90, 55), fill=(250, 250, 250))
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((35, 8, 125, 82), fill=255)
        defect_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(defect_mask).ellipse((70, 35, 90, 55), fill=255)

        results = {}
        for index, case in enumerate(CT_CASES + RGB_CASES):
            last_error = None
            for retry in range(8):
                try:
                    results[case] = apply_failure_case(
                        image,
                        "CT" if case.startswith("ct_") else "RGB",
                        case,
                        9_000_000 + index * 100 + retry,
                        object_mask=object_mask,
                        defect_mask=defect_mask,
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                self.fail(f"{case} exhausted severe-range retries: {last_error}")

        def parameters(case: str, record_type: str) -> dict:
            return next(
                record["parameters"]
                for record in results[case].records
                if record["type"] == record_type
            )

        alignment = parameters(
            "ct_cell_alignment_failure", "alignment_edge_crop"
        )
        self.assertTrue(
            0.92 <= alignment["target_outline_retained_ratio"] <= 0.96
        )
        self.assertTrue(0.90 <= alignment["retained_outline_ratio"] <= 0.98)

        motion = parameters("ct_acquisition_motion", "double_edge_ghosting")
        blur = parameters("ct_acquisition_motion", "directional_motion_blur")
        self.assertTrue(6.0 <= motion["offset_final_512_px"] <= 9.0)
        self.assertTrue(0.15 <= motion["shifted_weight"] <= 0.22)
        self.assertTrue(
            0.5 <= blur["displacement_range_final_512_px"] <= 1.2
        )

        projection = parameters(
            "ct_insufficient_projection_sampling", "radon_projection_drop"
        )
        if projection["subtype"] == "sparse_view":
            self.assertTrue(0.72 <= projection["retained_ratio"] <= 0.82)
        else:
            self.assertTrue(
                15.0 <= projection["removed_width_deg"] <= 25.0
            )
        reconstruction = parameters(
            "ct_insufficient_projection_sampling", "filtered_back_projection"
        )
        self.assertTrue(
            0.25 <= reconstruction["reconstruction_weight"] <= 0.40
        )

        low_signal = parameters("ct_low_signal_noise", "signal_to_transmission")
        photons = parameters("ct_low_signal_noise", "poisson_sampling")
        read_noise = parameters("ct_low_signal_noise", "read_noise")
        contrast = parameters(
            "ct_low_signal_noise", "low_contrast_attenuation"
        )
        self.assertTrue(0.78 <= low_signal["signal_factor"] <= 0.86)
        self.assertTrue(90.0 <= photons["photon_scale"] <= 130.0)
        self.assertTrue(0.001 <= read_noise["sigma_normalized"] <= 0.002)
        self.assertTrue(0.95 <= contrast["contrast_factor"] <= 1.00)

        dense = parameters(
            "ct_beam_hardening_metal_streak", "dense_material_mask"
        )
        streak = parameters(
            "ct_beam_hardening_metal_streak", "metal_anchored_streaks"
        )
        self.assertTrue(0.05 <= dense["attenuation"] <= 0.10)
        self.assertTrue(8 <= streak["streak_count"] <= 12)
        self.assertTrue(0.15 <= streak["decay_distance_ratio"] <= 0.22)
        self.assertTrue(
            all(
                0.4 <= width <= 0.8
                for width in streak["ray_widths_final_512_px"]
            )
        )

        timing = parameters("rgb_trigger_timing_failure", "timing_edge_crop")
        self.assertTrue(
            0.55 <= timing["target_outline_retained_ratio"] <= 0.68
        )
        self.assertTrue(0.52 <= timing["retained_outline_ratio"] <= 0.72)

        uneven = parameters("rgb_uneven_lighting", "lighting_gradient")
        self.assertIn(
            (uneven["dark_gain"], uneven["bright_gain"]),
            {(0.25, 1.65), (0.18, 1.85), (0.12, 2.05)},
        )
        self.assertTrue(0.45 <= uneven["output_asymmetry"] <= 0.60)
        self.assertGreaterEqual(
            uneven["output_asymmetry"] - uneven["baseline_asymmetry"], 0.25
        )

        glare = parameters(
            "rgb_reflection_glare", "surface_aware_specular_reflection"
        )
        self.assertEqual(len(glare["patches"]), 2)
        self.assertTrue(0.045 <= glare["core_object_area_ratio"] <= 0.12)
        self.assertTrue(
            all(0.70 <= patch["alpha"] <= 0.78 for patch in glare["patches"])
        )

        focus = parameters("rgb_focus_failure", "defocus_blur")
        self.assertTrue(7.5 <= focus["radius_final_space"] <= 10.0)

        under = parameters("rgb_underexposure", "linear_exposure_reduction")
        under_noise = parameters(
            "rgb_underexposure", "signal_dependent_shot_noise"
        )
        under_read = parameters("rgb_underexposure", "sensor_read_noise")
        self.assertTrue(0.18 <= under["exposure_factor"] <= 0.28)
        self.assertTrue(0.44 <= under["target_outline_mean_ratio"] <= 0.50)
        self.assertTrue(80.0 <= under_noise["photon_capacity"] <= 120.0)
        self.assertTrue(0.010 <= under_read["sigma"] <= 0.015)
        self.assertTrue(0.008 <= under_read["black_level"] <= 0.012)

        over = parameters("rgb_overexposure", "overexposure")
        self.assertTrue(2.10 <= over["exposure_factor"] <= 2.60)
        self.assertTrue(
            0.50 <= over["target_object_saturation_ratio"] <= 0.60
        )

        dust = parameters("rgb_surface_dust", "lens_dust_shadow")
        self.assertTrue(3 <= dust["shadow_count"] <= 4)
        self.assertTrue(1 <= dust["object_overlap_shadow_count"] <= 2)
        self.assertTrue(
            all(0.22 <= shadow["core_alpha"] <= 0.25 for shadow in dust["shadows"])
        )
        self.assertTrue(
            all(0.10 <= shadow["halo_alpha"] <= 0.125 for shadow in dust["shadows"])
        )

        hair = parameters("rgb_hair_contamination", "lens_fiber_shadow")
        self.assertEqual(hair["curve_count"], 2)
        self.assertTrue(
            all(0.18 <= curve["alpha"] <= 0.23 for curve in hair["curves"])
        )
        self.assertTrue(4.0 <= hair["halo_multiplier"] <= 5.0)
        self.assertTrue(
            all(0.08 <= alpha <= 0.10 for alpha in hair["halo_alphas"])
        )
        self.assertLessEqual(hair["frame_affected_ratio"], 0.35)

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
        with self.assertRaisesRegex(
            ValueError, "timing_crop_requires_object_mask"
        ):
            apply_failure_case(
                image,
                "RGB",
                "rgb_trigger_timing_failure",
                20260729,
                object_mask=None,
            )

    def test_ct_alignment_uses_gate_safe_same_size_crop_without_porosity(self) -> None:
        image = Image.new("L", (160, 90), 20)
        ImageDraw.Draw(image).rectangle((35, 8, 125, 82), fill=170)
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((35, 8, 125, 82), fill=255)

        directions = set()
        for seed in range(8):
            result = apply_failure_case(
                image,
                "CT",
                "ct_cell_alignment_failure",
                20260729 + seed,
                object_mask=object_mask,
                defect_mask=None,
                group_seed=777 + seed,
            )
            self.assertEqual(result.image.size, image.size)
            self.assertAlmostEqual(
                result.image.width / result.image.height,
                image.width / image.height,
                delta=0.001,
            )
            self.assertEqual(result.records[0]["type"], "alignment_edge_crop")
            directions.add(result.records[0]["parameters"]["direction"])
            retained = result.records[0]["parameters"]["retained_outline_ratio"]
            self.assertGreaterEqual(retained, 0.90)
            self.assertLessEqual(retained, 0.98)
        self.assertGreater(len(directions), 1)

        with self.assertRaisesRegex(
            ValueError, "alignment_crop_requires_object_mask"
        ):
            apply_failure_case(
                image,
                "CT",
                "ct_cell_alignment_failure",
                20260729,
                object_mask=None,
                defect_mask=None,
            )

    def test_rgb_gate_sensitive_cases_succeed_with_retry_budget(self) -> None:
        cases = (
            "rgb_overexposure",
            "rgb_focus_failure",
            "rgb_underexposure",
            "rgb_reflection_glare",
            "rgb_uneven_lighting",
        )
        variants = (
            ((45, 70, 95), (110, 145, 175)),
            ((105, 120, 135), (175, 195, 215)),
            ((20, 30, 45), (70, 95, 120)),
        )
        for variant_index, (background, foreground) in enumerate(variants):
            image = Image.new("RGB", (160, 90), background)
            ImageDraw.Draw(image).rectangle((35, 8, 125, 82), fill=foreground)
            object_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(object_mask).rectangle((35, 8, 125, 82), fill=255)
            defect_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(defect_mask).ellipse((68, 35, 92, 58), fill=255)
            for case_index, case in enumerate(cases):
                last_error = None
                for retry in range(8):
                    try:
                        result = apply_failure_case(
                            image,
                            "RGB",
                            case,
                            800_000
                            + variant_index * 10_000
                            + case_index * 100
                            + retry,
                            object_mask=object_mask,
                            defect_mask=defect_mask,
                        )
                        break
                    except ValueError as exc:
                        last_error = exc
                else:
                    self.fail(
                        f"{case} variant {variant_index} exhausted retries: "
                        f"{last_error}"
                    )

                baseline = np.asarray(image.convert("L"), dtype=np.float32)
                output = np.asarray(result.image.convert("L"), dtype=np.float32)
                region = np.asarray(object_mask) > 0
                if case == "rgb_overexposure":
                    saturation = float((output[region] >= 250).mean())
                    self.assertGreaterEqual(saturation, 0.15)
                    self.assertLessEqual(saturation, 0.75)
                elif case == "rgb_underexposure":
                    mean_ratio = float(output[region].mean()) / float(
                        baseline[region].mean()
                    )
                    self.assertGreaterEqual(mean_ratio, 0.40)
                    self.assertLessEqual(mean_ratio, 0.70)
                elif case == "rgb_reflection_glare":
                    parameters = result.records[0]["parameters"]
                    self.assertGreaterEqual(
                        parameters["outline_overlap_ratio"], 0.90
                    )
                    self.assertGreaterEqual(
                        parameters["core_object_area_ratio"], 0.01
                    )
                    self.assertLessEqual(
                        parameters["core_object_area_ratio"], 0.12
                    )
                    self.assertGreaterEqual(
                        parameters["defect_coverage_ratio"],
                        parameters["minimum_defect_coverage_ratio"],
                    )
                    self.assertTrue(
                        all(
                            2.0 * patch["half_length"] / patch["width"] >= 5.0
                            for patch in parameters["patches"]
                        )
                    )
                elif case == "rgb_uneven_lighting":
                    parameters = result.records[0]["parameters"]
                    self.assertGreaterEqual(parameters["output_asymmetry"], 0.25)
                    self.assertLessEqual(parameters["output_asymmetry"], 0.60)
                    self.assertGreaterEqual(
                        parameters["output_asymmetry"]
                        - parameters["baseline_asymmetry"],
                        0.15,
                    )

    def test_glare_fallback_keeps_the_same_strength_gate_on_a_narrow_outline(self) -> None:
        image = Image.new("RGB", (120, 240), (30, 40, 50))
        ImageDraw.Draw(image).rectangle((56, 5, 64, 235), fill=(130, 160, 190))
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((56, 5, 64, 235), fill=255)
        defect_mask = Image.new("L", image.size, 0)
        # Deliberately extend the annotation outside the battery outline.  Recorded gate
        # measurements must describe the visible, outline-clipped glare only.
        ImageDraw.Draw(defect_mask).rectangle((50, 20, 70, 43), fill=255)

        result = apply_failure_case(
            image,
            "RGB",
            "rgb_reflection_glare",
            123,
            object_mask=object_mask,
            defect_mask=defect_mask,
        )

        parameters = result.records[0]["parameters"]
        self.assertEqual(len(parameters["patches"]), 2)
        self.assertEqual(parameters["outline_overlap_ratio"], 1.0)
        self.assertGreaterEqual(parameters["core_object_area_ratio"], 0.045)
        self.assertLessEqual(parameters["core_object_area_ratio"], 0.12)
        self.assertGreaterEqual(
            parameters["defect_coverage_ratio"],
            parameters["minimum_defect_coverage_ratio"],
        )
        self.assertLessEqual(parameters["defect_coverage_ratio"], 0.70)

    def test_focus_gate_is_measured_in_final_512_pixel_space(self) -> None:
        image = Image.new("RGB", (1920, 1080), (45, 70, 95))
        ImageDraw.Draw(image).rectangle(
            (420, 90, 1500, 990), fill=(145, 175, 205)
        )
        object_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(object_mask).rectangle((420, 90, 1500, 990), fill=255)
        result = apply_failure_case(
            image,
            "RGB",
            "rgb_focus_failure",
            20260731,
            object_mask=object_mask,
        )
        baseline = np.asarray(
            image.convert("L").resize((512, 288), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        output = np.asarray(
            result.image.convert("L").resize(
                (512, 288), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )

        def rms_gradient(values: np.ndarray) -> float:
            horizontal = np.diff(values, axis=1)
            vertical = np.diff(values, axis=0)
            return float(
                np.sqrt(
                    np.mean(horizontal * horizontal)
                    + np.mean(vertical * vertical)
                )
            )

        ratio = rms_gradient(output) / rms_gradient(baseline)
        self.assertGreaterEqual(ratio, 0.15)
        self.assertLessEqual(ratio, 0.50)
        parameters = result.records[0]["parameters"]
        self.assertAlmostEqual(
            parameters["radius_source_space"],
            parameters["radius_final_space"] * 1920 / 512,
        )


if __name__ == "__main__":
    unittest.main()
