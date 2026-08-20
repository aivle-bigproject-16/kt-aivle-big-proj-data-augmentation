from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from quality_fail_augment.generator import generate, verify_dataset
from quality_fail_augment.geometry import Affine, extract_ct_roi, transform_label
from quality_fail_augment.planner import create_plan


def _label(name: str, image_id: int, battery_id: int, modality: str) -> dict:
    payload = {
        "data_info": {"battery_ids": battery_id},
        "swelling": {
            "battery_outline": [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]
        },
        "defects": [
            {
                "name": "porosity" if modality == "CT" else "scratch",
                "points": [30.0, 30.0, 50.0, 30.0, 50.0, 50.0, 30.0, 50.0],
            }
        ],
        "image_info": {
            "id": image_id,
            "file_name": name,
            "width": 100,
            "height": 100,
        },
    }
    if modality == "CT":
        payload["data_info"]["roi"] = [100, 100]
    return payload


def _write_raw(root: Path, count: int = 6) -> None:
    for modality in ("CT", "RGB"):
        image_dir = root / "Training" / "01.원천데이터" / modality / "images"
        label_dir = root / "Training" / "02.라벨링데이터" / modality / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = (
                f"CT_cell_pouch_{index + 1}_x_{index + 1}"
                if modality == "CT"
                else f"RGB_cell_cylindrical_{index + 1}_{index + 1}"
            )
            image = Image.new(
                "RGB",
                (100, 100),
                (25 + index * 5,) * 3 if modality == "CT" else (80 + index * 8, 130, 180),
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle((10 + index, 10, 90, 90), outline=(220, 220, 220), width=2)
            draw.point((index, index), fill=(255, 255, 255))
            image.save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.json").write_text(
                json.dumps(_label(f"{stem}.png", index + 1, index + 1, modality)),
                encoding="utf-8",
            )


def _config() -> dict:
    return {
        "ct_target": 4,
        "ct_augmented_target": 2,
        "ct_test_target": 2,
        "ct_test_fail_target": 1,
        "ct_failure_case_quotas": {
            "ct_cell_alignment_failure": 1,
            "ct_low_signal_noise": 1,
        },
        "rgb_target": 4,
        "rgb_augmented_target": 2,
        "rgb_test_target": 2,
        "rgb_test_fail_target": 1,
        "rgb_failure_case_quotas": {
            "rgb_surface_dust": 1,
            "rgb_hair_contamination": 1,
        },
        "reserve_per_modality": 1,
        "resize_long_side": 64,
        "upscale_small_images": True,
        "jpeg_quality": 90,
        "jpeg_subsampling": 0,
        "seed": 20260723,
        "max_augmentation_retries": 4,
        "ct_porosity_threshold": 0.25,
        "ct_bbox_policy_version": "v1.6_full_precision_ge_0.25",
        "ct_battery_id_start": 1900000001,
        "ct_battery_id_end": 1900100000,
        "rgb_battery_id_start": 1900100001,
        "rgb_battery_id_end": 1900199999,
        "ct_image_id_start": 2000000001,
        "rgb_image_id_start": 2000020001,
        "jobs": 1,
        "parallel_chunk_multiplier": 8,
        "preflight_per_stratum": 0,
        "systemic_consecutive_limit": 20,
        "systemic_cumulative_limit": 100,
    }


class PublicContractTests(unittest.TestCase):
    def test_actual_roi_location_and_flat_polygon_transform(self) -> None:
        source = _label("CT_cell_pouch_1_x_1.png", 1, 1, "CT")
        self.assertEqual(extract_ct_roi(source, 100, 100), (0.0, 0.0, 100.0, 100.0))
        transformed = transform_label(
            source,
            Affine(a=0.5, e=0.5, xoff=5, yoff=7),
            (50, 50),
            "CT",
            "fail",
            1900000001,
            2000000001,
            "out.jpg",
        )
        self.assertEqual(transformed["quality_class"], "fail")
        self.assertEqual(transformed["data_info"]["battery_ids"], 1900000001)
        self.assertEqual(transformed["image_info"]["id"], 2000000001)
        self.assertEqual(transformed["data_info"]["roi"], [0, 0, 50, 50])
        outline = transformed["swelling"]["battery_outline"]
        vertices = {
            (outline[index], outline[index + 1])
            for index in range(0, len(outline), 2)
        }
        self.assertEqual(
            vertices,
            {(10.0, 12.0), (50.0, 12.0), (50.0, 50.0), (10.0, 50.0)},
        )

    def test_plan_and_generate_four_json_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, plan_dir, output = root / "raw", root / "plan", root / "output"
            _write_raw(raw)
            metadata = create_plan(raw, _config(), plan_dir)
            self.assertEqual(metadata["selected_rows"], 8)
            self.assertEqual(metadata["package_version"], "1.8")
            with (plan_dir / "manifests" / "generation_plan.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                plan_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["synthetic_id"].startswith("QF18_") for row in plan_rows))
            self.assertTrue(
                {"case_seed", "group_key", "group_seed"}.issubset(plan_rows[0])
            )

            summary = generate(
                raw,
                _config(),
                plan_dir / "manifests" / "generation_plan.csv",
                output,
            )
            self.assertEqual(summary["counts"]["CT_pass"], 2)
            self.assertEqual(summary["counts"]["CT_fail"], 2)
            self.assertEqual(summary["counts"]["RGB_pass"], 2)
            self.assertEqual(summary["counts"]["RGB_fail"], 2)

            labels = list(output.glob("*/**/labels_json/*.json"))
            histories = list(output.glob("*/**/augmentation_json/*.augmentation.json"))
            self.assertEqual(len(labels), 8)
            self.assertEqual(len(histories), 4)
            self.assertTrue((output / "augmentation_json_4k_v1.8.zip").is_file())

            with (output / "manifests" / "dataset_manifest.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                self.assertEqual(row["quality_class"], row["quality_label"])
                self.assertEqual(row["image_relative_path"], row["image_path"])
                self.assertEqual(row["label_json_relative_path"], row["label_json_path"])
                self.assertEqual(
                    row["augmentation_json_relative_path"],
                    row["augmentation_json_path"],
                )
                label = json.loads((output / row["label_json_path"]).read_text(encoding="utf-8"))
                self.assertEqual(label["quality_class"], row["quality_label"])
                if row["quality_label"] == "pass":
                    self.assertEqual(row["augmentation_json_path"], "")
                    self.assertEqual(row["augmentation_json_sha256"], "")
                else:
                    history_path = output / row["augmentation_json_path"]
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    self.assertEqual(history["schema_version"], "1.1")
                    self.assertEqual(history["output_image_file"], Path(row["image_path"]).name)
                    self.assertEqual(history["label_json_file"], Path(row["label_json_path"]).name)
                    self.assertEqual(history["failure_case_count"], 1)
                    self.assertEqual(history["quality_label"], "fail")
                    self.assertTrue(
                        history["failure_case"]["source_reference"].startswith("v1.8:")
                    )
                    self.assertTrue(history["automatic_checks"]["passed"])
                    self.assertEqual(history["output"]["format"], "JPEG")
                    self.assertEqual(history["output"]["quality"], 90)

            self.assertTrue(
                (output / "manifests" / "generation_plan.csv").is_file()
            )

            partitions = {(row["modality"], row["partition"], row["quality_label"]) for row in rows}
            for modality in ("CT", "RGB"):
                self.assertIn((modality, "main", "pass"), partitions)
                self.assertIn((modality, "main", "fail"), partitions)
                self.assertIn((modality, "test", "pass"), partitions)
                self.assertIn((modality, "test", "fail"), partitions)
            verify_dataset(output)

    def test_pairing_is_scoped_by_raw_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_raw(raw)
            source = raw / "Training" / "01.원천데이터" / "CT" / "images" / "CT_cell_pouch_1_x_1.png"
            label = raw / "Training" / "02.라벨링데이터" / "CT" / "labels" / "CT_cell_pouch_1_x_1.json"
            validation_source = raw / "Validation" / "01.원천데이터" / "CT" / "images"
            validation_label = raw / "Validation" / "02.라벨링데이터" / "CT" / "labels"
            validation_source.mkdir(parents=True)
            validation_label.mkdir(parents=True)
            Image.open(source).save(validation_source / source.name)
            (validation_label / label.name).write_bytes(label.read_bytes())
            metadata = create_plan(raw, _config(), root / "plan")
            self.assertEqual(metadata["ambiguous_pair_count"], 0)

    def test_resume_removes_manifestless_partial_and_preserves_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, plan_dir, output = root / "raw", root / "plan", root / "output"
            _write_raw(raw)
            create_plan(raw, _config(), plan_dir)
            generate(
                raw,
                _config(),
                plan_dir / "manifests" / "generation_plan.csv",
                output,
            )
            partial = output / "CT" / "main" / "images" / "uncommitted.jpg"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial")
            resumed = generate(
                raw,
                _config(),
                plan_dir / "manifests" / "generation_plan.csv",
                output,
                resume=True,
            )
            self.assertFalse(partial.exists())
            self.assertEqual(sum(resumed["counts"].values()), 8)
            recovery = (
                output / "manifests" / "recovery_audit.csv"
            ).read_text(encoding="utf-8-sig")
            self.assertIn("uncommitted.jpg", recovery)

    def test_resume_stops_when_committed_file_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, plan_dir, output = root / "raw", root / "plan", root / "output"
            _write_raw(raw)
            create_plan(raw, _config(), plan_dir)
            generate(
                raw,
                _config(),
                plan_dir / "manifests" / "generation_plan.csv",
                output,
            )
            with (output / "manifests" / "dataset_manifest.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                first = next(csv.DictReader(handle))
            (output / first["label_json_path"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                generate(
                    raw,
                    _config(),
                    plan_dir / "manifests" / "generation_plan.csv",
                    output,
                    resume=True,
                )

    def test_generation_finishes_without_visual_qa_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, plan_dir, output = root / "raw", root / "plan", root / "output"
            _write_raw(raw)
            config = _config()
            create_plan(raw, config, plan_dir)
            summary = generate(
                raw,
                config,
                plan_dir / "manifests" / "generation_plan.csv",
                output,
            )
            self.assertEqual(sum(summary["counts"].values()), 8)
            self.assertFalse((output / "manifests" / "fail_visual_qa.csv").exists())


if __name__ == "__main__":
    unittest.main()
