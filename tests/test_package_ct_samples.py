from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from review_tools.package_ct_samples import package_samples


class PackageCtSamplesTests(unittest.TestCase):
    def test_production_annotation_and_augmentation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_root = root / "samples"
            sample_root.mkdir()
            source_image = root / "CT_cell_pouch_101_x_000.jpg"
            source_json = root / "CT_cell_pouch_101_x_000.json"
            image = Image.new("L", (100, 100), 45)
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 90, 90), fill=135)
            draw.rectangle((40, 40, 60, 60), fill=230)
            image.save(source_image)
            source_json.write_text(
                json.dumps(
                    {
                        "data_info": {
                            "battery_ids": 101,
                            "roi": [100, 100],
                        },
                        "swelling": {
                            "battery_outline": [
                                10,
                                10,
                                90,
                                10,
                                90,
                                90,
                                10,
                                90,
                            ]
                        },
                        "defects": [
                            {
                                "id": 1,
                                "name": "porosity",
                                "points": [
                                    40,
                                    40,
                                    60,
                                    40,
                                    60,
                                    60,
                                    40,
                                    60,
                                ],
                            }
                        ],
                        "image_info": {
                            "id": 1,
                            "file_name": source_image.name,
                            "width": 100,
                            "height": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (sample_root / "sample_manifest.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "failure_case",
                        "sample_index",
                        "output_file",
                        "source_image_name",
                        "source_image_path",
                        "source_json_path",
                        "seed",
                        "retry",
                        "augmentation_parameters",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "failure_case": "ct_low_signal_noise",
                        "sample_index": 1,
                        "output_file": "ct_low_signal_noise/01_test.png",
                        "source_image_name": source_image.name,
                        "source_image_path": source_image,
                        "source_json_path": source_json,
                        "seed": 123,
                        "retry": 0,
                        "augmentation_parameters": "[]",
                    }
                )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "resize_long_side": 64,
                        "upscale_small_images": True,
                        "jpeg_quality": 90,
                        "jpeg_subsampling": 0,
                        "jpeg_optimize": False,
                        "jpeg_progressive": False,
                        "ct_battery_id_start": 1900000001,
                        "ct_image_id_start": 2000000001,
                    }
                ),
                encoding="utf-8",
            )

            images_zip, labels_zip = package_samples(
                sample_root, config_path
            )
            with zipfile.ZipFile(images_zip) as archive:
                image_names = [
                    name
                    for name in archive.namelist()
                    if name != "matching_manifest.csv"
                ]
                self.assertEqual(len(image_names), 1)
                self.assertTrue(image_names[0].endswith(".jpg"))
            with zipfile.ZipFile(labels_zip) as archive:
                history_name = next(
                    name
                    for name in archive.namelist()
                    if name.endswith(".augmentation.json")
                )
                annotation_name = next(
                    name
                    for name in archive.namelist()
                    if "/labels_json/" in name
                )
                history = json.loads(archive.read(history_name))
                annotation = json.loads(archive.read(annotation_name))
            self.assertEqual(history["schema_version"], "1.1")
            self.assertEqual(history["quality_label"], "fail")
            self.assertTrue(history["is_augmented"])
            self.assertEqual(
                history["failure_case"]["id"], "ct_low_signal_noise"
            )
            self.assertEqual(
                history["label_json_file"], Path(annotation_name).name
            )
            self.assertEqual(
                annotation["image_info"]["file_name"],
                history["output_image_file"],
            )


if __name__ == "__main__":
    unittest.main()
