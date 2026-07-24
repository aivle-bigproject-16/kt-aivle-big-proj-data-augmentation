from __future__ import annotations

import unittest

from PIL import Image

from quality_fail_augment.geometry import Affine
from quality_fail_augment.preprocessing import (
    PreparedSource,
    apply_quality_transform,
    finalize_sample,
)


class PreprocessingStageTests(unittest.TestCase):
    def test_pass_skips_failure_case_and_keeps_stage_transform(self) -> None:
        image = Image.new("RGB", (20, 10), "white")
        source = PreparedSource(
            image=image,
            label={"defects": []},
            transform=Affine(xoff=-3, yoff=-4),
            object_mask=Image.new("L", image.size, 255),
            defect_mask=Image.new("L", image.size, 0),
            original_roi=[3, 4, 23, 14],
        )

        result = apply_quality_transform(
            source,
            modality="CT",
            quality="pass",
            failure_case="",
            item_seed=7,
            max_retries=8,
        )

        self.assertEqual(result.records, [])
        self.assertEqual(result.transform, source.transform)
        self.assertIs(result.image, source.image)

    def test_finalize_uses_one_matrix_for_resize_and_polygon(self) -> None:
        image = Image.new("RGB", (20, 10), "white")
        source = PreparedSource(
            image=image,
            label={
                "defects": [{"points": [2, 2, 10, 2, 10, 6, 2, 6]}],
                "swelling": {"battery_outline": [0, 0, 20, 0, 20, 10, 0, 10]},
            },
            transform=Affine(),
            object_mask=Image.new("L", image.size, 255),
            defect_mask=Image.new("L", image.size, 0),
            original_roi=None,
        )
        transformed = apply_quality_transform(
            source, "RGB", "pass", "", item_seed=7, max_retries=8
        )

        result = finalize_sample(
            transformed,
            modality="RGB",
            quality="pass",
            new_battery=100,
            new_image=200,
            image_name="sample.jpg",
            resize_long_side=10,
            allow_upscale=False,
        )

        self.assertEqual(result.image.size, (10, 5))
        self.assertEqual(result.transform.apply_point(10, 6), (5.0, 3.0))
        flat_points = result.label["defects"][0]["points"]
        points = set(zip(flat_points[::2], flat_points[1::2], strict=True))
        self.assertEqual(points, {(1.0, 1.0), (5.0, 1.0), (5.0, 3.0), (1.0, 3.0)})


if __name__ == "__main__":
    unittest.main()
