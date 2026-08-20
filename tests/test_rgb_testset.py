from __future__ import annotations

import unittest

from quality_fail_augment.rgb_testset import build_rgb_test_plan


def _row(
    synthetic_id: str,
    partition: str,
    battery: int,
    image: int,
    quality: str,
    failure_case: str = "",
) -> dict[str, str]:
    return {
        "synthetic_id": synthetic_id,
        "modality": "RGB",
        "partition": partition,
        "original_battery_id": str(battery),
        "new_battery_id": str(1_900_100_000 + battery),
        "new_image_id": str(2_000_000_000 + image),
        "raw_image_path": f"RGB/{battery}/{image}.png",
        "image_sha256": f"sha-{battery}-{image}",
        "pixel_hash": f"pixel-{battery}-{image}",
        "quality_label": quality,
        "quality_class": quality,
        "assignment": "augmented" if quality == "fail" else "original",
        "failure_case": failure_case,
        "item_seed": str(image),
        "case_seed": str(image + 1),
        "group_key": "",
        "group_seed": "",
        "config_sha256": "old",
    }


class RgbTestsetPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "ct_failure_case_quotas": {},
            "ct_test_failure_case_quotas": {},
            "rgb_failure_case_quotas": {},
            "rgb_test_failure_case_quotas": {},
        }

    def test_builds_exact_augmented_and_original_test_counts(self) -> None:
        rows = [_row("main", "main", 1, 1, "pass")]
        rows.extend(
            _row(
                f"test-{index}",
                "test",
                2,
                index,
                "fail",
                "rgb_underexposure",
            )
            for index in range(1, 5)
        )

        selected, config, audit = build_rgb_test_plan(rows, self.config, 4, 1)

        self.assertEqual(sum(row["quality_label"] == "fail" for row in selected), 1)
        self.assertEqual(sum(row["quality_label"] == "pass" for row in selected), 3)
        self.assertTrue(all(row["partition"] == "test" for row in selected))
        self.assertEqual(config["rgb_test_target"], 4)
        self.assertEqual(config["rgb_test_fail_target"], 1)
        self.assertEqual(audit["main_test_battery_overlap"], 0)
        failed = next(row for row in selected if row["quality_label"] == "fail")
        self.assertEqual(failed["failure_case"], "rgb_underexposure")
        self.assertEqual(failed["item_seed"], "1")
        self.assertEqual(failed["case_seed"], "2")

    def test_selects_augmented_rows_before_filling_original_quota(self) -> None:
        rows = [_row("main", "main", 1, 1, "pass")]
        rows.extend(_row(f"pass-{i}", "test", 2, i, "pass") for i in (1, 2))
        rows.append(
            _row("late-fail", "test", 2, 3, "fail", "rgb_underexposure")
        )

        selected, _, _ = build_rgb_test_plan(rows, self.config, 2, 1)

        self.assertEqual(
            {row["synthetic_id"] for row in selected}, {"pass-1", "late-fail"}
        )

    def test_rejects_missing_identity_instead_of_claiming_no_overlap(self) -> None:
        main = _row("main", "main", 1, 1, "pass")
        test = _row("test", "test", 2, 2, "fail", "rgb_underexposure")
        test["pixel_hash"] = ""

        with self.assertRaisesRegex(ValueError, "without pixel_hash"):
            build_rgb_test_plan([main, test], self.config, 1, 1)

    def test_rejects_battery_overlap_with_main(self) -> None:
        rows = [
            _row("main", "main", 1, 1, "pass"),
            _row("test", "test", 1, 2, "fail", "rgb_underexposure"),
        ]

        with self.assertRaisesRegex(ValueError, "original_battery_id"):
            build_rgb_test_plan(rows, self.config, 1, 1)

    def test_rejects_source_content_overlap_with_main(self) -> None:
        main = _row("main", "main", 1, 1, "pass")
        test = _row("test", "test", 2, 2, "fail", "rgb_underexposure")
        test["pixel_hash"] = main["pixel_hash"]

        with self.assertRaisesRegex(ValueError, "pixel_hash"):
            build_rgb_test_plan([main, test], self.config, 1, 1)


if __name__ == "__main__":
    unittest.main()
