from __future__ import annotations

import unittest
from unittest import mock

from PIL import Image

from quality_fail_augment.augment import (
    EXHAUSTED_SEARCH_MARKER,
    ExhaustedSearchError,
)
from quality_fail_augment.preprocessing.stages import (
    PreparedSource,
    apply_quality_transform,
)


def _source() -> PreparedSource:
    image = Image.new("RGB", (16, 16), (128, 128, 128))
    return PreparedSource(
        image=image,
        label={},
        transform=mock.MagicMock(),
        object_mask=None,
        defect_mask=None,
        porosity_mask=None,
        porosity_ids=(),
        original_roi=None,
    )


class RetryPolicyTests(unittest.TestCase):
    def test_exhausted_search_is_not_retried(self) -> None:
        """격자 전수 탐색 실패는 시드를 안 쓰므로 재시도가 같은 계산을 반복할 뿐이다."""
        calls = []

        def explode(*args, **kwargs):
            calls.append(args)
            raise ExhaustedSearchError("glare_no_gate_safe_geometry")

        with mock.patch(
            "quality_fail_augment.preprocessing.stages.apply_failure_case", explode
        ):
            with self.assertRaises(ValueError) as caught:
                apply_quality_transform(
                    _source(), "RGB", "fail", "rgb_reflection_glare", 1, max_retries=8
                )

        self.assertEqual(len(calls), 1)
        self.assertIn(EXHAUSTED_SEARCH_MARKER, str(caught.exception))

    def test_other_failures_still_use_every_retry(self) -> None:
        """게이트 실패는 파라미터가 시드에 따라 달라지므로 재시도에 값이 있다."""
        calls = []

        def explode(*args, **kwargs):
            calls.append(args)
            raise ValueError("quality_gate: glare core area is outside 4.5%..12%")

        with mock.patch(
            "quality_fail_augment.preprocessing.stages.apply_failure_case", explode
        ):
            with self.assertRaises(ValueError) as caught:
                apply_quality_transform(
                    _source(), "RGB", "fail", "rgb_reflection_glare", 1, max_retries=8
                )

        self.assertEqual(len(calls), 8)
        self.assertIn("retries exhausted", str(caught.exception))

    def test_success_on_a_later_attempt_is_kept(self) -> None:
        attempts = {"n": 0}

        def flaky(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("quality_gate: transient")
            result = mock.MagicMock()
            result.image = Image.new("RGB", (16, 16))
            result.records = []
            result.transform = mock.MagicMock()
            return result

        with mock.patch(
            "quality_fail_augment.preprocessing.stages.apply_failure_case", flaky
        ):
            sample = apply_quality_transform(
                _source(), "RGB", "fail", "rgb_reflection_glare", 1, max_retries=8
            )

        self.assertEqual(attempts["n"], 3)
        self.assertIsNotNone(sample.image)


if __name__ == "__main__":
    unittest.main()
