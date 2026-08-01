from __future__ import annotations

import unittest

import numpy as np

from quality_fail_augment.augment import GLARE_WIDE_ANCHOR_LIMIT, _defect_anchors


class DefectAnchorTests(unittest.TestCase):
    def test_anchors_land_on_the_mask(self) -> None:
        """빈 곳을 조준하면 확장 탐색이 의미가 없다. 앵커는 항상 결함 화소여야 한다."""
        mask = np.zeros((200, 200), dtype=bool)
        for centre in (20, 90, 170):
            mask[centre - 3 : centre + 3, centre - 3 : centre + 3] = True

        anchors = _defect_anchors(mask, GLARE_WIDE_ANCHOR_LIMIT)

        self.assertTrue(anchors)
        for x, y in anchors:
            self.assertTrue(mask[int(y), int(x)])

    def test_scattered_defect_yields_spread_anchors(self) -> None:
        """흩어진 결함에서 앵커가 한 덩어리에 몰리면 전체 중심과 다를 바 없다."""
        mask = np.zeros((200, 200), dtype=bool)
        for centre in (20, 90, 170):
            mask[centre - 3 : centre + 3, centre - 3 : centre + 3] = True

        anchors = _defect_anchors(mask, GLARE_WIDE_ANCHOR_LIMIT)

        self.assertGreaterEqual(len(anchors), 2)
        spread = max(anchors)[0] - min(anchors)[0] + max(anchors)[1] - min(anchors)[1]
        self.assertGreater(spread, 100)

    def test_single_clump_collapses_to_one_anchor(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[28:32, 28:32] = True

        anchors = _defect_anchors(mask, GLARE_WIDE_ANCHOR_LIMIT)

        self.assertTrue(anchors)
        for x, y in anchors:
            self.assertTrue(mask[int(y), int(x)])

    def test_empty_mask_returns_no_anchors(self) -> None:
        self.assertEqual(_defect_anchors(np.zeros((10, 10), dtype=bool), 4), [])

    def test_anchor_order_is_deterministic(self) -> None:
        """plan 이 고정돼 있어도 앵커 순서가 흔들리면 산출물이 재현되지 않는다."""
        rng = np.random.default_rng(7)
        mask = rng.random((120, 120)) > 0.97

        first = _defect_anchors(mask, GLARE_WIDE_ANCHOR_LIMIT)
        second = _defect_anchors(mask, GLARE_WIDE_ANCHOR_LIMIT)

        self.assertEqual(first, second)

    def test_limit_is_respected(self) -> None:
        rng = np.random.default_rng(11)
        mask = rng.random((120, 120)) > 0.95

        self.assertLessEqual(len(_defect_anchors(mask, 2)), 2)


if __name__ == "__main__":
    unittest.main()
