import unittest
import numpy as np
from rlsm_immtrack.metrics import frame_statistics, summarize_frames


class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        gt = np.array([[0., 0., 10., 20.], [5., 5., 10., 10.]])
        metrics = summarize_frames(frame_statistics(gt.copy(), gt))
        self.assertEqual(metrics["success_auc"], 1.)
        self.assertEqual(metrics["normalized_precision_auc"], 1.)
        self.assertEqual(metrics["precision_20px"], 1.)

    def test_absent_frames_are_excluded(self):
        gt = np.array([[0., 0., 10., 10.], [0., 0., 0., 0.], [np.nan, 0, 0, 0]])
        stats = frame_statistics(np.zeros((3, 4)), gt)
        self.assertEqual(len(stats["iou"]), 1)
        self.assertEqual(stats["iou"][0], 0.)

    def test_threshold_zero_is_included(self):
        gt = np.array([[0., 0., 10., 10.]])
        pred = np.array([[100., 100., 10., 10.]])
        result = summarize_frames(frame_statistics(pred, gt))
        self.assertAlmostEqual(result["success_auc"], 1 / 21)
        self.assertEqual(result["iou_lt_0.1"], 1.)

    def test_no_valid_ground_truth_raises(self):
        with self.assertRaises(ValueError):
            frame_statistics(np.zeros((1, 4)), np.zeros((1, 4)))


if __name__ == "__main__":
    unittest.main()
