import unittest
from argparse import Namespace

import numpy as np
import torch

from utils.eval import evalute


class TrajectoryMetricsTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = evalute(Namespace(roc=False, prediction_thresh=0.9))

    def test_complete_track_has_full_coverage(self):
        self.evaluator.matches = {
            "0": {
                "seg_pred": torch.full((8,), 0.95),
                "seg_gt": torch.ones(8),
                "trajectory_ids": np.ones(8),
                "trajectory_times": torch.tensor(
                    [0, 1, 10, 11, 20, 21, 30, 31],
                ),
            }
        }
        metrics = self.evaluator.evaluate_trajectory_metrics(
            thresh=0.9, bin_ms=10, correct_thresh=0.5,
        )
        self.assertEqual(metrics["trajectory_coverage"], 1.0)
        self.assertEqual(metrics["trajectory_longest_ratio"], 1.0)
        self.assertAlmostEqual(metrics["trajectory_fragmentation"], 0.25)

    def test_gap_reduces_longest_run(self):
        predictions = torch.tensor([
            0.95, 0.95, 0.95, 0.95,
            0.05, 0.05,
            0.95, 0.95,
        ])
        self.evaluator.matches = {
            "0": {
                "seg_pred": predictions,
                "seg_gt": torch.ones(8),
                "trajectory_ids": np.ones(8),
                "trajectory_times": torch.tensor(
                    [0, 1, 10, 11, 20, 21, 30, 31],
                ),
            }
        }
        metrics = self.evaluator.evaluate_trajectory_metrics(
            thresh=0.9, bin_ms=10, correct_thresh=0.5,
        )
        self.assertEqual(metrics["trajectory_coverage"], 0.75)
        self.assertEqual(metrics["trajectory_longest_ratio"], 0.5)
        self.assertEqual(metrics["trajectory_fragmentation"], 0.5)


if __name__ == "__main__":
    unittest.main()
