import unittest
from argparse import Namespace

import numpy as np
import torch

from dataset.ev_uav import EvUAV
from model.evspsegnet import patch_attention
from utils.tacloss import TrajectoryAwareContrastiveLoss
from utils.mbtcloss import MotionBidirectionalTrajectoryLoss


def loss_config():
    return Namespace(
        k=3, t=3, focal_gamma=2.0, tversky_alpha=0.6,
        tversky_beta=0.4, pos_balance_max=8.0, hard_pos_weight=1.0,
        hard_neg_weight=2.0, tversky_weight=1.0, instance_weight=0.1,
        motion_weight=0.1, embedding_margin=1.0,
        background_margin=0.75, motion_bin_ms=10.0,
        max_background_embeddings=32,
        support_weight=1.0, trajectory_weight=0.25, route_weight=0.05,
        motion_direction_weight=0.1, motion_speed_weight=0.1,
        trajectory_bin_ms=10.0, min_trajectory_events=2,
        motion_speed_scale=0.2,
    )


class MotionTargetTest(unittest.TestCase):
    def test_straight_track_points_in_positive_x(self):
        locations = np.array([
            [0, 4, 0], [1, 4, 1],
            [10, 4, 10], [11, 4, 11],
            [20, 4, 20], [21, 4, 21],
        ], dtype=np.int64)
        labels = np.ones(6, dtype=np.float32)
        instances = np.ones(6, dtype=np.float32)

        target, valid = EvUAV._trajectory_motion_targets(
            locations, labels, instances, bin_ms=10,
        )

        self.assertTrue(valid.all())
        np.testing.assert_allclose(target[:, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(target[:, 1], 0.0, atol=1e-6)

    def test_background_has_no_motion_target(self):
        locations = np.array([[0, 0, 0], [1, 1, 10]], dtype=np.int64)
        target, valid = EvUAV._trajectory_motion_targets(
            locations, np.zeros(2), np.zeros(2), bin_ms=10,
        )
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(target, 0)


class PatchAttentionTest(unittest.TestCase):
    def test_samples_do_not_leak_into_each_other(self):
        torch.manual_seed(4)
        module = patch_attention(8, (11, 9, 8), mode="sequence")
        features = torch.randn(7, 8)
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0],
            [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0],
        ], dtype=torch.int32)
        original = module._attend_features(features, indices, [4, 2, 2])
        modified = features.clone()
        modified[4:] += 100
        changed = module._attend_features(modified, indices, [4, 2, 2])
        torch.testing.assert_close(original[:4], changed[:4])

    def test_context_changes_a_patch(self):
        torch.manual_seed(5)
        module = patch_attention(8, (11, 9, 8), mode="sequence")
        features = torch.randn(3, 8)
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
        ], dtype=torch.int32)
        original = module._attend_features(features, indices, [3, 1, 1])
        modified = features.clone()
        modified[2, 0] += 20
        changed = module._attend_features(modified, indices, [3, 1, 1])
        self.assertGreater((original[0] - changed[0]).abs().max().item(), 1e-5)


class TACLTest(unittest.TestCase):
    def setUp(self):
        self.loss = TrajectoryAwareContrastiveLoss(loss_config())

    def test_segmentation_prefers_correct_predictions(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        correlation = torch.tensor([0.8, 0.4, 0.8, 0.2])
        correct = torch.tensor([0.9, 0.8, 0.1, 0.2])
        wrong = 1 - correct
        self.assertLess(
            self.loss._segmentation_loss(correct, labels, correlation),
            self.loss._segmentation_loss(wrong, labels, correlation),
        )

    def test_motion_loss_is_directional(self):
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        valid = torch.tensor([True, True])
        aligned = self.loss._motion_loss(target.clone(), target, valid)
        opposite = self.loss._motion_loss(-target, target, valid)
        self.assertLess(aligned, opposite)
        self.assertAlmostEqual(aligned.item(), 0.0, places=6)

    def test_instance_loss_backpropagates(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, 0.2], [0.0, 1.0],
        ], requires_grad=True)
        labels = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        instances = torch.tensor([1, 1, 2, 2, 0])
        batches = torch.zeros(5, dtype=torch.long)
        value = self.loss._instance_contrastive_loss(
            embeddings, labels, instances, batches,
        )
        value.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())


class MBTCTest(unittest.TestCase):
    def setUp(self):
        self.loss = MotionBidirectionalTrajectoryLoss(loss_config())

    def test_hard_structure_receives_more_weight(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        probabilities = torch.tensor([0.8, 0.8, 0.2, 0.2])
        easy = torch.zeros(4)
        hard = torch.ones(4)
        self.assertGreater(
            self.loss._segmentation_loss(probabilities, labels, hard),
            self.loss._segmentation_loss(probabilities, labels, easy),
        )

    def test_motion_loss_distinguishes_speed_and_direction(self):
        target = torch.tensor([[0.1, 0.0], [0.0, 0.05]])
        valid = torch.tensor([True, True])
        aligned = self.loss._motion_loss(target, target, valid)
        opposite = self.loss._motion_loss(-target, target, valid)
        self.assertLess(aligned[0], opposite[0])
        self.assertAlmostEqual(aligned[1].item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
