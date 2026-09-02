"""Motion-Bidirectional Trajectory Consistency loss for EV-UAV.

The loss is deliberately independent of persistent homology.  It uses labels
already supplied by EV-UAV to make two errors expensive at the same time:

* a positive event that breaks a target's local trajectory (a hard positive);
* a background event that belongs to a locally coherent predicted structure
  (a hard negative).

The same local support operator is used for predictions and labels.  An
auxiliary velocity head is trained only on foreground events, and confidence
is regularised over temporal bins of each annotated instance.  All auxiliary
branches are discarded at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class MotionBidirectionalTrajectoryLoss(nn.Module):
    """Structure-aware segmentation plus motion and track consistency terms."""

    def __init__(self, cfg, weight_clip_eps: float = 1e-5):
        super().__init__()
        self.k = int(cfg.k)
        self.t = int(cfg.t)
        self.volume = float(self.k * self.k * self.t)
        self.eps = weight_clip_eps
        self.gamma = float(cfg.focal_gamma)
        self.pos_balance_max = float(cfg.pos_balance_max)
        self.support_weight = float(cfg.support_weight)
        self.trajectory_weight = float(cfg.trajectory_weight)
        self.motion_direction_weight = float(cfg.motion_direction_weight)
        self.motion_speed_weight = float(cfg.motion_speed_weight)
        self.route_weight = float(cfg.route_weight)
        self.trajectory_bin_ms = float(cfg.trajectory_bin_ms)
        self.min_trajectory_events = int(cfg.min_trajectory_events)
        self.motion_speed_scale = float(cfg.motion_speed_scale)

        self.support_conv = spconv.SubMConv3d(
            1,
            1,
            kernel_size=[self.k, self.k, self.t],
            stride=1,
            padding=[self.k // 2, self.k // 2, self.t // 2],
            bias=False,
        )
        with torch.no_grad():
            self.support_conv.weight.fill_(1.0)
        for parameter in self.support_conv.parameters():
            parameter.requires_grad_(False)

    def _support(self, voxel, values, p2v_map):
        value_voxel_features = values.new_zeros(voxel.features.shape[0])
        value_voxel_features.scatter_add_(0, p2v_map, values)
        counts = values.new_zeros(voxel.features.shape[0])
        counts.scatter_add_(0, p2v_map, torch.ones_like(values))
        value_voxel = voxel.replace_feature(
            (value_voxel_features / counts.clamp_min(1.0)).reshape(-1, 1)
        )
        support = self.support_conv(value_voxel).features[:, 0]
        return support[p2v_map].clamp(0, self.volume) / self.volume

    def forward(
        self,
        voxel,
        p2v_map,
        predictions,
        label,
        auxiliary,
        locations,
        instance_ids,
        motion_target,
        motion_valid,
    ):
        probabilities = predictions[p2v_map].reshape(-1).clamp(
            self.eps, 1.0 - self.eps,
        )
        label = label.reshape(-1).to(probabilities.dtype)
        p2v_map = p2v_map.reshape(-1).long()

        # Ground-truth support is detached and used as a difficulty signal;
        # predicted support remains differentiable through the probabilities.
        pred_support = self._support(voxel, probabilities, p2v_map)
        gt_support = self._support(voxel, label, p2v_map).detach()
        # Positive gaps are hard according to GT support; coherent false
        # positives are hard according to the current prediction support.
        hard_structure = (
            label * (1.0 - gt_support)
            + (1.0 - label) * pred_support
        )
        segmentation = self._segmentation_loss(
            probabilities, label, hard_structure,
        )

        trajectory = self._trajectory_consistency(
            probabilities,
            label,
            locations,
            instance_ids,
        )
        predicted_motion = auxiliary.get("local_motion")
        if (
            predicted_motion is None
            or predicted_motion.shape[0] <= int(p2v_map.max().item())
        ):
            predicted_motion = auxiliary.get("motion")
        if predicted_motion is not None:
            predicted_motion = predicted_motion[p2v_map]
        motion = self._motion_loss(predicted_motion, motion_target, motion_valid)
        route = self._route_regularizer(auxiliary.get("routing"), probabilities)
        total = (
            segmentation
            + self.trajectory_weight * trajectory
            + self.motion_direction_weight * motion[0]
            + self.motion_speed_weight * motion[1]
            + self.route_weight * route
        )
        return total, {
            "seg": segmentation.detach(),
            "trajectory": trajectory.detach(),
            "motion_direction": motion[0].detach(),
            "motion_speed": motion[1].detach(),
            "route_balance": route.detach(),
            "pred_support": pred_support.detach().mean(),
            "gt_support": gt_support.detach().mean(),
            "hard_structure": hard_structure.detach().mean(),
        }

    def _segmentation_loss(self, probabilities, label, hard_structure):
        positives = label.sum().clamp_min(1.0)
        negatives = (1.0 - label).sum().clamp_min(1.0)
        balance = torch.sqrt(negatives / positives).clamp(
            1.0, self.pos_balance_max,
        )
        weight = 1.0 + self.support_weight * hard_structure
        positive = -balance * weight * (1 - probabilities).pow(self.gamma) * (
            probabilities.log()
        )
        negative = -weight * probabilities.pow(self.gamma) * (
            (1 - probabilities).log()
        )
        return torch.where(label > 0.5, positive, negative).mean()

    def _trajectory_consistency(self, probabilities, label, locations, instance_ids):
        """Penalise confidence valleys between adjacent bins of each target."""
        zero = probabilities.sum() * 0.0
        if locations.numel() == 0:
            return zero
        batch_ids = locations[:, 0].long()
        times = locations[:, 3].to(probabilities.dtype)
        losses = []
        foreground = (label > 0.5) & (instance_ids > 0)
        for batch_id in batch_ids.unique(sorted=True):
            sample = batch_ids == batch_id
            for instance_id in instance_ids[sample & foreground].unique(sorted=True):
                mask = sample & foreground & (instance_ids == instance_id)
                if int(mask.sum()) < self.min_trajectory_events:
                    continue
                local_probabilities = probabilities[mask]
                local_times = times[mask]
                bins = torch.floor(
                    (local_times - local_times.min()) / max(self.trajectory_bin_ms, 1e-3)
                ).long()
                num_bins = int(bins.max().item()) + 1
                if num_bins < 2:
                    continue
                sums = local_probabilities.new_zeros(num_bins)
                sums.scatter_add_(0, bins, local_probabilities)
                counts = local_probabilities.new_zeros(num_bins)
                counts.scatter_add_(0, bins, torch.ones_like(local_probabilities))
                occupied = counts > 0
                confidence = (sums / counts.clamp_min(1.0))[occupied]
                if confidence.numel() < 2:
                    continue
                # Adjacent target bins should not contain a sharp confidence
                # valley; this specifically targets fragmented trajectories.
                losses.append(F.relu(0.75 - confidence[:-1]).mean())
                losses.append((confidence[1:] - confidence[:-1]).pow(2).mean())
        return torch.stack(losses).mean() if losses else zero

    def _motion_loss(self, predicted, target, valid):
        zero = target.sum() * 0.0
        if predicted is None or not bool(valid.any()):
            return zero, zero
        predicted = predicted[valid]
        target = target.to(predicted.dtype)[valid]
        # Direction is scale-free, while the Huber term retains speed cues.
        pred_direction = F.normalize(predicted, dim=-1, eps=1e-6)
        target_direction = F.normalize(target, dim=-1, eps=1e-6)
        direction = (
            1.0 - (pred_direction * target_direction).sum(dim=-1)
        ).mean()
        speed = F.smooth_l1_loss(predicted, target, beta=0.02)
        return direction, speed

    @staticmethod
    def _route_regularizer(routing, reference):
        if routing is None:
            return reference.sum() * 0.0
        # Keep the batch-average usage balanced without forcing every event to
        # use every branch.  The task losses still decide local assignments.
        mean_usage = routing.mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        return (mean_usage - uniform).pow(2).sum()
