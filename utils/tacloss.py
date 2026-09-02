"""Trajectory-Aware Contrastive Loss (TACL) for EV-UAV.

TACL treats foreground events as samples from object trajectories rather than
independent binary labels.  It combines calibrated point classification with
three trajectory priors already available in EV-UAV's event annotations:

* locally isolated positives and locally coherent false positives are hard;
* events sharing an instance id should have a compact embedding;
* the predicted local motion should follow the instance centroid trajectory.

The instance id and motion targets are only used during training; inference
still requires raw events alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class TrajectoryAwareContrastiveLoss(nn.Module):
    def __init__(self, cfg, weight_clip_eps: float = 1e-5):
        super().__init__()
        self.k = cfg.k
        self.t = cfg.t
        self.volume = float(self.k * self.k * self.t)
        self.eps = weight_clip_eps

        self.focal_gamma = cfg.focal_gamma
        self.tversky_alpha = cfg.tversky_alpha
        self.tversky_beta = cfg.tversky_beta
        self.pos_balance_max = cfg.pos_balance_max
        self.hard_pos_weight = cfg.hard_pos_weight
        self.hard_neg_weight = cfg.hard_neg_weight
        self.tversky_weight = cfg.tversky_weight
        self.instance_weight = cfg.instance_weight
        self.motion_weight = cfg.motion_weight
        self.embedding_margin = cfg.embedding_margin
        self.background_margin = cfg.background_margin
        self.motion_bin_ms = cfg.motion_bin_ms
        self.max_background_embeddings = cfg.max_background_embeddings

        self.correlation_conv = spconv.SubMConv3d(
            1, 1, kernel_size=[self.k, self.k, self.t], stride=1,
            padding=[self.k // 2, self.k // 2, self.t // 2], bias=False,
        )
        self.correlation_conv.weight.data.fill_(1)
        self.correlation_conv.requires_grad_(False)

    def forward(
        self, voxel, p2v_map, predictions, label, auxiliary, locations,
        instance_ids, motion_target, motion_valid,
    ):
        probabilities = predictions[p2v_map].reshape(-1).clamp(self.eps, 1 - self.eps)
        label = label.reshape(-1).to(probabilities.dtype)

        correlation_voxel = self.correlation_conv(voxel)
        correlation = (
            correlation_voxel.features[p2v_map].reshape(-1) / self.volume
        ).clamp(0, 1)

        segmentation = self._segmentation_loss(probabilities, label, correlation)
        embeddings = auxiliary["embedding"][p2v_map]
        motion = auxiliary["motion"][p2v_map]
        instances = self._instance_contrastive_loss(
            embeddings, label, instance_ids, locations[:, 0].long(),
        )
        motion_consistency = self._motion_loss(
            motion, motion_target, motion_valid,
        )

        total = (
            segmentation
            + self.instance_weight * instances
            + self.motion_weight * motion_consistency
        )
        components = {
            "seg": segmentation.detach(),
            "instance": instances.detach(),
            "motion": motion_consistency.detach(),
            "correlation": correlation.detach().mean(),
        }
        return total, components

    def _segmentation_loss(self, probabilities, label, correlation):
        positives = label.sum().clamp_min(1)
        negatives = (1 - label).sum().clamp_min(1)
        positive_balance = torch.sqrt(negatives / positives).clamp(
            min=1, max=self.pos_balance_max,
        )

        # Missing pieces of a thin trajectory are hard positives. Conversely,
        # coherent background predictions are precisely the false-alarm mode
        # that the original (1-w) negative weighting tends to under-penalise.
        positive_hardness = 1 + self.hard_pos_weight * (1 - correlation)
        negative_hardness = 1 + self.hard_neg_weight * correlation
        positive_loss = -positive_balance * positive_hardness * (
            1 - probabilities
        ).pow(self.focal_gamma) * torch.log(probabilities)
        negative_loss = -negative_hardness * probabilities.pow(
            self.focal_gamma
        ) * torch.log1p(-probabilities)
        focal = torch.where(label > 0.5, positive_loss, negative_loss).mean()

        true_positive = (probabilities * label).sum()
        false_positive = (probabilities * (1 - label)).sum()
        false_negative = ((1 - probabilities) * label).sum()
        tversky = (true_positive + self.eps) / (
            true_positive
            + self.tversky_alpha * false_positive
            + self.tversky_beta * false_negative
            + self.eps
        )
        return focal + self.tversky_weight * (1 - tversky).pow(self.focal_gamma)

    def _instance_contrastive_loss(self, embeddings, label, instance_ids, batch_ids):
        zero = embeddings.sum() * 0
        losses = []
        for batch_id in batch_ids.unique(sorted=True):
            sample = batch_ids == batch_id
            foreground = sample & (label > 0.5) & (instance_ids > 0)
            ids = instance_ids[foreground].unique(sorted=True)
            if ids.numel() == 0:
                continue

            means = []
            pull_terms = []
            for instance_id in ids:
                instance_embedding = embeddings[foreground & (instance_ids == instance_id)]
                mean = instance_embedding.mean(dim=0)
                means.append(mean)
                pull_terms.append((instance_embedding - mean).pow(2).sum(dim=1).mean())
            means = torch.stack(means)
            loss = torch.stack(pull_terms).mean()

            if means.shape[0] > 1:
                distances = torch.pdist(means, p=2)
                loss = loss + F.relu(self.embedding_margin - distances).pow(2).mean()

            background = embeddings[sample & (label <= 0.5)]
            if background.shape[0] > 0:
                if background.shape[0] > self.max_background_embeddings:
                    # Evenly spaced deterministic sampling preserves reproducibility.
                    select = torch.linspace(
                        0, background.shape[0] - 1,
                        self.max_background_embeddings,
                        device=background.device,
                    ).long()
                    background = background[select]
                nearest = torch.cdist(background, means).min(dim=1).values
                loss = loss + F.relu(self.background_margin - nearest).pow(2).mean()
            losses.append(loss)
        return torch.stack(losses).mean() if losses else zero

    def _motion_loss(self, predicted_motion, target, valid):
        if not valid.any():
            return predicted_motion.sum() * 0
        predicted = F.normalize(predicted_motion[valid], dim=-1, eps=1e-6)
        return (1 - (predicted * target[valid]).sum(dim=-1)).mean()
