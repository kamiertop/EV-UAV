import os.path
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
import tqdm

from utils import args as _args


def run_test(model_path: str, cfg, *, device: str = "cuda:0") -> dict:
    """Load a checkpoint and evaluate on the test set.

    Returns a dict with keys ``iou``, ``seg_acc``, and optionally
    ``pd`` / ``fa`` (when ``cfg.roc`` is enabled).
    """
    from dataset.ev_uav import EvUAV
    from model.evspsegnet import evspsegnet

    net = evspsegnet(cfg).eval().to(device)
    try:
        net.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint architecture does not match the evaluation flags. "
            "Use the same --patch_attention, --motion_gd, --loss, and --width "
            "stored in the run's config.json."
        ) from error
    print(f"[eval] Loaded checkpoint: {model_path}")

    dataset = EvUAV(cfg, mode="test")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        num_workers=cfg.train_workers,
        pin_memory=True,
    )

    evaluator = evalute(cfg)
    pbar = tqdm.tqdm(
        total=len(loader), desc="Test", unit="video",
        unit_scale=True, position=0, leave=True,
    )

    for sample, ev in enumerate(loader):
        with torch.no_grad():
            x = dataset.voxelize_to_sparse(ev, device)
            label = ev["seg_label"].float().to(device)
            p2v_map = ev["p2v_map"].long().to(device)
            ev_locs = ev["locs"].float()
            idx = torch.as_tensor(ev["idx_label"], dtype=torch.long)
            ts = ev_locs[:, 3]

            preds, _, _ = net(x)
            preds = preds[p2v_map].squeeze().cpu()

            if cfg.eval:
                evaluator.matches[str(sample)] = {
                    "seg_pred": preds,
                    "seg_gt": label,
                    "trajectory_ids": idx,
                    "trajectory_times": ev_locs[:, 3].cpu(),
                    "trajectory_batches": ev_locs[:, 0].cpu(),
                }
                if cfg.roc:
                    evaluator.roc_update(
                        ts, preds, idx, label.cpu(), ev_locs,
                        thresh=cfg.prediction_thresh,
                    )

        pbar.update(1)

    results: dict = {}
    if cfg.eval:
        results["iou"] = evaluator.evaluate_semantic_segmantation_miou(
            thresh=cfg.prediction_thresh,
        )
        results["seg_acc"] = evaluator.evaluate_semantic_segmantation_accuracy(
            thresh=cfg.prediction_thresh, device=device,
        )
        results.update(evaluator.evaluate_trajectory_metrics(
            thresh=cfg.prediction_thresh,
            bin_ms=cfg.trajectory_bin_ms,
            correct_thresh=cfg.trajectory_correct_thresh,
        ))
        if cfg.roc:
            results["pd"], results["fa"] = evaluator.cal_roc()
    return results


class evalute():
    def __init__(self, cfg):
        self.matches = {}
        self.data = pd.DataFrame()
        self.prediction_thresh = getattr(cfg, "prediction_thresh", 0.9)

        if cfg.roc:
            self.pd_detT = cfg.pd_detT
            self.correct_thresh = cfg.correct_thresh
            self.whole_ev_num = 0  # 总共的事件点的数量
            self.obj_num = 0  # gt中目标的数量
            self.frame_num = 0  # 等效帧的数量
            self.correct_num = 0  # 不同置信度阈值下的正确预测的点的数量
            self.false_num = 0  # 不同置信度阈值下的虚警的点的数量

    def roc_update(self, ts, preds, idx, label, ev_locs, thresh=0.9):
        self.whole_ev_num += preds.shape[0]
        self.frame_num += int((ts.max() - ts.min()) / self.pd_detT)
        for i in range(int((ts.max() - ts.min()) / self.pd_detT + 1)):
            t_range = (ts > i * self.pd_detT) * (ts < (i + 1) * self.pd_detT)
            idx_frame, preds_frame, label_frame, ev_locs_frame = idx[t_range], preds[t_range], label[t_range], \
                ev_locs[:, 1:4][t_range]
            preds_frame_ori = preds_frame.clone()
            idx_list_frame = set(idx_frame.tolist())
            false_mask = np.zeros((260, 346), dtype=np.uint8)
            preds_frame[preds_frame_ori >= thresh] = 1
            preds_frame[preds_frame_ori < thresh] = 0

            # 计算检测Pd
            for idx_i in idx_list_frame:
                if idx_i != 0:  # 目标
                    self.obj_num += 1
                    preds_frame_i = preds_frame[idx_frame == idx_i]
                    label_frame_i = label_frame[idx_frame == idx_i]
                    num_correct_frame = (preds_frame_i == label_frame_i).sum()
                    if num_correct_frame / label_frame_i.sum() >= self.correct_thresh:  # 如果目标占比大于阈值，则认为检测准确
                        self.correct_num += 1

            # 计算虚警Fa
            false_ev = ev_locs_frame[(label_frame == 0) * (preds_frame == 1)]
            for ii in range(false_ev.shape[0]):
                false_mask[int(false_ev[:, 1][ii]), int(false_ev[:, 0][ii])] += 1  # 将虚警点映射为帧
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(false_mask, connectivity=8,
                                                                                    ltype=cv2.CV_32S)
            self.false_num += (num_labels - 1)

    def cal_roc(self):
        pd = self.correct_num / max(self.obj_num, 1)
        fa = self.false_num / max(self.frame_num * 346 * 260, 1)
        return pd, fa

    def evaluate_semantic_segmantation_miou(self, thresh=0.9):
        seg_gt_list = []
        seg_pred_list = []
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        seg_gt_all = torch.cat(seg_gt_list, dim=0).cpu()
        seg_pred_all = torch.cat(seg_pred_list, dim=0).cpu()
        seg_pred_all[seg_pred_all >= thresh] = 1
        seg_pred_all[seg_pred_all < thresh] = 0
        assert seg_gt_all.shape == seg_pred_all.shape
        iou_list = []
        for _index in seg_gt_all.unique():
            if _index == 1:
                intersection = ((seg_gt_all == _index) & (seg_pred_all == _index)).sum()
                union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
                iou = intersection.float() / union
                iou_list.append(iou)
        if not iou_list:
            return torch.tensor(0.0)
        iou_tensor = torch.stack(iou_list)
        miou = iou_tensor.mean()
        return miou

    def evaluate_semantic_segmantation_accuracy(self, thresh=0.9, device=None):
        device = device or f"cuda:{_args.cfg.gpu}"
        seg_gt_list = []
        seg_pred_list = []
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        seg_gt_all = torch.cat(seg_gt_list, dim=0).to(device)
        seg_pred_all = torch.cat(seg_pred_list, dim=0).to(device)
        seg_pred_all[seg_pred_all >= thresh] = 1
        seg_pred_all[seg_pred_all < thresh] = 0
        assert seg_gt_all.shape == seg_pred_all.shape
        correct = (seg_gt_all[seg_gt_all == 1] == seg_pred_all[seg_gt_all == 1]).sum()
        whole = (seg_gt_all == 1).sum()
        seg_accuracy = correct.float() / whole.float()
        return seg_accuracy

    def evaluate_trajectory_metrics(
        self,
        thresh=0.9,
        bin_ms=50.0,
        correct_thresh=0.1,
        min_events=2,
    ):
        """Report target temporal coverage, longest run, and fragmentation."""
        coverages = []
        longest_ratios = []
        fragmentations = []
        for record in self.matches.values():
            pred = torch.as_tensor(record["seg_pred"]).reshape(-1).numpy()
            ids = np.asarray(record.get("trajectory_ids", []))
            times = torch.as_tensor(
                record.get("trajectory_times", []),
            ).reshape(-1).numpy()
            batches = torch.as_tensor(
                record.get("trajectory_batches", np.zeros_like(ids)),
            ).reshape(-1).numpy()
            if (
                pred.shape[0] != ids.shape[0]
                or pred.shape[0] != times.shape[0]
                or pred.shape[0] != batches.shape[0]
            ):
                continue
            predicted = pred >= thresh
            for batch_id in np.unique(batches):
                sample = batches == batch_id
                for instance_id in np.unique(ids[sample & (ids > 0)]):
                    foreground = sample & (ids == instance_id)
                    if int(foreground.sum()) < min_events:
                        continue
                    track_times = times[foreground]
                    track_predictions = predicted[foreground]
                    bins = np.floor(
                        (track_times - track_times.min())
                        / max(float(bin_ms), 1e-6)
                    ).astype(np.int64)
                    occupied = np.unique(bins)
                    detected = np.asarray([
                        track_predictions[bins == bin_id].mean() >= correct_thresh
                        for bin_id in occupied
                    ], dtype=bool)
                    if detected.size == 0:
                        continue
                    coverages.append(float(detected.mean()))
                    padded = np.pad(detected.astype(np.int8), (1, 1))
                    starts = np.flatnonzero(
                        (padded[1:] == 1) & (padded[:-1] == 0)
                    )
                    ends = np.flatnonzero(
                        (padded[:-1] == 1) & (padded[1:] == 0)
                    )
                    longest = int((ends - starts).max()) if starts.size else 0
                    longest_ratios.append(float(longest / detected.size))
                    fragmentations.append(float(starts.size / detected.size))

        if not coverages:
            return {
                "trajectory_coverage": 0.0,
                "trajectory_longest_ratio": 0.0,
                "trajectory_fragmentation": 0.0,
                "trajectory_count": 0,
            }
        return {
            "trajectory_coverage": float(np.mean(coverages)),
            "trajectory_longest_ratio": float(np.mean(longest_ratios)),
            "trajectory_fragmentation": float(np.mean(fragmentations)),
            "trajectory_count": len(coverages),
        }
