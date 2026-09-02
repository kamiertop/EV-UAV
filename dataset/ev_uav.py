import os
from utils import args
import numpy as np
from dataset.basedataset import BaseDataLoader

class EvUAV(BaseDataLoader):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)

        self.mode = mode
        self.data_dir = os.path.join(self.data_dir,mode)
        self.file_list = sorted(
            filename for filename in os.listdir(self.data_dir)
            if filename.endswith(".npz")
        )
        self._cache = []
        for filename in self.file_list:
            with np.load(os.path.join(self.data_dir, filename)) as events:
                evs_norm = events['evs_norm']
                motion_target, motion_valid = self._trajectory_motion_targets(
                    events['ev_loc'], evs_norm[:, 4], evs_norm[:, 5],
                    getattr(configs, 'motion_bin_ms', 50.0),
                )
                # Copy arrays before closing the NPZ archive so samples remain
                # fully resident in memory and no file handles stay open.
                self._cache.append((
                    evs_norm[:, :4].copy(),
                    events['ev_loc'].copy(),
                    evs_norm[:, 4].copy(),
                    evs_norm[:, 5].copy(),
                    motion_target,
                    motion_valid,
                ))

    @staticmethod
    def _trajectory_motion_targets(ev_loc, labels, instance_ids, bin_ms):
        """Build per-event velocity targets in pixels/ms from track centroids."""
        target = np.zeros((len(ev_loc), 2), dtype=np.float32)
        valid = np.zeros(len(ev_loc), dtype=np.bool_)
        foreground_ids = np.unique(
            instance_ids[(labels > 0.5) & (instance_ids > 0)]
        )

        for instance_id in foreground_ids:
            event_indices = np.flatnonzero(
                (labels > 0.5) & (instance_ids == instance_id)
            )
            time = ev_loc[event_indices, 2]
            bins = np.floor((time - time.min()) / bin_ms).astype(np.int64)
            _, inverse = np.unique(bins, return_inverse=True)
            num_bins = inverse.max() + 1
            if num_bins < 2:
                continue

            counts = np.bincount(inverse, minlength=num_bins).clip(min=1)
            centroids = np.stack([
                np.bincount(
                    inverse, weights=ev_loc[event_indices, axis],
                    minlength=num_bins,
                ) / counts
                for axis in (0, 1)
            ], axis=1)
            centroid_time = np.bincount(
                inverse, weights=time, minlength=num_bins,
            ) / counts

            displacement = np.empty_like(centroids, dtype=np.float32)
            elapsed = np.empty(num_bins, dtype=np.float32)
            displacement[0] = centroids[1] - centroids[0]
            displacement[-1] = centroids[-1] - centroids[-2]
            elapsed[0] = centroid_time[1] - centroid_time[0]
            elapsed[-1] = centroid_time[-1] - centroid_time[-2]
            if num_bins > 2:
                displacement[1:-1] = centroids[2:] - centroids[:-2]
                elapsed[1:-1] = centroid_time[2:] - centroid_time[:-2]
            usable_bins = elapsed > 1e-3
            velocity = displacement / np.maximum(elapsed[:, None], 1e-6)
            event_usable = usable_bins[inverse]
            usable_events = event_indices[event_usable]
            target[usable_events] = velocity[inverse[event_usable]]
            valid[usable_events] = True
        return target, valid

    def __getitem__(self, num):
        evs_norm, ev_loc, seg_label, idx, motion_target, motion_valid = self._cache[num]


        if self.mode=='train':
            num_events = ev_loc.shape[0]
            if num_events >= self.configs.max_events_num:
                dowmsample_idx = np.random.choice(
                    num_events, self.configs.max_events_num, replace=False,
                )
                ev_loc = ev_loc[dowmsample_idx]
                evs_norm=evs_norm[dowmsample_idx]
                seg_label = seg_label[dowmsample_idx]
                idx = idx[dowmsample_idx]
                motion_target = motion_target[dowmsample_idx]
                motion_valid = motion_valid[dowmsample_idx]


        out={}
        out['ev_loc']=ev_loc
        out['evs_norm']=evs_norm
        out['seg_label']=seg_label
        out['idx'] = idx
        out['motion_target'] = motion_target
        out['motion_valid'] = motion_valid

        return out


    def __len__(self):
        return len(self.file_list)
