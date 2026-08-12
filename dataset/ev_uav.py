import os
from utils import args
import numpy as np
from dataset.basedataset import BaseDataLoader

class EvUAV(BaseDataLoader):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)

        self.mode = mode
        self.data_dir = os.path.join(self.data_dir,mode)
        self.file_list = os.listdir(self.data_dir)
        self._cache = []
        for filename in self.file_list:
            with np.load(os.path.join(self.data_dir, filename)) as events:
                evs_norm = events['evs_norm']
                # Copy arrays before closing the NPZ archive so samples remain
                # fully resident in memory and no file handles stay open.
                self._cache.append((
                    evs_norm[:, :4].copy(),
                    events['ev_loc'].copy(),
                    evs_norm[:, 4].copy(),
                    evs_norm[:, 5].copy(),
                ))

    def __getitem__(self, num):
        evs_norm, ev_loc, seg_label, idx = self._cache[num]


        if self.mode=='train':
            num_events = ev_loc.shape[0]
            if num_events >= args.cfg.max_events_num:
                dowmsample_idx = np.random.choice(num_events,args.cfg.max_events_num,replace=False)
                ev_loc = ev_loc[dowmsample_idx]
                evs_norm=evs_norm[dowmsample_idx]
                seg_label = seg_label[dowmsample_idx]
                idx = idx[dowmsample_idx]


        out={}
        out['ev_loc']=ev_loc
        out['evs_norm']=evs_norm
        out['seg_label']=seg_label
        out['idx'] = idx

        return out


    def __len__(self):
        return len(self.file_list)
