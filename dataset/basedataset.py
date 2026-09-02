import torch
import numpy as np
from torch.autograd import Function
import HAIS_OP
import spconv.pytorch as spconv
from utils import args

class Voxelization_Idx(Function):
    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.zeros(N, dtype=torch.int32)
        output_map = input_map.new()

        HAIS_OP.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None
voxelization_idx = Voxelization_Idx.apply

class Voxelization(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.zeros(M, C, device=feats.device)

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        HAIS_OP.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.zeros(N, C, device=d_output_feats.device)

        HAIS_OP.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None
voxelization = Voxelization.apply




class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, configs):
        self.configs = configs
        self.data_dir = configs.data_dir
        self.whole_t = configs.whole_t
        self.res = configs.res

    def custom_collate(self, batch):

        batch_size = len(batch)
        loc_batches=[]
        feature_batches=[]
        seg_label_batches=[]
        idx_label_batches = []
        motion_target_batches=[]
        motion_valid_batches=[]

        for i,ev in enumerate(batch):
            ev_loc = ev['ev_loc']
            loc = np.hstack((i * np.ones((ev_loc.shape[0], 1)), ev_loc))
            loc_batches.append(loc)

            feature = ev['evs_norm'][:,0:4]
            feature_batches.append(feature)

            seg_label = ev['seg_label']

            seg_label_batches.append(seg_label)

            idx_label =ev['idx']
            idx_label_batches.append(idx_label)
            motion_target_batches.append(ev['motion_target'])
            motion_valid_batches.append(ev['motion_valid'])





        locs_batches = np.concatenate(loc_batches, axis=0)
        seg_label_batches = np.concatenate(seg_label_batches, axis=0)
        idx_label_batches = np.concatenate(idx_label_batches, axis=0)


        locs_batches = torch.from_numpy(locs_batches).to(torch.int64).contiguous()
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs_batches, batch_size, 4)

        feature_batches = torch.from_numpy(np.concatenate(feature_batches, axis=0)).contiguous()
        feature_batches =feature_batches.float()

        output = {}

        output['features'] = feature_batches
        output['voxel_locs'] = voxel_locs
        output['v2p_map'] = v2p_map
        output['batch_size'] = batch_size
        output['seg_label'] = torch.from_numpy(seg_label_batches)
        output['p2v_map'] = p2v_map
        output['locs'] = locs_batches
        output['idx_label'] = idx_label_batches
        output['motion_target'] = torch.from_numpy(
            np.concatenate(motion_target_batches, axis=0)
        ).float()
        output['motion_valid'] = torch.from_numpy(
            np.concatenate(motion_valid_batches, axis=0)
        ).bool()

        return output

    def voxelize_to_sparse(self, batch, device):
        voxel_feats = voxelization(
            batch['features'].to(device), batch['v2p_map'].to(device), 4
        )
        spatial_shape = np.array([11*32, 9*32, 256*32])
        return spconv.SparseConvTensor(
            voxel_feats, batch['voxel_locs'].int().to(device),
            spatial_shape, batch['batch_size'],
        )
