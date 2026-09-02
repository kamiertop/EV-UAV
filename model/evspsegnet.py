import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import spconv.pytorch as spconv
import functools
from spconv.pytorch import functional as Fsp
from model.basemodel import GDBlock
import HAIS_OP
from utils import args as _args
import math
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class patch_attention(spconv.SparseModule,):
    """Self-attention over pooled sparse patches, independently per sample.

    ``nn.MultiheadAttention`` expects a sequence dimension.  The original
    implementation inserted the singleton dimension before the sparse-patch
    dimension, making the sequence length equal to one.  Consequently every
    attention map was exactly ``[[1]]``.  This implementation explicitly
    groups patches by batch id and attends over each sample's patch sequence.
    """

    def __init__(self, channel,in_sptial_size,att_sptial_size=(11,9,8),indice_key='pa1', mode='sequence'):
        super(patch_attention, self).__init__()
        self.mode = mode
        maxpool_num = math.log2(int(in_sptial_size[0]/att_sptial_size[0]))
        self.layers = nn.ModuleList()
        for i in range(int(maxpool_num)):
            self.layers.append(spconv.SparseMaxPool3d((2,2,4),stride=(2,2,4),indice_key=indice_key+str(i)))

        num_heads = 1 if mode == "legacy" else (4 if channel % 4 == 0 else 1)
        self.position_mlp = None if mode == "legacy" else nn.Sequential(
            nn.Linear(3, channel), nn.GELU(), nn.Linear(channel, channel),
        )
        self.pre_norm = None if mode == "legacy" else nn.LayerNorm(channel)
        self.multihead_attn=nn.MultiheadAttention(
            embed_dim=channel, num_heads=num_heads, batch_first=True,
        )
        self.post_norm = None if mode == "legacy" else nn.LayerNorm(channel)
        self.inver_layer=nn.ModuleList()
        for i in range(int(maxpool_num)):
            self.inver_layer.append(spconv.SparseInverseConv3d(channel, channel, (2,2,4), indice_key=indice_key+str(int(maxpool_num-1)-i), bias=False,))
        self.conv = spconv.SubMConv3d(
            channel, channel, 1, stride=1,
            padding=1 if mode == "legacy" else 0, bias=False,
        )

    def _attend_features(self, features, indices, spatial_shape):
        """Dense part of patch attention, separated for CPU unit testing."""
        output = torch.empty_like(features)
        coords = indices[:, 1:].to(dtype=features.dtype)
        shape = torch.as_tensor(
            spatial_shape, dtype=features.dtype, device=features.device,
        ).clamp_min(1)
        position = self.position_mlp(coords / shape)
        batch_ids = indices[:, 0]

        for batch_id in batch_ids.unique(sorted=True):
            mask = batch_ids == batch_id
            # batch_first=True: (batch=1, sequence=patches, channels).
            sequence = self.pre_norm(features[mask] + position[mask]).unsqueeze(0)
            attended, _ = self.multihead_attn(
                sequence, sequence, sequence, need_weights=False,
            )
            output[mask] = self.post_norm(
                features[mask] + attended.squeeze(0)
            )
        return output

    def forward(self, x):
        identity = x.features
        for m in self.layers:
            x=m(x)
        if self.mode == "legacy":
            # Exact shape semantics of the released implementation: every
            # patch forms a separate batch with a one-token sequence.
            attended, _ = self.multihead_attn(
                x.features.unsqueeze(1), x.features.unsqueeze(1),
                x.features.unsqueeze(1), need_weights=False,
            )
            attended = attended.squeeze(1)
        else:
            attended = self._attend_features(
                x.features, x.indices, x.spatial_shape,
            )
        x = x.replace_feature(attended)
        for m in self.inver_layer:
            x=m(x)
        x = x.replace_feature(x.features + identity)
        x = self.conv(x)

        return x


def post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    indice_key=None,
    stride=1,
    padding=0,
    conv_type='subm',
    norm_fn=None,
    algo=None,
    dilations=(1, 2, 3, 4),
    ad_channels=16,
    gd_mode='fixed',
    route_temperature=1.0,
    motion_speed_scale=0.2,
):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False,algo=algo)
    elif conv_type == 'gd':
        conv = GDBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm_fn,
            dilations=dilations,
            indice_key=indice_key,
            bias=False,
            ad_channels=ad_channels,
            mode=gd_mode,
            route_temperature=route_temperature,
            motion_speed_scale=motion_speed_scale,
        )

    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature( self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature( out.features + identity)
        out = out.replace_feature(self.relu(out.features))

        return out

class evspsegnet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        input_channels = cfg.input_channel
        width=cfg.width

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, width, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(width),
            nn.ReLU(),
        )
        motion_gd = getattr(cfg, "motion_gd", "shallow")
        route_temperature = getattr(cfg, "route_temperature", 1.0)
        motion_speed_scale = getattr(cfg, "motion_speed_scale", 0.2)
        temporal_dilations = getattr(
            cfg, "motion_temporal_dilations", (1, 2, 4, 8),
        )
        motion_dilations = tuple(
            (spatial, spatial, temporal)
            for spatial, temporal in zip((1, 2, 3, 4), temporal_dilations)
        )

        def block(*block_args, gd_stage=None, **block_kwargs):
            use_motion = (
                motion_gd == "all"
                or (motion_gd == "encoder" and gd_stage and gd_stage.startswith("encoder"))
                or (motion_gd == "shallow" and gd_stage == "encoder1")
            )
            return post_act_block(
                *block_args,
                gd_mode="motion" if use_motion else "fixed",
                dilations=motion_dilations if use_motion else (1, 2, 3, 4),
                route_temperature=route_temperature,
                motion_speed_scale=motion_speed_scale,
                **block_kwargs,
            )

        self.conv1 = spconv.SparseSequential(
            block(
                width, width, 3, norm_fn=norm_fn, padding=1,
                indice_key='subm1', conv_type='gd', gd_stage='encoder1',
            ),
        )

        self.conv2 = spconv.SparseSequential(
            block(width, 2*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv2', conv_type='spconv'),
            block(
                2*width, 2*width, 3, norm_fn=norm_fn, padding=1,
                indice_key='subm2', conv_type='gd', ad_channels=16,
                gd_stage='encoder2',
            ),
        )
        attention_mode = getattr(cfg, "patch_attention", "sequence")
        self.pa2 = patch_attention(
            2*width, (176, 144, 2048), indice_key='pa2', mode=attention_mode,
        )

        self.conv3 = spconv.SparseSequential(

            block(2*width, 4*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv3', conv_type='spconv'),
            block(
                4*width, 4*width, 3, norm_fn=norm_fn, padding=1,
                indice_key='subm3', conv_type='gd', ad_channels=8,
                gd_stage='encoder3',
            ),
        )
        self.pa3 = patch_attention(
            4*width, (88, 72, 512), indice_key='pa3', mode=attention_mode,
        )

        self.conv4 = spconv.SparseSequential(

            block(4*width, 4*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv4', conv_type='spconv'),
            block(
                4*width, 4*width, 3, norm_fn=norm_fn, padding=1,
                indice_key='subm4', conv_type='gd', ad_channels=0,
                gd_stage='encoder4',
            ),
        )
        self.pa4 = patch_attention(
            4*width, (44, 36, 256), indice_key='pa4', mode=attention_mode,
        )

        # decoder
        self.conv_up_t4 = SparseBasicBlock(4*width, 4*width, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(
            8*width, 4*width, 3, norm_fn=norm_fn, padding=1,
            indice_key='subm4', conv_type='gd', gd_stage='decoder4',
        )
        self.inv_conv4 = block(4*width, 4*width, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')


        self.conv_up_t3 = SparseBasicBlock(4*width, 4*width, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(
            8*width, 4*width, 3, norm_fn=norm_fn, padding=1,
            indice_key='subm3', conv_type='gd', gd_stage='decoder3',
        )
        self.inv_conv3 = block(4*width, 2*width, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')


        self.conv_up_t2 = SparseBasicBlock(2*width, 2*width, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(
            4*width, 2*width, 3, norm_fn=norm_fn,
            indice_key='subm2', conv_type='gd', gd_stage='decoder2',
        )
        self.inv_conv2 = block(2*width, width, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')


        self.conv_up_t1 = SparseBasicBlock(width, width, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(
            2*width, width, 3, norm_fn=norm_fn,
            indice_key='subm1', conv_type='gd', gd_stage='decoder1',
        )

        self.conv5 = spconv.SparseSequential(
            block(
                width, width, 3, norm_fn=norm_fn, padding=1,
                indice_key='subm1', conv_type='gd', gd_stage='refine',
            )
        )

        self.semantic_linear = nn.Sequential(
            nn.Linear(width, 1),
            nn.Sigmoid()
        )
        # Training-only auxiliary predictions used by TACL.  They are kept
        # lightweight and can be ignored when using the original STC loss.
        loss_name = getattr(cfg, "loss", "stc")
        embedding_dim = getattr(cfg, "embedding_dim", 8)
        self.trajectory_embedding = (
            nn.Linear(width, embedding_dim) if loss_name == "tacl" else None
        )
        self.trajectory_motion = (
            nn.Linear(width, 2)
            if loss_name in {"tacl", "mbtc"} else None
        )
        self.motion_speed_scale = motion_speed_scale
        self.motion_gd = motion_gd
        self._motion_blocks = self._collect_motion_blocks()

    def _collect_motion_blocks(self):
        blocks = []
        for module in self.modules():
            if isinstance(module, GDBlock) and module.motion_prediction is not None:
                blocks.append(module)
            elif isinstance(module, GDBlock):
                gdconv = module.gdconv[0]
                if hasattr(gdconv, "motion_head"):
                    blocks.append(module)
        return blocks

    @staticmethod
    def _routing_summary(blocks):
        routing = [block.routing_weights for block in blocks]
        routing = [weights for weights in routing if weights is not None]
        if not routing:
            return None, None
        distributions = torch.stack([weights.mean(dim=0) for weights in routing])
        entropies = torch.stack([
            -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log())
            .sum(dim=1).mean()
            for weights in routing
        ])
        return distributions.mean(dim=0), entropies.mean()
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = x.replace_feature(torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.replace_feature( features.view(n, out_channels, -1).sum(dim=2))
        return x


    def forward(self, input):
        x = self.conv_input(input)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.pa2(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.pa3(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.pa4(x_conv4)

        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        features = x_up1.features
        output = self.semantic_linear(features)
        voxel = x_up1.replace_feature(output)
        shallow_motion = None
        if self._motion_blocks:
            candidate = self._motion_blocks[0].motion_prediction
            if candidate is not None:
                shallow_motion = candidate
        fallback_motion = None
        if self.trajectory_motion is not None:
            fallback_motion = (
                torch.tanh(self.trajectory_motion(features))
                * self.motion_speed_scale
            )
        route_distribution, route_entropy = self._routing_summary(self._motion_blocks)
        shallow_routing = None
        if self._motion_blocks:
            candidate = self._motion_blocks[0].routing_weights
            if candidate is not None:
                shallow_routing = candidate
        auxiliary = {
            "embedding": (
                torch_f.normalize(
                    self.trajectory_embedding(features), dim=-1, eps=1e-6,
                )
                if self.trajectory_embedding is not None else None
            ),
            # The final head is aligned with x_up1 and therefore with p2v_map.
            # The local router motion is exposed separately for diagnostics.
            "motion": fallback_motion if fallback_motion is not None else shallow_motion,
            "local_motion": shallow_motion,
            "routing": shallow_routing,
            "route_distribution": route_distribution,
            "route_entropy": route_entropy,
        }
        return output,voxel,auxiliary
