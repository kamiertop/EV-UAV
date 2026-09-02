import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import  SparseModule


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_c
        width = cfg.width

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels,norm_fn, stride=1):
        super(Shortcut, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SparseAgvPool(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x_features = x.features
        agv = torch.mean(x_features,dim=0)
        return agv


class SEModule(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = SparseAgvPool()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc(se)
        x = x.replace_feature(se * x.features)
        return x


class GDConv(SparseModule):
    def __init__(self, out_channels, norm_fn, stride=1, dilations=(1, 2, 3, 4), indice_key=None):
        super().__init__()
        num_splits = len(dilations)
        assert (out_channels % num_splits == 0)
        temp = out_channels // num_splits
        convs = []
        for d in dilations:
            convs.append(spconv.SubMConv3d(temp, temp, kernel_size=3, padding=d, dilation=d,stride=stride))
        self.convs = nn.ModuleList(convs)
        self.num_splits=num_splits
        self.temp = temp

    def forward(self,x):
        res = []
        for i in range(self.num_splits):
            features_split = x.features[:,i*self.temp:(i+1)*self.temp]
            x_split = spconv.SparseConvTensor(features_split,x.indices,x.spatial_shape,x.batch_size)
            res.append(self.convs[i](x_split).features)
        x_features = torch.cat(res,dim=1)
        x = x.replace_feature(x_features)

        return x


class MotionConditionedGDConv(SparseModule):
    """Grouped dilated sparse convolution with event-wise branch routing.

    The released GDSC block assigns one fixed channel group to every dilation
    branch.  This variant preserves that inexpensive grouped computation, then
    predicts a local velocity and uses it together with the point feature to
    gate the branch groups for every active voxel.  A zero-initialised router
    starts from uniform weights, so the initial feature scale matches GDConv.

    ``last_motion`` and ``last_routing`` are transient tensors from the latest
    forward pass.  EV-SpSegNet uses the full-resolution motion prediction for
    auxiliary trajectory supervision and logs the routing distribution.
    """

    def __init__(
        self,
        out_channels,
        norm_fn,
        stride=1,
        dilations=(1, 2, 3, 4),
        indice_key=None,
        route_temperature=1.0,
        motion_speed_scale=0.2,
    ):
        super().__init__()
        del norm_fn, indice_key  # Kept for the same constructor contract as GDConv.
        self.num_splits = len(dilations)
        if out_channels % self.num_splits != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by "
                f"{self.num_splits} motion branches"
            )
        self.temp = out_channels // self.num_splits
        self.route_temperature = float(route_temperature)
        self.motion_speed_scale = float(motion_speed_scale)

        self.convs = nn.ModuleList([
            spconv.SubMConv3d(
                self.temp,
                self.temp,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                stride=stride,
            )
            for dilation in dilations
        ])

        hidden = max(out_channels // 2, 8)
        self.motion_head = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.router = nn.Sequential(
            nn.Linear(out_channels + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_splits),
        )
        # Uniform initial routing makes R * softmax(logits) equal to one.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.last_motion = None
        self.last_routing = None

    def _route(self, features):
        motion = torch.tanh(self.motion_head(features)) * self.motion_speed_scale
        speed = torch.linalg.vector_norm(motion, dim=1, keepdim=True)
        direction = F.normalize(motion, dim=1, eps=1e-6)
        logits = self.router(torch.cat((features, direction, speed), dim=1))
        routing = torch.softmax(logits / self.route_temperature, dim=1)
        return motion, routing

    def forward(self, x):
        motion, routing = self._route(x.features)
        outputs = []
        for branch_index, convolution in enumerate(self.convs):
            start = branch_index * self.temp
            stop = (branch_index + 1) * self.temp
            branch = spconv.SparseConvTensor(
                x.features[:, start:stop],
                x.indices,
                x.spatial_shape,
                x.batch_size,
            )
            branch_features = convolution(branch).features
            # Multiplication by R preserves the original feature magnitude for
            # uniform routing while allowing sample-dependent scale selection.
            gate = self.num_splits * routing[:, branch_index:branch_index + 1]
            outputs.append(gate * branch_features)

        self.last_motion = motion
        self.last_routing = routing
        return x.replace_feature(torch.cat(outputs, dim=1))



class GDBlock(SparseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_fn,
        dilations=(1, 2, 3, 4),
        indice_key=None,
        bias=False,
        ad_channels=16,
        mode="fixed",
        route_temperature=1.0,
        motion_speed_scale=0.2,
    ):
        """

        :rtype: object
        """
        super().__init__()

        self.shortcut = Shortcut(in_channels,out_channels,norm_fn)
        add_channel = ad_channels
        self.pwconv=spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels+add_channel, kernel_size=1, padding=1, bias=bias),
            norm_fn(out_channels+add_channel),
            nn.ReLU(),
        )

        gdconv_cls = GDConv if mode == "fixed" else MotionConditionedGDConv
        gdconv_kwargs = {}
        if mode == "motion":
            gdconv_kwargs.update(
                route_temperature=route_temperature,
                motion_speed_scale=motion_speed_scale,
            )
        self.gdconv=spconv.SparseSequential(
            gdconv_cls(
                out_channels + add_channel,
                norm_fn,
                stride,
                dilations=dilations,
                indice_key=indice_key,
                **gdconv_kwargs,
            ),
            norm_fn(out_channels+add_channel),
            nn.ReLU(),
        )

        self.se = SEModule(out_channels+add_channel,reduction=2)

        self.conv3=spconv.SparseSequential(
            spconv.SubMConv3d(out_channels+add_channel, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels),
        )

        self.act = spconv.SparseSequential(nn.ReLU())


    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        identity = self.shortcut(identity)
        x = self.pwconv(input)
        x = self.gdconv(x)
        x = self.se(x)
        x = self.conv3(x)

        x = x.replace_feature(x.features+identity.features)
        x = self.act(x)

        return x

    @property
    def motion_prediction(self):
        gdconv = self.gdconv[0]
        return getattr(gdconv, "last_motion", None)

    @property
    def routing_weights(self):
        gdconv = self.gdconv[0]
        return getattr(gdconv, "last_routing", None)


def Downsample_block(in_channels, out_channels, kernel_size,norm_fn, stride=2, padding=1,bias=False,indice_key=None ):
    m = spconv.SparseSequential(
        spconv.SparseConv3d(in_channels, out_channels, [3,3,5], stride=[2,2,4], padding=padding,bias=False,indice_key=indice_key),
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m
