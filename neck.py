import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType


@MODELS.register_module()
class MultiScaleAttentionFusionCpu(BaseModule):
    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate=0.0,
        drop_path=0.0,
        layer_scale=1.0,
        num_fusion_block=2,
        norm_cfg=dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg=dict(type="GELU"),
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(*[
                SimpleFusionBlock(
                    out_channel,
                    groups,
                    norm_cfg,
                    act_cfg
                ) for _ in range(num_fusion_block)
            ]) for _ in range(self.num_levels)
        ])

    def forward(self, inputs):
        outputs = self.channel_mapper(inputs)
        outputs = [block(feat) for block, feat in zip(self.fusion_blocks, outputs)]
        return tuple(outputs)


class SimpleFusionBlock(nn.Module):
    def __init__(self, channels, groups, norm_cfg, act_cfg):
        super().__init__()
        self.conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.conv(x)

