import torch
import torch.nn as nn
from mmcv.cnn import (
    build_norm_layer,
    build_activation_layer,
)
from mmengine.model import BaseModule
from mmdet.registry import MODELS


class StemLayer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, act_cfg=dict(type="GELU"), norm_cfg=dict(type='GN', num_groups=1, requires_grad=True)):
        super().__init__()
        stem_norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=1, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=1, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        stem = self.stem(x)
        x = self.proj(stem)
        x = self.norm(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, channels, norm_cfg=dict(type='GN', num_groups=1, requires_grad=True)):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(norm_cfg, 2 * channels)[1]

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class InternImageBlock(nn.Module):
    def __init__(self, channels, depth, groups, act_cfg=dict(type="GELU"),
                 norm_cfg=dict(type='GN', num_groups=1, requires_grad=True), post_norm=False, downsample=True):
        super().__init__()
        self.post_norm = post_norm
        self.blocks = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False)
            for _ in range(depth)
        ])
        if not self.post_norm:
            self.norm = build_norm_layer(norm_cfg, channels)[1]
        self.downsample = DownsampleLayer(channels=channels, norm_cfg=norm_cfg) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample:
            x = self.downsample(x)
        return (x, x_) if return_wo_downsample else x


@MODELS.register_module()
class LightInternImage(BaseModule):
    def __init__(self, channels=128, depths=[8, 8, 4], groups=[4, 8, 16],
                 act_cfg=dict(type="GELU"), norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
                 post_norm=False, out_indices=(1, 2, 3), init_cfg=None,
                 drop_rate=0.0, drop_path_rate=0.0, layer_scale=1.0):
        super().__init__(init_cfg)
        self.num_levels = len(depths)
        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale = layer_scale

        self.patch_embed = StemLayer(
            in_chans=3, embed_dim=channels, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageBlock(
                channels=int(channels * 2 ** i),
                depth=depths[i],
                groups=groups[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

    def forward(self, x):
        x = self.patch_embed(x)
        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.contiguous())
        return seq_out
