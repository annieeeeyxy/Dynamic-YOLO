import torch
import torch.nn as nn
from mmcv.cnn import is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmdet.registry import MODELS
from mmdet.models.layers import DyReLU
from mmdet.models.dense_heads.atss_head import ATSSHead


@MODELS.register_module()
class SepDecoupleHead(ATSSHead):
    def __init__(self, num_classes=80, in_channels=128, feat_channels=128, stacked_convs=2,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type="GELU"), with_objectness=False, **kwargs):

        self.act_cfg = act_cfg
        self.with_objectness = with_objectness  # 提前赋值，避免父类调用_init_layers时报错

        super().__init__(num_classes, in_channels, feat_channels=feat_channels,
                         stacked_convs=stacked_convs, norm_cfg=norm_cfg, **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for _ in range(self.stacked_convs):
            self.cls_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))
            self.reg_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))

        self.dy_yolo_cls = nn.Conv2d(self.in_channels, self.num_base_priors * self.cls_out_channels, kernel_size=1)
        self.dy_yolo_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4, kernel_size=1)

        if self.with_objectness:
            self.dy_yolo_obj = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.dy_yolo_cls, std=0.01, bias=bias_cls)
        normal_init(self.dy_yolo_reg, std=0.01)
        if self.with_objectness:
            normal_init(self.dy_yolo_obj, std=0.01, bias=bias_cls)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []

        for x in feats:
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            cls_score = self.dy_yolo_cls(cls_feat)
            bbox_pred = self.dy_yolo_reg(reg_feat)

            if self.with_objectness:
                obj_score = self.dy_yolo_obj(reg_feat)
                cls_score = cls_score * obj_score

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return tuple(cls_scores), tuple(bbox_preds)
