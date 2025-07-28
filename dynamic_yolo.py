# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector


@MODELS.register_module()
class DynamicYOLO(SingleStageDetector):
    """Implementation of DynamicYOLO."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def loss(self, inputs, data_samples, **kwargs):
        # Ensure pad_shape is present using img_shape (which equals pad_shape after Pad step)
        for data_sample in data_samples:
            if 'pad_shape' not in data_sample.metainfo:
                data_sample.set_metainfo(dict(pad_shape=data_sample.img_shape))

        x = self.extract_feat(inputs)
        return self.bbox_head.loss(x, data_samples)

    def forward_train(self, images, batch_data_samples, **kwargs):
        x = self.extract_feat(images)

        for data_sample in batch_data_samples:
            pad_shape = data_sample.metainfo.get('pad_shape', None)
            if pad_shape is None:
                pad_shape = data_sample.metainfo.get('img_shape', None)
            if pad_shape is None:
                raise ValueError('pad_shape and img_shape are both missing in metainfo')

            # ✅ 设置到 DetDataSample 本身
            data_sample.set_metainfo({'pad_shape': pad_shape})

            # ✅ 设置到 gt_instances（AnchorHead 要求的地方）
            if hasattr(data_sample, 'gt_instances'):
                data_sample.gt_instances.pad_shape = pad_shape

        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses



