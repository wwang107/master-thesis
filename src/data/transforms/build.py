# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T

FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ]
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)
                      ), 'DATASET.OUTPUT_SIZE should be list or tuple'
    if is_train:
        pass
    else:
        pass
    # coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    # if cfg.DATASET.WITH_CENTER:
        # coco_flip_index.append(17)
    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    else:
        raise ValueError(
            'Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
    coco_flip_index = FLIP_CONFIG[dataset_name]

    transforms = T.Compose(
        [
            # T.RandomAffineTransform(
            #     input_size,
            #     output_size,
            #     max_rotation,
            #     min_scale,
            #     max_scale,
            #     scale_type,
            #     max_translate,
            #     scale_aware_sigma=cfg.DATASET.SCALE_AWARE_SIGMA
            # ),
            # T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            # T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
