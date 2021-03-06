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
    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    else:
        raise ValueError(
            'Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
    coco_flip_index = FLIP_CONFIG[dataset_name]
    input_size = cfg.DATASET.INPUT_SIZE
    output_size = cfg.DATASET.OUTPUT_SIZE
    min_scale = cfg.DATASET.MIN_SCALE
    max_scale = cfg.DATASET.MAX_SCALE
    max_rotation = cfg.DATASET.MAX_ROTATION
    max_translate = cfg.DATASET.MAX_TRANSLATE

    if is_train:
        transforms = T.Compose(
            [   
                T.RandomAffineTransform(
                    input_size,
                    output_size,
                    max_rotation,
                    min_scale,
                    max_scale,
                    max_translate,
                ),
                T.RandomHorizontalFlip(coco_flip_index, output_size, prob=0.5),
                T.ToTensor()
            ]
        )
    else:
        transforms = T.Compose(
            [   
                T.Resize(input_size, output_size),
                T.ToTensor()
            ]
        )
    
    return transforms
