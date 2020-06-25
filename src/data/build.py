# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .transforms import build_transforms

def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)
    
    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
    dataset = coco_kpt(
        cfg, 
        dataset_name,
        cfg.DATASET.DATA_FORMAT,
        transforms=transforms)

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):

    dataset = build_dataset(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset)
    return data_loader


def make_test_dataloader(cfg):

    return data_loader, dataset
