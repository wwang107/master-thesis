from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .PanopticDataset import PanopticDataset as cmu_kpt
from .PanopticDataset import ReplicateViewPanoptic as replicate_cmu_kpt
from .transforms import build_transforms, transforms

from .target_generators import HeatmapGenerator
from .target_generators import DirectionalKeypointsGenerator

coco_skeleton = {
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"],

    "skeleton": [
        [16, 14], [14, 12], [17, 15], [15, 13],
        [12, 13], [6, 12], [7, 13], [6, 7],
        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
        [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
}


def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)
        
    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    directional_keypoint_generator = DirectionalKeypointsGenerator(
        cfg.DATASET.NUM_JOINTS,
        coco_skeleton['skeleton'])

    num_pair = len(coco_skeleton['skeleton'])
    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS + num_pair*2,
        cfg.DATASET.SIGMA)

    dataset = coco_kpt(
        cfg,
        dataset_name,
        cfg.DATASET.DATA_FORMAT,
        directional_keypoint_generator,
        heatmap_generator,
        transforms=transforms)

    return dataset

def build_CMU_dataset(cfg, is_train, replicate_view=False):
    directional_keypoint_generator = DirectionalKeypointsGenerator(
        cfg.DATASET.NUM_JOINTS,
        coco_skeleton['skeleton'])

    num_pair = len(coco_skeleton['skeleton'])
    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS + num_pair*2,
        cfg.DATASET.SIGMA)
    if replicate_view:
        dataset = replicate_cmu_kpt(
            cfg,
            cfg.DATASET.NUM_VIEW,
            heatmap_generator,
            directional_keypoint_generator,
            is_train
            )
    else:
        dataset = cmu_kpt(
            cfg,
            cfg.DATASET.NUM_VIEW,
            heatmap_generator,
            directional_keypoint_generator,
            is_train
            )

    return dataset


def make_dataloader(cfg, dataset_name='coco', is_train=True, replicate_view=False):
    if dataset_name == 'coco':
        dataset = build_dataset(cfg, is_train)
        data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=8, batch_size=16, collate_fn=collate_fn)
    else:
        dataset = build_CMU_dataset(cfg, is_train, replicate_view)
        is_shuffle = True if is_train else False
        num_wokers = cfg.DATASET.CMU_NUM_WORKERS
        batch_size = cfg.DATASET.CMU_BATCH_SIZE
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            num_workers=num_wokers,
            batch_size=batch_size,
            shuffle=is_shuffle,
            collate_fn=None)
    
    return data_loader


def collate_fn(batch):
    r"""
        batch: is a list of dictionary with (image, heatmap, joints, directional_keypoints)
    """
    collated_batch = {'images': [], 'heatmaps': [], 'masks': [],
                      'joints': [], 'directional_keypoints': []}

    for sample in batch:
        for key in sample:
            ele = sample[key]
            if type(ele).__name__ == 'ndarray':
                ele = torch.from_numpy(ele)
            collated_batch[key].append(ele)

    collated_batch['images'] = torch.stack(collated_batch['images'])
    collated_batch['heatmaps'] = torch.stack(collated_batch['heatmaps'])
    collated_batch['masks'] = torch.stack(collated_batch['masks'])
    return collated_batch


def make_test_dataloader(cfg):

    pass
