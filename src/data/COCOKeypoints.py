from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import pycocotools
from .COCODataset import CocoDataset

class CocoKeypoints(CocoDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 direction_keypoints_generator=None,
                 heatmap_generator=None,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT)

        self.num_joints = cfg.DATASET.NUM_JOINTS

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.direction_keypoints_generator = direction_keypoints_generator

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)
        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints = self.get_joints(anno)

        if self.transforms:
            img, mask, joints = self.transforms(
                img, mask, joints
            )
        dir_keypoints = self.direction_keypoints_generator(joints, 0.1)        
        heatmap = self.heatmap_generator(dir_keypoints)

        return {'images': img, 'heatmaps': heatmap, 'joints': joints, 'directional_keypoints': dir_keypoints}

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5

    def get_joints(self, anno):
        num_people = len(anno)

        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

        return joints
