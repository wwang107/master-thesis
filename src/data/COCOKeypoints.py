from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import pycocotools
from .COCODataset import CocoDataset


class CocoKeypoints(CocoDataset):
    def __init__(
        self,
        cfg,
        dataset_name,
        remove_images_without_annotations,
        direction_keypoints_generator=None,
        heatmap_generator=None,
        transforms=None,
    ):
        super().__init__(cfg.DATASET.COCOROOT, dataset_name, cfg.DATASET.DATA_FORMAT)

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
        mask = np.array(self.get_mask(anno, idx), dtype=np.uint8)
        anno = [obj for obj in anno if obj["iscrowd"] == 0 or obj["num_keypoints"] > 0]

        joints = self.get_joints(anno)
        bboxes = self.get_bboxes(anno)
        if self.transforms:
            img, mask, joints, bboxes= self.transforms(img, mask, joints, bboxes)
        dir_keypoints = self.direction_keypoints_generator(joints, 0.1)
        heatmap = self.heatmap_generator(
            np.concatenate((joints, dir_keypoints), axis=1)
        )

        return {
            "bboxes": bboxes,
            "images": img,
            "masks": mask,
            "heatmaps": heatmap,
            "joints": joints,
            "directional_keypoints": dir_keypoints,
        }

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info["height"], img_info["width"]))

        for obj in anno:
            # if obj['iscrowd']:
            #     rle = pycocotools.mask.frPyObjects(
            #         obj['segmentation'], img_info['height'], img_info['width'])
            #     m += pycocotools.mask.decode(rle)
            if obj["num_keypoints"] > 0:
                rles = pycocotools.mask.frPyObjects(
                    obj["segmentation"], img_info["height"], img_info["width"]
                )
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m > 0.5

    def get_bboxes(self, anno):
        num_people = len(anno)
        bboxes = np.zeros((num_people, 4))

        for i, obj in enumerate(anno):
            bboxes[i] = np.array(obj['bbox'])
        return bboxes

    def get_joints(self, anno):
        num_people = len(anno)

        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, : self.num_joints, :3] = np.array(obj["keypoints"]).reshape(
                [-1, 3]
            )

        return joints
    
    def evaluate_with_gt_bbox(self, cfg, hms):
        channels, height, width = hms[0].size()
        for idx, hm in enumerate(hms):
            img, anno = super().__getitem__(idx)
        
        return 
