
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, SegmentationMapsOnImage


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints):
        for t in self.transforms:
            image, mask, joints = t(image, mask, joints)
        return image, mask, joints

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, mask, joints):
        return F.to_tensor(image), F.to_tensor(mask.astype(int)), joints


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size

    def __call__(self, image, mask, joints):
        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            joints = joints[:, self.flip_index]
            joints[:, :, 0] = self.output_size - joints[:, :, 0] - 1

        return image, mask, joints


class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 max_translate):
        self.input_size = input_size
        self.output_size = output_size

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_translate = max_translate

    def __call__(self, image, mask, joints):
        scale = self.output_size/self.input_size # scale factor to transform input size to output size
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=(-self.max_rotation, self.max_rotation),
                scale=(self.min_scale, self.max_scale),
                translate_px={
                    "x": (-self.max_translate, self.max_translate),
                    "y": (-self.max_translate, self.max_translate)}
            ),
            iaa.Resize(self.input_size)] # resize both x and y, so the output image is a squre image
        ).to_deterministic()

        for i, person in enumerate(joints):
            kpt = KeypointsOnImage.from_xy_array(person[:, :2], image.shape)
            kps_aug = seq.augment_keypoints(kpt)
            joints[i,:,:2] = KeypointsOnImage.to_xy_array(kps_aug) * scale # scale the joint coordinates to output coordinate
        
        seq_mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        image_aug, mask_aug = seq(image=image, segmentation_maps=seq_mask)
        return image_aug, mask_aug.get_arr(), joints
