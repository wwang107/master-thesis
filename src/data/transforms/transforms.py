
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
from imgaug.augmentables import Keypoint, KeypointsOnImage


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
        return F.to_tensor(image), mask, joints


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
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    def __call__(self, image, mask, joints):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1

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
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=(-self.max_rotation, self.max_rotation),
                scale=(self.min_scale, self.max_scale),
                translate_px={
                    "x": (-self.max_translate, self.max_translate),
                    "y": (-self.max_translate, self.max_translate)}
            ),
            iaa.Resize(self.input_size)]
        ).to_deterministic()
        for i, pr in enumerate(joints):
            kpt = KeypointsOnImage.from_xy_array(pr[:, :2], image.shape)
            kps_aug = seq.augment_keypoints(kpt)
            joints[i,:,:2] = KeypointsOnImage.to_xy_array(kps_aug)
            
        image_aug = seq.augment_image(image)

        from utils.vis import save_valid_image
        save_valid_image(
            image_aug, joints, '/home/weiwang/master-thesis/images/test.png')

        return image, mask, joints
