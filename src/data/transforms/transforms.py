
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
import numpy as np

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

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


class Resize(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, image, mask, joints):
        h, w = image.shape[:2]
        scale_height, scale_width = self.output_size/h, self.output_size/w

        image = cv2.resize(image,(self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.output_size, self.output_size))
        joints[:, :, 0] = joints[:, :, 0] * scale_width
        joints[:, :, 1] = joints[:, :, 1] * scale_height

        return image, mask, joints


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size

    def __call__(self, image, mask, joints):
        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            mask = mask[:, ::-1] - np.zeros_like(mask)
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

    def _get_affine_matrix(self, center, scale, translation, rot=0):
        tx = translation[0]
        ty = translation[1]
        T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        R = cv2.getRotationMatrix2D(center, rot, scale)
        R = np.concatenate((R, [[0, 0, 1]]))
        M = T @ R
        return M

    def _get_affined_joints(self, joints, mat):
        shape = joints.shape
        joints_coords = joints[:, :, :2]

        joints_coords = joints_coords.reshape(-1, 2)
        joints_coords = (np.concatenate(
            (joints_coords, joints_coords[:, 0:1]*0+1), axis=1) @ mat.T)[:, :2]
        joints[:, :, :2] = joints_coords.reshape((shape[0], shape[1], 2))
        return joints

    def __call__(self, image, mask, joints):
        h, w = image.shape[:2]
        center = (h/2+.5, w/2+.5)
        scale_height, scale_width = self.output_size/h, self.output_size/w

        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation
        if self.max_translate > 0:
            aug_dxdy = (
                np.random.randint(
                    -self.max_translate, self.max_translate),
                np.random.randint(
                    -self.max_translate, self.max_translate))

        M = self._get_affine_matrix(center, aug_scale, aug_dxdy, aug_rot)
        image = cv2.resize(
            cv2.warpAffine(image, M[:2, :], (w, h)),
            (self.input_size, self.input_size))
        mask = cv2.resize(
            cv2.warpAffine(mask, M[:2, :], (w, h)),
            (self.output_size, self.output_size))
        mask = mask.astype(np.float32)
        joints = self._get_affined_joints(joints, M)
        joints[:, :, 0] = joints[:, :, 0] * scale_width
        joints[:, :, 1] = joints[:, :, 1] * scale_height

        return image, mask, joints
