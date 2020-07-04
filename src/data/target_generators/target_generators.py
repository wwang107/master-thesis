from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import List

class DirectionalKeypointsGenerator():
    def __init__(self, num_joints, skeleton: List[List]):
        self.num_joints = num_joints
        self.skeleton = np.array(skeleton) - 1

    def __call__(self, joints, ratio):
        num_link = len(self.skeleton)
        direct_keypoints = np.zeros((joints.shape[0], 2*num_link, 3))
        for k in range(joints.shape[0]):
            for i, sk in enumerate(self.skeleton):
                if joints[k, sk[0], 2] > 0 and joints[k, sk[1], 2] > 0:
                    for j, t in enumerate([ratio, 1 - ratio]):
                        direct_keypoints[k, 2*i+j, 0] = \
                            (1-t) * joints[k, sk[0], 0] + \
                            t * joints[k, sk[1], 0]
                        direct_keypoints[k, 2*i+j, 1] = \
                            (1 - t) * joints[k, sk[0], 1] + \
                            t * joints[k, sk[1], 1]
                        direct_keypoints[k, 2*i+j, 2] = 1
                    
        return direct_keypoints


class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)
                             ), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)
                             ), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms