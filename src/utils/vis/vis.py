import numpy as np
import torchvision
import torch
import cv2
from . import VIS_CONFIG
import math


def add_joints(image, joints, dataset='COCO'):
    """Draw joint on the given image

    Parameters
    ----------
    iamge : ndarray \n
        2D array containing data with `float` type.
    arr2 : ndarray \n
        1D mask array(containing data with boolean type).
    """
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']
    idx_color = VIS_CONFIG[dataset]['idx_color']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(
                joint[1])), 1, idx_color[i], 2)
            # cv2.putText(image, str(i), (int(joint[0]), int(
            #     joint[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 255))

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], idx_color[part_idx[pair[0]]])

    return image


def add_direct_keypoints(image, keypoints):
    for keypt in keypoints:
        if keypt[2] > 0:
            cv2.circle(image, (int(keypt[0]), int(
                keypt[1])), 1, (255, 255, 255), -1)
    return image


def save_batch_joint_and_keypoint(file_name, batch_image, batch_joints, batch_keypoints, scale=1, dataset='COCO'):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    nrow = 8
    padding = 10
    grid = torchvision.utils.make_grid(batch_image, nrow=nrow, padding=padding)
    ndarr = grid.permute(1, 2, 0).mul(255)\
        .clamp(0, 255).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            keypoints = batch_keypoints[k]
            joints[:, :, 0] = x * width + padding + joints[:, :, 0]*scale
            joints[:, :, 1] = y * height + padding + joints[:, :, 1]*scale
            keypoints[:, :, 0] = x * width + padding + keypoints[:, :, 0]*scale
            keypoints[:, :, 1] = y * height + \
                padding + keypoints[:, :, 1]*scale
            for i in range(joints.shape[0]):
                add_joints(ndarr, joints[i, :, :], dataset)
                add_direct_keypoints(ndarr, keypoints[i, :, :])
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def make_heatmaps(image, heatmaps):
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.mul(255)\
            .clamp(0, 255)\
            .byte()\
            .cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def save_batch_maps(
        batch_image,
        batch_maps,
        batch_mask,
        normalize=False
):
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_maps.size(0)
    num_joints = batch_maps.size(1)
    map_height = batch_maps.size(2)
    map_width = batch_maps.size(3)

    grid_image = np.zeros(
        (batch_size*map_height, (num_joints+1)*map_width, 3),
        dtype=np.uint8
    )

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
            .clamp(0, 255)\
            .byte()\
            .permute(1, 2, 0)\
            .cpu().numpy()
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        maps = batch_maps[i]

        image_with_hms = make_heatmaps(image, maps)

        height_begin = map_height * i
        height_end = map_height * (i + 1)

        grid_image[height_begin:height_end, :, :] = image_with_hms
        if batch_mask is not None:
                mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
                grid_image[height_begin:height_end, :map_width, :] = \
                    grid_image[height_begin:height_end, :map_width, :] * mask
    return cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    
