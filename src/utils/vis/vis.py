from typing import List
import numpy as np
import torchvision
import torch
import cv2

from data.transforms.transforms import Normalize
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


def save_batch_sequence_image_with_joint(batch_image,
                                         batch_joints,
                                         num_person,
                                         file_name,
                                         padding=2
                                         ):
    '''
    batch_image: [batch_size, channel, height, width, view, frames]
    batch_joints: [batch_size, num_person, num_joints, 3, view, frames],
    num_person: [batch_size]
    }
    '''
    n_batch = batch_image.size(0)
    n_frame = batch_image.size(5)
    n_view = batch_image.size(4)
    assert n_view == batch_joints.size(4)

    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    for i in range(n_batch):
        grid_image = np.zeros(
            (n_frame * height,  n_view * width, 3),
            dtype=np.uint8)
        # swap the color channel to the first dimension
        image = batch_image[i].permute((1, 2, 0, 3, 4))
        for x in range(n_view):
            for y in range(n_frame):
                width_begin = x*width
                width_end = (x+1)*width-padding
                height_begin = y*height
                height_end = (y+1)*height-padding
                grid_image[height_begin:height_end,
                           width_begin:width_end, :] = image[:, :, :, x, y]
                for p in range(num_person[i]):
                    joints = batch_joints[i, p]

                    for joint in joints[:, :, x, y]:
                        joint[0] = x * width + padding + joint[0]
                        joint[1] = y * height + padding + joint[1]
                        cv2.circle(grid_image, (int(joint[0]), int(joint[1])), 2,
                                   [0, 255, 255], 2)
        cv2.imwrite('{}_frame_{}.png'.format(file_name, i), grid_image)


def save_batch_multi_view_with_heatmap(batch_image, batch_heatmaps, file_name, normalize=True):
    num_view = batch_heatmaps.size(4)
    multi_view_images = []
    for k in range(num_view):
        file_name_view = file_name + '_{}'.format(k)
        multi_view_images.append(
            save_batch_heatmaps_multi(
                batch_image=batch_image[:, :, :, :, k],
                batch_heatmaps=batch_heatmaps[:, :, :, :, k],
                file_name=file_name_view, normalize=normalize)
        )
    return multi_view_images


def save_batch_image_with_joints_multi(batch_image,
                                       batch_joints,
                                       batch_joints_vis,
                                       num_person,
                                       file_name,
                                       nrow=8,
                                       padding=2):
    '''
    batch_image: [batch_size, channel, height, width, view, frames]
    batch_joints: [batch_size, num_person, num_joints, 3, view, frames],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    '''
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

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
            for n in range(num_person[k]):
                joints = batch_joints[k, n]
                joints_vis = batch_joints_vis[k, n]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                                   [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def __draw_point(image: np.ndarray, points: np.ndarray, color: tuple)->None:
    for pt in points:
        loc = (int(pt[1]), int(pt[0]))
        cv2.circle(image, loc, 1,color)
def save_keypoint_detection(batch_image,
                            batch_heatmaps,
                            batch_fp_points,
                            batch_fn_points,
                            batch_tp_points,
                            num_joints_to_show = 17,
                            normalize = True):
    if normalize:
        batch_image = batch_image.clone().float()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(
        1) if num_joints_to_show is None else num_joints_to_show
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .permute(1, 2, 0)\
                              .byte()\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        
        fp_points = batch_fp_points[i]
        tp_points = batch_tp_points[i]
        fn_points = batch_fn_points[i]
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            
            if len(tp_points[j]) > 0: 
                __draw_point(masked_image, tp_points[j][0],(255,0,0))
            if len(fp_points[j]) > 0:
                __draw_point(masked_image, fp_points[j][0],(0,0,255))
            if len(fn_points[j]) > 0:
                __draw_point(masked_image, fn_points[j][0],(0,255,255))

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    return grid_image


def save_batch_keypoints(batch_image: torch.Tensor,
                         batch_heatmaps: torch.Tensor,
                         gt_keypoints: torch.Tensor,
                         pred_keypoints: List[List],
                         num_joints_to_show: int = None,
                         normalize: bool = True) -> np.array:
    '''
   batch_image: [batch_size, channel, height, width]
   batch_heatmaps: ['batch_size, num_joints, height, width]
   gt_keypoints: ['batch_size, num_max_person, num_joints, 2d_coordinates_and_visibility]
   pred_keypoints: ['batch_size, num_joints, 2d_coordinates]
   file_name: saved file name
   '''
    if normalize:
        batch_image = batch_image.clone().float()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(
        1) if num_joints_to_show is None else num_joints_to_show
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .permute(1, 2, 0)\
                              .byte()\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)

        pred_keypoint = pred_keypoints[i]
        gt_keypoint = gt_keypoints[i]
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            for p in gt_keypoint:
                if p[j, 2] > 0:
                    cv2.circle(masked_image, (int(p[j, 1]), int(p[j, 0])), 1,
                               [0, 0, 255], 2)
            pts = pred_keypoint[j]
            for pt in pts:
                cv2.circle(masked_image, (int(pt[1]), int(pt[0])), 1,
                           [0, 255, 255], 2)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    return grid_image


def save_batch_heatmaps_multi(batch_image, batch_heatmaps, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone().float()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    # batch_image = batch_image.flip(1)
    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .permute(1, 2, 0)\
                              .byte()\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    return grid_image
