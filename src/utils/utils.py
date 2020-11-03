
from typing import List
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from torch._C import dtype


def find_keypoints_from_heatmaps(batch_heatmaps: torch.Tensor,
                                 num_type_keypoint: int = 17,
                                 threshold: float = 0.0) -> List[List[List]]:
    num_batch = batch_heatmaps.size(0)
    batch_heatmaps = batch_heatmaps.cpu().numpy()
    keypoints = [[[] for i in range(num_type_keypoint)]
                 for b in range(num_batch)]

    for b in range(num_batch):
        for j in range(num_type_keypoint):
            hm = batch_heatmaps[b, j, :, :]
            lbl, num_cluster = label(hm >= threshold)
            centroids = center_of_mass(
                hm, lbl, [i for i in range(1, num_cluster+1)])
            if isinstance(centroids, tuple):
                # only one peak
                if np.isnan(centroids[0]):
                    continue
                else:
                    keypoints[b][j].append(centroids)
            elif isinstance(centroids, list):
                # multiple peaks
                for centroid in centroids:
                    if np.isnan(centroid[0]):
                        continue
                    else:
                        keypoints[b][j].append(centroid)
            if len(keypoints[b][j]) > 0:
                keypoints[b][j] = np.array(keypoints[b][j])
    return keypoints


def match_detected_groundtruth_keypoint(batch_gt_keypoints: torch.Tensor, batch_detected_keypoints: np.ndarray, thresh: np.float = np.float('inf')):
    """
    gt_keypoints: [batch_size, num_max_person, num_joints, 2d_coordinates_and_visibility], Note that 2d_coordinates_and_visibility has order of yxv
    detected_keypoints: [batch_size, num_joints, num_keypoints], Note that the order of 2d_coordinate is yx
    """
    num_batch_size = batch_gt_keypoints.size(0)
    num_joints = len(batch_detected_keypoints[0])
    assert num_batch_size == len(
        batch_gt_keypoints), 'groundtruth keypoints and detected keypoints should have the same batch size'

    tp = [[[] for i in range(num_joints)]
          for i in range(num_batch_size)]  # true positive
    fp = [[[] for i in range(num_joints)]
          for i in range(num_batch_size)]  # false positive
    fn = [[[] for i in range(num_joints)]
          for i in range(num_batch_size)]  # false negative
    tp_dist = 0
    result = {'false positive': {'points': [], 'num': 0},
              'true positive': {'points': [], 'num': 0},
              'false negative': {'points': [], 'num': 0}
              }
    
    total_gt_joints = 0
    for i in range(num_batch_size):
        for j, detected_joints in enumerate(batch_detected_keypoints[i]):
            gt_joints = batch_gt_keypoints[i, :, j, 0:2]
            # visible = batch_gt_keypoints[i, :, j, 2] > 0
            # gt_joints = batch_gt_keypoints[i, :, j, 0:2][visible]
            total_gt_joints += gt_joints.shape[0]
            if len(detected_joints) > 0 and gt_joints.shape[0] == 0:
                fp[i][j].append(detected_joints)
                result['false positive']['num'] += len(detected_joints)
            elif len(detected_joints) == 0 and gt_joints.shape[0] > 0:
                fn[i][j].append(gt_joints)
                result['false negative']['num'] += gt_joints.shape[0]
            else:
                fn_inds = np.ones(gt_joints.shape[0], dtype=np.bool)
                tp_inds = np.zeros(detected_joints.shape[0], dtype=np.bool)
                
                cost_matrix = cdist(detected_joints, gt_joints)
                paired_det_inds, paired_gt_inds = linear_sum_assignment(cost_matrix)
                tp_condition = cost_matrix[paired_det_inds,paired_gt_inds] <= thresh
                tp_dist += np.sum(np.sum(cost_matrix[paired_det_inds[tp_condition], paired_gt_inds[tp_condition]]))
                tp_inds[paired_det_inds[tp_condition]] = True
                fn_inds[paired_gt_inds[tp_condition]] = False
                fp_inds = ~tp_inds
                
                num_tp = np.sum(tp_inds)
                num_fn = np.sum(fn_inds)
                num_fp = np.sum(fp_inds)

                if num_fn > 0:
                    fn[i][j].append(gt_joints[fn_inds])
                    result['false negative']['num'] += num_fn
                if num_tp > 0:
                    tp[i][j].append(detected_joints[tp_inds])
                    result['true positive']['num'] += num_tp
                if num_fp > 0:
                    fp[i][j].append(detected_joints[fp_inds])
                    result['false positive']['num'] += num_fp
                
            assert result['false negative']['num'] + result['true positive']['num'] == total_gt_joints
    result['false positive']['points'] = fp
    result['false negative']['points'] = fn
    result['true positive']['points'] = tp

    return result
