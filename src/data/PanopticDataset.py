import random
import json
import torch
import cv2
import numpy as np
from data import panutils
from pathlib import Path
from torch.utils.data import Dataset
from utils.transform import get_affine_transform, affine_transform, get_scale
from tqdm import tqdm
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'

CMU_TO_COCO_JOINT_LABEL = {
    1: 0, # nose 
    3: 5, # shouder_l
    4: 7, # elbow_l
    5: 9, # wrist_l
    6: 11, # hip_l
    7: 13, # knee_l
    8: 15, # ankle_l
    9: 6, # shoulder_r
    10: 8, # elbow_r
    11: 10, # wrist_r
    12: 12, # hip_r
    13: 14, # knee_r
    14: 16, # ankle_r
    15: 2, # eye_r
    16: 1, # eye_l
    17: 4, # ear_r
    18: 3 # ear_l
    }
"""
Panoptic Joint Label:
0: Neck
1: Nose
2: BodyCenter (center of hips)
3: lShoulder
4: lElbow
5: lWrist,
6: lHip
7: lKnee
8: lAnkle
9: rShoulder
10: rElbow
11: rWrist
12: rHip
13: rKnee
14: rAnkle
15: rEye
16: lEye
17: rEar
18: lEar
"""
WIDTH = 1920
HEIGHT = 1080
HD_CAMERA_ID = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
HD_IMG = 'hdImgs'
BODY_EDGES = np.array([[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [
    8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11], [11, 12]])-1


class PanopticDataset(Dataset):
    """
    A dataset for CMU Panoptic
    """

    def __init__(self, root,
                cfg,
                heatmap_generator=None,
                keypoint_generator=None,
                is_load_path_only=True, 
                is_coco_keypoint=True):
        super().__init__()
        self.files = {}
        self.cfg = cfg
        self.keypoint_generator = keypoint_generator
        self.heatmap_generator = heatmap_generator
        self.is_load_path_only = is_load_path_only
        self.is_coco_keypoint = is_coco_keypoint
        self.root_id = 2 # let hip to be the root joint
        self.interval = 2
        self.num_frames_in_subseq = 16
        self.num_view = 5
        self.num_joints = 17
        self.num_row_per_joints = 3
        self.num_max_people = 10
        self.num_directional_keypoint = 38
        
        rootDir = Path(root)
        self.seq_names = [x.name for x in rootDir.iterdir() if x.is_dir()]
        for seq_name in self.seq_names:
            count = 0
            valid_frames = [] # frame with skeleton is defined as valid frame
            self.files[seq_name] = {'cams_matrix': {}, 'pose3d': [], 'hd_img':[]}
            with open(rootDir.joinpath(seq_name, 'calibration_{0}.json'.format(seq_name))) as cfile:
                calib = json.load(cfile)
            # Cameras are identified by a tuple of (panel#,node#)
            cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
            # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera
            print('Loading hd images path...')
            for cam_id in HD_CAMERA_ID:
                if cam_id in cameras:
                    cam = cameras[cam_id]
                    cam['K'] = np.matrix(cam['K'])
                    cam['distCoef'] = np.array(cam['distCoef'])
                    cam['R'] = np.matrix(cam['R'])
                    cam['t'] = np.array(cam['t']).reshape((3, 1))
                    self.files[seq_name]['cams_matrix'][cam_id] = cam
                    # self.files[seq_name]['hd_imgs'][cam_id] = {}
                    # for img_path in rootDir.joinpath(seq_name, HD_IMG, '{0:02d}_{1:02d}'.format(cam_id[0], cam_id[1])).glob('*.jpg'):
                    #     self.files[seq_name]['hd_imgs'][cam_id][img_path.stem] = img_path
            print('Done')
            
            all_pose3d = []
            all_hd_img = []
            # Load skeleton
            print('Loading skeleton...')
            skel_json_paths = list(rootDir.joinpath(
                seq_name, 'hdPose3d_stage1_coco19').glob('*.json'))
            skel_json_paths = skel_json_paths[len(
                skel_json_paths) // 2 - 50: len(skel_json_paths) // 2 + 50]
            for bframe_path in tqdm(skel_json_paths, total=len(skel_json_paths)):
                with open(bframe_path) as sfile:
                    bodies = json.load(sfile)['bodies']
                pose3d_frame = []
                hd_img_frame = {}
                if len(bodies) == 0:
                    continue
                
                for body in bodies:
                    pose3d = np.array(body['joints19']).reshape((-1, 4)).transpose()
                    joints_vis = pose3d[-1, :] > 0.1
                    if not joints_vis[self.root_id]:
                            continue
                    pose3d_frame.append(pose3d)
                for cam_id in HD_CAMERA_ID:
                    postfix = bframe_path.name.replace('body3DScene', '')
                    prefix = '{:02d}_{:02d}'.format(cam_id[0], cam_id[1])
                    image = rootDir.joinpath(seq_name, 'hdImgs', prefix, prefix + postfix)
                    image = Path(str(image).replace('json', 'jpg'))
                    hd_img_frame[cam_id] = image
                all_pose3d.append(pose3d_frame)
                all_hd_img.append(hd_img_frame)
            
            self.files[seq_name]['pose3d'] = all_pose3d
            self.files[seq_name]['hd_img'] = all_hd_img
            print('Done')

    
    def __getitem__(self, idx):
        hm = np.zeros((
            self.num_directional_keypoint,
            self.cfg.DATASET.OUTPUT_SIZE[1], self.cfg.DATASET.OUTPUT_SIZE[0],
            self.num_view,
            self.num_frames_in_subseq))
        img = np.zeros((
            self.cfg.DATASET.OUTPUT_SIZE[1], self.cfg.DATASET.OUTPUT_SIZE[0],
            3,
            self.num_view,
            self.num_frames_in_subseq), np.uint8)
        pose2d = np.zeros((
            self.num_max_people,
            3,
            self.num_joints,
            self.num_view,
            self.num_frames_in_subseq), np.float32)
        keypoint2d = np.zeros((
            self.num_max_people,
            self.num_directional_keypoint,
            3,
            self.num_view,
            self.num_frames_in_subseq), np.float32)
        num_person = 0
        c = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        s = get_scale((WIDTH, HEIGHT), self.cfg.DATASET.OUTPUT_SIZE)
        r = 0
        trans = get_affine_transform(c, s, r, self.cfg.DATASET.OUTPUT_SIZE)
        
        seq_name = self.seq_names[idx]
        n_frames = len(self.files[seq_name]['pose3d'])
        start = random.randint(0, n_frames - self.interval * self.num_frames_in_subseq)
        end = start + self.interval * self.num_frames_in_subseq
        
        subseq_pose3d = self.files[seq_name]['pose3d'][start:end:self.interval]
        subseq_hdImg = self.files[seq_name]['hd_img'][start:end:self.interval]
        for f, frame3d in enumerate(subseq_pose3d):
            for k, cam_id in enumerate(HD_CAMERA_ID):
                for p, pose3d in enumerate(frame3d):
                    pose3d = self.mapKeypointsToCOCO(pose3d)
                    pose2d[p, :,:,k,f] = self.map3DkeypointsTo2d(pose3d, self.files[seq_name]['cams_matrix'][cam_id])
                    for i in range(self.num_joints):
                        pose2d[p, 0:2, i, k, f] = affine_transform(
                            pose2d[p, 0:2, i, k, f], trans)
                        
                    num_person = p+1
                keypoint2d[0:num_person, :, :, k, f] = self.keypoint_generator(
                    pose2d[0:num_person, :, :, k, f].transpose((0, 2, 1)), 0.2)
                heatmap = self.heatmap_generator(
                    keypoint2d[0:num_person, :, :, k, f])
                img[:, :, :,k, f] = \
                    cv2.warpAffine(
                    cv2.imread(str(subseq_hdImg[f][cam_id])),
                    trans,
                    (int(self.cfg.DATASET.OUTPUT_SIZE[0]), int(self.cfg.DATASET.OUTPUT_SIZE[1])),
                    flags=cv2.INTER_LINEAR)
                hm[:,:,:,k,f] = heatmap

        return {'heatmap': torch.from_numpy(hm), 'img': torch.from_numpy(img), 'keypoint2d': keypoint2d, 'num_person': num_person}
    
    def __len__(self):
        return len(self.seq_names)
    
    def readSkeletonFromPath(self, path):
        skeleton = []
        with open(path) as sfile:
            bframe = json.load(sfile)
        for body in bframe['bodies']:
            keypoint = np.array(body['joints19']).reshape(
                (-1, 4)).transpose()
            skeleton.append(self.mapKeypointsToCOCO(keypoint))
        
        return skeleton

    def mapKeypointsToCOCO(self, cmu_keypoint):
        """
        Map the keypoint to COCO dataset format.
        """
        coco_keypoint = np.zeros((4,17),np.float)
        for cmu_coco_idx in CMU_TO_COCO_JOINT_LABEL.items():
            cmu_idx = cmu_coco_idx[0]
            coco_idx = cmu_coco_idx[1]
            coco_keypoint[:,coco_idx] = cmu_keypoint[:,cmu_idx]

        return coco_keypoint

    
    # def visualize3DKeypoint(self, filename, seq_name, camera_id, idx):
        
    #     framesList = sorted(self.files[seq_name]['skeletons'].keys())
        
        
    #     img = plt.imread(self.files[seq_name]['hd_imgs'][camera_id]['{:02d}_{:02d}_{}'.format(
    #         camera_id[0], camera_id[1], framesList[idx])])
    #     cam = self.files[seq_name]['cams_matrix'][camera_id]
    #     if (self.is_load_path_only):
    #         path = self.files[seq_name]['skeletons'][framesList[idx]]
    #         keypoint3d = []
    #         with open(path) as sfile:
    #             bframe = json.load(sfile)
    #         for body in bframe['bodies']:
    #             keypoint3d.append(
    #                 np.array(body['joints19']).reshape((-1, 4)).transpose())
    #     else:
    #         keypoint3d = self.files[seq_name]['skeletons'][framesList[idx]]
        
    #     plt.figure(figsize=(15, 15))
    #     plt.imshow(img)
    #     for keypoint in keypoint3d:
    #         pt = panutils.projectPoints(keypoint[0:3, :],
    #                                 cam['K'], cam['R'], cam['t'],
    #                                 cam['distCoef'])


    #         plt.plot(pt[0, :], pt[1, :], '.')

    #         # Plot edges for each bone
    #         for i,edge in enumerate(BODY_EDGES):
    #             plt.plot(pt[0, edge], pt[1, edge])

    #         for ip in range(pt.shape[1]):
    #                 plt.text(pt[0, ip], pt[1, ip]-5,
    #                         '{0}'.format(ip))

    #         plt.draw()
    #     plt.savefig('{}.png'.format(filename))
    #     plt.close()
            
    def map3DkeypointsTo2d(self, keypoints3d, camera):
        keypoints2d = panutils.projectPoints(keypoints3d[0:3, :],
                                             camera['K'], camera['R'], camera['t'],
                                             camera['distCoef'])
        return keypoints2d

    def showHDViewAnd2DKeypoints(self, hd_image, keypoint2d):
        pass



