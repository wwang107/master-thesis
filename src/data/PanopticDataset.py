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

CMU_TO_COCO_JOINT_LABEL = {
    1: 0,  # nose
    3: 5,  # shouder_l
    4: 7,  # elbow_l
    5: 9,  # wrist_l
    6: 11,  # hip_l
    7: 13,  # knee_l
    8: 15,  # ankle_l
    9: 6,  # shoulder_r
    10: 8,  # elbow_r
    11: 10,  # wrist_r
    12: 12,  # hip_r
    13: 14,  # knee_r
    14: 16,  # ankle_r
    15: 2,  # eye_r
    16: 1,  # eye_l
    17: 4,  # ear_r
    18: 3,  # ear_l
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
TRAINING_CAMERA_ID =  list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
VALIDATION_CAMERA_ID = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
HD_IMG = "hdImgs"
BODY_EDGES = (
    np.array(
        [
            [1, 2],
            [1, 4],
            [4, 5],
            [5, 6],
            [1, 3],
            [3, 7],
            [7, 8],
            [8, 9],
            [3, 13],
            [13, 14],
            [14, 15],
            [1, 10],
            [10, 11],
            [11, 12],
        ]
    )
    - 1
)

TRAIN_LIST = ["170407_haggling_a1", "160226_haggling1", "161202_haggling1"]

VAL_LIST = ["160906_pizza1"]


class PanopticDataset(Dataset):
    """
    A dataset for CMU Panoptic
    """

    def __init__(
        self, 
        cfg,
        num_view,
        heatmap_generator=None, 
        keypoint_generator=None,
        is_train=True
    ):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.num_frames_in_subseq = cfg.DATASET.NUM_FRAME_PER_SUBSEQ
        self.num_view = num_view 
        self.num_max_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.keypoint_generator = keypoint_generator
        self.heatmap_generator = heatmap_generator
        self.root_id = 2  # let hip to be the root joint
        self.num_joints = 17
        self.num_row_per_joints = 3
        self.num_directional_keypoint = 55
        self.use_cameras = TRAINING_CAMERA_ID if is_train else VALIDATION_CAMERA_ID

        self.files = {}
        self.files["sub_seq_data"] = []
        rootDir = Path(cfg.DATASET.CMUROOT)
        self.seq_names = TRAIN_LIST if is_train else VAL_LIST
        self.seq_names = sorted(self.seq_names)

        for seq_name in self.seq_names:
            self.files[seq_name] = {"cams_matrix": {}}
            with open(
                rootDir.joinpath(seq_name, "calibration_{0}.json".format(seq_name))
            ) as cfile:
                calib = json.load(cfile)
            # Cameras are identified by a tuple of (panel#,node#)
            cameras = {(cam["panel"], cam["node"]): cam for cam in calib["cameras"]}
            # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera
            print("Loading hd images path...")
            for cam_id in self.use_cameras:
                if cam_id in cameras:
                    cam = cameras[cam_id]
                    cam["K"] = np.matrix(cam["K"])
                    cam["distCoef"] = np.array(cam["distCoef"])
                    cam["R"] = np.matrix(cam["R"])
                    cam["t"] = np.array(cam["t"]).reshape((3, 1))
                    self.files[seq_name]["cams_matrix"][cam_id] = cam

            # Load skeleton
            skel_json_paths = sorted(
                list(
                    rootDir.joinpath(seq_name, "hdPose3d_stage1_coco19").glob("*.json")
                )
            )

            start_frame = (
                self.num_frames_in_subseq if is_train else len(skel_json_paths) // 4
            )
            step = self.num_frames_in_subseq
            end_frame = len(skel_json_paths)
            print("Loading skeleton...")
            for i in tqdm(range(start_frame, end_frame, step), desc=seq_name):
                pose3d_subseq = []
                hd_img_subseq = {}
                subseq_data = {"seq_name": None, "3d_pose": None, "hd_img": {}}
                for j in range(self.num_frames_in_subseq, 0, -1):
                    with open(skel_json_paths[i - j]) as sfile:
                        bodies = json.load(sfile)["bodies"]
                    if len(bodies) == 0:
                        break

                    poses3d = []
                    for body in bodies:
                        pose3d = np.array(body["joints19"]).reshape((-1, 4)).transpose()
                        joints_vis = pose3d[-1, :] > 0.1
                        if not joints_vis[self.root_id]:
                            continue
                        poses3d.append(pose3d)
                    if len(poses3d) == 0:
                        break
                    pose3d_subseq.append(poses3d)

                    for cam_id in self.use_cameras:
                        if cam_id not in hd_img_subseq:
                            hd_img_subseq[cam_id] = []
                        postfix = skel_json_paths[i - j].name.replace("body3DScene", "")
                        prefix = "{:02d}_{:02d}".format(cam_id[0], cam_id[1])
                        image = rootDir.joinpath(
                            seq_name, "hdImgs", prefix, prefix + postfix
                        )
                        image = Path(str(image).replace("json", "jpg"))
                        hd_img_subseq[cam_id].append(image)

                if len(pose3d_subseq) != self.num_frames_in_subseq:
                    continue
                else:
                    subseq_data["seq_name"] = seq_name
                    subseq_data["3d_pose"] = pose3d_subseq
                    subseq_data["hd_img"] = hd_img_subseq
                    self.files["sub_seq_data"].append(subseq_data)

    def __len__(self):
        return len(self.files["sub_seq_data"])

    def __getitem__(self, idx):
        subseq_pose3d = self.files["sub_seq_data"][idx]["3d_pose"]
        subseq_hdImg = self.files["sub_seq_data"][idx]["hd_img"]
        seq_name = self.files["sub_seq_data"][idx]["seq_name"]
        camera_ids = (
            random.sample(TRAINING_CAMERA_ID, self.num_view)
            if self.is_train
            else VALIDATION_CAMERA_ID[:self.num_view]
        )
        krt = np.zeros((3,4,len(camera_ids)))
        hm = np.zeros(
            (
                self.num_directional_keypoint,
                self.cfg.DATASET.OUTPUT_SIZE[1],
                self.cfg.DATASET.OUTPUT_SIZE[0],
                self.num_view,
                self.num_frames_in_subseq,
            ),
            dtype=np.float32,
        )
        img = np.zeros(
            (
                self.cfg.DATASET.INPUT_SIZE,
                self.cfg.DATASET.INPUT_SIZE,
                3,
                self.num_view,
                self.num_frames_in_subseq,
            ),
            np.uint8,
        )
        pose2d = np.zeros(
            (
                self.num_max_people,
                3,
                self.num_joints,
                self.num_view,
                self.num_frames_in_subseq,
            ),
            np.float32,
        )
        keypoint2d = np.zeros(
            (
                self.num_max_people,
                self.num_directional_keypoint,
                3,
                self.num_view,
                self.num_frames_in_subseq,
            ),
            np.float32,
        )
        num_person = np.zeros(self.num_frames_in_subseq, np.int)
        c = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        s = get_scale(
            (WIDTH, HEIGHT), (self.cfg.DATASET.INPUT_SIZE, self.cfg.DATASET.INPUT_SIZE)
        )
        s_output = get_scale(
            (WIDTH, HEIGHT), self.cfg.DATASET.OUTPUT_SIZE
        )
        
        for k, cam_id in enumerate(camera_ids):
            r = (random.random() * 2 - 1) * 60 if self.is_train else 0
            trans = get_affine_transform(
                c, s, r, (self.cfg.DATASET.INPUT_SIZE, self.cfg.DATASET.INPUT_SIZE)
            )
            trans_output = get_affine_transform(c, s_output, r, self.cfg.DATASET.OUTPUT_SIZE)
            trans_output = np.vstack([trans_output, [0,0,1]])
            for f, frame3d in enumerate(subseq_pose3d):
                krt[:,:,k] = self.composeProjectionMatrix(self.files[seq_name]["cams_matrix"][cam_id]['K'],
                                                   self.files[seq_name]["cams_matrix"][cam_id]['R'],
                                                   self.files[seq_name]["cams_matrix"][cam_id]['t'],
                                                   trans_output)
                for p, pose3d in enumerate(frame3d):
                    pose3d = self.mapKeypointsToCOCO(pose3d)
                    pose2d[p, :, :, k, f] = self.map3DkeypointsTo2d(
                        pose3d, self.files[seq_name]["cams_matrix"][cam_id])
                    keypoint2d[p, 0:17, :, k, f] =  pose2d[p, :, 0:17, k, f].transpose()
                    keypoint2d[p, 17:, :, k, f] = self.keypoint_generator(np.expand_dims(pose2d[p, :, :, k, f].transpose(), axis=0), 0.2)
                    x_check = np.bitwise_and(keypoint2d[p, :,0, k, f] >= 0, keypoint2d[p, :,0, k, f] <= WIDTH - 1)
                    y_check = np.bitwise_and(keypoint2d[p, :,1, k, f] >= 0, keypoint2d[p, :,1, k, f] <= HEIGHT - 1)
                    check = np.bitwise_and(x_check, y_check)
                    keypoint2d[p,:,2,k,f] = check * 1.0

                    for i in range(55):
                        keypoint2d[p, i,0:2, k, f] = affine_transform(
                            keypoint2d[p, i, 0:2, k, f], trans_output
                        )
                num_person[f] = p + 1

                heatmap = self.heatmap_generator(
                    keypoint2d[0:num_person[f], :, :, k, f]
                )
                a = cv2.imread(str(subseq_hdImg[cam_id][f]))
                try:
                    if a.shape[0] < 0 or a.shape[1] < 0:
                        print(str(subseq_hdImg[cam_id][f]))
                except AttributeError:
                    print(str(subseq_hdImg[cam_id][f]))
                img[:, :, :, k, f] = cv2.warpAffine(
                    cv2.imread(str(subseq_hdImg[cam_id][f])),
                    trans,
                    (self.cfg.DATASET.INPUT_SIZE, self.cfg.DATASET.INPUT_SIZE),
                    flags=cv2.INTER_LINEAR,
                )
                hm[:, :, :, k, f] = heatmap
        
        keypoint2d[:,:,[0,1],:,:] = keypoint2d[:,:,[1,0],:,:] # swap xyv to yxv
        img = img.transpose(2, 0, 1, 3, 4)
        return {
            "heatmap": torch.from_numpy(hm),
            "img": torch.from_numpy(img),
            "keypoint2d": torch.from_numpy(keypoint2d),
            "num_person": num_person,
            "KRT": torch.from_numpy(krt)
        }

    def readSkeletonFromPath(self, path):
        skeleton = []
        with open(path) as sfile:
            bframe = json.load(sfile)
        for body in bframe["bodies"]:
            keypoint = np.array(body["joints19"]).reshape((-1, 4)).transpose()
            skeleton.append(self.mapKeypointsToCOCO(keypoint))

        return skeleton

    def mapKeypointsToCOCO(self, cmu_keypoint):
        """
        Map the keypoint to COCO dataset format.
        """
        coco_keypoint = np.zeros((4, 17), np.float)
        for cmu_coco_idx in CMU_TO_COCO_JOINT_LABEL.items():
            cmu_idx = cmu_coco_idx[0]
            coco_idx = cmu_coco_idx[1]
            coco_keypoint[:, coco_idx] = cmu_keypoint[:, cmu_idx]

        return coco_keypoint

    def map3DkeypointsTo2d(self, keypoints3d, camera, affine_mat = None):
        keypoints2d = panutils.projectPoints(
            keypoints3d[0:3, :],
            camera["K"],
            camera["R"],
            camera["t"],
            camera["distCoef"],
            affine_mat
        )
        return keypoints2d

    def composeProjectionMatrix(self,K,R,T, affine_mat):
        R_homo = np.zeros((4,4))
        K_homo = np.zeros((3,4))

        R_homo[0:3,0:3] = R
        R_homo[0:3,3] = T[:,0] # [R|t]
        R_homo[3,3] = 1
        K_homo[0:3,0:3] = affine_mat*K
        K_homo[2,3] = 1

        out = K_homo@R_homo
        return out
class ReplicateViewPanoptic(PanopticDataset):
    def __init__(self, cfg, num_view, heatmap_generator=None, keypoint_generator=None, is_train=True) -> None:
        super().__init__(cfg, 1, heatmap_generator, keypoint_generator, is_train)
        self.num_replication = num_view
    
    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        hm = data['heatmap'].repeat(1,1,1,self.num_replication,1)
        img = data['img'].repeat(1,1,1,self.num_replication,1)
        keypoint2d = data['keypoint2d'].repeat(1,1,1,self.num_replication,1)
        num_person = data['num_person']
        return {
            "heatmap": hm,
            "img": img,
            "keypoint2d": keypoint2d,
            "num_person": num_person
        }