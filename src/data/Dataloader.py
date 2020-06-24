from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os.path
from visualize.visualizor import visualizor


class COCOHuamnDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader. 

    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
        "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    """

    def __init__(self, dataDir, dataType, gussianStd=0.5, transform=None, augmentTransform=None, debug=True):
        annFileInstance = '{}/annotations/instances_{}.json'.format(
            dataDir, dataType)
        annFileKeypoint = '{}/annotations/person_keypoints_{}.json'.format(
            dataDir, dataType)
        # self.numJoints = 17  # coco
        self.dataDir = dataDir
        self.dataType = dataType
        self.cocoIns = COCO(annFileInstance)
        self.coocKpt = COCO(annFileKeypoint)
        self.transform = transform
        self.augmentTransform = augmentTransform
        self.catId = self.cocoIns.getCatIds(catNms=['person'])
        self.ids = self.cocoIns.getImgIds(catIds=self.catId)
        self.skeleton = np.array(self.coocKpt.loadCats(self.catId)[0]['skeleton']) - 1

        self.sigma = 10
        self.heatmapSize = np.array((320,320)) # (width, height)
        

    def __getitem__(self, index):
        painter = visualizor()
        img = self.__loadImage(105)
        x,y,v = self.__loadKeyPoints(105)
        dx, dy = self.__addDirectionalKpt(x, y)
        X = np.concatenate((x, dx),axis=1)
        Y = np.concatenate((y, dy),axis=1)
        heatmap = self.__createHeatmap(X, Y, img.size, self.heatmapSize)
        
    def __addDirectionalKpt(self, x, y, ratio=.9):
        def interpolation(x0, y0, x1, y1, t):
            return (
                x0 + t * (x1 - x0),
                y0 + t * (y1 - y0)
                )
        directX = []
        directY = []
        for sk in self.skeleton:
            v0 = interpolation(x[:, sk[0]], y[:, sk[0]],
                               x[:, sk[1]], y[:, sk[1]], ratio)
            v1 = interpolation(x[:, sk[0]], y[:, sk[0]],
                               x[:, sk[1]], y[:, sk[1]], 1-ratio)
            directX.append(v0[0])
            directY.append(v0[1])
            directX.append(v1[0])
            directY.append(v1[1])

        return (
            np.array(directX).transpose(), 
            np.array(directY).transpose()
            )

    def __createHeatmap(self, keypointsX, keypointsY, imgSize, heatmapSize):
        numJoint = keypointsX.shape[1]
        target = np.zeros((numJoint,
                           imgSize[0],
                           imgSize[1]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(numJoint):
            feat_stride = imgSize / heatmapSize
            mu_x = (keypointsX[:, joint_id] / feat_stride[0]).astype(int)
            mu_y = (keypointsY[:, joint_id] / feat_stride[1]).astype(int)
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size, mu_y + tmp_size ]

            xx, yy = np.meshgrid(
                np.arange(heatmapSize[0]), np.arange((heatmapSize[1])))
            
            for i in range(mu_x.shape[0]):
                g = np.exp(- ((xx- mu_x[i]) ** 2 + (yy - mu_y[i]) ** 2) /
                           (2 * self.sigma ** 2))

        return target, target_weight

            
    def __loadKeyPoints(self, index:int):
        validKeyPoints = []
        cocoKpt = self.coocKpt
        annIds = cocoKpt.getAnnIds(
            imgIds=self.ids[index], catIds=self.catId, iscrowd=None)
        annObjs = cocoKpt.loadAnns(annIds)

        for obj in annObjs:
            if max(obj['keypoints']) == 0:
                continue
            else:
                validKeyPoints.append(obj['keypoints'])

        kptArray = np.array(validKeyPoints)
        x = kptArray[:, 0::3]
        y = kptArray[:, 1::3]
        v = kptArray[:, 2::3]

        return (x, y, v)

    def __loadImage(self, index):
        path = self.cocoIns.loadImgs(self.ids[index])[0]['file_name']
        return Image.open(os.path.join(self.dataDir, 'images', self.dataType, path)).convert('RGB')

    def __len__(self):
        return len(self.ids)
