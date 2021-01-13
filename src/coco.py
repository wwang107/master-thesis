import cv2
import torch
from pycocotools.coco import COCO
from utils.vis import VIS_CONFIG
from data.build import build_dataset
from config.defualt import get_cfg_defaults
from matplotlib import pyplot as plt
from utils.vis.vis import save_batch_maps
from data.COCOKeypoints import CocoKeypoints
from data.PanopticDataset import PanopticDataset

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
        if joint[2] == 1:
            cv2.circle(image, (int(joint[0]), int(
                joint[1])), 12, (1,1,1),-1 )
        if joint[2] == 2:
            cv2.circle(image, (int(joint[0]), int(
                joint[1])), 12, idx_color[i], -1)
            # cv2.putText(image, str(i), (int(joint[0]), int(
            #     joint[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 255))

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], idx_color[part_idx[pair[0]]])

    return image


cfg = get_cfg_defaults()

def panoptic():
    from data.target_generators.target_generators import DirectionalKeypointsGenerator, HeatmapGenerator
    from utils.multiview import findFundamentalMat
    coco_skeleton = {
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"],

    "skeleton": [
        [16, 14], [14, 12], [17, 15], [15, 13],
        [12, 13], [6, 12], [7, 13], [6, 7],
        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
        [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    }

    directional_keypoint_generator = DirectionalKeypointsGenerator(
        cfg.DATASET.NUM_JOINTS,
        coco_skeleton['skeleton'])

    num_pair = len(coco_skeleton['skeleton'])
    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS + num_pair*2,
        cfg.DATASET.SIGMA)
    cmu = PanopticDataset(cfg,5, heatmap_generator, directional_keypoint_generator)
    
    data_obj = cmu.__getitem__(1000)
    
    frame = 0
    cams = (0,1,2,3,4)
    person_id = 0

    img1 = data_obj['img'][...,cams[0], 0]
    img2 = data_obj['img'][...,cams[1], 0]
    min = float(img1.min())
    max = float(img1.max())
    img1.add_(-min).div_(max - min + 1e-5)
    min = float(img2.min())
    max = float(img2.max())
    img2.add_(-min).div_(max - min + 1e-5)
    hm1 = data_obj['heatmap'][...,cams[0], 0]
    hm2 = data_obj['heatmap'][...,cams[1], 0]
    p1 = data_obj['KRT'][...,cams[0]]
    p2 = data_obj['KRT'][...,cams[1]]
    kpt1 = data_obj['keypoint2d'][...,cams[0],0][person_id][[0,7,8],0:2]
    kpt2 = data_obj['keypoint2d'][...,cams[1],0][person_id][[0,7,8],0:2]
    vis2 = vis1 = data_obj['keypoint2d'][...,cams[1],0][person_id][:,2]
    vis1 = data_obj['keypoint2d'][...,cams[0],0][person_id][:,2]

    img1_resized = cv2.resize(img1.cpu().permute(1,2,0).numpy(), cfg.DATASET.OUTPUT_SIZE)
    img2_resized = cv2.resize(img2.cpu().permute(1,2,0).numpy(), cfg.DATASET.OUTPUT_SIZE)
    
    f_mat12 = findFundamentalMat(p1,p2)
    kpt1_homo = torch.cat([kpt1,torch.ones_like(kpt1[:,0:1])], dim=1).to(f_mat12)
    kpt1_homo = kpt1_homo.transpose(0,1)
    l2 = torch.matmul(f_mat12, kpt1_homo).squeeze()

    pt1_in_im2 = lambda x: (l2[0,:] * x + l2[2,:]) / (-l2[1,:])
    y1_a = pt1_in_im2(0).numpy()
    y1_b = pt1_in_im2(10000).numpy()

    fig = plt.figure(figsize=(60, 60))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(img1_resized)
    ax2.imshow(img2_resized)

    ax1.scatter(kpt1[:,0],kpt1[:,1], color='yellow')
    ax2.scatter(kpt2[:,0],kpt2[:,1], color='yellow')
    ax2.plot([0, 10000], [y1_a, y1_b])
    ax2.set_ylim([cfg.DATASET.OUTPUT_SIZE[0], 0])
    ax2.set_xlim([0, cfg.DATASET.OUTPUT_SIZE[1]])
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

def with_aug():
    from data.transforms import transforms as T
    from data.transforms import FLIP_CONFIG
    coco_flip_index = FLIP_CONFIG['COCO']
    input_size = cfg.DATASET.INPUT_SIZE
    output_size = cfg.DATASET.OUTPUT_SIZE
    min_scale = cfg.DATASET.MIN_SCALE
    max_scale = cfg.DATASET.MAX_SCALE
    max_rotation = cfg.DATASET.MAX_ROTATION
    max_translate = cfg.DATASET.MAX_TRANSLATE

    transforms_origin = T.Compose(
            [   
                T.Resize(input_size, output_size),
                T.ToTensor()
            ]
        )
    
    transforms_affine = T.Compose(
            [   
                T.RandomAffineTransform(
                    input_size,
                    output_size,
                    max_rotation,
                    min_scale,
                    max_scale,
                    max_translate,
                ),
                # T.RandomHorizontalFlip(coco_flip_index, output_size, prob=0.5),
                T.ToTensor()
            ]
        )

    transforms_flip = T.Compose(
            [   
                T.RandomAffineTransform(
                    input_size,
                    output_size,
                    0,
                    1,
                    1,
                    0,
                ),
                T.RandomHorizontalFlip(coco_flip_index, output_size, prob=1.0),
                T.ToTensor()
            ]
        )
    
    coco_origin = CocoKeypoints(cfg, cfg.DATASET.TEST, True, None, None, transforms_origin)
    coco_affine = CocoKeypoints(cfg, cfg.DATASET.TEST, True, None, None, transforms_affine)
    coco_flip = CocoKeypoints(cfg, cfg.DATASET.TEST, True, None, None, transforms_flip)
    fig = plt.figure(figsize=(40, 40))
    idx = [29,36,40]
    cols = 3

    for i,id in enumerate(idx):
        origin_img = fig.add_subplot(len(idx),cols,i*cols+1)
        data = coco_origin.__getitem__(id)
        img = data['images']
        joints = data['joints']
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
        img = img.cpu().permute(1,2,0).numpy()
        for person in joints:
            img_joint = add_joints(img, person)
        origin_img.imshow(img)
        origin_img.axis('off')

        affine_img = fig.add_subplot(len(idx),cols,i*cols+2)
        data = coco_affine.__getitem__(id)
        img = data['images']
        joints = data['joints']
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
        img = img.cpu().permute(1,2,0).numpy()
        for person in joints:
            img_joint = add_joints(img, person)
        affine_img.imshow(img_joint)
        affine_img.axis('off')

        flip_img = fig.add_subplot(len(idx),cols,i*cols+3)
        data = coco_flip.__getitem__(id)
        img = data['images']
        joints = data['joints']
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
        img = img.cpu().permute(1,2,0).numpy()
        for person in joints:
            img_joint = add_joints(img, person)
        flip_img.imshow(img_joint)
        flip_img.axis('off')

        if i == len(idx)-1:
            origin_img.text(0.5,-0.1, "original", size=12, ha="center", transform=origin_img.transAxes)
            affine_img.text(0.5,-0.1, "affined", size=12, ha="center", transform=affine_img.transAxes)
            flip_img.text(0.5,-0.1, "left-right flipped", size=12, ha="center", transform=flip_img.transAxes)
    plt.tight_layout()
    plt.show()

     
def without_aug():
    coco = build_dataset(cfg=cfg, is_train=False)
    fig = plt.figure(figsize=(40, 40))
    idx = [29,36,40]
    cols = 3
    for i,id in enumerate(idx):
        print(i)
        ax_img = fig.add_subplot(len(idx),cols,i*cols+1)
        data = coco.__getitem__(id)

    

        img = data['images']
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
        img = img.cpu().permute(1,2,0).numpy()

        joints = data['joints']
        ax_img.imshow(img)
        ax_img.axis('off')

        ax_joint = fig.add_subplot(len(idx),cols,i*cols+2)
        img_joint = img.copy()
        for person in joints:
            img_joint = add_joints(img_joint, person)
        ax_joint.imshow(img_joint)
        ax_joint.axis('off')

        mask = data['masks']
        ax_mask = fig.add_subplot(len(idx),cols,i*cols+3)
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.axis('off')

        if i == len(idx)-1:
            ax_img.text(0.5,-0.1, "(a)", size=12, ha="center", transform=ax_img.transAxes)
            ax_joint.text(0.5,-0.1, "(b)", size=12, ha="center", transform=ax_joint.transAxes)
            ax_mask.text(0.5,-0.1, "(c)", size=12, ha="center", transform=ax_mask.transAxes)


    plt.show()

    data1 = coco.__getitem__(29)
    data2 = coco.__getitem__(40)

    heatmap1 = torch.from_numpy(data1['heatmaps']).unsqueeze(dim=0)[:,0:17].max(dim=1, keepdim=True)[0]
    heatmap2 = torch.from_numpy(data2['heatmaps']).unsqueeze(dim=0)[:,0:17].max(dim=1, keepdim=True)[0]
    images = torch.cat((data1['images'].unsqueeze(dim=0), data2['images'].unsqueeze(dim=0)))
    hms = torch.cat((heatmap1, heatmap2))
    vis_heatmap = save_batch_maps(images, hms)
    cv2.imwrite("coco-heatmap.png", vis_heatmap)

if __name__ == "__main__":
    # with_aug()
    panoptic()