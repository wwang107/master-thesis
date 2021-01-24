from numpy.core.numeric import zeros_like
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import dataloader
from utils.load_model import load_model_state_dict
from models.resnet.model import CustomizedResnet
from models.unet.TemporalUnet import TemporalUnet
from models.fusion_net.model import FusionNet
from models.aggregate.model import AggregateModel
from config.defualt import get_cfg_defaults
from data.build import build_dataset, make_dataloader
from callbacks.callbacks import LogConfusionTable, LogModelHeatmaps
from models.epipolar.EpipolarTransformer import Epipolar
from utils.multiview import camera_center
from utils.vis.vis import add_joints
from tqdm import tqdm 
import cv2
import math
import matplotlib.pyplot as plt
from data.build import build_CMU_dataset
from utils.triangulation import Pose, generate_3d_cloud, cluster_3d_cloud, extract_poses, get_cmap, find_nearest_pose, calculate_pckh3d
import numpy as np


def load_weight(model, state_dict):
    model.load_state_dict(state_dict)
    return model

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2) * 10000

def calculate_prediction(hms, bboxes, imgs = None, method='max'):
    
    if imgs is not None:
        imgs = imgs.cpu().clone().float()
        min = float(imgs.min())
        max = float(imgs.max())
        imgs.add_(-min).div_(max - min + 1e-5)

    batch_num, max_person = bboxes.size(0), bboxes.size(1)
    batch_predictions = torch.zeros((batch_num, max_person, 17,3)).to(bboxes)
    nms = torch.nn.Threshold(0.0, 0)
    for i,hm in enumerate(hms):
        for k,bbox in enumerate(bboxes[i]):
            pred_joint = torch.zeros(17,3).to(hm)
            if bbox[4]<1:
                continue
            x_ul, y_ul, w, h = int(bbox[0]), int(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
            roi = hm[:,y_ul:y_ul+h, x_ul:x_ul+w]
            if method != 'avg':
                roi_reshape = torch.reshape(roi,(-1, w*h))
                maxval, maxidx = torch.max(roi_reshape, dim=1)
                x_loc = maxidx % w
                y_loc = maxidx // w
                batch_predictions[i,k,:,0] = x_ul + x_loc
                batch_predictions[i,k,:,1] = y_ul + y_loc 
                batch_predictions[i,k,:,2] = 1 
    return batch_predictions

def get_true_positive(batch_gt, batch_pred, batch_num_person, scale=0.5):
    tp_num = torch.zeros((17)).to(batch_gt)
    gt_num = torch.zeros((17)).to(batch_gt)
    for i, gt in enumerate(batch_gt):
        num_person = batch_num_person[i]
        gt = gt[0:num_person]
        pred = batch_pred[i]
        for id in range(gt.size(0)):
            gt_person = gt[id]
            head_length = torch.sqrt(torch.sum((gt_person[1,0:2] - gt_person[2,0:2])**2)) * scale
            visible = gt_person[:,2] > 0
            pred_person = pred[id]
            joint_differece = torch.sqrt(torch.sum((pred_person[:,0:2] - gt_person[:,0:2])**2, dim = 1))
            tp_num = tp_num + (joint_differece <= head_length) * 1.0
            gt_num = gt_num + visible * 1.0
    return tp_num, gt_num

def main(hparams):
    print(hparams)
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels, out_channels, num_feature = 55, 55, hparams.num_feat
    num_levels = hparams.num_level
    replicate_view = hparams.replicate_view
    is_train_input_encoder = hparams.input_encoder

    resnet = CustomizedResnet(use_pretrained=True)
    resnet = load_weight(resnet,load_model_state_dict("pretrain/resnet50/train_on_cmu/best_18.pth", device))
    # print(resnet)

    fuse_model = FusionNet(2*in_channels, out_channels, num_feature, input_frame=1)
    model = AggregateModel(resnet, Epipolar(debug=False), None, fuse_model, None,
                           weighted_mse_loss, in_channels, out_channels, 
                           train_input_heatmap_encoder=is_train_input_encoder, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    fusion_state_dict = torch.load('pretrain/fusion-model/with-resnet50/epoch=19.ckpt')
    model.load_state_dict(fusion_state_dict['state_dict'])
    temporal_model = TemporalUnet(in_channels, out_channels, num_feature,9)
    print(temporal_model)
    old_model = AggregateModel(None, None, None, None, temporal_model,
                           weighted_mse_loss, in_channels, out_channels, 
                           train_input_heatmap_encoder=is_train_input_encoder, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    
    temporal_state_dict = torch.load('pretrain/temporal-9-frame-model/epoch=5.ckpt')
    old_model.load_state_dict(temporal_state_dict['state_dict'], strict=False)
    model.temporal_encoder = old_model.temporal_encoder
    data_loader = {
        # 'train': make_dataloader(cfg, dataset_name='cmu', is_train=True, replicate_view=replicate_view),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False, replicate_view=replicate_view)
    }
    
    model = model.to(device)
    pckh = {'resnet50':{'tp':torch.zeros(17).to(device),'gt':torch.zeros(17).to(device), 'pckh': torch.zeros(17).to(device)},
            'fusion_net':{'tp':torch.zeros(17).to(device),'gt':torch.zeros(17).to(device), 'pckh': torch.zeros(17).to(device)},
            'temporal_fusion':{'tp':torch.zeros(17).to(device),'gt':torch.zeros(17).to(device), 'pckh': torch.zeros(17).to(device)}}
    gt_num = torch.zeros(17).to(device)

    parameters = {
    'scale_to_mm': 10.0,
    'meanshift_radius_in_mm': 200.0,
    'meanshift_gamma': 10000.0,
    'points3d_merge_distance_mm': 50.0
    }

    tot_person = 0
    tp = np.zeros(17)
    used_joint = np.asarray([5,6,7,8,9,10,11,12,13,14,15,16])
    with torch.no_grad():
        for data in tqdm(data_loader['valid']):
            batch_imgs = data['img'].to(device)
            batch_krt = data['KRT'].to(device)
            batch_keypoint3d = data['keypoint3d']
            batch_keypoint = data['keypoint2d'].to(device)[:,:,0:17]
            batch_bboxes = data['bboxes'].to(device)
            batch_num_person = data['num_person'].to(device)

            out = model(batch_imgs, batch_krt)
            B,C,H,W,V,F = out['input_heatmap_encoder'].size()
            
            input_hms = out['input_heatmap_encoder'][:,0:17]
            fusion_hms = out['fusion_net']
            temporal_hms = out['temporal_encoder']

            # # Prepare the input of 3d pose estimation
            # hm = []
            # cameras = []
            # for i in range(V):
            #     hm.append(input_hms[0,..., i, F//2].cpu().numpy().transpose(1, 2, 0))
            #     cameras.append(batch_krt[0, ..., i].cpu())
            
            # Points3d = []
            # for jid in tqdm(range(55)):
            #     HMs = []
            #     for h in hm:
            #         HMs.append(h[:,:,jid])
            #     # HMs = [hm[0][:,:,jid], hm[1][:,:,jid], hm[2][:,:,jid], hm[3][:,:,jid]]
            #     points3d, values = generate_3d_cloud(HMs, cameras, Axs=None)
            #     if isinstance(points3d, list):
            #         Points3d.append([])
            #         continue
            #     points3d, values = cluster_3d_cloud(points3d, values, Cameras=cameras, Axs=None)
            #     Points3d.append((points3d, values))

            # num_person = batch_num_person[0,F//2]
            # if num_person == 0:
            #     continue
            # else:
            #     gt_poses = batch_keypoint3d[0,0:num_person,:,:,F//2].cpu().numpy()
            #     poses = extract_poses(Points3d, scale2mm=parameters['scale_to_mm'])
            #     if len(poses)>0:
            #         gt_joints, est_joints = find_nearest_pose(gt_poses, poses)
            #         tp += calculate_pckh3d(gt_joints, est_joints[:,0:3])
            #     tot_person += num_person
            
            # print(tp[used_joint]/tot_person.cpu().numpy())
            # np.savetxt('test_results/input-cmu-5-view.txt',tp[used_joint]/tot_person.cpu().numpy())

            # imgs = batch_imgs[0,..., F//2]
            # imgs = imgs.clone().cpu().float()
            # min = float(imgs.min())
            # max = float(imgs.max())
            # imgs.add_(-min).div_(max - min + 1e-5)

            # fig = plt.figure(figsize=(12, 12))
            # Axs = []
            # for i in range(5):
            #     ax = fig.add_subplot(2, 3, i+1)
            #     ax.set_xlim([0, 255/2]); ax.set_ylim([255/2, 0])
            #     Axs.append(ax)

            # Axs[0].imshow(cv2.resize(imgs[...,0].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            # Axs[1].imshow(cv2.resize(imgs[...,1].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            # Axs[2].imshow(cv2.resize(imgs[...,2].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            # Axs[3].imshow(cv2.resize(imgs[...,3].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            # Axs[4].imshow(cv2.resize(imgs[...,4].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            
            # cmap = get_cmap(len(poses))
            # for ax, cam in zip(Axs, cameras):
            #     for i, pose in enumerate(poses):
            #         if pose.count_limbs() > 5:
            #             pose.plot(ax, cam, cmap(i))

            # plt.show()

            for i in range(V):
                for j in range(F):
                    input_prediction = calculate_prediction(input_hms[...,i,j], batch_bboxes[...,i,j])
                    tp, gt = get_true_positive(batch_keypoint[...,i,j], input_prediction, batch_num_person[:,j])
                    pckh['resnet50']['tp'] += tp
                    pckh['resnet50']['gt'] += gt
            print(torch.true_divide(pckh['resnet50']['tp'], pckh['resnet50']['gt']))
            # for i in range(V):
            #     for j in range(F):
            #         fusion_prediction = calculate_prediction(fusion_hms[...,i,j], batch_bboxes[...,i,j])
            #         tp, gt = get_true_positive(batch_keypoint[...,i,j], fusion_prediction, batch_num_person[:,j])
            #         pckh['fusion_net']['tp'] += tp
            #         pckh['fusion_net']['gt'] += gt

            # for i in range(V):
            #     temporal_prediction = calculate_prediction(temoral_hms[...,i], batch_bboxes[...,i,F//2])
            #     tp, gt = get_true_positive(batch_keypoint[...,i,F//2], temporal_prediction, batch_num_person[:,F//2])
            #     pckh['temporal_fusion']['tp'] += tp
            #     pckh['temporal_fusion']['gt'] += gt
        # print('end')
        # pckh = tp/tot_person.cpu().numpy()
        
    for model_name in pckh.keys():
        pckh[model_name]['pckh'] = torch.true_divide(pckh[model_name]['tp'], pckh[model_name]['gt'])
    
    print(pckh)
    torch.save(pckh, 'test_results/pckh.pth')
        
        
        




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--images_dir', default='images')
    parser.add_argument('--resnet_weight_dir', default='pretrain/CutomizeResNet-experiment/best_13.pth')
    parser.add_argument('--temporal_encoder', action='store_true')
    parser.add_argument('--view_encoder', action='store_true')
    parser.add_argument('--input_encoder', action='store_true')
    parser.add_argument('--replicate_view', action='store_true')
    parser.add_argument('-num_feat', default=55)
    parser.add_argument('-num_level', default=2)
    args = parser.parse_args()
    main(args)
