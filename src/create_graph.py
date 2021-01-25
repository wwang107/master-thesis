import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.load_model import load_model_state_dict
from models.resnet.model import CustomizedResnet
from models.unet.TemporalUnet import TemporalUnet
from models.fusion_net.model import FusionNet
from models.aggregate.model import AggregateModel
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from models.epipolar.EpipolarTransformer import Epipolar
from tqdm import tqdm 
from utils.triangulation import Pose, generate_3d_cloud, cluster_3d_cloud, extract_poses, get_cmap
from utils.vis.vis import add_joint_matplot
import cv2
import matplotlib.pyplot as plt

def load_weight(model, state_dict):
    model.load_state_dict(state_dict)
    return model

def main(hparams):
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels, out_channels, num_feature = 55, 55, hparams.num_feat
    replicate_view = hparams.replicate_view
    is_train_input_encoder = hparams.input_encoder

    # Baseline
    baseline = CustomizedResnet(use_pretrained=True)
    baseline = load_weight(baseline,load_model_state_dict("pretrain/resnet50/train_on_cmu/best_18.pth", device))
    baseline = baseline.to(device)

    # Full-framework
    resnet = CustomizedResnet(use_pretrained=True)
    resnet = load_weight(resnet,load_model_state_dict("pretrain/resnet50/best_68.pth", device))
    fuse_model = FusionNet(2*in_channels, out_channels, num_feature, input_frame=1)

    model = AggregateModel(resnet, Epipolar(debug=False), None, fuse_model, None,
                           None, in_channels, out_channels, 
                           train_input_heatmap_encoder=is_train_input_encoder, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    fusion_state_dict = torch.load('pretrain/fusion-model/with-resnet50/epoch=19.ckpt')
    model.load_state_dict(fusion_state_dict['state_dict'])
    temporal_model = TemporalUnet(in_channels, out_channels, num_feature,9)
    old_model = AggregateModel(None, None, None, None, temporal_model,
                           None, in_channels, out_channels, 
                           train_input_heatmap_encoder=is_train_input_encoder, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    
    temporal_state_dict = torch.load('pretrain/temporal-9-frame-model/epoch=5.ckpt')
    old_model.load_state_dict(temporal_state_dict['state_dict'], strict=False)
    model.temporal_encoder = old_model.temporal_encoder
    model = model.to(device)
    
    data_loader = {
        # 'train': make_dataloader(cfg, dataset_name='cmu', is_train=True, replicate_view=replicate_view),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False, replicate_view=replicate_view)
    }

    parameters = {
    'scale_to_mm': 10.0,
    'meanshift_radius_in_mm': 200.0,
    'meanshift_gamma': 10000.0,
    'points3d_merge_distance_mm': 50.0
    }

    plot_num = 0
    with torch.no_grad():
        for data in tqdm(data_loader['valid']):
            batch_imgs = data['img'].to(device)
            batch_krt = data['KRT'].to(device)
            batch_keypoint3d = data['keypoint3d']
            batch_keypoint = data['keypoint2d'].to(device)
            batch_bboxes = data['bboxes'].to(device)
            batch_num_person = data['num_person'].to(device)

            out = model(batch_imgs, batch_krt)
            B,C,H,W,V,F = out['input_heatmap_encoder'].size()
            input_hms = out['input_heatmap_encoder'][0].cpu().numpy().transpose(1, 2, 0,3,4)
            fusion_hms = out['fusion_net'][0].cpu().numpy().transpose(1, 2, 0,3,4)
            temporal_hms = out['temporal_encoder'][0].cpu().numpy().transpose(1, 2, 0,3)
            
            baseline_hms = torch.zeros((1,55,128,128,5)).to(device)
            for v in range(5):
                baseline_hms[..., v] = baseline(batch_imgs[...,v,9//2])
            baseline_hms = baseline_hms[0].cpu().numpy().transpose(1, 2, 0,3)
            # 2D graph
            
            

            # 3D graph
            cameras = []
            Point3d_baseline_heatmap = []
            Point3d_input_heatmap = []
            Point3d_fusion_heatmap = []
            Point3d_temporal_heatmap = []

            for i in range(V):
                cameras.append(batch_krt[0, ..., i].cpu())

            # BASELINE_HEATMAP
            for jid in tqdm(range(55)):
                HMs_baseline_heatmap = []
                for v in range(V):
                    HMs_baseline_heatmap.append(baseline_hms[:,:,jid,v])

                points3d, values = generate_3d_cloud(HMs_baseline_heatmap, cameras, Axs=None)
                if isinstance(points3d, list):
                    Point3d_baseline_heatmap.append([])
                    continue
                points3d, values = cluster_3d_cloud(points3d, values, Cameras=cameras, Axs=None)
                Point3d_baseline_heatmap.append((points3d, values))


            # INPUT_HEATMAP
            for jid in tqdm(range(55)):
                HMs_input_heatmap = []
                for v in range(V):
                    HMs_input_heatmap.append(input_hms[:,:,jid,v,F//2])

                points3d, values = generate_3d_cloud(HMs_input_heatmap, cameras, Axs=None)
                if isinstance(points3d, list):
                    Point3d_input_heatmap.append([])
                    continue
                points3d, values = cluster_3d_cloud(points3d, values, Cameras=cameras, Axs=None)
                Point3d_input_heatmap.append((points3d, values))
            
            # FUSINO_HEATMAP
            for jid in tqdm(range(55)):
                HMs_fusion_heatmap = []
                for v in range(V):
                    HMs_fusion_heatmap.append(fusion_hms[:,:,jid,v,F//2])

                points3d, values = generate_3d_cloud(HMs_fusion_heatmap, cameras, threshold=0.3,Axs=None)
                if isinstance(points3d, list):
                    Point3d_fusion_heatmap.append([])
                    continue
                points3d, values = cluster_3d_cloud(points3d, values, Cameras=cameras, Axs=None)
                Point3d_fusion_heatmap.append((points3d, values))

            # TEMPORAL HEATMAP
            for jid in tqdm(range(55)):
                HMs_temoral_heatmap = []
                for v in range(V):
                    HMs_temoral_heatmap.append(temporal_hms[:,:,jid,v])

                points3d, values = generate_3d_cloud(HMs_temoral_heatmap, cameras, threshold=0.25, Axs=None)
                if isinstance(points3d, list):
                    Point3d_temporal_heatmap.append([])
                    continue
                points3d, values = cluster_3d_cloud(points3d, values, Cameras=cameras, Axs=None)
                Point3d_temporal_heatmap.append((points3d, values))

            poses_baseline = extract_poses(Point3d_baseline_heatmap, scale2mm=parameters['scale_to_mm'])
            poses_input = extract_poses(Point3d_input_heatmap, scale2mm=parameters['scale_to_mm'])
            poses_fusion_heatmap = extract_poses(Point3d_fusion_heatmap, scale2mm=parameters['scale_to_mm'])
            poses_temporal_heatmap = extract_poses(Point3d_temporal_heatmap, scale2mm=parameters['scale_to_mm'])

            imgs = batch_imgs[0,..., F//2]
            imgs = imgs.clone().cpu().float()
            min = float(imgs.min())
            max = float(imgs.max())
            imgs.add_(-min).div_(max - min + 1e-5)

            fig = plt.figure(figsize=(60, 60))
            Axs = []
            for r in range(5): # gt # baseline # input # fusion # temporal 
                for c in range(5):
                    ax = fig.add_subplot(5, 5, r*5+c+1)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_xlim([0, 255/2]); ax.set_ylim([255/2, 0])
                    Axs.append(ax)
                    if r == 0:
                        if c==0:
                            ax.set_title('View 0', {'fontsize': 40},color='k')
                        if c==1:
                            ax.set_title('View 1', {'fontsize': 40},color='k')
                        if c==2:
                            ax.set_title('View 2', {'fontsize': 40},color='k')
                        if c==3:
                            ax.set_title('View 3', {'fontsize': 40},color='k')
                        if c==4:
                            ax.set_title('View 4', {'fontsize': 40},color='k')
                    if c==0:
                        if r==0:
                            ax.set_ylabel('Ground truth', {'fontsize': 40})
                        if r==1:
                            ax.set_ylabel('Baseline', {'fontsize': 40})
                        if r==2:
                            ax.set_ylabel('2D backbone', {'fontsize': 40})
                        if r==3:
                            ax.set_ylabel('View fusion', {'fontsize': 40})
                        if r==4:
                            ax.set_ylabel('Temporal fusion', {'fontsize': 40})
                    
                    Axs[r*5+c].imshow(cv2.resize(imgs[...,c].permute(1, 2, 0).numpy(), dsize=None, fx=1/2, fy=1/2))
            
            batch_keypoint = batch_keypoint[0,...,F//2].cpu().numpy()
            n = batch_num_person[0,...,F//2].cpu().numpy()
            cmap = get_cmap(10)
            for c, ax in enumerate(Axs[0*5+0:0*5+5]):
                img = cv2.resize(imgs[...,c].permute(1, 2, 0)\
                                            .cpu().numpy(), dsize=None, fx=1/2, fy=1/2)
                ax.imshow(img)
                for i,joints in enumerate(batch_keypoint[0:n]):
                    img = add_joint_matplot(img, joints[0:17,...,c],ax, cmap(i))
                
            cmap = get_cmap(len(poses_baseline))
            for ax, cam in zip(Axs[1*5+0:1*5+5], cameras):
                for i, pose in enumerate(poses_baseline):
                    if pose.count_limbs() > 5:
                        pose.plot(ax, cam, cmap(i))
            
            cmap = get_cmap(len(poses_input))
            for ax, cam in zip(Axs[2*5+0:2*5+5], cameras):
                for i, pose in enumerate(poses_input):
                    if pose.count_limbs() > 5:
                        pose.plot(ax, cam, cmap(i))
            
            cmap = get_cmap(len(poses_fusion_heatmap))
            for ax, cam in zip(Axs[3*5+0:3*5+5], cameras):
                for i, pose in enumerate(poses_fusion_heatmap):
                    if pose.count_limbs() > 5:
                        pose.plot(ax, cam, cmap(i))
            
            cmap = get_cmap(len(poses_temporal_heatmap))
            for ax, cam in zip(Axs[4*5+0:4*5+5], cameras):
                for i, pose in enumerate(poses_temporal_heatmap):
                    if pose.count_limbs() > 5:
                        pose.plot(ax, cam, cmap(i))
            plt.show()
            fig.savefig('test_results/3D-{}.png'.format(plot_num))
            plot_num += 1

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