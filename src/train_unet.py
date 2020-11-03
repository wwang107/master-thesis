from typing import Dict, List
import torch
import cv2
import numpy as np
from torchvision import models
from models.losses import JointsMSELoss
# from model.unet.model import UNet3D
from models.aggregate.model import AggregateModel
from data.build import make_dataloader
from utils.load_model import load_model_state_dict
from utils.vis.vis import save_batch_multi_view_with_heatmap, save_batch_keypoints, save_keypoint_detection
from utils.vis.graph import group_bar_plot
# from model.resnet.resnet import CustomizedResnet
from config.defualt import get_cfg_defaults
from pathlib import Path
from datetime import datetime
from utils.writer.writer import TensorBoardWriter
from utils.load_model import load_checkpoint
from utils.utils import find_keypoints_from_heatmaps, match_detected_groundtruth_keypoint
from trainer.model_trainer import save_checkpoint

PATH_TO_CHECKPOINT = '/home/wei/master-thesis/pretrain/unet-baseline/best_0.pth'
PATH_TO_HEATMAP_MODEL = '/home/wei/master-thesis/pretrain/CutomizeResNet-experiment/best_13.pth'


def main():
    cfg = get_cfg_defaults()
    # log_dir = str(Path.cwd().joinpath(
    #     'runs', 'myunet', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')).resolve())
    # tsboard = TensorBoardWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agg_model = AggregateModel(
        in_channels=55, out_channels=55, num_feature=110, num_levels=2).to(device)
        # weight_resnet=load_model_state_dict(PATH_TO_HEATMAP_MODEL, 'cpu'),
        # weight_unet=load_model_state_dict(PATH_TO_CHECKPOINT, 'cpu'))
    agg_model = agg_model.load_weight(load_model_state_dict(PATH_TO_HEATMAP_MODEL, device), load_model_state_dict(PATH_TO_CHECKPOINT, device))
    # unet = UNet3D(in_channels=55, out_channels=55,
    #               num_feature=110, num_levels=2)
    # unet = unet.to(device)
    # heatmap_model = CustomizedResnet()
    # heatmap_model.load_state_dict(load_model_state_dict(PATH_TO_HEATMAP_MODEL))
    # heatmap_model.to(device)

    data_loader = {
        'train': make_dataloader(cfg, dataset_name='cmu', is_train=True),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False)
    }
    criterion = JointsMSELoss(use_target_weight=True)
    num_view = cfg.DATASET.NUM_VIEW
    num_frame = cfg.DATASET.NUM_FRAME_PER_SUBSEQ

    lowest_loss = float('inf')

    # stats
    stats = {'resnet': {'tp': 0, 'fp': 0, 'fn': 0},
             'unet': {'tp': 0, 'fp': 0, 'fn': 0}}

    for epoch in range(1):
        for phase in ['valid']:
            epoch_loss = {phase: 0}

            # if phase == 'train':
            #     unet.train()  # Set model to training mode
            #     num_view = cfg.DATASET.NUM_VIEW
            # else:
            #     unet.eval()   # Set model to evaluate mode
            #     num_view = 4
            #     valid_images = {}
            #     for i in range(num_view):
            #         valid_images['input_heatmap_cam_view_{}'.format(i)] = []
            #         valid_images['predict_heatmap_cam_view_{}'.format(i)] = []
            #         valid_images['predict_groundtruth_cam_view_{}'.format(i)] = [
            #         ]

            running_loss = 0.0
            # Iterate over data.
            for i, data in enumerate(data_loader[phase]):
                print('{}/{}'.format(i, len(data_loader[phase])))

                # optimizer.zero_grad()

                batch_kpt, batch_hm_gt, batch_img, num_person = \
                    data['keypoint2d'], data['heatmap'], data['img'], data['num_person']
                batch_hm_gt, batch_img = batch_hm_gt.to(
                    device), batch_img.float().to(device)
                print(agg_model)
                agg_model(data)
                
                # heatmap_input = torch.zeros(batch_hm_gt.size()).cuda(device)
                # batch_img = batch_img.permute((0, 3, 1, 2, 4, 5))
                # with torch.set_grad_enabled(phase == 'train'):
                #     for cam_view in range(num_view):
                #         for frame in range(num_frame):
                #             heatmap_input[:, :, :, :, cam_view, frame] = heatmap_model(
                #                 batch_img[:, :, :, :, cam_view, frame])

                # forward
                # track history if only in train
                    # loss = 0.0
                    # batch_hm_gt = batch_hm_gt.to(device)
                    # x = unet(heatmap_input[:, :, :, :, :, -1])
                    # for k in range(num_view):
                    #     loss += criterion(x[:, :, :, :, k],
                    #                       batch_hm_gt[:, :, :, :, k, -1])
                    # if phase == 'train':
                    #     loss.backward()
                    #     optimizer.step()

                    # if phase == 'valid':
                    #     for k in range(num_view):
                    #         unet_detected_keypoint = find_keypoints_from_heatmaps(
                    #             x[:, :, :, :, k], threshold=0.5)
                    #         unet_result = match_detected_groundtruth_keypoint(
                    #             batch_kpt[:, :, :, :, k, -1], unet_detected_keypoint, 1)
                    #         resnet_detected_keypoint = find_keypoints_from_heatmaps(
                    #             heatmap_input[:, :, :, :, k, -1], threshold=0.5)
                    #         resnet_result = match_detected_groundtruth_keypoint(
                    #             batch_kpt[:, :, :, :, k, -1], resnet_detected_keypoint, 1)
                            # stats['resnet']['fp']+= resnet_result['false positive']['num']
                            # stats['resnet']['tp']+= resnet_result['true positive']['num']
                            # stats['resnet']['fn']+= resnet_result['false negative']['num']
                            # stats['unet']['fp']+= unet_result['false positive']['num']
                            # stats['unet']['tp']+= unet_result['true positive']['num']
                            # stats['unet']['fn']+= unet_result['false negative']['num']
                            # unet_img = save_keypoint_detection(batch_image=batch_img[:, :, :, :, k, -1],
                            #                                    batch_heatmaps=x[:,
                            #                                                     :, :, :, k],
                            #                                    batch_fp_points=unet_result['false positive']['points'],
                            #                                    batch_fn_points=unet_result['false negative']['points'],
                            #                                    batch_tp_points=unet_result['true positive']['points'])
                            # resnet_img = save_keypoint_detection(batch_image=batch_img[:, :, :, :, k, -1],
                            #                                      batch_heatmaps=heatmap_input[:,
                            #                                                                   :, :, :, k, -1],
                            #                                      batch_fp_points=resnet_result['false positive']['points'],
                            #                                      batch_fn_points=resnet_result['false negative']['points'],
                            #                                      batch_tp_points=resnet_result['true positive']['points'])
                            # group_bar_plot(
                            #     [[unet_result['false positive']['num'], unet_result['true positive']['num'], unet_result['false negative']['num']],
                            #      [resnet_result['false positive']['num'], resnet_result['true positive']['num'], resnet_result['false negative']['num']]], ['unet', 'resnet'], ['fp', 'tp', 'fn']
                            # )
                            # test_img = save_batch_keypoints(batch_image=batch_img[:,:,:,:,k,-1],
                            #                     batch_heatmaps=x[:,:,:,:,k],
                            #                     gt_keypoints=batch_kpt[:,:,:,:,k,-1],
                            #                     pred_keypoints=centroids,
                            #                     num_joints_to_show=17)
                            # test_img = save_batch_heatmaps_multi(batch_image=batch_img[:,:,:,:,k,-1], batch_heatmaps=batch_hm_gt[:,:,:,:,k,-1])
                            # cv2.imwrite('unet_{}.png'.format(k), unet_img)
                            # cv2.imwrite('resnet_{}.png'.format(k), resnet_img)

                # statistics
            #     running_loss += loss.item()
            #     if i % 100 == 0:    # print every 100 mini-batches
            #         print('[%s][%d, %3d/%3d] loss: %.3f' %
            #               (phase, epoch, i, len(data_loader[phase]), loss.item()))
            #         if phase == 'valid':
            #             a = save_batch_multi_view_with_heatmap(
            #                 batch_img[:, :, :, :, :, -1], heatmap_input, 'test')
            #             b = save_batch_multi_view_with_heatmap(
            #                 batch_img[:, :, :, :, :, -1], x, 'test')
            #             c = save_batch_multi_view_with_heatmap(
            #                 batch_img[:, :, :, :, :, -1], batch_hm_gt, 'test')

            #             for k in range(num_view):
            #                 valid_images['input_heatmap_cam_view_{}'.format(
            #                     k)].append(np.flip(a[k], 2))
            #                 valid_images['predict_heatmap_cam_view_{}'.format(
            #                     k)].append(np.flip(b[k], 2))
            #                 valid_images['predict_groundtruth_cam_view_{}'.format(
            #                     k)].append(np.flip(c[k], 2))
            #     print('[%s][%d, %3d/%3d] loss: %.3f' %
            #           (phase, epoch, i, len(data_loader[phase]), loss.item()))

            # epoch_loss[phase] = running_loss/len(data_loader[phase])
            # if phase == 'valid':
            #     tsboard.add_images('valid', valid_images, epoch)
            #     tsboard.add_scalar('valid', epoch_loss[phase], epoch)
            #     del valid_images
            #     if epoch_loss[phase] < lowest_loss:
            #         lowest_loss = epoch_loss[phase]
            #         save_checkpoint(unet, optimizer, epoch,
            #                         epoch_loss[phase], log_dir, 'best')

            # elif phase == 'train':
            #     tsboard.add_scalar('train', epoch_loss[phase], epoch)
            #     save_checkpoint(unet, optimizer, epoch,
            #                     epoch_loss[phase], log_dir, 'checkpoint')
    # tot_sample = len(data_loader['valid']) * cfg.DATASET.CMU_BATCH_SIZE * num_view
    # bar_chart_data = []
    # bar_chart_data.append([
    #     stats['resnet']['tp']/tot_sample,
    #     stats['resnet']['fp']/tot_sample,
    #     stats['resnet']['fn']/tot_sample])
    # bar_chart_data.append([
    #     stats['unet']['tp']/tot_sample,
    #     stats['unet']['fp']/tot_sample,
    #     stats['unet']['fn']/tot_sample])
    # group_bar_plot(bar_chart_data, ['resnet', 'unet'], ['tp','fp','fn'], file_name='mean')


if __name__ == "__main__":
    main()
