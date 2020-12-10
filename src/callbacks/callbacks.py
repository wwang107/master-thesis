import cv2
import os
import torch
from pytorch_lightning import Callback
from utils.vis.vis import save_batch_multi_view_with_heatmap
from utils.utils import pad_heatmap_with_replicate_frame, find_keypoints_from_heatmaps, match_detected_groundtruth_keypoint
from utils.vis.vis import save_keypoint_detection
class LogConfusionTable(Callback):
   def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        confusion_table = outputs['stats']
        stats_keys = [available_key for available_key in confusion_table if confusion_table[available_key] != None]
    
        tp_dist = {}
        tp = {}
        fp = {}
        fn = {}

        for encoder_key in stats_keys:
            tp[encoder_key] = confusion_table[encoder_key]['true positive']
            fn[encoder_key] = confusion_table[encoder_key]['false negative']
            fp[encoder_key] = confusion_table[encoder_key]['false positive']
            tp_dist[encoder_key] = confusion_table[encoder_key]['distance']
        
        pl_module.logger.experiment.add_scalars("ture positive", tp, global_step=pl_module.global_step)
        pl_module.logger.experiment.add_scalars("false negative", fn, global_step=pl_module.global_step)
        pl_module.logger.experiment.add_scalars("false positive", fp, global_step=pl_module.global_step)
        pl_module.logger.experiment.add_scalars("ture positive distance", tp_dist, global_step=pl_module.global_step)


class LogModelHeatmaps(Callback):
    def __init__(self, log_dir:str, num_frame:int, logging_batch_interval: int= 20):
        self.logging_batch_interval = logging_batch_interval
        self.log_dir = log_dir
        self.middle_frame = num_frame // 2

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (batch_idx) % self.logging_batch_interval != 0:
            return
        self.shared_step(trainer, pl_module, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (batch_idx) % self.logging_batch_interval != 0:
            return

        global_step = pl_module.global_step
        epoch = pl_module.current_epoch
        prefix = os.path.join(self.log_dir, 'ver_' + str(trainer.logger.version))
        os.makedirs(prefix, exist_ok=True)
        batch_images = batch['img'].float().to(pl_module.device)
        batch_gt_keypoint = batch['keypoint2d']

        input_heatmap = pl_module.get_input_heatmaps(batch_images)

        num_view = input_heatmap.size(4)
        num_frame = input_heatmap.size(5)

        for f in range(0, input_heatmap.size(5),3):
                visualization = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, f] ,input_heatmap[:,:,:,:,:, f])
                for view, image in enumerate(visualization):
                    file_name = os.path.join(prefix, '{}_test_epoch_{}_step_{}_view_{}_frame_{}.png'.format('input_encoder', epoch, global_step, view, f))
                    cv2.imwrite(str(file_name), image)
        
        if pl_module.temporal_encoder != None:
            pad = (pl_module.num_frame // 2, pl_module.num_frame // 2)
            pad_input_heatmap = pad_heatmap_with_replicate_frame(input_heatmap, pad)
            out_heatmap = pl_module.temporal_encoder(pad_input_heatmap)
            for f in range(0, out_heatmap.size(5),3):
                visualization = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, f] ,out_heatmap[:,:,:,:,:, f])
                for view, image in enumerate(visualization):
                    file_name = os.path.join(prefix, '{}_test_epoch_{}_step_{}_view_{}_frame_{}.png'.format('temporal_encoder', epoch, global_step, view, f))
                    cv2.imwrite(str(file_name), image)

            file_name = 'confusion_metrics_{}_test_epoch_{}_step_{}'.format('temporal_encoder', epoch, global_step)
            file_name = os.path.join(prefix, file_name)
            self.visualize_confusion_metrics(file_name, batch_images, out_heatmap, batch_gt_keypoint)
        
        if pl_module.camera_view_encoder != None:
            out_heatmap = pl_module.camera_view_encoder(input_heatmap)
            
            for f in range(0, out_heatmap.size(5),3):
                visualization = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, f] ,out_heatmap[:,:,:,:,:, f])
                for view, image in enumerate(visualization):
                    file_name = os.path.join(prefix, '{}_test_epoch_{}_step_{}_view_{}_frame_{}.png'.format('camera_encoder', epoch, global_step, view, f))
                    cv2.imwrite(str(file_name), image)

            file_name = 'confusion_metrics_{}_test_epoch_{}_step_{}'.format('camera_encoder', epoch, global_step)
            file_name = os.path.join(prefix, file_name)
            self.visualize_confusion_metrics(file_name, batch_images, out_heatmap, batch_gt_keypoint)
        
        if pl_module.fusion_net != None:
            proj_mats = batch['KRT']
            fused_heatmaps = torch.zeros_like(input_heatmap)
            for f in range(num_frame):
                for v in range(num_view):
                    index = [v]
                    index.extend([i for i in range(num_view) if i != v])
                    index = torch.LongTensor(index).to(input_heatmap.device)
                    epipolar_heatmaps, unfused_heatmaps = pl_module.get_epipolar_heatmap(input_heatmap[...,index,f], proj_mats[...,index])
                    fused_heatmap = pl_module.fusion_net(torch.cat((epipolar_heatmaps, unfused_heatmaps[:,:,:,:,0:1,:]), dim=1))
                    fused_heatmaps[:,:,:,:,v,f] = fused_heatmap[...,0,0]
                visualization = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, f] ,fused_heatmaps[...,f])
                for view, image in enumerate(visualization):
                    file_name = os.path.join(prefix, '{}_test_epoch_{}_step_{}_view_{}_frame_{}.png'.format('fusion_net', epoch, global_step, view, f))
                    cv2.imwrite(str(file_name), image)
            
            file_name = 'confusion_metrics_{}_test_epoch_{}_step_{}'.format('fusion_net', epoch, global_step)
            file_name = os.path.join(prefix, file_name)
            self.visualize_confusion_metrics(file_name, batch_images, fused_heatmaps, batch_gt_keypoint)
        
    def visualize_confusion_metrics(self, file_name, batch_images, batch_heatmaps, batch_gt_keypoint):
        frame = 0
        num_view = batch_heatmaps.size(4)
        for v in range(num_view):
            batch_detections = find_keypoints_from_heatmaps(batch_heatmaps[:, :, :, :, v, frame], threshold=0.5)
            confusion_metrics = match_detected_groundtruth_keypoint(batch_gt_keypoint[:, :, :, :, v, frame], batch_detections)
            img = save_keypoint_detection(batch_images[:, :, :, :, v, frame], 
                                        batch_heatmaps[:, :, :, :, v, frame], 
                                        confusion_metrics['false positive']['points'], 
                                        confusion_metrics['false negative']['points'],
                                        confusion_metrics['true positive']['points'])
            
            cv2.imwrite('{}_view_{}.png'.format(file_name, v), img)

    def shared_step(self, trainer, pl_module, batch):
        global_step = pl_module.global_step
        epoch = pl_module.current_epoch
        prefix = os.path.join(self.log_dir, 'ver_' + str(trainer.logger.version))
        os.makedirs(prefix, exist_ok=True)

        batch_images = batch['img']
        proj_mat = batch['KRT']
        out = pl_module(batch_images.float().to(pl_module.device), proj_mat.to(pl_module.device))
        for encoder in out:
            if out[encoder] != None:
                if encoder == 'temporal_encoder':
                    heatmaps = out[encoder] 
                if encoder == 'camera_view_encoder': 
                    heatmaps = out[encoder]
                    if isinstance(heatmaps, tuple):
                        fused_heatmap, unfused_heatmap = heatmaps
                        fused_heatmap = fused_heatmap.cpu()
                        unfused_heatmap = unfused_heatmap.cpu()
                        vis_fused = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, self.middle_frame],fused_heatmap)
                        vis_unfused = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, self.middle_frame],unfused_heatmap)

                        for view, image in enumerate(vis_fused):
                            file_name = os.path.join(prefix, 'fused_{}_epoch_{}_step_{}_view_{}.png'.format(encoder, epoch, global_step, view))
                            cv2.imwrite(str(file_name), image)
                        for view, image in enumerate(vis_unfused):
                            file_name = os.path.join(prefix, 'unfused_{}_epoch_{}_step_{}_view_{}.png'.format(encoder, epoch, global_step, view))
                            cv2.imwrite(str(file_name), image)
                    else:
                        heatmaps = heatmaps.cpu()
                        visualization = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, self.middle_frame],heatmaps)
                        for view, image in enumerate(visualization):
                            file_name = os.path.join(prefix, '{}_epoch_{}_step_{}_view_{}.png'.format(encoder, epoch, global_step, view))
                            cv2.imwrite(str(file_name), image)
                
                if encoder == 'fusion_net':
                    heatmaps = out[encoder]
                    vis_fused = save_batch_multi_view_with_heatmap(batch_images[:,:,:,:,:, self.middle_frame],heatmaps)
                    for view, image in enumerate(vis_fused):
                        file_name = os.path.join(prefix, 'fused_{}_epoch_{}_step_{}_view_{}.png'.format(encoder, epoch, global_step, view))
                        cv2.imwrite(str(file_name), image)

