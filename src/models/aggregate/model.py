import torch
import pytorch_lightning as pl
import numpy as np
import cv2
import matplotlib.pylab as plt
from models.epipolar.EpipolarTransformer import Epipolar
from utils.multiview import findFundamentalMat
from utils.utils import find_keypoints_from_heatmaps, match_detected_groundtruth_keypoint, pad_heatmap_with_replicate_frame
from utils.vis.vis import save_batch_image_with_joints_multi, save_batch_maps, save_batch_multi_view_with_heatmap
class AggregateModel(pl.LightningModule):
    '''
    Aggregate Model composed by a resnet that take 256x256 RGB, then output 64x64 heatmaps and by an unet that takes 64x64 heatmaps 
    '''

    def __init__(self,
                 input_heatmap_encoder,
                 epipolar_transformer = None,
                 camera_view_encoder=None,
                 fusion_encoder = None,
                 temporal_encoder=None,
                 loss=None,
                 in_channels=55,
                 out_channels=55,
                 heatmap_size=(64, 64),
                 num_camera_can_see=5,
                 num_frame_can_see=15,
                 train_input_heatmap_encoder: bool = False):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_view = num_camera_can_see
        self.num_frame = num_frame_can_see
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_input_heatmap_encoder = train_input_heatmap_encoder
        self.input_heatmap_encoder = input_heatmap_encoder
        self.epipolar_transformer = epipolar_transformer
        self.camera_view_encoder = camera_view_encoder
        self.fusion_net = fusion_encoder
        self.temporal_encoder = temporal_encoder
        self.loss = loss

        self.confusion_table = {}
        self.confusion_table['input_heatmap_encoder'] = {
            'fp': 0, 'tp': 0, 'tn': 0}
        if self.camera_view_encoder != None:
            self.confusion_table['camera_view_encoder'] = {
                'fp': 0, 'tp': 0, 'tn': 0}
        if self.temporal_encoder != None:
            self.confusion_table['temporal_encoder'] = {
                'fp': 0, 'tp': 0, 'tn': 0}

    def get_input_heatmaps(self, x):
        num_view = x.size(4)
        num_frame = x.size(5)
        self.input_heatmap_encoder.train()
        input_heatmaps = torch.zeros((x.size(
            0), self.in_channels, *self.heatmap_size, num_view, num_frame)).to(self.device)
        
        for k in range(num_view):
            for f in range(num_frame):
                input_heatmaps[:, :, :, :, k, f] = self.input_heatmap_encoder(
                    x[:, :, :, :, k, f])
        return input_heatmaps
    
    def get_epipolar_heatmap(self,feats, proj_mats, imgs = None, keypoints = None):
        num_view = feats.size(4) 

        ref_feat = feats[:,:,:,:,0]
        ref_p = proj_mats[:,:,:,0]
        fuse_sum = torch.zeros(ref_feat.size()).to(ref_feat)
        for j in range(1, num_view):
            src_feat = feats[:,:,:,:,j]
            src_p = proj_mats[:,:,:,j]
            fuse = self.epipolar_transformer(ref_feat, src_feat, ref_p, src_p, 
                                    imgs[...,0] if imgs != None else None,
                                    imgs[...,j] if imgs != None else None,
                                    keypoints[..., 0] if keypoints != None else None,
                                    keypoints[..., j] if keypoints != None else None)
            fuse_sum += fuse
        
        fuse_sum = fuse_sum.view(*fuse_sum.size(),1)
        return fuse_sum.view(*fuse_sum.size(),1)

    def get_temporal_heatmap(self, x):
        temporal_heatmaps = self.temporal_encoder(x)
        temporal_heatmaps = torch.squeeze(temporal_heatmaps, dim=5)
        return temporal_heatmaps

    def get_camera_heatmap(self, x, proj_mats, imgs=None, keypoints=None):
        fused_heatmap, unfused_heatmap = self.camera_view_encoder(x, proj_mats, 
        imgs[...,0] if imgs != None else None, 
        keypoints[...,0] if keypoints != None else None)
        if self.fusion_net != None:
            concat_input = torch.cat((fused_heatmap, unfused_heatmap), dim=1)
            out = self.fusion_net(concat_input)
            return out, unfused_heatmap
        else:
            return fused_heatmap, unfused_heatmap

    def forward(self, x, proj_mats = None, keypoint = None):
        '''
        param: x the image
        '''
        results = {'input_heatmap_encoder': None,
                   'temporal_encoder': None, 'camera_view_encoder': None, 'fusion_net': None}
        is_train_input_heatmap_encoder = self.train_input_heatmap_encoder and self.training

        with torch.set_grad_enabled(is_train_input_heatmap_encoder):
            input_heatmaps = self.get_input_heatmaps(x)
        if not is_train_input_heatmap_encoder:
            input_heatmaps = input_heatmaps.detach()

        if self.epipolar_transformer != None and self.fusion_net != None:
            fused_heatmaps = torch.zeros(input_heatmaps.size()).to(input_heatmaps)
            with torch.no_grad():
                for f in range(self.num_frame):
                    for v in range(self.num_view):
                        index = [v]
                        index.extend([i for i in range(self.num_view) if i != v])
                        index = torch.LongTensor(index).to(input_heatmaps.device)
                        epipolar_heatmaps = self.get_epipolar_heatmap(input_heatmaps[...,index,f], proj_mats[..., index], x[..., index,f], keypoint[...,f] if keypoint != None else None)
                        fused_heatmaps[...,v:v+1,f:f+1] = self.fusion_net(torch.cat((epipolar_heatmaps, input_heatmaps[:,:,:,:,v:v+1,f:f+1]), dim=1))

            results['epipolar_transoformer'] = epipolar_heatmaps
            results['fusion_net'] = fused_heatmaps
            
        results['input_heatmap_encoder'] = input_heatmaps
        if self.temporal_encoder:
            if self.fusion_net != None:
                results['temporal_encoder'] = self.get_temporal_heatmap(
                    fused_heatmaps)
            else:
                results['temporal_encoder'] = self.get_temporal_heatmap(
                    input_heatmaps)
        if self.camera_view_encoder:
            results['camera_view_encoder'] = self.get_camera_heatmap(
                input_heatmaps, proj_mats
            )

        return results

    def shared_step(self, batch, batch_index):
        batch_imgs = batch['img'].float()
        batch_gt_heatmaps = batch['heatmap']
        proj_mats = batch['KRT']
        batch_keypoint = batch['keypoint2d']
        out = self(batch_imgs, proj_mats, batch_keypoint)
        middle_frame = self.num_frame//2

        if out['temporal_encoder'] == None and out['camera_view_encoder'] == None:
            if out['fusion_net'] != None:
                fused_heatmap = out['fusion_net']
                weight = (batch_gt_heatmaps[:, :, :, :, 0:1, :] > 0.1) * \
                1.0 + (batch_gt_heatmaps[:, :, :, :,0:1, :] <= 0.1) * 0.1
                t_loss = self.loss(
                    fused_heatmap, batch_gt_heatmaps[:, :, :, :, 0:1, :], weight)
                return t_loss

        if out['temporal_encoder'] != None and out['camera_view_encoder'] == None:
            temporal_heatmaps = out['temporal_encoder']
            weight = (batch_gt_heatmaps[:, :, :, :, :, middle_frame] > 0.1) * \
                1.0 + (batch_gt_heatmaps[:, :, :, :,
                                            :, middle_frame] <= 0.1) * 0.1
            t_loss = self.loss(
                temporal_heatmaps, batch_gt_heatmaps[:, :, :, :, :, middle_frame], weight)
            return t_loss

        elif out['camera_view_encoder'] != None and out['temporal_encoder'] == None:
            camera_view_heatmaps = out['camera_view_encoder']
            if isinstance(camera_view_heatmaps, tuple):
                fused_heatmap, unfused_heatmap = camera_view_heatmaps
                weight = (batch_gt_heatmaps[...,0,middle_frame].view(*fused_heatmap.size()) > 0.1) * \
                    1.0 + (batch_gt_heatmaps[...,0,middle_frame].view(*fused_heatmap.size()) <= 0.1) * 0.1
                c_loss = self.loss(
                    fused_heatmap, batch_gt_heatmaps[...,0,middle_frame].view(*fused_heatmap.size()), weight)
                return c_loss

        elif out['camera_view_encoder'] != None and out['temporal_encoder'] != None:
            temporal_heatmaps = out['temporal_encoder']
            camera_view_heatmaps = out['camera_view_encoder']
            weight = (batch_gt_heatmaps[:, :, :, :, :, middle_frame] > 0.1) * \
                1.0 + (batch_gt_heatmaps[:, :, :, :,
                                            :, middle_frame] <= 0.1) * 0.1
            t_loss = self.loss(
                temporal_heatmaps, batch_gt_heatmaps[:, :, :, :, :, middle_frame], weight)
            c_loss = self.loss(
                camera_view_heatmaps[:, :, :, :, :, middle_frame], batch_gt_heatmaps[:, :, :, :, :, middle_frame], weight)

            return c_loss + t_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        if self.camera_view_encoder != None and self.temporal_encoder == None:
            loss_key = 'training_step/camera_encoder'
            self.log(loss_key, loss)
            return loss
        elif self.camera_view_encoder == None and self.temporal_encoder != None:
            loss_key = 'training_step/temporal_encoder'
            self.log(loss_key, loss)
            return loss
        elif self.camera_view_encoder == None and self.temporal_encoder == None:
            if self.fusion_net != None:
                loss_key = 'training_step/fusion_net'
                self.log(loss_key, loss)
                return loss
        else:
            raise NotImplementedError()
        
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        if self.camera_view_encoder != None and self.temporal_encoder == None:
            loss_key = 'validation_step_avg_loss/camera_encoder'
            self.log(loss_key, loss, on_epoch=True)
            return loss
        elif self.camera_view_encoder == None and self.temporal_encoder != None:
            loss_key = 'validation_step_avg_loss/temporal_encoder'
            self.log(loss_key, loss, on_epoch=True)
            return loss
        elif self.camera_view_encoder == None and self.temporal_encoder == None:
            if self.fusion_net != None:
                loss_key = 'validation_step_avg_loss/fusion_net'
                self.log(loss_key, loss, on_epoch=True)
                return loss
    
    def test_step(self, batch, batch_index):
        batch_imgs = batch['img'].float()
        batch_gt_keypoint = batch['keypoint2d']
        proj_mats = batch['KRT']
        stats = {'temporal_encoder': None, 'camera_view_encoder': None, 'input_heatmap_encoder': None}
        
        input_heatmaps = self.get_input_heatmaps(batch_imgs)
        input_metric = self.calculate_confusion_metrics(input_heatmaps, batch_gt_keypoint)
        stats['input_heatmap_encoder'] = input_metric
        if self.camera_view_encoder != None:
            fused_heatmap = self.get_camera_heatmap(input_heatmaps, proj_mats, batch_imgs, batch_gt_keypoint)
            camera_encoder_metric = self.calculate_confusion_metrics(fused_heatmap, batch_gt_keypoint)
            stats['camera_view_encoder'] = camera_encoder_metric

        if self.temporal_encoder != None:
            if self.fusion_net:
                fused_heatmaps = torch.zeros(input_heatmaps.size()).to(input_heatmaps)
                for f in range(self.num_frame):
                    for v in range(self.num_view):
                        index = [v]
                        index.extend([i for i in range(self.num_view) if i != v])
                        index = torch.LongTensor(index).to(input_heatmaps.device)
                        epipolar_heatmaps = self.get_epipolar_heatmap(input_heatmaps[...,index,f], proj_mats[..., index])
                        fused_heatmaps[...,v:v+1,f:f+1] = self.fusion_net(torch.cat((epipolar_heatmaps, input_heatmaps[:,:,:,:,v:v+1,f:f+1]), dim=1))
                input_heatmaps = fused_heatmaps
            pad = (self.num_frame // 2, self.num_frame // 2)
            padded_input_heatmaps = pad_heatmap_with_replicate_frame(input_heatmaps, pad)
            out_heatmap = self.temporal_encoder(padded_input_heatmaps)
            temporal_encoder_metric = self.calculate_confusion_metrics(out_heatmap, batch_gt_keypoint)
            stats['temporal_encoder'] = temporal_encoder_metric
        return stats
    
    def test_epoch_end(self, outputs) -> None:
        total_num_test_batch = len(outputs)
        distance_devider = {'temporal_encoder': total_num_test_batch if self.temporal_encoder != None else None, 
                  'camera_view_encoder': total_num_test_batch if self.camera_view_encoder != None else None,
                  'input_heatmap_encoder': total_num_test_batch if self.input_heatmap_encoder != None else None,
                  'fusion_net':total_num_test_batch if self.fusion_net != None else None} 
        result = {'temporal_encoder': {'false positive':0, 'true positive':0, 'false negative':0, 'true positive distance': 0} if self.temporal_encoder != None else None, 
                  'camera_view_encoder': {'false positive':0, 'true positive':0, 'false negative':0, 'true positive distance': 0} if self.camera_view_encoder != None else None,
                  'input_heatmap_encoder': {'false positive':0, 'true positive':0, 'false negative':0, 'true positive distance': 0} if self.input_heatmap_encoder != None else None,
                  'fusion_net': {'false positive':0, 'true positive':0, 'false negative':0, 'true positive distance': 0} if self.fusion_net != None else None}
        
        for stats in outputs:
            for encoder in stats.keys():    
                if stats[encoder] != None: 
                    result[encoder]['false positive'] += stats[encoder]['false positive']
                    result[encoder]['false negative'] += stats[encoder]['false negative']
                    result[encoder]['true positive'] += stats[encoder]['true positive']
                    if stats[encoder]['true positive distance'] == None:
                        distance_devider[encoder] -= 1
                    else:
                        result[encoder]['true positive distance'] += stats[encoder]['true positive distance']

        for encoder in result:
            if result[encoder] != None:
                self.log('{}/false positive'.format(encoder), result[encoder]['false positive']/total_num_test_batch)
                self.log('{}/true positive'.format(encoder), result[encoder]['true positive']/total_num_test_batch)
                self.log('{}/false negative'.format(encoder), result[encoder]['false negative']/total_num_test_batch)
                if distance_devider[encoder] == 0:
                    self.log('{}/true positive distance'.format(encoder), None)
                else:
                    self.log('{}/true positive distance'.format(encoder), result[encoder]['true positive distance']/distance_devider[encoder])       

    def calculate_confusion_metrics(self, batch_heatmaps, batch_gt_keypoint):
        num_frame = batch_heatmaps.size(5)
        num_camera = batch_heatmaps.size(4)
        num_batch_size = batch_heatmaps.size(0)

        result = {'false negative':0, 'false positive':0, 'true positive':0, 'true positive distance':0}
        for k in range(num_camera):
            for f in range(num_frame):
                batch_detections = find_keypoints_from_heatmaps(batch_heatmaps[:, :, :, :, k, f], threshold=0.5)
                confusion_metrics = match_detected_groundtruth_keypoint(batch_gt_keypoint[:, :, :, :, k, f], batch_detections)
                result['false negative'] += confusion_metrics['false negative']['num']
                result['false positive'] += confusion_metrics['false positive']['num']
                result['true positive'] += confusion_metrics['true positive']['num']
                result['true positive distance'] += confusion_metrics['distance']
        result['true positive distance'] = result['true positive distance']/result['true positive'] if result['true positive']>0 else None 
        result['false negative'] = result['false negative']/num_batch_size/num_frame/num_camera 
        result['false positive'] = result['false positive']/num_batch_size/num_frame/num_camera
        result['true positive'] = result['true positive']/num_batch_size/num_frame/num_camera
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
