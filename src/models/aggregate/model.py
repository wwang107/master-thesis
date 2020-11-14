import torch
import pytorch_lightning as pl
from utils.utils import find_keypoints_from_heatmaps, match_detected_groundtruth_keypoint

class AggregateModel(pl.LightningModule):
    '''
    Aggregate Model composed by a resnet that take 256x256 RGB, then output 64x64 heatmaps and by an unet that takes 64x64 heatmaps 
    '''

    def __init__(self,
                 input_heatmap_encoder,
                 camera_view_encoder=None,
                 temporal_encoder=None,
                 loss=None,
                 in_channels=55,
                 out_channels=55,
                 heatmap_size=(64, 64),
                 num_camera_can_see=6,
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
        self.camera_view_encoder = camera_view_encoder
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
        self.input_heatmap_encoder.train()
        input_heatmaps = torch.zeros((x.size(
            0), self.in_channels, *self.heatmap_size, self.num_view, self.num_frame)).to(self.device)
        for k in range(self.num_view):
            for f in range(self.num_frame):
                input_heatmaps[:, :, :, :, k, f] = self.input_heatmap_encoder(
                    x[:, :, :, :, k, f])
        return input_heatmaps

    def get_temporal_heatmap(self, x):
        temporal_heatmaps = self.temporal_encoder(x)
        temporal_heatmaps = torch.squeeze(temporal_heatmaps, dim=5)
        return temporal_heatmaps

    def get_camera_heatmap(self, x):
        camera_heatmaps = self.camera_view_encoder(x)
        return camera_heatmaps

    def forward(self, x):
        '''
        param: x the image
        '''
        results = {'input_heatmap_encoder': None,
                   'temporal_encoder': None, 'camera_view_encoder': None}

        with torch.set_grad_enabled(self.train_input_heatmap_encoder):
            input_heatmaps = self.get_input_heatmaps(x)
        if self.train_input_heatmap_encoder:
            input_heatmaps = input_heatmaps.detach()

        results['input_heatmap_encoder'] = input_heatmaps
        if self.temporal_encoder:
            results['temporal_encoder'] = self.get_temporal_heatmap(
                input_heatmaps)
        if self.camera_view_encoder:
            results['camera_view_encoder'] = self.camera_view_encoder(
                input_heatmaps)

        return results

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
            self.log(loss_key, loss)
            return loss
        else:
            raise NotImplementedError()
    
    def test_step(self, batch, batch_index):
        batch_imgs = batch['img'].float()
        stats = {'temporal_encoder': None, 'camera_view_encoder': None, 'input_heatmap_encoder': None}
        
        input_heatmaps = self.get_input_heatmaps(batch_imgs)
        if self.camera_view_encoder != None:
            camera_heatmap = self.get_camera_heatmap(input_heatmaps) 

        if self.temporal_encoder != None:
            pad = (self.num_frame // 2, self.num_frame // 2)
            input_heatmaps = torch.nn.functional.pad(input_heatmaps, pad)
            

        

    def shared_step(self, batch, batch_index):
        batch_imgs = batch['img'].float()
        batch_gt_heatmaps = batch['heatmap']
        out = self(batch_imgs)
        middle_frame = self.num_frame//2

        if out['temporal_encoder'] == None and out['camera_view_encoder'] == None:
            raise NotImplementedError()

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
            weight = (batch_gt_heatmaps > 0.1) * \
                1.0 + (batch_gt_heatmaps <= 0.1) * 0.1
            c_loss = self.loss(
                camera_view_heatmaps, batch_gt_heatmaps, weight)
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

    # def validation_step(self, batch, batch_idx):
    #     stats = {'temporal_encoder': None,
    #              'camera_view_encoder': None, 'input_heatmap_encoder': None}

    #     batch_imgs = batch['img'].float()
    #     batch_gt_hm = batch['heatmap']
    #     batch_gt_keypoint = batch['keypoint2d']
    #     only_use_middle_frame = True if self.temporal_encoder != None else False
    #     middle_frame = self.num_frame // 2

    #     input_heatmaps = self.get_input_heatmaps(batch_imgs)
    #     i_results, t_results, c_results = self.calculate_confusion_table(
    #         input_heatmaps, batch_gt_keypoint, only_use_middle_frame)
    #     stats['input_heatmap_encoder'] = self.get_average_confusion_table(
    #         i_results)

    #     if self.temporal_encoder == None and self.camera_view_encoder == None:
    #         raise NotImplementedError()

    #     if self.temporal_encoder != None and self.camera_view_encoder == None:
    #         temporal_heatmaps = self.get_temporal_heatmap(input_heatmaps)
    #         weight = (batch_gt_hm[:, :, :, :, :, middle_frame] > 0.1) * \
    #             1.0 + (batch_gt_hm[:, :, :, :, :, middle_frame] <= 0.1) * 0.1

    #         t_loss = self.loss(
    #             temporal_heatmaps, batch_gt_hm[:, :, :, :, :, middle_frame], weight)

    #         stats['temporal_encoder'] = self.get_average_confusion_table(
    #             t_results)
            
    #         self.log('avg_validation/temporal_encoder', t_loss, on_epoch=True)
    #         loss = t_loss

    #     elif self.camera_view_encoder != None and self.temporal_encoder == None:
    #         camera_view_heatmaps = self.get_camera_heatmap(input_heatmaps)
    #         if only_use_middle_frame:
    #             weight = (batch_gt_hm[:, :, :, :, :, middle_frame] > 0.1) * \
    #                 1.0 + (batch_gt_hm[:, :, :, :, :,
    #                                    middle_frame] <= 0.1) * 0.1
    #             c_loss = self.loss(
    #                 camera_view_heatmaps[:, :, :, :, :, middle_frame], batch_gt_hm[:, :, :, :, :, middle_frame], weight)
    #         else:
    #             weight = (batch_gt_hm > 0.1) * \
    #                 1.0 + (batch_gt_hm <= 0.1) * 0.1
    #             c_loss = self.loss(
    #                 camera_view_heatmaps, batch_gt_hm, weight)
    #         stats['camera_view_encoder'] = self.get_average_confusion_table(
    #             c_results)
    #         self.log('avg_validation/camera_view_encoder', c_loss, on_epoch=True)
    #         loss = c_loss

    #     elif self.camera_view_encoder != None and self.temporal_encoder != None:
    #         temporal_heatmaps = self.get_temporal_heatmap(input_heatmaps)
    #         camera_view_heatmaps = self.get_camera_heatmap(input_heatmaps)
    #         weight = (batch_gt_hm[:, :, :, :, :, middle_frame] > 0.1) * \
    #             1.0 + (batch_gt_hm[:, :, :, :, :, middle_frame] <= 0.1) * 0.1

    #         t_loss = self.loss(
    #             temporal_heatmaps, batch_gt_hm[:, :, :, :, :, middle_frame], weight)
    #         c_loss = self.loss(
    #             camera_view_heatmaps[:, :, :, :, :, middle_frame], batch_gt_hm[:, :, :, :, :, middle_frame], weight)
    #         stats['temporal_encoder'] = self.get_average_confusion_table(
    #             t_results)
    #         stats['camera_view_encoder'] = self.get_average_confusion_table(
    #             c_results)

    #         self.log('avg_validation/temporal_encoder', t_loss, on_epoch=True)
    #         self.log('avg_validation/camera_view_encoder',
    #                  c_loss, on_epoch=True)
    #         loss = t_loss + c_loss

    #     return {'loss':loss, 'stats': stats}

    def get_average_confusion_table(self, confusion_results):
        total = len(confusion_results)
        stats = {'false positive': 0, 'true positive': 0, 'false negative': 0}
        dist = 0.0
        distCount = 0
        for result in confusion_results:
            for key in stats.keys():
                stats[key] += result[key]['num']
            if result['distance'] != None:
                dist += result['distance']
                distCount += 1
        if distCount != 0:
            stats['distance'] = dist / distCount
        else:
            stats['distance'] = 99999
        for key in stats.keys():
            stats[key] = stats[key] / total
        return stats

    def calculate_confusion_table(self, input_heatmaps, batch_gt_keypoint, only_use_middle_frame):
        middle_frame = self.num_frame//2
        i_results = []
        t_results = []
        c_results = []

        for k in range(self.num_view):
            if only_use_middle_frame:
                i_detected_keypoint = find_keypoints_from_heatmaps(
                    input_heatmaps[:, :, :, :, k, middle_frame], threshold=0)
                i_results.append(match_detected_groundtruth_keypoint(
                    batch_gt_keypoint[:, :, :, :, k, middle_frame], i_detected_keypoint, 1))
            else:
                for f in range(self.num_frame):
                    i_detected_keypoint = find_keypoints_from_heatmaps(
                        input_heatmaps[:, :, :, :, k, f], threshold=0)
                    i_results.append(match_detected_groundtruth_keypoint(
                        batch_gt_keypoint[:, :, :, :, k, f], i_detected_keypoint, 1))

        if self.temporal_encoder != None:
            temporal_heatmaps = self.get_temporal_heatmap(input_heatmaps)
            for k in range(self.num_view):
                t_detected_keypoint = find_keypoints_from_heatmaps(
                    temporal_heatmaps[:, :, :, :, k], threshold=0.0)
                t_results.append(match_detected_groundtruth_keypoint(
                    batch_gt_keypoint[:, :, :, :, k, middle_frame], t_detected_keypoint, 1))

        if self.camera_view_encoder != None:
            camera_view_heatmaps = self.get_camera_heatmap(input_heatmaps)
            for k in range(self.num_view):
                if only_use_middle_frame:
                    c_detected_keypoint = find_keypoints_from_heatmaps(
                        camera_view_heatmaps[:, :, :, :, k, middle_frame], threshold=0.0)
                    c_results.append(match_detected_groundtruth_keypoint(
                        batch_gt_keypoint[:, :, :, :, k, middle_frame], c_detected_keypoint, 1))
                else:
                    for f in range(self.num_frame):
                        c_detected_keypoint = find_keypoints_from_heatmaps(
                            camera_view_heatmaps[:, :, :, :, k, f], threshold=0)
                        c_results.append(match_detected_groundtruth_keypoint(
                            batch_gt_keypoint[:, :, :, :, k, f], c_detected_keypoint, 1))

        return i_results, t_results, c_results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
