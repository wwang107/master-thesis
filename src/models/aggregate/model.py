import torch
import pytorch_lightning as pl
from utils.utils import find_keypoints_from_heatmaps, match_detected_groundtruth_keypoint, pad_heatmap_with_replicate_frame
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
        is_train_input_heatmap_encoder = self.train_input_heatmap_encoder and self.training

        with torch.set_grad_enabled(is_train_input_heatmap_encoder):
            input_heatmaps = self.get_input_heatmaps(x)
        if not is_train_input_heatmap_encoder:
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
        batch_gt_keypoint = batch['keypoint2d']
        stats = {'temporal_encoder': None, 'camera_view_encoder': None, 'input_heatmap_encoder': None}
        
        input_heatmaps = self.get_input_heatmaps(batch_imgs)
        input_metric = self.calculate_confusion_metrics(input_heatmaps, batch_gt_keypoint)
        stats['input_heatmap_encoder'] = input_metric
        if self.camera_view_encoder != None:
            out_heatmap = self.get_camera_heatmap(input_heatmaps)
            camera_encoder_metric = self.calculate_confusion_metrics(out_heatmap, batch_gt_keypoint)
            stats['camera_view_encoder'] = camera_encoder_metric

        if self.temporal_encoder != None:
            pad = (self.num_frame // 2, self.num_frame // 2)
            padded_input_heatmaps = pad_heatmap_with_replicate_frame(input_heatmaps, pad)
            out_heatmap = self.temporal_encoder(padded_input_heatmaps)
            temporal_encoder_metric = self.calculate_confusion_metrics(out_heatmap, batch_gt_keypoint)
            stats['temporal_encoder'] = temporal_encoder_metric 
        return stats
    
    def test_epoch_end(self, outputs) -> None:
        result = {'temporal_encoder': {'false positive':0, 'true positive':0, 'false negative':0} if self.temporal_encoder != None else None, 
                  'camera_view_encoder': {'false positive':0, 'true positive':0, 'false negative':0} if self.camera_view_encoder != None else None,
                  'input_heatmap_encoder': {'false positive':0, 'true positive':0, 'false negative':0} if self.input_heatmap_encoder != None else None}
        totol_num_test_batch = len(outputs)
        
        for stats in outputs:
            for encoder in stats.keys():
                fp = 0
                fn = 0
                tp = 0
                if stats[encoder] != None: 
                    result[encoder]['false positive'] += stats[encoder]['false positive']
                    result[encoder]['false negative'] += stats[encoder]['false negative']
                    result[encoder]['true positive'] += stats[encoder]['true positive']

        for encoder in result:
            if result[encoder] != None:
                self.log('{}/false positive'.format(encoder), result[encoder]['false positive']/totol_num_test_batch)
                self.log('{}/true positive'.format(encoder), result[encoder]['true positive']/totol_num_test_batch)
                self.log('{}/false negative'.format(encoder), result[encoder]['false negative']/totol_num_test_batch)       

        

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

    # def get_average_confusion_table(self, confusion_results):
    #     total = len(confusion_results)
    #     stats = {'false positive': 0, 'true positive': 0, 'false negative': 0}
    #     dist = 0.0
    #     distCount = 0
    #     for result in confusion_results:
    #         for key in stats.keys():
    #             stats[key] += result[key]['num']
    #         if result['distance'] != None:
    #             dist += result['distance']
    #             distCount += 1
    #     if distCount != 0:
    #         stats['distance'] = dist / distCount
    #     else:
    #         stats['distance'] = 99999
    #     for key in stats.keys():
    #         stats[key] = stats[key] / total
    #     return stats

    def calculate_confusion_metrics(self, batch_heatmaps, batch_gt_keypoint):
        num_frame = batch_heatmaps.size(5)
        num_camera = batch_heatmaps.size(4)
        num_batch_size = batch_heatmaps.size(0)

        result = {'false negative':0, 'false positive':0, 'true positive':0}
        for k in range(num_camera):
            for f in range(num_frame):
                batch_detections = find_keypoints_from_heatmaps(batch_heatmaps[:, :, :, :, k, f])
                confusion_metrics = match_detected_groundtruth_keypoint(batch_gt_keypoint[:, :, :, :, k, f], batch_detections, 1)
                result['false negative'] += confusion_metrics['false negative']['num']
                result['false positive'] += confusion_metrics['false positive']['num']
                result['true positive'] += confusion_metrics['true positive']['num']
        result['false negative'] = result['false negative']/num_batch_size/num_frame/num_camera 
        result['false positive'] = result['false positive']/num_batch_size/num_frame/num_camera
        result['true positive'] = result['true positive']/num_batch_size/num_frame/num_camera
        
        return result
        # middle_frame = self.num_frame//2
        # i_results = []
        # t_results = []
        # c_results = []

        # for k in range(self.num_view):
        #     if only_use_middle_frame:
        #         i_detected_keypoint = find_keypoints_from_heatmaps(
        #             input_heatmaps[:, :, :, :, k, middle_frame], threshold=0)
        #         i_results.append(match_detected_groundtruth_keypoint(
        #             batch_gt_keypoint[:, :, :, :, k, middle_frame], i_detected_keypoint, 1))
        #     else:
        #         for f in range(self.num_frame):
        #             i_detected_keypoint = find_keypoints_from_heatmaps(
        #                 input_heatmaps[:, :, :, :, k, f], threshold=0)
        #             i_results.append(match_detected_groundtruth_keypoint(
        #                 batch_gt_keypoint[:, :, :, :, k, f], i_detected_keypoint, 1))

        # if self.temporal_encoder != None:
        #     temporal_heatmaps = self.get_temporal_heatmap(input_heatmaps)
        #     for k in range(self.num_view):
        #         t_detected_keypoint = find_keypoints_from_heatmaps(
        #             temporal_heatmaps[:, :, :, :, k], threshold=0.0)
        #         t_results.append(match_detected_groundtruth_keypoint(
        #             batch_gt_keypoint[:, :, :, :, k, middle_frame], t_detected_keypoint, 1))

        # if self.camera_view_encoder != None:
        #     camera_view_heatmaps = self.get_camera_heatmap(input_heatmaps)
        #     for k in range(self.num_view):
        #         if only_use_middle_frame:
        #             c_detected_keypoint = find_keypoints_from_heatmaps(
        #                 camera_view_heatmaps[:, :, :, :, k, middle_frame], threshold=0.0)
        #             c_results.append(match_detected_groundtruth_keypoint(
        #                 batch_gt_keypoint[:, :, :, :, k, middle_frame], c_detected_keypoint, 1))
        #         else:
        #             for f in range(self.num_frame):
        #                 c_detected_keypoint = find_keypoints_from_heatmaps(
        #                     camera_view_heatmaps[:, :, :, :, k, f], threshold=0)
        #                 c_results.append(match_detected_groundtruth_keypoint(
        #                     batch_gt_keypoint[:, :, :, :, k, f], c_detected_keypoint, 1))

        return i_results, t_results, c_results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
