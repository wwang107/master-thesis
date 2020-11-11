import cv2
import os
from pytorch_lightning import Callback
from utils.vis.vis import save_batch_multi_view_with_heatmap


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
        self.num_frame = num_frame // 2

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        heatmaps_dict = outputs['heatmaps']
        prefix = os.path.join(self.log_dir, 'ver_' + str(trainer.logger.version))
        os.makedirs(prefix, exist_ok=True)
        
        encoders = [key for key in heatmaps_dict if heatmaps_dict[key] != None]
        global_step = pl_module.global_step
        epoch = pl_module.current_epoch
        for encoder in encoders:
            images = heatmaps_dict[encoder]['images'][:,:,:,:,:, self.num_frame // 2].cpu()
            
            if encoder == 'temporal_encoder':
                heatmaps = heatmaps_dict[encoder]['heatmaps']
            else:
                heatmaps = heatmaps_dict[encoder]['heatmaps'][:,:,:,:,:, self.num_frame // 2]

            heatmaps = heatmaps.cpu()
            visualization = save_batch_multi_view_with_heatmap(images,heatmaps)
            for view, image in enumerate(visualization):
                file_name = os.path.join(prefix, '{}_epoch_{}_step_{}_view_{}.png'.format(encoder, epoch, global_step, view))
                cv2.imwrite(str(file_name), image)
        
        
