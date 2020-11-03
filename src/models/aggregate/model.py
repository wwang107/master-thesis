import torch
import pytorch_lightning as pl
from models.resnet.model import CustomizedResnet
from models.unet.model import UNet3D, TemporalResnet


class AggregateModel(pl.LightningModule):
    '''
    Aggregate Model composed by a resnet that take 256x256 RGB, then output 64x64 heatmaps and by an unet that takes 64x64 heatmaps 
    '''

    def __init__(self,
                input_heatmap_encoder,
                camera_view_encoder = None,
                temporal_encoder = None,
                loss = None,
                in_channels=55,
                out_channels=55,
                heatmap_size=(64,64),
                num_camera_can_see=6,
                num_frame_can_see=15,
                is_train_restnet: bool = False):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_camera_view = num_camera_can_see
        self.num_frame = num_frame_can_see
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_train_restnet = is_train_restnet
        self.input_heatmap_encoder = input_heatmap_encoder
        self.camera_view_encoder = camera_view_encoder
        self.temporal_encoder = temporal_encoder
        self.loss = loss
        # if weight_resnet is not None:
        #     self.resnet.load_state_dict(weight_resnet)
        # if weight_unet is not None:
        #     self.unet.load_state_dict(weight_unet)

    def forward(self, x):
        '''
        param: x the image
        '''
        batch_heatmaps = torch.zeros((x.size(0), self.in_channels, *self.heatmap_size, self.num_camera_view, self.num_frame)).to(self.device)
        with torch.set_grad_enabled(self.is_train_restnet):
            for k in range(self.num_camera_view):
                for f in range(self.num_frame):
                    batch_heatmaps[:,:,:,:,k,f] = self.input_heatmap_encoder(x[:,:,:,:,k,f])

        t_results = self.temporal_encoder(batch_heatmaps)
        return t_results

    def training_step(self, batch, batch_idx):
        batch_imgs = batch['img'].float()
        batch_gt_hm = batch['heatmap'][:,:,:,:,:,self.num_frame//2]
        temporal_hm = self(batch_imgs)
        t_loss = self.loss(temporal_hm, batch_gt_hm, batch_gt_hm>0)
        print(t_loss)
        return t_loss

    def validation_step(self, batch, batch_idx):
        batch_imgs = batch['img'].float()
        batch_gt_hm = batch['heatmap'][:,:,:,:,:,self.num_camera_view//2]
        temporal_hm = self(batch_imgs)
        t_loss = self.loss(temporal_hm, batch_gt_hm, batch_gt_hm>0)
        self.log('validation loss', t_loss, prog_bar=True)
        return t_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    
