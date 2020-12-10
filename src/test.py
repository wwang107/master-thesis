import torch
import pytorch_lightning as pl
from data.build import make_dataloader
from models.epipolar.EpipolarTransformer import Epipolar
from models.unet.TemporalUnet import TemporalUnet
from models.resnet.model import CustomizedResnet
from models.aggregate.model import AggregateModel
from callbacks.callbacks import LogConfusionTable, LogModelHeatmaps
from config.defualt import get_cfg_defaults

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2) * 10000

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    in_channels, out_channels, num_feature = 55, 55, 55
    num_camera_can_see = 5
    num_frame_can_see = 9
    replicate_view = False
    is_train_input_encoder = False

    # Initialize model
    backbone = CustomizedResnet(use_pretrained=False)
    fuse_model = TemporalUnet(2*in_channels, out_channels, num_feature, input_frame=1)
    epiploar = Epipolar(debug=False)
    temporal_model = TemporalUnet(in_channels, out_channels, num_feature, 
                                  input_frame=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)

    # Initialize Aggregate model
    agg_model = AggregateModel(backbone, epiploar, None, fuse_model, temporal_model, 
                               weighted_mse_loss, 
                               in_channels, out_channels, 
                               train_input_heatmap_encoder=is_train_input_encoder, 
                               num_camera_can_see=num_camera_can_see,
                               num_frame_can_see=num_frame_can_see)

    # load pretrained weight
    state_dict = torch.load('pretrain/fusion-model/epoch=19.ckpt', map_location=device)['state_dict']
    agg_model.load_state_dict(state_dict, strict=False)

    # Test
    data_loader = {
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False, replicate_view=replicate_view)
    }
    trainer = pl.Trainer(gpus=1, 
                        max_epochs= 20,
                        limit_test_batches=3,
                        callbacks=[LogModelHeatmaps(log_dir='images', num_frame=num_frame_can_see)])
    
    trainer.test(agg_model, test_dataloaders=data_loader['valid'])

if __name__ == "__main__":
    main()