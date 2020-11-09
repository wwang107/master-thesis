import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.load_model import load_model_state_dict
from models.resnet.model import CustomizedResnet
from models.unet.TemporalUnet import TemporalUnet
from models.unet.BaselineMultiViewModel import BaselineMultiViewModel
from models.aggregate.model import AggregateModel
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from callbacks.callbacks import LogConfusionTable, LogModelHeatmaps

def load_weight(model, state_dict):
    model.load_state_dict(state_dict)
    return model

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def main(hparams):
    
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_channels, out_channels, num_feature = 55, 55, hparams.num_feat
    num_levels = hparams.num_level
    resnet = CustomizedResnet(use_pretrained=False, fix_encoder_params=False)
    resnet = load_weight(resnet,load_model_state_dict(hparams.resnet_weight_dir, device))
    resnet = resnet.deactive_batchnorm()
    camera_view_model = BaselineMultiViewModel(in_channels, out_channels,
                  num_feature=num_feature, num_levels=num_levels, kernel_size=(3,3,5)) if hparams.view_encoder else None
    temporal_model = TemporalUnet(in_channels, out_channels, num_feature) if hparams.temporal_encoder else None
    
    data_loader = {
        'train': make_dataloader(cfg, dataset_name='cmu', is_train=True),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False)
    }
    trainer = pl.Trainer(gpus=hparams.gpus, max_epochs= 20, callbacks=[LogConfusionTable(), LogModelHeatmaps(log_dir=hparams.images_dir, num_frame=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)])
    model = AggregateModel(resnet, camera_view_model, temporal_model,
                           weighted_mse_loss, in_channels, out_channels, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    trainer.fit(model, train_dataloader=data_loader['train'], val_dataloaders=data_loader['valid'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--images_dir', default='images')
    parser.add_argument('--resnet_weight_dir', default='/home/wei/master-thesis/pretrain/CutomizeResNet-experiment/best_13.pth')
    parser.add_argument('--temporal_encoder', action='store_true')
    parser.add_argument('--view_encoder', action='store_true')
    parser.add_argument('--num_feat', default=110)
    parser.add_argument('--num_level', default=2)
    args = parser.parse_args()
    main(args)
