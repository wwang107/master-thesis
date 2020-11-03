import torch
import pytorch_lightning as pl
from models.resnet.model import CustomizedResnet
from models.unet.model import UNet3D, TemporalResnet
from models.aggregate.model import AggregateModel
from config.defualt import get_cfg_defaults
from data.build import make_dataloader

def load_weight(self, model, state_dict):
    model.load_state_dict(state_dict)
    return model

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def main():
    cfg = get_cfg_defaults()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = {
        # 'train': make_dataloader(cfg, dataset_name='cmu', is_train=True),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False)
    }
    trainer = pl.Trainer(gpus=1)

    in_channels, out_channels, num_feature = 55, 55, 110
    num_levels = 2
    resnet = CustomizedResnet(use_pretrained=False, fix_encoder_params=False)
    unet = UNet3D(in_channels, out_channels,
                  num_feature=num_feature, num_levels=num_levels)
    temporal_restnet = TemporalResnet(in_channels, out_channels, num_feature)
    model = AggregateModel(resnet, unet, temporal_restnet,
                           weighted_mse_loss, in_channels, out_channels, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    trainer.fit(model, train_dataloader=data_loader['valid'])


if __name__ == "__main__":
    main()
