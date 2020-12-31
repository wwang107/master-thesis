import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.load_model import load_model_state_dict
from models.resnet.model import CustomizedResnet
from models.unet.TemporalUnet import TemporalUnet
from models.fusion_net.model import FusionNet
from models.aggregate.model import AggregateModel
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from callbacks.callbacks import LogConfusionTable, LogModelHeatmaps
from models.epipolar.EpipolarTransformer import Epipolar
from utils.multiview import camera_center

def load_weight(model, state_dict):
    model.load_state_dict(state_dict)
    return model

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2) * 10000


def main(hparams):
    print(hparams)
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels, out_channels, num_feature = 55, 55, hparams.num_feat
    num_levels = hparams.num_level
    replicate_view = hparams.replicate_view
    is_train_input_encoder = hparams.input_encoder

    resnet = CustomizedResnet(use_pretrained=True)
    resnet = load_weight(resnet,load_model_state_dict("/home/wei/master-thesis/best_50.pth", device))
    


    fuse_model = FusionNet(2*in_channels, out_channels, num_feature, input_frame=1)
    model = AggregateModel(resnet, Epipolar(debug=False), None, fuse_model, None,
                           weighted_mse_loss, in_channels, out_channels, 
                           train_input_heatmap_encoder=is_train_input_encoder, num_camera_can_see=cfg.DATASET.NUM_VIEW, num_frame_can_see=cfg.DATASET.NUM_FRAME_PER_SUBSEQ)
    data_loader = {
        'train': make_dataloader(cfg, dataset_name='cmu', is_train=True, replicate_view=replicate_view),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False, replicate_view=replicate_view)
    }
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs= 20,
                         limit_val_batches=0.5,
                        #  limit_test_batches=3,
                         callbacks=[LogModelHeatmaps(log_dir=hparams.images_dir, num_frame=cfg.DATASET.NUM_FRAME_PER_SUBSEQ),
                                    ModelCheckpoint(monitor='validation_step_avg_loss/temporal_encoder', save_top_k=3)])
    trainer.fit(model, train_dataloader=data_loader['train'], val_dataloaders=data_loader['valid'])
    trainer.test(test_dataloaders=data_loader['valid'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--images_dir', default='images')
    parser.add_argument('--resnet_weight_dir', default='pretrain/CutomizeResNet-experiment/best_13.pth')
    parser.add_argument('--temporal_encoder', action='store_true')
    parser.add_argument('--view_encoder', action='store_true')
    parser.add_argument('--input_encoder', action='store_true')
    parser.add_argument('--replicate_view', action='store_true')
    parser.add_argument('-num_feat', default=55)
    parser.add_argument('-num_level', default=2)
    args = parser.parse_args()
    main(args)
