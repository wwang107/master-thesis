from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from utils.vis.handlers import VisulizationHandler
from model.model import CustomizedResnet, train_model
from model.losses import RegLoss, WeightedRegLoss

import torch


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.freeze()
    vis_handler = VisulizationHandler(cfg.DATASET.INPUT_SIZE,
                                      cfg.DATASET.OUTPUT_SIZE,
                                      '/home/weiwang/master-thesis/images')
    # data_loaders = {'train': make_dataloader(cfg, is_train=True),
    #                 'val': make_dataloader(cfg, is_train=False)}
    data_loaders = {'train': make_dataloader(cfg, is_train=True)}
    model = CustomizedResnet()
    optimizer = torch.optim.Adam(model.parameters())
    loss = WeightedRegLoss()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, data_loaders, loss, optimizer, device, vis_handler)
