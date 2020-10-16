import torch
import os
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from model.resnet.resnet import CustomizedResnet
from trainer.model_trainer import train_model
from model.losses import WeightedRegLoss
from utils.writer.writer import TensorBoardWriter
from pathlib import Path
from datetime import datetime


COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [
        12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def load_checkpoint(model, optimizer, path):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return model, optimizer, epoch, loss

if __name__ == "__main__":
    log_dir = str(Path.cwd().joinpath(
        'runs', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')).resolve())
    cfg = get_cfg_defaults()
    cfg.freeze()
    tsboard = TensorBoardWriter(log_dir)
    data_loaders = {'train': make_dataloader(cfg, is_train=True),
                    'val': make_dataloader(cfg, is_train=False)}
    model = CustomizedResnet()
    optimizer = torch.optim.Adam(model.parameters())
    loss = WeightedRegLoss()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, epoch, _= \
        load_checkpoint(
        model,
        optimizer,
        '/home/weiwang/master-thesis/runs/2020-10-02-16:52:11/checkpoint_15.pth')
    trained_model, loss = train_model(model, data_loaders, loss, optimizer, device, checkpt_dir=log_dir, writer=tsboard, num_epochs= 20)


