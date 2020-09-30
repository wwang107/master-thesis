import os
from random import sample

from torch.utils.data import dataloader
import data
from data.transforms import build_transforms
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from utils.vis.handlers import VisulizationHandler
from model.model import CustomizedResnet, train_model
from model.losses import RegLoss, WeightedRegLoss
from utils.writer.writer import TensorBoardWriter
from pathlib import Path
from datetime import datetime
from data.PanopticDataset import PanopticDataset
from data.target_generators import HeatmapGenerator, DirectionalKeypointsGenerator
from utils.vis.vis import save_batch_heatmaps_multi, save_batch_sequence_image_with_joint
from torch.utils.data import DataLoader

COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [
        12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

if __name__ == "__main__":
    log_dir = str(Path.cwd().joinpath(
        'runs', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')).resolve());
    cfg = get_cfg_defaults()
    cfg.freeze()
    # # vis_handler = VisulizationHandler(cfg.DATASET.INPUT_SIZE,
    # #                                   cfg.DATASET.OUTPUT_SIZE,
    # #                                   '/home/weiwang/master-thesis/images')
    # tsboard = TensorBoardWriter(log_dir)
    # data_loaders = {'train': make_dataloader(cfg, is_train=True),
    #                 'val': make_dataloader(cfg, is_train=False)}
    # # data_loaders = {'train': make_dataloader(cfg, is_train=True)}
    # model = CustomizedResnet()
    # optimizer = torch.optim.Adam(model.parameters())
    # loss = WeightedRegLoss()
    # print(torch.cuda.is_available())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_model(model, data_loaders, loss, optimizer, device, writer=tsboard)
    

    is_load_path_only = True
    dataset = PanopticDataset(
        'dataset/panoptic', cfg,
        heatmap_generator=HeatmapGenerator(cfg.DATASET.OUTPUT_SIZE, len(COCO_SKELETON)*2, cfg.DATASET.SIGMA),
        keypoint_generator=DirectionalKeypointsGenerator(
            cfg.DATASET.NUM_JOINTS, COCO_SKELETON),
        is_load_path_only=is_load_path_only)
    dataloader = DataLoader(dataset, batch_size=1)
    camera_id = (0, 0)
    seq_name = '170407_haggling_a1'

    for data in dataloader:
        batch_img, batch_heatmap, batch_keypoint2d, num_person = \
            data['img'], data['heatmap'], data['keypoint2d'], data['num_person']
        batch_img = batch_img.float().permute((0, 3, 1, 2, 4,5))
        save_batch_sequence_image_with_joint(
            batch_img, batch_keypoint2d, num_person, 'test')
