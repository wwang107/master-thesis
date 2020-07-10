from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from utils.vis.handlers import VisulizationHandler
from model.model import CustomizedResnet, train_model
import torch


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.freeze()
    vis_handler = VisulizationHandler(cfg.DATASET.INPUT_SIZE,
                                      cfg.DATASET.OUTPUT_SIZE,
                                      '/home/weiwang/master-thesis/images')
    data_loaders = {'train': make_dataloader(cfg, is_train=False),
                    'val': make_dataloader(cfg, is_train=False)}
    model = CustomizedResnet()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    loss = torch.nn.BCELoss()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, data_loaders, loss, optimizer, device, vis_handler)
        
        
        # vis_handler.save_batch_joints_and_directional_keypoints_plot(
        #     batch_images, batch_joints, batch_keypoints)
        # vis_handler.save_batch_heatmaps(
        #     batch_images, batch_heatmaps, batch_masks=None)
