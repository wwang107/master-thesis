from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from utils.vis.handlers import VisulizationHandler


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.freeze()
    vis_handler = VisulizationHandler(cfg.DATASET.INPUT_SIZE,
                                      cfg.DATASET.OUTPUT_SIZE, 
                                      '/home/weiwang/master-thesis/images')
    data_loader = make_dataloader(cfg, is_train=True)
    for i, data in enumerate(data_loader):
        batch_images, batch_heatmaps, batch_joints, batch_keypoints = \
            data["images"], data["heatmaps"], data["joints"], data["directional_keypoints"]
        vis_handler.save_batch_joints_and_directional_keypoints_plot(
            batch_images, batch_joints, batch_keypoints)
        vis_handler.save_batch_heatmaps(batch_images, batch_heatmaps, batch_masks= None)

            
