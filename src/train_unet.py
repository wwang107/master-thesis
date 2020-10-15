import torch
from model.losses import JointsMSELoss
from model.unet3d.unet3d import UNet3D
from data.build import make_dataloader
from utils.load_model import load_model_state_dict
from utils.vis.vis import save_batch_multi_view_with_heatmap
from model.resnet.resnet import CustomizedResnet
from config.defualt import get_cfg_defaults
from pathlib import Path
from datetime import datetime
from utils.writer.writer import TensorBoardWriter
from trainer.model_trainer import save_checkpoint
import copy

PATH_TO_HEATMAP_MODEL = '/home/weiwang/master-thesis/runs/2020-10-02-16:52:11/checkpoint_15.pth'
ROOT_TO_CMU_DATA = "/home/weiwang/master-thesis/dataset/panoptic"

def main():
    cfg = get_cfg_defaults()
    log_dir = str(Path.cwd().joinpath(
        'runs', 'unet', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')).resolve())
    tsboard = TensorBoardWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet3D(in_channels=55, out_channels=55, final_sigmoid=False, f_maps=110, num_levels=2,
              is_segmentation=False, num_groups=55)
    unet = unet.to(device)
    heatmap_model = CustomizedResnet()
    heatmap_model.load_state_dict(load_model_state_dict(PATH_TO_HEATMAP_MODEL))
    heatmap_model.to(device)

    data_loader = { 
        'train': make_dataloader(cfg, dataset_name='cmu', is_train=True),
        'valid': make_dataloader(cfg, dataset_name='cmu', is_train=False),
	}
    criterion = JointsMSELoss(use_target_weight=True)
    num_view = cfg.DATASET.NUM_VIEW
    num_frame = cfg.DATASET.NUM_FRAME_PER_SUBSEQ
    optimizer = torch.optim.Adam(unet.parameters())
    
    best_model_wts = copy.deepcopy(unet.state_dict())
    lowest_loss = float('inf')
    for epoch in range(20):
        for phase in ['train', 'valid']:
            if phase == 'train':
                unet.train()  # Set model to training mode
            else:
                unet.eval()   # Set model to evaluate mode
                valid_images = {}
                for i in range(num_view):
                    valid_images['input_heatmap_cam_view_{}'.format(i)] = []
                    valid_images['predict_heatmap_cam_view_{}'.format(i)] = []
                    valid_images['predict_groundtruth_cam_view_{}'.format(i)] = []
                
            running_loss = 0.0
            # Iterate over data.
            for i, data in enumerate(data_loader[phase]):
                
                optimizer.zero_grad()

                batch_kpt, batch_hm_gt, batch_img, num_person = \
                data['keypoint2d'], data['heatmap'], data['img'], data['num_person']
                batch_hm_gt, batch_img = batch_hm_gt.to(device), batch_img.float().to(device)
                heatmap_input = torch.zeros(batch_hm_gt.size()).cuda(device)
                batch_img = batch_img.permute((0, 3, 1, 2, 4, 5))
                with torch.no_grad():
                    for cam_view in range(num_view):
                        for frame in range(num_frame):
                            heatmap_input[:,:,:,:,cam_view, frame]= heatmap_model(batch_img[:,:,:,:,cam_view, frame])

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0.0
                    batch_hm_gt = batch_hm_gt.to(device)
                    x = unet(heatmap_input[:, :, :, :, :, -1])
                    for k in range(num_view):
                        loss += criterion(x[:,:,:,:,k], batch_hm_gt[:,:,:,:,k,-1])
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item()
                if i % 100 == 0:    # print every 100 mini-batches
                        print('[%s][%d, %3d/%3d] loss: %.3f' %
                                (phase,epoch, i, len(data_loader[phase]), loss.item()))
                        if phase == 'valid':
                            a = save_batch_multi_view_with_heatmap(
                                batch_img[:, :, :, :, :, -1], heatmap_input, 'test')
                            b = save_batch_multi_view_with_heatmap(
                                batch_img[:, :, :, :, :, -1], x, 'test')
                            c = save_batch_multi_view_with_heatmap(
                                batch_img[:, :, :, :, :, -1], batch_hm_gt, 'test')
                            
                            for k in range(num_view):
                                valid_images['input_heatmap_cam_view_{}'.format(k)].append(a[k])
                                valid_images['predict_heatmap_cam_view_{}'.format(k)].append(b[k])
                                valid_images['predict_groundtruth_cam_view_{}'.format(k)].append(c[k])
                print('[%s][%d, %3d/%3d] loss: %.3f' %
                      (phase, epoch, i, len(data_loader[phase]), loss.item()))
            epoch_loss = running_loss/len(data_loader['valid'])
            if phase == 'valid':
                tsboard.add_images('valid',valid_images,epoch)
                tsboard.add_scalar('valid', epoch_loss, epoch)

                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_model_wts = copy.deepcopy(unet.state_dict())
                    save_checkpoint(unet, optimizer, epoch,
                                    epoch_loss, log_dir, 'best')

            elif phase == 'train':
                tsboard.add_scalar('train', epoch_loss, epoch)
                save_checkpoint(unet, optimizer, epoch,
                                loss.item(), log_dir, 'checkpoint')
                    
if __name__ == "__main__":
    main()
