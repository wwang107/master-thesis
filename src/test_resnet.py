import torch
import os
from config.defualt import get_cfg_defaults
from data.build import make_dataloader
from models.resnet.model import CustomizedResnet
from utils.vis.vis import add_joints
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

# def load_checkpoint(model, optimizer, path):
#     # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
#     checkpoint = torch.load(path)
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.cuda()

#     return model, optimizer, epoch, loss

    
def calculate_prediction(hms, bboxes, imgs = None, method='avg'):
    if imgs is not None:
        imgs = imgs.cpu().clone().float()
        min = float(imgs.min())
        max = float(imgs.max())
        imgs.add_(-min).div_(max - min + 1e-5)

    batch_predictions = []
    nms = torch.nn.Threshold(0.6, 0)
    for i,hm in enumerate(hms):
        predictions = []
        for k,bbox in enumerate(bboxes[i]):
            pred_joint = torch.zeros(17,3).to(hm)
            x_ul, y_ul, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            roi = nms(hm[:,y_ul:y_ul+h, x_ul:x_ul+w])
            
            denom = torch.sum(torch.sum(roi, dim=-1), dim=-1)
            vis = denom>=1e-5
            

            y_grid, x_grid = torch.meshgrid(torch.arange(y_ul, y_ul + h), torch.arange(x_ul, x_ul + w))
            grid = torch.cat((x_grid.unsqueeze(dim=2),y_grid.unsqueeze(dim=2)), dim=2).unsqueeze(0).expand(17,-1,-1,-1).to(hm)
            x_loc = torch.sum(torch.sum(grid[...,0] * roi, dim=-1),
                              dim=-1) / denom
            y_loc = torch.sum(torch.sum(grid[...,1] * roi, dim=-1),
                              dim=-1) / denom
            pred_joint[:,0] = x_loc
            pred_joint[:,1] = y_loc
            pred_joint[vis,2] = 1
            predictions.append(pred_joint)
        if imgs is not None:
            img = imgs[i].mul(255)\
                              .clamp(0, 255)\
                              .permute(1, 2, 0)\
                              .byte()\
                              .cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for person in predictions:
                add_joints(img, person*2)
            cv2.imwrite('test.png',img)
        batch_predictions.append(torch.stack(predictions))

    return batch_predictions

if __name__ == "__main__":
    # log_dir = str(Path.cwd().joinpath(
    #     'runs', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')).resolve())
    cfg = get_cfg_defaults()
    cfg.freeze()
    data_loaders = {'val': make_dataloader(cfg, is_train=False)}
   
    weight = torch.load('pretrain/resnet50/best_68.pth')['model_state_dict']
    model = CustomizedResnet()
    model.load_state_dict(weight)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model.eval()

    for data in data_loaders['val']:
        print(len(data_loaders['val']))
        with torch.no_grad():
            imgs, joints, bboxes = data['images'].to(device).float(), data['joints'], data['bboxes']
            hms = model(imgs)[:,0:17]
            predictions = calculate_prediction(hms, bboxes, imgs)
        # imgs = F.interpolate(imgs, 128)
        # batch_image = imgs.cpu().clone().float()
        # min = float(batch_image.min())
        # max = float(batch_image.max())
        # batch_image.add_(-min).div_(max - min + 1e-5)
        # batch_image = batch_image.permute(0,2,3,1).numpy()
        
        # fig = plt.figure(figsize=(60, 60))
        # ax = fig.add_subplot(111)
        # ax.axis('off')
        # ax.imshow(batch_image[0])

        # for bbox in bboxes[0]:
        #     rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
        #     ax.add_patch(rect)
        
        
        # plt.tight_layout()
        # plt.show()



