import torch
import time
import copy
from torch import mode
from torch.optim import optimizer
from utils.vis.vis import save_batch_maps
from pathlib import Path


def train_model(model, dataloaders, criterion, optimizer, device, checkpt_dir, writer=None, num_epochs=20):
    since = time.time()

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            valid_images = {'prediction': [],
                            'groundtruth': [], 'direction-keypoint': []}
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                images, joints, keypoints, heatmaps, masks = data['images'], data[
                    'joints'], data['directional_keypoints'], data['heatmaps'], data['masks']
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                masks = masks.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps, masks)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                
                if i % 100 == 99:    # print every 100 mini-batches
                    print('%s: [%d, %3d/%3d] loss: %.3f' %
                          (phase, epoch, i,len(dataloaders[phase]), loss.item()))
                    if phase == 'val':
                        valid_images['prediction'].append(
                            save_batch_maps(images, outputs, masks))
                        valid_images['groundtruth'].append(
                            save_batch_maps(images, heatmaps, masks))
                        valid_images['direction-keypoint'].append(
                            save_batch_maps(images, outputs, masks))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if phase == 'train':
                    writer.add_scalar('train', epoch_loss, epoch)
            # deep copy the model
            if phase == 'val':
                save_checkpoint(model, optimizer, epoch,
                                loss.item(), checkpt_dir, 'checkpoint')
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_checkpoint(model, optimizer, epoch, epoch_loss, checkpt_dir, 'best')
                    
                
                writer.add_scalar('valid', epoch_loss, epoch)
                writer.add_images('valid', valid_images, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('lowest loss val: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss

def save_checkpoint(model, optimizer, epoch, loss, path, name):
    
    torch.save(
        {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, 
        Path.joinpath(Path(path), '{}_{}.pth'.format(name, epoch)))
    