import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import time
import copy


class CustomizedResnet(nn.Module):
    def __init__(self, use_pretrained=True, fix_encoder_params=False):
        super().__init__()
        resnet18 = models.resnet18(pretrained=use_pretrained)
        self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 55, 1)
        )

        if fix_encoder_params:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        r"""
        Parmeter
        --------------------------
        x has a dimension of [N,channel,Height,Width]
        """
        x = self.encoder(x)
        return self.decoder(x)


def train_model(model, dataloaders, criterion, optimizer, device, visulizor=None, num_epochs=25):
    since = time.time()

    val_loss_history = []

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            iteration_loss = 0.0
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
                iteration_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, iteration_loss / 100))
                    iteration_loss = 0.0
                    visulizor.save_batch_heatmaps(
                        images, outputs, batch_masks=masks, file_name="{}-{}".format(epoch, i))
                    visulizor.save_batch_heatmaps(
                        images, heatmaps, batch_masks=masks, file_name="{}-{}-gt".format(epoch, i))
                    visulizor.save_batch_joints_and_directional_keypoints_plot(
                        images, joints, keypoints, file_name="{}-{}-joints".format(epoch, i))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('lowest loss val: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss
