import torch
import torch.nn as nn
import torchvision.models as models
 
class ResidualBlock(nn.Module):
   def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
       '''
       '''
       super().__init__()
       padding = kernel_size//2
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
           nn.ReLU())
       self.conv2 = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
           nn.ReLU())
       self.conv3 = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
           nn.ReLU())
 
   def forward(self, x):
       residual = x
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x += residual
       return x
 
class CustomizedResnet(nn.Module):
   def __init__(self, use_pretrained=True, fix_encoder_params=True):
       super().__init__()
       self.fix_encoder_params = fix_encoder_params
       resnet18 = models.resnet18(pretrained=use_pretrained)
       self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(512, 512, 2, stride=2),
           ResidualBlock(512,512,3),
           nn.ConvTranspose2d(512, 256, 2, stride=2),
           ResidualBlock(256,256,3),
           nn.ConvTranspose2d(256, 128, 2, stride=2),
           ResidualBlock(128,128,3),
           nn.Conv2d(128,55,1)
       )
 
   def forward(self, x):
       r"""
       Parmeter
       --------------------------
       x has a dimension of [N,channel,Height,Width]
       """
       with torch.set_grad_enabled(not self.fix_encoder_params):
           x = self.encoder(x)
       return self.decoder(x)
 
   def deactive_batchnorm(self):
       for m in self.encoder.modules():
           if isinstance(m, nn.BatchNorm2d):
               m.eval()
           if isinstance(m, nn.Dropout2d):
               m.eval()
       return self