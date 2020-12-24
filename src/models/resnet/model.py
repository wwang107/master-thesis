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
           nn.ReLU(), 
           nn.BatchNorm2d(out_channels))
       self.conv2 = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
           nn.ReLU(),  
           nn.BatchNorm2d(out_channels))
       self.conv3 = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
           nn.ReLU(),  
           nn.BatchNorm2d(out_channels))
 
   def forward(self, x):
       residual = x
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x += residual
       return x

class FCNHead(nn.Module):
	def __init__(self, in_channels, channels):
		super().__init__()
		inter_channels = in_channels // 4
		self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 3, padding=1))
    
	def forward(self, x):
		return self.conv(x) 
class CustomizedResnet(nn.Module):
   def __init__(self, use_pretrained=True, fix_encoder_params=True):
        super().__init__()
        self.fix_encoder_params = fix_encoder_params
        # resnet18 = models.resnet18(pretrained=use_pretrained)
        # self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 2, stride=2),
        #     nn.ReLU(),nn.BatchNorm2d(256),
        #     ResidualBlock(256,256,3),
        #     nn.ConvTranspose2d(256, 128, 2, stride=2),
        #     nn.ReLU(),nn.BatchNorm2d(128),
        #     ResidualBlock(128,128,3),
        #     nn.ConvTranspose2d(128, 55, 2, stride=2),
        #     nn.ReLU(),nn.BatchNorm2d(55),
        #     ResidualBlock(55,55,3),
        #     nn.ConvTranspose2d(55, 55, 2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(55,55,3,padding=1)
        
        resnet50 = models.segmentation.fcn_resnet50(pretrained=use_pretrained)
        self.encoder = nn.Sequential(*list(resnet50.children())[:-2])
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(2048, 1024, 2, stride=2), nn.ReLU(),nn.BatchNorm2d(1024),
		# 	nn.Conv2d(1024, 512, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(512),
		# 	nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(),nn.BatchNorm2d(256),
		# 	FCNHead(256, 55)
		# )

        self.downsample = nn.Sequential(nn.Conv2d(2048,2048,1,2, bias=False),nn.BatchNorm2d(2048))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2), nn.ReLU(),nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ReLU(),nn.BatchNorm2d(512),
			nn.Conv2d(512, 256, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),nn.BatchNorm2d(128),
			FCNHead(128, 55)
		)

   def forward(self, x):
       r"""
       Parmeter
       --------------------------
       x has a dimension of [N,channel,Height,Width]
       """
       with torch.set_grad_enabled(not self.fix_encoder_params):
            x = self.encoder(x)['out']
       x = self.downsample(x)
       return self.decoder(x)
 
   def deactive_batchnorm(self):
       for m in self.encoder.modules():
           if isinstance(m, nn.BatchNorm2d):
               m.eval()
           if isinstance(m, nn.Dropout2d):
               m.eval()
       return self