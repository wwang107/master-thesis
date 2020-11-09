import torch
import pytorch_lightning as pl
from torch import nn
from models.unet.blocks import ResidualBlock, TemporalResidualBlock, Encoder, Decoder

class TemporalUnet(pl.LightningModule):
    '''
    An encoder structure that have a hardcoded depth and fixed recpetive field.
    Currently, only support depth of 4 and receptive field of 15
    '''
    def __init__(self, in_channels: int, out_channels: int,num_feature:int) -> None:
        super().__init__()
        depth = 3
        encoder_kernel_size = (3,3,3)
        decoder_kernel_size = (3,3,1)
        encoders = []
        decoders = []
        
        self.input_conv = self.InputLayer(in_channels, num_feature)
        for i in range(depth):
            if i == 0:
                encoder = Encoder(TemporalResidualBlock, num_feature, num_feature, encoder_kernel_size, pow(2,i), apply_pooling = False)
            else:
                encoder = Encoder(TemporalResidualBlock, num_feature, num_feature, encoder_kernel_size, pow(2,i), apply_pooling = True, pool_kernel = (2,2,1))
            encoders.append(encoder)
        
        for i in range(depth - 1):
            decoder = Decoder(TemporalResidualBlock, num_feature, num_feature, decoder_kernel_size, 1, scale_factor = (2,2,1))
            decoders.append(decoder)

        self.input_conv = self.InputLayer(in_channels, num_feature)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.last_conv = ResidualBlock(num_feature, out_channels, (3,3,1), 1)
    
    def forward(self, x):
        '''
        Forward pass each camera view indidvidually, the model will share weight across different view
        params\n
        x: with 6 dimensions [batch, num_joints, height, width, num_views, num_frames]
        '''
        results = []
        num_view = x.size(4)
        input_view = [x[:,:,:,:,k,:] for k in range(num_view)]
        for k in range(num_view):
            encoders_features = []
            out = input_view[k]
            out = self.input_conv(out)
            for encoder in self.encoders:
                out = encoder(out)
                encoders_features.insert(0, out)
            
            encoders_features = encoders_features[1:]
            for decoder, encoder_features in zip(self.decoders, encoders_features):
                encoder_features = torch.mean(encoder_features, dim = 4, keepdim=True)
                out = decoder(out, encoder_features)

            results.append(self.last_conv(out))

        return torch.stack(results, dim=4)

    class InputLayer(nn.Module):
        def __init__(self, in_channels, out_channels) -> None:
            super().__init__()
            self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels,
                        kernel_size=(1, 1, 1), dilation=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
            self.conv3 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels,
                        kernel_size=(1, 1, 1), dilation=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())

        def forward(self, x):
            residual = self.conv1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + residual
            return x

# input = torch.zeros((1,55,64,64,5,15))
# model = TemporalUnet(55,55,128)
# out = model(input[:,:,:,:,:,:])