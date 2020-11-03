import pytorch_lightning as pl
import torch
from torch import nn
from models.unet.blocks import ResidualBlock, TemporalResidualBlock, Encoder, Decoder


class UNet3D(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, num_feature: int, num_levels: int, has_skip_conn_encoder_decoder=True, kernel_size=3) -> None:
        super(UNet3D, self).__init__()
        f_maps = [num_feature * pow(2, i)
                  for i in range(num_levels)]
        encoders = []
        decoders = []

        for i in range(len(f_maps)):
            out_features_num = f_maps[i]
            if i == 0:
                encoder = Encoder(ResidualBlock, in_channels, out_features_num, kernel_size)
            else:
                encoder = Encoder(ResidualBlock, f_maps[i-1], out_features_num, kernel_size)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))

        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] if has_skip_conn_encoder_decoder else \
                reversed_f_maps[i] + reversed_f_maps[i+1]
            decoder = Decoder(
                in_feature_num, reversed_f_maps[i+1], kernel_size)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.last_conv = ResidualBlock(f_maps[0], out_channels, kernel_size)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)
            
        return self.last_conv(x)


class TemporalResnet(pl.LightningModule):
    '''
    An encoder structure that have a hardcoded depth and fixed recpetive field.
    Currently, only support depth of 4 and receptive field of 15
    '''
    def __init__(self, in_channels: int, out_channels: int,num_feature:int) -> None:
        super().__init__()
        kernel_size = 3
        input_conv = self.InputLayer(in_channels, num_feature)
        encoders = []
        encoders.append(input_conv)
        for i in range(3):
            encoder = TemporalResidualBlock(num_feature,num_feature, kernel_size, pow(2, i), stride_size=1)
            encoders.append(encoder)
        
        self.encoders = nn.Sequential(*encoders)
        self.last_conv = ResidualBlock(num_feature, out_channels, 1, 1)
    
    def forward(self, x):
        '''
        Forward the 6-dim tensor x, the model will share weight across different view
        params\n
        x: with 6 dimensions [batch, num_joints, height, width, num_views, num_frames]
        '''
        results = []
        num_view = x.size(4)
        for k in range(num_view):
            out = self.encoders(x[:,:,:,:,k,:])
            results.append(self.last_conv(out))

        return torch.stack(results)

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
        
        

        
        