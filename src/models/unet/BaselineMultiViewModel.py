from typing import Tuple
import pytorch_lightning as pl
import torch
from torch import nn
from models.unet.blocks import ResidualBlock, Encoder, Decoder

class BaselineMultiViewModel(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, num_feature: int, num_levels: int, kernel_size:Tuple) -> None:
        super().__init__()
        f_maps = [num_feature * pow(2, i)
                  for i in range(num_levels)]
        encoders = []
        decoders = []
        
        for i in range(len(f_maps)):
            out_features_num = f_maps[i]
            if i == 0:
                encoder = Encoder(ResidualBlock, in_channels, out_features_num, kernel_size, apply_pooling=False)
            else:
                encoder = Encoder(ResidualBlock, f_maps[i-1], out_features_num, kernel_size, apply_pooling=True, pool_kernel = (2,2,1))

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))

        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            decoder = Decoder(
                ResidualBlock, in_feature_num, reversed_f_maps[i+1], kernel_size, scale_factor=(2,2,1))
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.last_conv = ResidualBlock(f_maps[0], out_channels, kernel_size, dilation=1)

    def forward(self, x):
        results = []
        num_frame = x.size(5)
        input_view = [x[:,:,:,:,:,f] for f in range(num_frame)]

        for k in range(num_frame):
            encoders_features = []
            out = input_view[k]
            for encoder in self.encoders:
                out = encoder(out)
                encoders_features.insert(0, out)

            encoders_features = encoders_features[1:]

            for decoder, encoder_features in zip(self.decoders, encoders_features):
                out = decoder(out, encoder_features)
                out = self.last_conv(out)
            results.append(out)

        return torch.stack(results, dim=5)


        
        
# input = torch.zeros((1,55,64,64,5,15))
# model = BaselineMultiViewModel(55,55,2,3,(3,3,5))

# for i in range(15):
#     out = model(input[:,:,:,:,:,i])