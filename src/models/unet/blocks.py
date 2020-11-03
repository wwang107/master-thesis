from typing import Tuple
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation_size: int = 1) -> None:
        super(ResidualBlock, self).__init__()
        assert kernel_size % 2 != 0, "kernel size can only be odd number"
        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=(kernel_size, kernel_size, 1),
                               padding=(padding, padding, 0))
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=(kernel_size, kernel_size, kernel_size),
                      padding=(padding, padding, padding)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=(kernel_size, kernel_size, kernel_size),
                      padding=(padding, padding, padding)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        return x


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, diliation_size: int = 1, stride_size: int = 1) -> None:
        '''
        '''
        super().__init__()
        assert kernel_size % 2 != 0
        self.t = kernel_size
        self.d = diliation_size

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//2, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels//2, out_channels,
                      kernel_size=(1, 1, self.t), dilation=(1, 1, self.d), stride=(1, 1, stride_size)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=(1, 1, 1), dilation=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        residual = self.__get_sliced_input(x, self.t, self.d)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        return x

    def __get_sliced_input(self, x, kernel_size: int, dilation: int):
        rf = self.__get_receptive_field(kernel_size, dilation)
        pad = rf//2
        length = x.size(4)
        start = pad
        end = length - pad
        return x[:, :, :, :, start:end]

    def __get_receptive_field(self, kernel_size: int, dilation: int) -> Tuple[int, int]:
        return (kernel_size-1) * dilation + 1


class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor, mode='nearest') -> None:
        super(UpSampleBlock, self).__init__()
        self.block = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, basic_block, in_channels: int, out_channels: int, kernel_size: int, dilation_size: int = 1, apply_pooling: bool = True) -> None:
        super(Encoder, self).__init__()
        if apply_pooling:
            self.conv_block = nn.Sequential(
                nn.MaxPool3d(kernel_size=2),
                basic_block(in_channels, out_channels, kernel_size, dilation_size))
        else:
            self.conv_block = basic_block

    def forward(self, x):
        return self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size) -> None:
        super(Decoder, self).__init__()
        self.up_sample_block = UpSampleBlock(scale_factor=2, mode='nearest')
        self.conv_block = ResidualBlock(in_channel, out_channels, kernel_size)

    def forward(self, x, encoder_feature=None):
        out = self.up_sample_block(x)
        out = self.conv_block(out)
        if encoder_feature != None:
            out = out + encoder_feature
        return out
