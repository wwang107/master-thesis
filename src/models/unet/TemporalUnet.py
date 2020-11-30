import torch
import pytorch_lightning as pl
from torch import nn
from models.unet.blocks import ResidualBlock, TemporalResidualBlock, Encoder, Decoder

class TemporalUnet(pl.LightningModule):
    '''
    An encoder structure that have a hardcoded depth and fixed recpetive field.
    Currently, only support depth of 4 and receptive field of 15
    '''
    def __init__(self, in_channels: int, out_channels: int,num_feature:int, input_frame:int, epipolar_transfomer:nn.Module=None) -> None:
        super().__init__()
        depth = 4
        encoder_kernel_size = (3,3,3)
        input_frame_at_depth = input_frame
        # encoder_kernel_size = (3,3,1)
        decoder_kernel_size = (3,3,1)
        f_maps = [num_feature * pow(2,i) for i in range(0, depth)]
        encoders = []
        decoders = []
        
        for i in range(depth):
            dilation = pow(2,i)
            eff_k = self.effective_kernel_size(encoder_kernel_size[2], dilation)
            if input_frame_at_depth <= 1:
                dilation = 1
                encoder_kernel_size = (3,3,1)
            if i == 0:
                # encoder = Encoder(TemporalResidualBlock, in_channels, f_maps[i], encoder_kernel_size, pow(2,i), apply_pooling = False)
                encoder = Encoder(TemporalResidualBlock, in_channels, f_maps[i], encoder_kernel_size, dilation, apply_pooling = False)
            else:
                # encoder = Encoder(TemporalResidualBlock, f_maps[i-1], f_maps[i], encoder_kernel_size, pow(2,i), apply_pooling = True, pool_kernel = (2,2,1))
                encoder = Encoder(TemporalResidualBlock, f_maps[i-1], f_maps[i], encoder_kernel_size, dilation, apply_pooling = True, pool_kernel = (2,2,1))
            input_frame_at_depth = self.output_size(input_frame_at_depth, eff_k)
            encoders.append(encoder)
        
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            decoder = Decoder(TemporalResidualBlock, in_feature_num, reversed_f_maps[i+1], decoder_kernel_size, 1, scale_factor = (2,2,1))
            decoders.append(decoder)

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.epipolar = epipolar_transfomer
        self.last_conv = ResidualBlock(f_maps[0], out_channels, (3,3,1), 1, use_batch_norm=False)
    
    def forward(self, x, proj_mat=None, imgs = None, keypoints = None):
        '''
        Forward pass each camera view indidvidually, the model will share weight across different view
        params\n
        x: with 6 dimensions [batch, num_joints, height, width, num_views, num_frames]
        proj_mat: 3-by-4 projection matrix that composed by KRT matrices [batch, 3, 4, num_views]
        '''
        results = []
        feats = []
        num_view = x.size(4)
        input_view = [x[:,:,:,:,k,:] for k in range(num_view)]
        for k in range(num_view):
            encoders_features = []
            out = input_view[k]
            for encoder in self.encoders:
                out = encoder(out)
                encoders_features.insert(0, out)
            
            encoders_features = encoders_features[1:]
            for decoder, encoder_features in zip(self.decoders, encoders_features):
                encoder_features = torch.mean(encoder_features, dim = 4, keepdim=True)
                out = decoder(out, encoder_features)

            feats.append(out.squeeze(dim=4))
        
        if self.epipolar != None and proj_mat != None:
            unfused = feats[0].clone()
            ref_feat = feats[0]
            ref_p = proj_mat[:,:,:,0]
            for j in range(1, len(feats)):
                src_feat = feats[j]
                src_p = proj_mat[:,:,:,j]
                fuse = self.epipolar(ref_feat, src_feat, ref_p, src_p, 
                                     imgs[...,0] if imgs != None else None,
                                     imgs[...,j] if imgs != None else None,
                                     keypoints[..., 0] if keypoints != None else None,
                                     keypoints[..., 1] if keypoints != None else None)
                ref_feat += fuse
            
            ref_feat = self.last_conv(ref_feat.view(*ref_feat.size(),1))
            unfused = self.last_conv(unfused.view(*unfused.size(),1))
            return ref_feat.view(*ref_feat.size(),1), unfused.view(*unfused.size(),1)

        else:
            for j in range(0, len(feats)):
                ref_feat = feats[j]
                ref_feat = ref_feat.view(*ref_feat.size(),1)
                results.append(self.last_conv(ref_feat))
            return torch.stack(results, dim=4)
        
    def effective_kernel_size(self, kernel_size, dilation):
        return kernel_size + (kernel_size-1)*(dilation -1)
    
    def output_size(self, input_size, kernel_size):
        return input_size - kernel_size + 1
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

# input = torch.zeros((1,55,64,64,4,15))
# model = TemporalUnet(55,55,128)
# out = model(input[:,:,:,:,:,:])