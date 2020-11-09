import torch.nn as nn
import torchvision.models as models


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

    def deactive_batchnorm(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.Dropout2d):
                m.eval()
        return self
