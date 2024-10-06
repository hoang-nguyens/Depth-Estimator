# import library
# Encoder block ( use mobilenet v3)
# Up sampling layer: using bilinear interpolation
# Decoder block
# combine to Encoder - Decoder architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.mobilenet_v3_large(pretrained = True)

    def forward(self, x):
        features = [x]

        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]))

        return features
''' mobilenet_v3_large structure( with batch_size = 1 )
Feature 0 shape: (1, 3, 224, 224)
Feature 1 shape: (1, 16, 112, 112)
Feature 2 shape: (1, 16, 112, 112)
Feature 3 shape: (1, 24, 56, 56)
Feature 4 shape: (1, 24, 56, 56)
Feature 5 shape: (1, 40, 28, 28)
Feature 6 shape: (1, 40, 28, 28)
Feature 7 shape: (1, 40, 28, 28)
Feature 8 shape: (1, 80, 14, 14)
Feature 9 shape: (1, 80, 14, 14)
Feature 10 shape: (1, 80, 14, 14)
Feature 11 shape: (1, 80, 14, 14)
Feature 12 shape: (1, 112, 14, 14)
Feature 13 shape: (1, 112, 14, 14)
Feature 14 shape: (1, 160, 7, 7)
Feature 15 shape: (1, 160, 7, 7)
Feature 16 shape: (1, 160, 7, 7)
Feature 17 shape: (1, 960, 7, 7)

'''
class UpSampling(nn.Sequential):
    def __init__(self, skip_input, output):
        super(UpSampling, self).__init__()
        self.skip_input = skip_input
        self.output = output

        self.conv2d_A = nn.Conv2d(self.skip_input, self.output, kernel_size= 3, padding=1, stride = 1)
        self.leakyReLU_A = nn.LeakyReLU(0.2)
        self.conv2d_B = nn.Conv2d(self.output, self.output, kernel_size= 3, padding=1, stride = 1)
        self.leakyReLU_B = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        x_interpolation = F.interpolate(x, size = [concat_with.size(2), concat_with.size(3)], mode = 'bilinear', align_corners=True)
        out = torch.cat([x_interpolation, concat_with], dim = 1)
        out = self.conv2d_A(out)
        out = self.leakyReLU_A(out)
        out = self.conv2d_B(out)
        out = self.leakyReLU_B(out)
        return out

class Decoder(nn.Module):
    def __init__(self, num_features = 960, scale_factor = 0.6 ):
        super().__init__()
        features = int( num_features * scale_factor)
        self.conv2d_1 = nn.Conv2d(num_features, features, kernel_size=3, padding=1, stride=1 )

        self.up_1 = UpSampling(features // 1 + 160, features // 2)
        self.up_2 = UpSampling(features // 2 + 112, features // 2)
        self.up_3 = UpSampling(features // 2 + 80, features // 4)
        self.up_4 = UpSampling(features // 4 + 40, features // 8)
        self.up_5 = UpSampling(features // 8 + 24, features // 8)
        self.up_6 = UpSampling(features // 8 + 16, features // 16)

        self.conv2d_2 = nn.Conv2d(features // 16, 1, padding = 1, kernel_size=3, stride=1)

    def forward(self, features):
        x_0, x_1, x_2, x_3, x_4, x_5, x_6 = features[17], features[15], features[13], features[10], features[6], features[4], features[2]
        out = self.conv2d_1(x_0)
        out = self.up_1(out, x_1)
        out = self.up_2(out, x_2)
        out = self.up_3(out, x_3)
        out = self.up_4(out, x_4)
        out = self.up_5(out, x_5)
        out = self.up_6(out, x_6)
        out = self.conv2d_2(out)
        return out

class MobileDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out




