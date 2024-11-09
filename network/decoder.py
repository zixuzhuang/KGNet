import torch
import torch.nn as nnF
import torch.nn as nn
import torch.nn.functional as F

from network.decoder_modules import ConvBlock


class Decoder(nn.Module):
    def __init__(self, out_size, num_classes, channels=(8, 16, 32, 64, 512), strides=(2, 2, 2, 2, 4), **kwargs):
        super().__init__()
        nb_filter = channels

        res_unit = ConvBlock
        self.conv3_1 = res_unit(nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[1], nb_filter[0], **kwargs)
        self.convds0 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.size = []
        size = out_size
        for stride in strides:
            self.size.append(size // stride)
            size = size // stride
        self.size.append(out_size)

    def upsample(self, inputs, target_size):
        return F.interpolate(inputs, size=target_size, mode="bilinear", align_corners=False)

    def forward(self, x4):
        x3 = self.conv3_1(self.upsample(x4, self.size[3]))
        x2 = self.conv2_2(self.upsample(x3, self.size[2]))
        x1 = self.conv1_3(self.upsample(x2, self.size[1]))
        x0 = self.conv0_4(self.upsample(x1, self.size[0]))

        out = F.interpolate(self.convds0(x0), size=self.size[-1], mode="bilinear", align_corners=False)
        return out
