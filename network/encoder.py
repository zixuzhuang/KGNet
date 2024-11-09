import logging

import timm
import torch
import torch.nn as nn
from einops import rearrange


class Encoder(nn.Module):
    def __init__(
        self,
        model_name="resnet18",
        output_channels=512,
        pretrained=False,
    ):
        super().__init__()
        # Patch Encoder Layer
        _cnn = timm.create_model(model_name, pretrained=pretrained)
        self.encoder = nn.Sequential(
            nn.Sequential(_cnn.conv1, _cnn.bn1, _cnn.act1, _cnn.maxpool),
            _cnn.layer1,
            _cnn.layer2,
            _cnn.layer3,
            _cnn.layer4,
            _cnn.global_pool,
        )
        self.input_proj = nn.Linear(_cnn.num_features, output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.input_proj(x)
        return x


class Encoders(nn.Module):
    def __init__(self, num_views=3, model_name="resnet18", output_channels=512, pretrained=False):
        super().__init__()
        self.num_views = num_views
        self.cnns = nn.ModuleList([Encoder(model_name, output_channels, pretrained) for _ in range(num_views)])

    def forward(self, patches, keepidx):
        n = patches.shape[0]
        h = torch.zeros(
            [n, self.num_views, 512],
            device=patches.device,
            dtype=patches.dtype,
        )
        for i in range(self.num_views):
            idx = keepidx[i] if keepidx else torch.arange(n)
            x = patches[idx, i : i + 1, :, :].repeat(1, 3, 1, 1)
            h[idx, i] = h[idx, i] + self.cnns[i](x)
        return h
