from pprint import pprint

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from einops import rearrange
from vit_pytorch import Transformer

# from tasks.CNN.net import CNN
from utils.utils import freeze, get_stage_size, make_cnn
from utils.zxClass import Config


class Replace_T(nn.Module):
    def __init__(self, g_dim, x_ps, x_dim, head_n, head_d, dropout=0.1):
        super().__init__()
        self.xproj = nn.Conv2d(in_channels=x_dim, out_channels=g_dim, kernel_size=x_ps)
        self.norm = nn.LayerNorm(g_dim)
        self.ff = Transformer(dim=g_dim, depth=1, heads=8, dim_head=64, mlp_dim=64)

    def forward(self, x, h, g):
        x = self.xproj(x).squeeze()
        h = self.norm(h)
        h = self.ff(h[None, ...])[0] + h
        h = h + x
        return h