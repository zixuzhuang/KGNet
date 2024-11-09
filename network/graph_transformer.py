import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm_out = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm_out(x)
        x = self.net(x)
        return x


class GraphAttention(nn.Module):
    def __init__(self, g_dim, head_n, dropout):
        super().__init__()
        # Tranformer
        inner_dim = g_dim // head_n
        self.head_num = head_n
        self.scale = g_dim**-0.5
        self.to_qkv = nn.Linear(g_dim, g_dim * 3, bias=False)
        self.to_out = nn.Dropout(dropout)
        # PreNorm
        self.norm_in = nn.LayerNorm(g_dim)

    def forward(self, s: torch.Tensor, g: dgl.DGLGraph):
        s = s[None, ...]
        s = self.norm_in(s)
        with g.local_scope():
            qkv = self.to_qkv(s).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b n) h d", h=self.head_num), qkv)
            # b=1, n=num_patches, h=head_num, d=head_dim
            # print(q.shape, k.shape, v.shape)
            g.srcdata.update({"q": q, "v": v})
            g.dstdata.update({"k": k})

            # compute edge attention.
            g.apply_edges(fn.u_mul_v("q", "k", "att"))
            g.edata["att"] = g.edata["att"] * self.scale
            g.edata["att"] = edge_softmax(g, g.edata["att"])

            # message passing
            g.update_all(fn.u_mul_e("v", "att", "m"), fn.sum("m", "rst"))
            rst = g.dstdata["rst"]

            # recovery shape
            rst = rearrange(rst, "n h d -> n (h d)")[None, ...]
            rst = self.to_out(rst)[0]
        return rst


class GraphTransformer(nn.Module):
    def __init__(self, g_dim, head_n, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(g_dim)
        self.attention = GraphAttention(g_dim, head_n, dropout=dropout)
        self.ff = FeedForward(g_dim, hidden_dim=g_dim * 4, dropout=dropout)

    def forward(self, g, h):
        for i in range(self.depth):
            h = self.norm(h)
            h = self.ff(self.attention(h, g)) + h
        # merge
        return h
