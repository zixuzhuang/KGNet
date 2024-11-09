import logging

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from network.dataloader import KGNetData
from network.decoder import Decoder
from network.encoder import Encoders
from network.graph_transformer import GraphTransformer
from network.pos_embed import pos_encoding
from network.transformer_layer import Transformer
from utils.Config import Config

MAX_NODE_NUM = 600


class KneeGraphNetwork(nn.Module):
    def __init__(self, num_cls=3, pretrain_from_imagenet=False):
        super().__init__()

        # Patch Encoders
        patch_dim = 512
        self.patch_encoder = Encoders(pretrained=pretrain_from_imagenet, output_channels=patch_dim)
        # Position embedding
        self.patch_dim = patch_dim
        # Patch Fusion
        self.patch_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.patch_fusion = Transformer(dim=patch_dim, depth=3, head_num=4, head_dim=512, mlp_dim=2048, dropout=0.3)
        # Graph Transformer Layer
        g_dim = 512
        self.gtn = GraphTransformer(g_dim=g_dim, head_n=4)
        self.xproj = nn.Linear(patch_dim, g_dim)

        # Graph Pooling and Classifier
        self.gclassifier = nn.Linear(g_dim, num_cls)

        # Decoder for Pre-Training
        self.decoder_seg = Decoder(out_size=64, num_classes=8 * 3)
        self.decoder_rec = Decoder(out_size=64, num_classes=3)

        # Ablation 1: Transformer
        self.ttn = Transformer(
            dim=g_dim,
            head_num=4,
            depth=3,
            head_dim=g_dim,
            mlp_dim=g_dim * 4,
            dropout=0.1,
        )

    def encode(self, data: KGNetData, ratio1=0.3, ratio2=0.3, mask=False):
        patches = data.patch.clone()
        keepidx = None
        if mask:
            self.random_masking_vertex(data, ratio1, ratio2)
            keepidx = data.keep
        patch_embs = self.patch_encoder(patches, keepidx)
        pos_embs = pos_encoding(data.pos, d_model=self.patch_dim)
        patch_embs = patch_embs + pos_embs[:, None, :]
        patch_token = self.patch_token.repeat(patch_embs.shape[0], 1, 1)
        patch_embs = torch.cat([patch_token, patch_embs], dim=1)
        patch_embs = self.patch_fusion(patch_embs)[:, 0]
        return patch_embs

    def decode(self, x):
        out = {}
        out["seg"] = rearrange(self.decoder_seg(x[..., None, None]), "n (v m) h w -> n v m h w", v=3)
        out["rec"] = rearrange(self.decoder_rec(x[..., None, None]), "n (v m) h w -> n v m h w", v=3)
        return out

    def graphconv(self, x, g: dgl.DGLGraph):
        x = self.xproj(x)
        x = self.gtn(g, x)
        return x

    def pretrain(self, data: KGNetData):
        x = self.encode(data, mask=True)
        x = self.graphconv(x, data.graph)
        out = self.decode(x)
        return out

    def finetune(self, data: KGNetData):
        x = self.encode(data)
        x = self.graphconv(x, data.graph)
        out = self.clshead(x, data.graph)
        return out

    def multitask(self, data: KGNetData):
        x = self.encode(data, mask=True)
        x = self.graphconv(x, data)
        out = self.decode(x)
        out.update(self.clshead(x, data.graph))
        return out

    def clshead(self, x, g):
        out = {}
        with g.local_scope():
            g.ndata["x"] = x
            mx = dgl.mean_nodes(g, "x")
        out["cls"] = self.gclassifier(mx)
        return out

    def load_pretrain(self, pre_path: str):
        if pre_path:
            logging.info(f"Loading pretrain weight from {pre_path}.")
            load_dict = torch.load(pre_path).state_dict()
            model_dict = self.state_dict()
            load_dict = {k: v for k, v in load_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(load_dict)
            self.load_state_dict(model_dict)
            logging.info("Loading completed.")
        else:
            logging.info("Using default weight.")

    @torch.no_grad()
    def random_masking_vertex(self, data: KGNetData, ratio1: float, ratio2: float):
        """
        Perform per-sample random masking by per-vertex shuffling.
        Per-vertex shuffling is done by argsort random noise.
        x   : [V, N, 3, ps, ps], 3-channel patch
        mask_ratio1: float, the num of masked WHOLE vertices
        mask_ratio2: float, the num of masked ONLY A VIEW
        """
        N = data.N
        n_rec = int(N * ratio1)
        n_mix = int(N * ratio2 / 3)
        n_seg = N - n_rec - n_mix * 3
        noise = torch.rand(N, device=data.device)  # noise in [0, 1]

        # sort noise for each sample
        idx_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove

        # keep the first subset
        idx_seg = idx_shuffle[:n_seg]  # index of fully kept vertices
        idx_rec = idx_shuffle[n_seg : n_seg + n_rec]  # index of fully masked vertices
        idx_mix = [
            idx_shuffle[n_seg + n_rec + i * n_mix : n_seg + n_rec + (i + 1) * n_mix] for i in range(3)
        ]  # index of partly masked vertices
        idx_mask = [torch.cat([idx_rec, idx_mix[i]], dim=0) for i in range(3)]
        idx_keep = [torch.cat([idx_seg, idx_mix[(i + 1) % 3], idx_mix[(i + 2) % 3]], dim=0) for i in range(3)]

        data.mask = idx_mask
        data.keep = idx_keep
        data.idx_rec = idx_rec  # index of fully masked vertices
        data.idx_mix = idx_mix  # index of masked [v1, v2, v3]
        data.idx_seg = idx_seg  # index of fully kept vertices
        return


class KneeLoss(object):
    def __init__(self, device, dataset_name) -> None:
        self.seg_func = nn.CrossEntropyLoss().to(device)
        self.rec_func = nn.MSELoss().to(device)
        self.les_func = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1.25, 1.0])).to(device)
        if dataset_name == "inhouse":
            self.cls_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0])).to(device)
        elif dataset_name == "mrnet":
            self.cls_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0])).to(device)

    def rec_(self, data: KGNetData, pred: torch.Tensor):
        true = data.patch
        idx_view = torch.cat(
            [torch.zeros_like(data.mask[i], device=data.device) + i for i in range(3)],
            dim=0,
        )
        idx_node = torch.cat(data.mask, dim=0)
        _pred = pred[idx_node, idx_view, 0]
        _true = true[idx_node, idx_view]
        loss = self.rec_func(_pred, _true)
        return loss

    def seg_(self, data: KGNetData, pred: torch.Tensor):
        true = data.seg
        idx_view = torch.cat(
            [torch.zeros_like(data.keep[i], device=data.device) + i for i in range(3)],
            dim=0,
        )
        idx_node = torch.cat(data.keep, dim=0)
        _pred = pred[idx_node, idx_view]
        _true = true[idx_node, idx_view]
        loss = self.seg_func(_pred, _true)
        return loss

    def rec(self, data: KGNetData, pred):
        return self.rec_(data, pred["rec"])

    def seg(self, data: KGNetData, pred):
        return self.seg_(data, pred["seg"])

    def cls(self, data: KGNetData, pred):
        true = data.grade
        return self.cls_func(pred["cls"], true)

    def les(self, data: KGNetData, pred):
        true = data.les
        return self.les_func(pred["cls"], true)

    def pre(self, data: KGNetData, pred):
        loss_rec = self.rec_(data, pred["rec"])
        loss_seg = self.seg_(data, pred["seg"])
        return loss_rec + loss_seg

    def multi(self, data: KGNetData, pred):
        loss_rec = self.rec_(data, pred["rec"])
        loss_seg = self.seg_(data, pred["seg"])
        loss_cls = self.cls_func(pred["cls"], data.grade)
        return loss_rec + loss_seg + loss_cls
