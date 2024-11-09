import logging
import os
import time

import torch
from einops import rearrange

import wandb
from CSNet.dataloader_ import CSNetData
from utils.utils import mkdirs
from utils.utils_img import ImagePrinter, show_seg, show_slices


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label


def cal_dice(preds, label, num_cls=2):
    """
    preds: [N, K, H, W], torch.float32, K is the num of cls
    label: [N, H, W], torch.long, label map
    dice: [3, K], torch.float32, dice of each cls
    """
    # label = rearrange(label, "k v p q -> (k v) p q")
    # preds = rearrange(preds, "k v c p q -> (k v) c p q")
    preds = torch.argmax(preds, dim=1)  # [N, H, W]

    true_one_hot = get_one_hot(label, num_cls)
    pred_one_hot = get_one_hot(preds, num_cls)

    inter_sum = torch.sum(true_one_hot * pred_one_hot, dim=[0])
    true_sum = torch.sum(true_one_hot, dim=[0])
    pred_sum = torch.sum(pred_one_hot, dim=[0])
    dice_matrix = torch.stack([true_sum, pred_sum, inter_sum], dim=0)
    return dice_matrix


class Result(object):
    def __init__(self, num_seg, save_path) -> None:
        super().__init__()
        self.epoch = 1
        self.best_epoch = 1
        self.best_result = 1e5
        self.max_vertices = 3
        self.num_seg = num_seg
        self.imgprinter = ImagePrinter(nrow=6, seg_cls=num_seg, alpha=0.5)
        self.save_path = save_path
        return

    @torch.no_grad()
    def eval(self, data, pred, loss):
        # cal loss for seg and mae
        pred_seg = pred["seg"]

        idx_view = torch.cat(
            [torch.zeros_like(data.keep[i], device=data.device) + i for i in range(3)],
            dim=0,
        )
        idx_node = torch.cat(data.keep, dim=0)

        self.loss.append(loss)
        self.dice.append(
            cal_dice(
                pred_seg[idx_node, idx_view],
                data.seg[idx_node, idx_view],
                num_cls=self.num_seg,
            )
        )
        self.add_img(data, torch.argmax(pred_seg, dim=2))
        return

    def init(self):
        self.st = time.time()
        self.dice = []
        self.loss = []
        self.imgs = {"cvr": [], "mvs": []}
        self.paint = 1
        return

    @torch.no_grad()
    def add_img(self, data: CSNetData, seg: torch.Tensor):
        """
        add 2 vertices to show pre-train result in each batch
        input:
            data.label: [3, N, ps, ps]
            data.patch: [3, N, C , ps, ps]
            pred_seg:   [3, N, sn, ps, ps]
        output:
            list of [_true_mae, _pred_mae, _true_seg, _pred_seg]
        """
        if len(self.imgs["mvs"]) // 3 >= self.max_vertices and self.paint:
            self.paint = 0
            self.show()
            return

        if len(self.imgs["mvs"]) // 3 >= self.max_vertices:
            return

        view_idx = len(self.imgs["mvs"]) // self.max_vertices

        noise = torch.randint(0, data.idx_mix[view_idx].shape[0], (1,)).item()
        idx = data.idx_mix[view_idx][noise]
        self.imgs["cvr"].append([data.patch[idx], data.seg[idx], seg[idx], view_idx])

        noise = torch.randint(0, data.idx_seg.shape[0], (1,)).item()
        idx = data.idx_seg[noise]
        self.imgs["mvs"].append([data.patch[idx], data.seg[idx], seg[idx], -1])
        return

    @torch.no_grad()
    def stastic(self, epoch: int, datatype="test"):
        self.epoch = epoch
        log_text = []

        dice = torch.stack(self.dice, dim=0)
        dice = torch.sum(dice, dim=0)
        smooth = 1e-3
        dice = (2.0 * dice[2] + smooth) / (dice[1] + dice[0] + smooth)
        dice = dice.reshape(-1)

        dice_title = ["BG", "FB", "FC", "M", "TB", "TC", "PB", "PC"]
        titles = ["dataset"] + dice_title
        items = [datatype.upper()] + [dice[_].item() for _ in range(len(dice))]
        log_title = "|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        log_result = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"

        self.time = time.time() - self.st

        loss = torch.mean(torch.stack(self.loss, dim=0), dim=0)
        dice = torch.mean(dice).item()
        log_text.append(f"SEG DIC: {dice:.3f} LOSS: {loss:.3f}")
        log_text.append(f"\nEVAL TIME: {self.time:.1f}s")
        log_text.append((log_title + log_result).format(*titles, *items))
        if loss < self.best_result:
            self.best_epoch = epoch
            self.best_result = loss
        logging.info("\n".join(log_text))

        self.dice = dice
        self.loss = loss
        return 

    def show(self):
        """ """
        mkdirs(self.save_path.replace(".log", ""))
        for _task, _item in self.imgs.items():
            # to [n*3, ps, ps]
            true_rec = rearrange(
                torch.stack([_[0] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            true_seg = rearrange(
                torch.stack([_[1] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            pred_seg = rearrange(
                torch.stack([_[2] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            idx_mask_view = torch.tensor([_[3] for _ in _item])
            # to [n*3, c, ps, ps]
            true_seg = self.imgprinter.patch_add_seg(true_rec, true_seg)
            pred_seg = self.imgprinter.patch_add_seg(true_rec, pred_seg)
            # to [n, 3, c, ps, ps]
            true_seg = rearrange(true_seg, "(v n) c h w -> n v c h w", v=3)
            pred_seg = rearrange(pred_seg, "(v n) c h w -> n v c h w", v=3)
            # add red edge to partly masked patch
            if _task == "mvs":
                idx_node = torch.arange(0, len(true_seg), 1, device=true_seg.device)
                pred_seg[idx_node, idx_mask_view, 0, 0, :] = 255
                pred_seg[idx_node, idx_mask_view, 0, :, 0] = 255
                pred_seg[idx_node, idx_mask_view, 0, -1, :] = 255
                pred_seg[idx_node, idx_mask_view, 0, :, -1] = 255
            idx_kind = torch.arange(0, 2, 1, device=true_seg.device).repeat(
                len(true_seg) * 3
            )
            idx_view = rearrange(
                torch.arange(0, 3, 1, device=true_seg.device).repeat(2),
                "(a b) -> (b a)",
                a=2,
            ).repeat(len(true_seg))
            idx_node = rearrange(
                torch.arange(0, len(true_seg), 1, device=true_seg.device)
                .reshape(-1, 1)
                .repeat(1, 6),
                "a b-> (a b)",
            )

            image = torch.stack([true_seg, pred_seg], dim=0)
            image = image[idx_kind, idx_node, idx_view, ...]
            self.imgprinter.save_patches(
                image,
                path=os.path.join(self.save_path.replace(".log", f"/{_task}.png")),
            )
            # print(os.path.join(self.save_path.replace(".log", f"/{_task}.png")))
        return
