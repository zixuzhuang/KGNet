import logging
import os
import time

import torch
from einops import rearrange

# import wandb
from CSNet.dataloader_ import CSNetData
from utils.utils import mkdirs
from utils.utils_img import ImagePrinter, show_seg, show_slices


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

    def init(self):
        self.st = time.time()
        self.loss = []
        self.imgs = {"cpr": [], "cvr": []}
        self.paint = 1
        return

    @torch.no_grad()
    def eval(self, data, pred, loss):
        # cal loss for seg and mae
        pred_rec = pred["rec"]
        self.loss.append(loss)
        self.add_img(data, pred_rec[:, :, 0])
        return

    @torch.no_grad()
    def add_img(self, data: CSNetData, rec: torch.Tensor):
        """
        add 2 vertices to show pre-train result in each batch
        input:
            data.label: [3, N, ps, ps]
            data.patch: [3, N, C , ps, ps]
            pred_mae:   [3, N, C , ps, ps]
        output:
            list of [_true_mae, _pred_mae, _true_seg, _pred_seg]
        """
        if len(self.imgs["cpr"]) // 3 >= self.max_vertices and self.paint:
            self.paint = 0
            self.show()
            return

        if len(self.imgs["cpr"]) // 3 >= self.max_vertices:
            return

        view_idx = len(self.imgs["cpr"]) // self.max_vertices

        noise = torch.randint(0, data.idx_rec.shape[0], (1,)).item()
        idx = data.idx_rec[noise]
        self.imgs["cpr"].append([data.patch[idx], rec[idx], -1])

        noise = torch.randint(0, data.idx_mix[view_idx].shape[0], (1,)).item()
        idx = data.idx_mix[view_idx][noise]
        self.imgs["cvr"].append([data.patch[idx], rec[idx], view_idx])

        return

    @torch.no_grad()
    def stastic(self, epoch: int):
        self.epoch = epoch
        self.time = time.time() - self.st
        loss = torch.mean(torch.stack(self.loss, dim=0), dim=0)

        log_text = []
        log_text.append(f"\nEVAL TIME: {self.time:.1f}s")
        log_text.append(f"LOSS: {loss:.3f}")
        wandb.log({"test_loss": loss}, step=epoch)
        if loss < self.best_result:
            self.best_epoch = epoch
            self.best_result = loss
        logging.info("\n".join(log_text))

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
            pred_rec = rearrange(
                torch.stack([_[1] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            idx_mask_view = torch.tensor([_[2] for _ in _item])
            # to [n*3, c, ps, ps]
            true_rec = self.imgprinter.patch_to_img(true_rec)
            pred_rec = self.imgprinter.patch_to_img(pred_rec)
            # to [n, 3, c, ps, ps]
            true_rec = rearrange(true_rec, "(v n) c h w -> n v c h w", v=3)
            pred_rec = rearrange(pred_rec, "(v n) c h w -> n v c h w", v=3)
            # add red edge to partly masked patch
            if _task == "cvr":
                idx_node = torch.arange(0, len(true_rec), 1, device=true_rec.device)
                pred_rec[idx_node, idx_mask_view, 0, 0, :] = 255
                pred_rec[idx_node, idx_mask_view, 0, :, 0] = 255
                pred_rec[idx_node, idx_mask_view, 0, -1, :] = 255
                pred_rec[idx_node, idx_mask_view, 0, :, -1] = 255
            idx_kind = torch.arange(0, 2, 1, device=true_rec.device).repeat(
                len(true_rec) * 3
            )
            idx_view = rearrange(
                torch.arange(0, 3, 1, device=true_rec.device).repeat(2),
                "(a b) -> (b a)",
                a=2,
            ).repeat(len(true_rec))
            idx_node = rearrange(
                torch.arange(0, len(true_rec), 1, device=true_rec.device)
                .reshape(-1, 1)
                .repeat(1, 2 * 3),
                "a b-> (a b)",
            )

            image = torch.stack([true_rec, pred_rec], dim=0)
            image = image[idx_kind, idx_node, idx_view, ...]
            self.imgprinter.save_patches(
                image,
                path=os.path.join(self.save_path.replace(".log", f"/{_task}.png")),
            )
            # print(os.path.join(self.save_path.replace(".log", f"/{_task}.png")))
        return
