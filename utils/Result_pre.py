import logging
import os
import time

import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from SimpleITK import DICOMOrient as DCMO
from SimpleITK import GetArrayFromImage as GAFI
from SimpleITK import ReadImage as RI

import graph_construction.config as cfg
# from utils.Config import Config
# import wandb
from network.dataloader import KGNetData
from graph_construction.methods.patch import to_image_coordinate
from utils.utils import mkdirs
from utils.utils_img import ImagePrinter, show_slices


def normalize(image, img_max=0.995, img_min=0.005, t1=None, t2=None):
    assert type(image) is torch.Tensor or np.ndarray
    if type(image) is torch.Tensor:
        t_max = torch.quantile(image, img_max)
        t_min = torch.quantile(image, img_min)
    else:
        t_max = np.percentile(image, img_max * 100)
        t_min = np.percentile(image, img_min * 100)
    if t1 is not None:
        t_max = t1
    if t2 is not None:
        t_min = t2
    image = (image - t_min) / (t_max - t_min)
    image[image > 1] = 1
    image[image < 0] = 0
    return image, t_max, t_min


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label


def shift_matrix(matrix):
    return torch.cat((matrix[-1:], matrix[:-1]))


def replace_patches(images, patches, poses, scale=1.0, select=[]):
    images = images[:, np.newaxis, ...].repeat(3, axis=1)
    num_patches = patches.shape[0]
    poses = poses
    ps = 64
    #  pad 64 to height and width
    images = np.pad(images, ((0, 0), (0, 0), (ps, ps), (ps, ps)), mode="constant")
    patches = patches.cpu().numpy()
    for patch_idx in range(num_patches):
        if select and patch_idx not in select:
            continue
        pos = poses[patch_idx]
        patch = patches[patch_idx]
        images[
            pos[0],
            :,
            int(pos[1] * scale) + ps // 2 : int(pos[1] * scale) + ps // 2 * 3,
            int(pos[2] * scale) + ps // 2 : int(pos[2] * scale) + ps // 2 * 3,
        ] = patch
    images = images[:, :, ps:-ps, ps:-ps]
    return images


def draw_patch_edges(patch, linewidth, color):
    color = torch.tensor(color, dtype=torch.float32)
    for lw in range(linewidth):
        for c in range(3):
            patch[..., c, lw, :] = color[c]
            patch[..., c, -lw, :] = color[c]
            patch[..., c, :, lw] = color[c]
            patch[..., c, :, -lw] = color[c]
    return patch


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


class Patch:
    def __init__(self) -> None:
        self.pred_seg = None
        self.true_seg = None
        self.pred_rec = None
        self.true_rec = None
        self.input = None
        self.output = None

    def __len__(self):
        return len(self.pred)


class Result(object):
    def __init__(self, num_seg, save_path) -> None:
        super().__init__()
        self.epoch = 1
        self.best_epoch = 1
        self.best_result = 1e5
        self.max_vertices = 12
        self.num_seg = num_seg
        self.imgprinter = ImagePrinter(nrow=12, seg_cls=num_seg, alpha=0.5)
        self.save_path = save_path
        return

    @torch.no_grad()
    def eval(self, data, pred, loss):
        # cal loss for seg and mae
        pred_rec = pred["rec"]
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
        self.add_img(data, pred_rec[:, :, 0], torch.argmax(pred_seg, dim=2))
        return

    def visualize(self, data, pred):
        case = data.name[0]
        seleted_nodes = [24, 63, 267]
        views = ["sag", "cor", "axi"]
        orient = ["PIL", "LIP", "LPI"]

        path = f"data/mri2/{case}/sag/org.nii.gz"
        _data = np.load(f"data/graph4/{case}.npz", allow_pickle=True)
        _pos = np.array(_data["pos"], dtype=np.float32)

        org = {}
        arr = {}
        pos = {}
        t1 = []
        t2 = []
        for view, ori in zip(views, orient):
            org[view] = DCMO(RI(path.replace("sag", view)), ori)
            arr[view], s1, s2 = normalize(GAFI(org[view]).astype(np.float32))
            pos[view] = to_image_coordinate(org[view], _pos)
            t1.append(s1)
            t2.append(s2)
        print(f"patch mean: {data.mean} and std: {data.std}")

        patch = Patch()
        patch.true_rec = data.patch
        patch.pred_rec = pred["rec"].squeeze(2)
        patch.true_seg = data.seg
        patch.pred_seg = torch.argmax(pred["seg"], dim=2)

        num_views = data.patch.shape[1]
        num_nodes = data.patch.shape[0]
        print(f"num_views: {num_views}, num_nodes: {num_nodes}")
        print(f"pred_rec: {patch.pred_rec.shape}, pred_seg: {patch.pred_seg.shape}")

        # rearrange the (pred / true) patches to the original intensity

        for i in range(num_views):
            std, mean = data.std[i], data.mean[i]
            _img = patch.pred_rec[:, i] * std + mean
            patch.pred_rec[:, i] = normalize(_img, t1=t1[i], t2=t2[i])[0]
            _img = data.patch[:, i] * std + mean
            patch.true_rec[:, i] = normalize(_img, t1=t1[i], t2=t2[i])[0]

        idx_rec = data.idx_rec.unsqueeze(0).repeat(3, 1)
        idx_seg = data.idx_seg.unsqueeze(0).repeat(3, 1)
        idx_mix = torch.stack(data.idx_mix, dim=0)
        print(f"num_rec_nodes: {idx_rec.shape[1]}")
        print(f"num_seg_nodes: {idx_seg.shape[1]}")
        print(f"num_mix_nodes: {idx_mix.shape[1]}")
        # print(f"rec_nodes: \n{idx_rec[0]}")
        # print(f"seg_nodes: \n{idx_seg[0]}")
        # print(f"mix_nodes:\n{idx_mix[0]}\n{idx_mix[1]}\n{idx_mix[2]}")
        idx_rec_patch = torch.cat([idx_rec, idx_mix], dim=1)
        idx_seg_patch = idx_seg
        for i in range(num_views - 1):
            idx_mix = shift_matrix(idx_mix)
            idx_seg_patch = torch.cat([idx_seg_patch, idx_mix], dim=1)

        patch_shape = [num_nodes, num_views, 3, *patch.pred_seg.shape[-2:]]
        pred_seg_patch = torch.zeros(patch_shape, dtype=torch.uint8)
        true_seg_patch = torch.zeros(patch_shape, dtype=torch.uint8)
        pred_rec_patch = torch.zeros(patch_shape, dtype=torch.uint8)
        true_rec_patch = torch.zeros(patch_shape, dtype=torch.uint8)

        # transform to uint8
        for i in range(num_views):
            true_rec_patch[:, i] = self.imgprinter.patch_to_img(patch.true_rec[:, i])
            pred_rec_patch[:, i] = self.imgprinter.patch_to_img(patch.pred_rec[:, i])
            true_seg_patch[:, i] = self.imgprinter.patch_add_seg(
                patch.true_rec[:, i], patch.true_seg[:, i]
            )
            pred_seg_patch[:, i] = self.imgprinter.patch_add_seg(
                patch.true_rec[:, i], patch.pred_seg[:, i]
            )
        patch.true_rec = true_rec_patch
        patch.true_seg = true_seg_patch
        patch.pred_rec = pred_rec_patch
        patch.pred_seg = pred_seg_patch
        print(f"true_rec: {patch.true_rec.shape}, true_seg: {patch.true_seg.shape}")
        print(f"pred_rec: {patch.pred_rec.shape}, pred_seg: {patch.pred_seg.shape}")

        patch.output = torch.zeros_like(patch.true_rec, dtype=torch.float32)
        for idx_node in range(num_nodes):
            for idx_view in range(num_views):
                if idx_node in idx_rec_patch[idx_view]:
                    patch.output[idx_node, idx_view] = patch.pred_rec[
                        idx_node, idx_view
                    ]
                else:
                    patch.output[idx_node, idx_view] = patch.pred_seg[
                        idx_node, idx_view
                    ]
        patch.output = patch.output.numpy().astype(np.uint8)

        # 遍历每个 patch 图像并添加索引
        patch.input = torch.zeros_like(patch.true_rec, dtype=torch.float32)
        for idx_node in range(num_nodes):
            for idx_view in range(num_views):
                if idx_node in idx_rec_patch[idx_view]:
                    patch.input[idx_node, idx_view] = torch.zeros_like(
                        patch.input[idx_node, idx_view]
                    )
                if idx_node in idx_rec:
                    patch.input[idx_node, idx_view] = draw_patch_edges(
                        patch.input[idx_node, idx_view], 2, [255, 255, 255]
                    )
                else:
                    patch.input[idx_node, idx_view] = patch.true_rec[idx_node, idx_view]
                    patch.input[idx_node, idx_view] = draw_patch_edges(
                        patch.input[idx_node, idx_view], 2, [255, 192, 0]
                    )
        patch.input = patch.input / 255.0

        # 遍历每个 patch 图像并添加索引
        for i, patch_image in enumerate(patch.output):
            for j in range(num_views):
                if i not in seleted_nodes and seleted_nodes:
                    continue
                img = Image.fromarray(patch_image[j].transpose(1, 2, 0))  # 转换为 PIL 图像
                draw = ImageDraw.Draw(img)
                # 添加索引文本
                font = ImageFont.load_default()  # 使用默认字体
                draw.text((30, 30), str(i), fill=(255, 255, 255), font=font)  # 在图像上添加索引
                patch.output[i, j] = np.array(img).transpose(2, 0, 1)
        patch.output = torch.tensor(patch.output / 255.0)

        # print(patch.output.shape, patch.output.dtype, patch.output.max())
        # print(patch.input.shape, patch.input.dtype, patch.input.max())

        spacing = list(org["sag"].GetSpacing())
        scale = [spacing[0] / cfg.STD_SPACING]
        for i in range(1, 3):
            scale.append(1.0)

        # 打印选定的三个节点的视图，这里修改seleted_nodes输出不同结果
        for i, view in enumerate(views):
            slice_img = replace_patches(
                arr[view],
                patch.output[:, i],
                pos[view],
                scale[i],
                select=seleted_nodes,
            )
            show_slices(
                slice_img,
                norm=False,
                path=f"results/figures/{case}_slice_{view}_select.png",
            )
            slice_img = replace_patches(
                arr[view], patch.output[:, i], pos[view], scale[i]
            )
            show_slices(
                slice_img,
                norm=False,
                path=f"results/figures/{case}_slice_{view}_output.png",
            )
            slice_img = replace_patches(
                arr[view], patch.input[:, i], pos[view], scale[i]
            )
            show_slices(
                slice_img,
                norm=False,
                path=f"results/figures/{case}_slice_{view}_input.png",
            )
            slice_img = arr[view]
            show_slices(
                slice_img,
                norm=False,
                path=f"results/figures/{case}_slice_{view}_origin.png",
            )

        # 打印选定的三种节点作为可视化
        for idx_node in range(num_nodes):
            for idx_view in range(num_views):
                if idx_node in idx_rec_patch[idx_view] and idx_node in seleted_nodes:
                    self.imgprinter.save_image(
                        torch.zeros_like(patch.pred_rec[idx_node, idx_view]),
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_rec_in.png",
                    )
                    self.imgprinter.save_image(
                        patch.pred_rec[idx_node, idx_view],
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_rec_out.png",
                    )
                    self.imgprinter.save_image(
                        patch.true_rec[idx_node, idx_view],
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_rec_gt.png",
                    )
                if idx_node in idx_seg_patch[idx_view] and idx_node in seleted_nodes:
                    self.imgprinter.save_image(
                        true_rec_patch[idx_node, idx_view],
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_seg_in.png",
                    )
                    self.imgprinter.save_image(
                        patch.pred_seg[idx_node, idx_view],
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_seg_out.png",
                    )
                    self.imgprinter.save_image(
                        patch.true_seg[idx_node, idx_view],
                        f"results/figures/{case}_patch_{idx_node}_{views[idx_view]}_seg_gt.png",
                    )
        return

    def init(self):
        self.st = time.time()
        self.dice = []
        self.loss = []
        self.imgs = {"cpr": [], "cvr": [], "mvs": []}
        self.paint = 1
        return

    @torch.no_grad()
    def add_img(self, data: KGNetData, rec: torch.Tensor, seg: torch.Tensor):
        """
        add 2 vertices to show pre-train result in each batch
        input:
            data.label: [3, N, ps, ps]
            data.patch: [3, N, C , ps, ps]
            pred_mae:   [3, N, C , ps, ps]
            pred_seg:   [3, N, sn, ps, ps]
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
        self.imgs["cpr"].append(
            [data.patch[idx], rec[idx], data.seg[idx], seg[idx], -1]
        )

        noise = torch.randint(0, data.idx_mix[view_idx].shape[0], (1,)).item()
        idx = data.idx_mix[view_idx][noise]
        self.imgs["cvr"].append(
            [data.patch[idx], rec[idx], data.seg[idx], seg[idx], view_idx]
        )

        noise = torch.randint(0, data.idx_seg.shape[0], (1,)).item()
        idx = data.idx_seg[noise]
        self.imgs["mvs"].append(
            [data.patch[idx], rec[idx], data.seg[idx], seg[idx], -1]
        )
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
        log_text.append(f"\nEVAL TIME: {self.time:.1f}s")
        log_text.append((log_title + log_result).format(*titles, *items))

        loss = torch.mean(torch.stack(self.loss, dim=0), dim=0)
        dice = torch.mean(dice).item()
        log_text.append(f"SEG DIC: {dice:.3f} LOSS: {loss:.3f}")
        if loss < self.best_result:
            self.best_epoch = epoch
            self.best_result = loss
        logging.info("\n".join(log_text))

        return dice, loss

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
            true_seg = rearrange(
                torch.stack([_[2] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            pred_seg = rearrange(
                torch.stack([_[3] for _ in _item], dim=0),
                "n v h w -> (v n) h w",
            )
            idx_mask_view = torch.tensor([_[4] for _ in _item])
            # to [n*3, c, ps, ps]
            true_seg = self.imgprinter.patch_add_seg(true_rec, true_seg)
            pred_seg = self.imgprinter.patch_add_seg(true_rec, pred_seg)
            true_rec = self.imgprinter.patch_to_img(true_rec)
            pred_rec = self.imgprinter.patch_to_img(pred_rec)
            # to [n, 3, c, ps, ps]
            true_rec = rearrange(true_rec, "(v n) c h w -> n v c h w", v=3)
            pred_rec = rearrange(pred_rec, "(v n) c h w -> n v c h w", v=3)
            true_seg = rearrange(true_seg, "(v n) c h w -> n v c h w", v=3)
            pred_seg = rearrange(pred_seg, "(v n) c h w -> n v c h w", v=3)
            # add red edge to partly masked patch
            if _task == "cvr":
                idx_node = torch.arange(0, len(true_rec), 1, device=true_rec.device)
                pred_rec[idx_node, idx_mask_view, 0, 0, :] = 255
                pred_rec[idx_node, idx_mask_view, 0, :, 0] = 255
                pred_rec[idx_node, idx_mask_view, 0, -1, :] = 255
                pred_rec[idx_node, idx_mask_view, 0, :, -1] = 255
            idx_kind = torch.arange(0, 4, 1, device=true_rec.device).repeat(
                len(true_rec) * 3
            )
            idx_view = rearrange(
                torch.arange(0, 3, 1, device=true_rec.device).repeat(4),
                "(a b) -> (b a)",
                a=4,
            ).repeat(len(true_rec))
            idx_node = rearrange(
                torch.arange(0, len(true_rec), 1, device=true_rec.device)
                .reshape(-1, 1)
                .repeat(1, 12),
                "a b-> (a b)",
            )

            image = torch.stack([true_rec, pred_rec, true_seg, pred_seg], dim=0)
            image = image[idx_kind, idx_node, idx_view, ...]
            self.imgprinter.save_patches(
                image,
                path=os.path.join(self.save_path.replace(".log", f"/{_task}.png")),
            )
            # print(os.path.join(self.save_path.replace(".log", f"/{_task}.png")))
        return
