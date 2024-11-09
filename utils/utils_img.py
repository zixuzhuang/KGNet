import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch as th
import torchvision
from einops import rearrange
from torch.nn.functional import one_hot
from torchvision.transforms import InterpolationMode, Resize


def normalize(image, img_max=0.995, img_min=0.005):
    assert type(image) is th.Tensor or np.ndarray
    if type(image) is th.Tensor:
        t_max = th.quantile(image, img_max)
        t_min = th.quantile(image, img_min)
    else:
        t_max = np.percentile(image, img_max * 100)
        t_min = np.percentile(image, img_min * 100)
    image = (image - t_min) / (t_max - t_min)
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def resize(image, size: int, bl_mode=True):
    assert type(image) is th.Tensor or np.ndarray
    if type(image) is np.ndarray:
        image = torch.tensor(image, requires_grad=False)
    if bl_mode:
        image = Resize([size, size], interpolation=InterpolationMode.BILINEAR)(image)
    else:
        image = Resize([size, size], interpolation=InterpolationMode.NEAREST)(image)
    return image


def to_binary_mask(label, num_cls=3):
    # Validate input type
    if not isinstance(label, (th.Tensor, np.ndarray)):
        raise TypeError("label must be a PyTorch Tensor or a NumPy ndarray")
    # Initialize binary mask with appropriate type and shape
    if isinstance(label, th.Tensor):
        _label = th.zeros((num_cls, *label.shape), dtype=th.bool)
    else:
        _label = np.zeros((num_cls, *label.shape), dtype=np.bool8)
    # Use vectorized operations to create the binary mask
    for i in range(num_cls):
        _label[i] = (label == i + 1)
    return _label


def make_colormap(mapName="Set1", numCls=6):
    cmap = plt.get_cmap(mapName)
    cNorm = colors.Normalize(vmin=0, vmax=numCls + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    cm = [
        tuple((np.array(scalarMap.to_rgba(_)[:-1]) * 255).astype(np.int16).tolist())
        for _ in range(numCls)
    ]
    return cm


def show_slices(img, nrow=5, path="test_slice.png"):
    assert type(img) is not th.Tensor or np.ndarray
    if type(img) is th.Tensor:
        img = img.cpu().clone().detach()
    else:
        img = th.tensor(img, requires_grad=False)
    if img.max() > 1.0:
        img = normalize(img.type(torch.float32))
    img = (img * 255).type(torch.uint8)
    img = torch.repeat_interleave(img[None, ...], 3, 0).transpose(0, 1)
    img = torchvision.utils.make_grid(img, padding=1, nrow=nrow)
    img = img.transpose(0, 1).transpose(1, 2).numpy()
    plt.imsave(path, img, cmap="gray")


def show_seg(img, seg, save_path="test_seg.png", num_cls=6, alpha=0.6, nrow=5):
    assert type(img) is not th.Tensor or np.ndarray
    assert type(seg) is not th.Tensor or np.ndarray
    assert seg.max().item() < num_cls + 1
    if type(img) is th.Tensor:
        img = img.cpu().clone().detach()
    else:
        img = th.tensor(img, requires_grad=False)
    if type(seg) is th.Tensor:
        seg = seg.cpu().clone().detach()
    else:
        seg = th.tensor(seg, requires_grad=False)

    if img.max() > 1.0:
        img = normalize(img)
    img = (img * 255).type(torch.uint8)
    img = torch.repeat_interleave(img[:, None, ...], 3, 1)
    img = torchvision.utils.make_grid(img, padding=1, nrow=nrow)

    seg = to_binary_mask(seg, num_cls)
    seg = torchvision.utils.make_grid(seg.transpose(0, 1), padding=1, nrow=nrow)

    cm = make_colormap(numCls=num_cls)
    img = torchvision.utils.draw_segmentation_masks(
        image=img, masks=seg, alpha=alpha, colors=cm
    )
    img = img.transpose(0, 1).transpose(1, 2).numpy()
    plt.imsave(save_path, img)


def show_nii(
    org_path,
    seg_path=None,
    save_path="test_nii.png",
    num_cls=6,
    alpha=0.25,
    nrow=5,
    invert=False,
):
    org_data = sitk.ReadImage(org_path)
    org_data = sitk.DICOMOrient(org_data, "PIL")
    org_img = sitk.GetArrayFromImage(org_data).astype(np.float32)

    if seg_path is None:
        show_slices(org_img)
    else:
        seg_data = sitk.ReadImage(seg_path)
        seg_data = sitk.DICOMOrient(seg_data, "PIL")
        seg_img = sitk.GetArrayFromImage(seg_data).astype(np.int32)
        show_seg(
            org_img,
            seg_img,
            save_path=save_path,
            num_cls=num_cls,
            alpha=alpha,
            nrow=nrow,
        )
    return


class ImagePrinter(object):
    def __init__(
        self, norm=False, pad=1, pad_value=255, nrow=5, seg_cls=3, alpha=0.3
    ) -> None:
        self.norm = norm
        self.pad = pad
        self.pad_value = pad_value
        self.nrow = nrow
        self.seg_cls = seg_cls
        self.alpha = alpha

    def __call__(self, img, seg=None, path="test.png"):
        img = self.to_tensor(img)
        if self.norm:
            img = (img - img.min()) / (img.max() - img.min())
        img = self.to_uint8(img)
        img = self.add_channel(img)

        if seg is not None:
            seg = self.to_tensor(seg).type(torch.long)
            img = self.make_mask(img, seg)
        self.save_image(img, path)

    def to_tensor(self, img):
        assert type(img) is not th.Tensor or np.ndarray
        if type(img) is th.Tensor:
            img = img.cpu().clone().detach()
        else:
            img = th.tensor(img, requires_grad=False)
        return img

    def to_uint8(self, img: torch.Tensor):
        if img.max() > 1.00005:
            # print("check add 0.5")
            img = (img + 0.5) / 2
        # print("to uint8", img.max(), img.min(), img.mean(), img.std())
        img = (img * 255).type(torch.int16)
        img[img > 255] = 255
        img[img < 0] = 0
        return img.type(torch.uint8)

    def to_masked(self, img: torch.Tensor, seg: torch.Tensor):
        img = img[:, None, ...].repeat([1, 3, 1, 1])
        seg = one_hot(seg, self.seg_cls).type(torch.bool)
        seg = seg[..., 1:]
        seg = rearrange(seg, "s h w c -> s c h w")
        cm = make_colormap(numCls=self.seg_cls - 1)
        for i in range(img.shape[0]):
            img[i] = torchvision.utils.draw_segmentation_masks(
                image=img[i], masks=seg[i], alpha=self.alpha, colors=cm
            )
        return img

    def add_channel(self, img: torch.Tensor):
        if len(img.shape) == 2:
            img = img[None, ...].repeat([3, 1, 1])
        elif len(img.shape) == 3:
            img = img[:, None, ...].repeat([1, 3, 1, 1])
            img = torchvision.utils.make_grid(
                img, padding=self.pad, pad_value=self.pad_value, nrow=self.nrow
            ).type(torch.uint8)
        return img

    def make_mask(self, img: torch.Tensor, seg: torch.Tensor):
        seg = one_hot(seg, self.seg_cls).type(torch.bool)
        seg = seg[..., 1:]
        if len(seg.shape) == 3:
            seg = rearrange(seg, "h w c -> c h w")
        if len(seg.shape) == 4:
            seg = rearrange(seg, "s h w c -> c s h w")
            seg = torchvision.utils.make_grid(
                seg.transpose(0, 1), padding=self.pad, nrow=self.nrow
            )
        cm = make_colormap(numCls=self.seg_cls - 1)
        img = torchvision.utils.draw_segmentation_masks(
            image=img, masks=seg, alpha=self.alpha, colors=cm
        )
        return img

    def make_colormap(mapName="Set1", numCls=6):
        cmap = plt.get_cmap(mapName)
        cNorm = colors.Normalize(vmin=0, vmax=numCls + 1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        cm = [
            tuple((np.array(scalarMap.to_rgba(_)[:-1]) * 255).astype(np.int16).tolist())
            for _ in range(numCls)
        ]
        return cm

    def save_image(self, img, path):
        img = img.transpose(0, 1).transpose(1, 2).numpy()
        plt.imsave(path, img)

    def save_patches(self, img, path):
        img = torchvision.utils.make_grid(img, padding=self.pad, nrow=self.nrow)
        img = img.transpose(0, 1).transpose(1, 2).numpy()
        plt.imsave(path, img)

    def patch_add_seg(self, img, seg):
        img = self.to_tensor(img)
        img = self.to_uint8(img)
        seg = self.to_tensor(seg).type(torch.long)
        img = self.to_masked(img, seg)
        return img

    def patch_to_img(self, img):
        img = self.to_tensor(img)
        img = self.to_uint8(img)
        img = img[:, None, ...].repeat([1, 3, 1, 1])
        return img


def show_slices(img, nrow=5, norm=False, path="test_slice.png"):
    assert type(img) is not th.Tensor or np.ndarray
    if type(img) is th.Tensor:
        img = img.cpu().clone().detach()
    else:
        img = th.tensor(img, requires_grad=False)
    if img.max() > 1.0 and norm:
        img = normalize(img.type(torch.float32))
    img = (img * 255).type(torch.int16)
    img[img > 255] = 255
    img[img < 0] = 0
    if len(img.shape) < 4:
        img = img[:, None, ...].repeat([1, 3, 1, 1])
    img = torchvision.utils.make_grid(img, padding=1, nrow=nrow, pad_value=255).type(
        torch.uint8
    )
    img = img.transpose(0, 1).transpose(1, 2).numpy()
    plt.imsave(path, img, cmap="gray")
