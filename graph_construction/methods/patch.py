import dgl
import numpy as np
import SimpleITK as sitk
import torch
from SimpleITK import GetArrayFromImage as GAFI
from torchvision.transforms import CenterCrop, Pad

import graph_construction.config as cfg
from graph_construction.MRIData import MRIData
from utils.utils_img import show_nii, show_seg


def paintingPatch(arr, sxy):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    shape = arr.shape[1:]
    arr = Pad([p, p, p, p], fill=0.0)(arr)
    mask = torch.zeros(arr.shape, dtype=bool)
    for _sxy in sxy:
        s, x, y = _sxy
        s, x, y = int(_sxy[0]), int(_sxy[1]), int(_sxy[2])
        mask[s, x + p1 : x + p2, y + p1 : y + p2] = 1
    mask = CenterCrop(shape)(mask)
    return mask


def extract_patch(arr, sxy):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    arr = Pad([p, p, p, p], fill=0)(arr)
    patch = []
    for _sxy in sxy:
        s, x, y = _sxy[0], _sxy[1], _sxy[2]
        s, x, y = int(s), int(x), int(y)
        patch.append(arr[s, x + p1 : x + p2, y + p1 : y + p2])
    return torch.stack(patch)


def to_world_space(sitkimg, points):
    origin = np.array(sitkimg.GetOrigin()).reshape([1, 1, 3])
    spacing = np.array(sitkimg.GetSpacing()).reshape([1, 3])
    direction = np.array(sitkimg.GetDirection()).reshape([3, 3])
    points = np.flip(points, axis=1) * spacing  # to y, x, s
    points = np.matmul(points, direction.transpose())
    points = points + origin
    points = points[0]
    # print(points.shape)
    return points


def to_image_coordinate(sitkimg, points):
    origin = np.array(sitkimg.GetOrigin()).reshape([1, 1, 3])
    spacing = np.array(sitkimg.GetSpacing()).reshape([1, 3])
    direction = np.array(sitkimg.GetDirection()).reshape([3, 3])
    size = np.flip(sitkimg.GetSize(), axis=0)

    points = points - origin
    points = np.matmul(points, np.linalg.inv(direction.transpose()))
    points = points[0]
    points = points / spacing
    points = np.flip(points, axis=1)
    points = np.round(points, 0).astype(np.int32)
    for i in range(3):
        arr = points[:, i]
        arr[arr >= size[i]] = size[i] - 1
        arr[arr < 0] = 0
        points[:, i] = arr
    return points


def extractPatch(subject: MRIData):

    subject.pos = to_world_space(subject.data[subject.major_view], subject.v_3d)
    for view in subject.views:
        data = subject.data[view]
        image_coordinate = to_image_coordinate(data, subject.pos)
        subject.patch[view] = extract_patch(subject.org[view], image_coordinate).type(torch.float32)
        subject.patch_seg[view] = extract_patch(subject.seg[view], image_coordinate).type(torch.uint8)

    if subject.have_lesion:
        image_coordinate = to_image_coordinate(subject.data[subject.major_view], subject.pos)
        subject.patch_les = extract_patch(subject.les[subject.major_view], image_coordinate).type(torch.uint8)
        subject.patch_les = torch.max(subject.patch_les, dim=1)[0]
        subject.patch_les = torch.max(subject.patch_les, dim=1)[0]

    # show_seg(data.arr_org, mask_sag, "./mask_sag.webp", 4, 0.6, 5)
    # show_seg(arr_org_cor, mask_cor, "./mask_cor.webp", 4, 0.6, 5)
    # show_seg(arr_org_axi, mask_axi, "./mask_axi.webp", 4, 0.6, 5)
    return subject
