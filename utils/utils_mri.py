import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def mri_proj(data):
    origin = np.array(data.GetOrigin()).reshape([1, 3])
    spacing = np.array(data.GetSpacing()).reshape([1, 3])
    direction = np.array(data.GetDirection()).reshape([3, 3])
    shape = np.array(data.GetSize()).reshape([3])
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    H, W, S = shape[0], shape[1], shape[2]

    x = np.arange(H).astype(np.float32)
    y = np.arange(W).astype(np.float32)
    xx, yy = np.meshgrid(x, y)

    s = np.arange(S).astype(np.float32)
    xyz_projected = []
    for s_idx in s:
        _shape = xx.shape

        zz = np.zeros_like(xx, dtype=np.float32) + s_idx
        xyz = (
            np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1) * spacing
        )
        xyz = np.matmul(xyz, direction.transpose()) + origin
        xyz_projected.append(
            np.concatenate(
                [xyz.transpose().reshape(3, *_shape), image[int(s_idx)][None, ...]],
                axis=0,
            )
        )
    return xyz_projected


def to_world_coordinate(sitkimg, points):
    origin = np.array(sitkimg.GetOrigin()).reshape([1, 1, 3])
    spacing = np.array(sitkimg.GetSpacing()).reshape([1, 3])
    direction = np.array(sitkimg.GetDirection()).reshape([3, 3])
    points = np.flip(points, axis=1)  # to y, x, s
    points = points * spacing
    points = np.matmul(points, direction.transpose())
    points = points + origin
    points = points[0]
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


if __name__ == "__main__":
    data = sitk.ReadImage("./data/mri/sag/00006/org.nii.gz")
    proj = mri_proj(data)[10]
    print(proj.shape)
