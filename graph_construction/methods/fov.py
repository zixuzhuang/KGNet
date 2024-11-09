import skimage.feature
import skimage.measure
import skimage.morphology
import torch

from graph_construction.MRIData import MRIData
from utils.utils_img import normalize, to_binary_mask
import graph_construction.config as cfg


def adjustFOV(subject: MRIData):

    # Extract bone segmentation
    bone = to_binary_mask(subject.seg[subject.major_view], num_cls=6)[torch.tensor(subject.bones_idx) - 1]
    petalla = bone[2]
    subject.surface = torch.zeros((len(subject.bones_idx), *subject.seg[subject.major_view].shape))

    # Height limitation
    
    h_min = max(petalla.nonzero()[:, 1].min().item() - 30, 0)
    h_max = h_min + 300

    # Width limitation
    w_min = max(petalla.nonzero()[:, 2].min().item() + 10, 0)
    w_max = w_min + 300

    # Extract surface
    for b in range(len(subject.bones_idx)):
        for s in range(subject.shape[0]):
            slice = bone[b, s].clone().numpy()
            for _ in range(3):
                slice = skimage.morphology.binary_dilation(slice)
            slice = skimage.morphology.binary_dilation(slice).astype(int) - slice.astype(int)
            subject.surface[b, s] = torch.tensor(slice)

    # Update bone segmentation FOV
    subject.surface[..., :h_min, :] = 0
    subject.surface[..., h_max:, :] = 0
    subject.surface[..., :, :w_min] = 0
    subject.surface[..., :, w_max:] = 0

    return subject
