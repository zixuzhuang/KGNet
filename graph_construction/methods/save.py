import os

import numpy as np
import torch

import graph_construction.config as cfg
from graph_construction.MRIData import MRIData


def saveData(subject: MRIData, save_path):
    """
    Saves data from a MRIData object to a compressed .npz file.

    Args:
        subject (MRIData): The MRIData object containing the data to be saved.
    """
    patch_org = {view: subject.patch[view] for view in subject.views}
    patch_seg = {view: subject.patch_seg[view] for view in subject.views}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        graph=subject.graph,
        org=patch_org,
        seg=patch_seg,
        les=subject.patch_les,
        pos=subject.pos,
        grade=subject.grade,
    )

    print(f"Information: patch size {subject.patch[subject.major_view].shape}, save to {save_path}")

    return
