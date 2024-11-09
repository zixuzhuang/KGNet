from os import listdir
from os.path import exists, join

import numpy as np
import SimpleITK as sitk
import torch
from SimpleITK import GetArrayFromImage as GAFI

from graph_construction import config as cfg


def resize_nifti(sitk_data, interpolation_method):
    """
    Resize a NIfTI image to a new spacing specified by cfg.STD_SPACING using the specified interpolation method.

    Parameters:
        sitk_data (SimpleITK.Image): The input image to be resized.
        interpolation_method (str): 'linear' for linear interpolation or 'nearest' for nearest neighbor interpolation.

    Returns:
        SimpleITK.Image: The resized image.
        The corresponding PyTorch tensor.
    """
    # Assume cfg.STD_SPACING is set somewhere in your configuration settings
    new_spacing = [cfg.STD_SPACING, cfg.STD_SPACING, sitk_data.GetSpacing()[-1]]

    # Calculate the new size based on the old and new spacing
    original_size = sitk_data.GetSize()
    original_spacing = sitk_data.GetSpacing()
    new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]

    # Set the interpolation method
    if interpolation_method == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation_method == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("Unknown interpolation method: choose 'linear' or 'nearest'")

    # Create the resample object
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(sitk_data.GetOrigin())
    resample.SetOutputDirection(sitk_data.GetDirection())
    resample.SetInterpolator(interpolator)

    # Execute the resampling
    resized_image = resample.Execute(sitk_data)

    # Convert the SimpleITK image to a numpy array and then to the specified tensor type
    if interpolation_method == "linear":
        resized_array = torch.tensor(GAFI(resized_image), dtype=torch.float32)
    else:
        resized_array = torch.tensor(GAFI(resized_image).astype(np.int32), dtype=torch.int32)
    return resized_image, resized_array


def read_data(file_path, orient):
    """
    Reads and orients a DICOM image, then converts it to a specified data type tensor.

    Parameters:
        file_path (str): Path to the DICOM file to be read.
        orient (str): Key to look up the orientation in the orients_dict.
        data_type (str): Desired type of the output tensor ('float' or 'int').

    Returns:
        The oriented SimpleITK image.
    """
    # Dictionary defining orientation mappings
    orients_dict = {"sag": "PIL", "cor": "LIP", "axi": "LPI"}

    # Read and orient the DICOM image using SimpleITK
    data = sitk.DICOMOrient(sitk.ReadImage(file_path), orients_dict[orient])

    return data


class MRIData(object):
    def __init__(self, subject_folder, major_view) -> None:
        super().__init__()
        self.name = subject_folder.split("/")[-1]
        self.data = {}
        self.org = {}
        self.seg = {}
        self.les = {}

        # Read nifti data
        self.major_view = major_view
        files = [f for f in listdir(subject_folder) if f.endswith(".nii.gz")]
        self.views = list(set([f.split("_")[0] for f in files]))
        for view in self.views:
            data_org = read_data(join(subject_folder, f"{view}_org.nii.gz"), view)
            self.data[view], self.org[view] = resize_nifti(data_org, "linear")
            data_seg = read_data(join(subject_folder, f"{view}_seg.nii.gz"), view)
            data_seg, self.seg[view] = resize_nifti(data_seg, "nearest")

        # Read lesion file from major view for ablation study
        self.have_lesion = True if exists(join(subject_folder, f"{major_view}_les.nii.gz")) else False
        if self.have_lesion:
            data_les = read_data(join(subject_folder, f"{major_view}_les.nii.gz"), major_view)
            data_les, arr_les = resize_nifti(data_les, "nearest")
            # different bones have different grade label, normalize them
            arr_les[arr_les == 3] = 1
            arr_les[arr_les == 4] = 2
            arr_les[arr_les == 5] = 1
            arr_les[arr_les == 6] = 2
            self.les[major_view] = arr_les

        self.grade = arr_les.max() if self.have_lesion else -1

        # Space infomation
        self.shape = self.org[major_view].shape

        # graph parematers
        self.graph = None
        self.surface = None
        # edge parameters
        self.t_dist = cfg.PATCH_SIZE * (1 - cfg.LAP_RATIO)
        self.edges = None
        # vertex parameters
        self.patch = {view: None for view in self.views}
        self.patch_seg = {view: None for view in self.views}
        self.patch_les = None

        self.v_2d = []
        self.v_3d = []
        self.v_idx = []

        self.pos = None
        # # Save path
        # # self.label = torch.tensor(args.grade, dtype=torch.long)
        # self.save_graph = args.graph
        return


if __name__ == "__main__":
    test_path = "data/MRI_inhouse/00001/cor_org.nii.gz"
    data, array = read_data(test_path, "cor", "float")
    data_resized = resize_nifti(data, "linear")
    print(f"The original image has size {data.GetSize()} and spacing {data.GetSpacing()}")
    print(f"The resized image has size {data_resized.GetSize()} and spacing {data_resized.GetSpacing()}")
