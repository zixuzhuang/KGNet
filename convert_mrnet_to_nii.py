import os

import cv2
import numpy as np
import SimpleITK as sitk
import argparse

from graph_construction.MRIData import read_data


def resize_images(images):
    """
    Resize a batch of images to be twice as large as their original size.

    Args:
    images (numpy.ndarray): A numpy array of shape (N, H, W) where
                            N is the number of images,
                            H is the height of the images,
                            W is the width of the images.

    Returns:
    numpy.ndarray: A numpy array containing the resized images.
    """
    # Determine the new dimensions, doubling each spatial dimension
    new_height = images.shape[1] * 2
    new_width = images.shape[2] * 2

    # Initialize an array to store the resized images
    resized_images = np.zeros((images.shape[0], new_height, new_width), dtype=np.uint8)

    # Resize each image
    for idx, image in enumerate(images):
        resized_images[idx] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_images


def convert(array, ref_data, view):
    """
    Convert a numpy array to a SimpleITK image, using another SimpleITK image as a reference
    for image properties such as spacing, origin, and direction.

    Args:
    array (numpy.ndarray): The NumPy array to convert.
    ref_data (SimpleITK.Image): The reference SimpleITK image to copy properties from.

    Returns:
    SimpleITK.Image: A new SimpleITK image based on the input array but with properties copied from the reference image.
    """
    if view == "axi":
        array = np.flip(array, axis=2)
    elif view == "cor":
        array = np.flip(array, axis=0)
        array = np.flip(array, axis=2)

    # Create a SimpleITK image from the NumPy array
    converted_image = sitk.GetImageFromArray(array)
    spacing = list(ref_data.GetSpacing())
    spacing[-1] = 3

    # Copy metadata from the reference image
    converted_image.SetSpacing(spacing)
    converted_image.SetOrigin(ref_data.GetOrigin())
    converted_image.SetDirection(ref_data.GetDirection())

    return converted_image


def create_nifti_image(mrnet_file, ref_file, view, save_path):
    """
    Create and save a NIfTI image from MRNet data with resizing and reference properties.

    Args:
    mrnet_file (str): Path to the MRNet .npy file.
    ref_file (str): Path to the reference NIfTI file.
    view (str): The view to use ('sag', 'cor', 'axi').
    save_path (str): Path to save the resulting NIfTI image.
    """
    # Load the reference file for image properties
    ref_data = read_data(ref_file, view)

    # Load the MRNet file
    mrnet_data = np.load(mrnet_file)

    # Resize the MRNet data
    resized_data = resize_images(mrnet_data)

    # Convert to a SimpleITK image using the reference data
    nifti_img = convert(resized_data, ref_data, view)

    # Save the NIfTI image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitk.WriteImage(nifti_img, save_path)
    print(f"NIfTI image saved to {save_path}")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Convert MRNet data to NIfTI format.")

    # Add arguments
    parser.add_argument(
        "--mrnet_file",
        default="data/npy_mrnet_example/0001/sag_org.npy",
        type=str,
        help="Path to the MRNet .npy file.",
    )
    parser.add_argument(
        "--ref_file",
        default="data/nifti_inhouse_example/00001/sag_org.nii.gz",
        type=str,
        help="Path to the reference NIfTI file.",
    )
    parser.add_argument(
        "--view",
        default="sag",
        type=str,
        choices=["sag", "cor", "axi"],
        help="The view to use ('sag', 'cor', 'axi').",
    )
    parser.add_argument(
        "--save_path",
        default="data/nifti_mrnet_example/0001_sag/sag_org.nii.gz",
        type=str,
        help="Path to save the resulting NIfTI image.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the function
    create_nifti_image(args.mrnet_file, args.ref_file, args.view, args.save_path)


# views_fullname = {"sag": "sagittal", "cor": "coronal", "axi": "axial"}
# ref_data = {}


# for view in ["sag", "cor", "axi"]:
#     ref_path = f"data/MRI_inhouse_example/00001/{view}_org.nii.gz"
#     selected_view = "sag" if "sag" in ref_path else "cor" if "cor" in ref_path else "axi"
#     ref_data[view] = read_data(ref_path, selected_view)

# for dataset in ["valid"]: # "train"
#     subjects = sorted(os.listdir(f"data/RAW_MRNet/{dataset}/axial/"))
#     subjects = [_ for _ in subjects if _.endswith(".npy")]
#     for subject in subjects:
#         for view in ["sag", "cor", "axi"]:
#             in_path = f"data/RAW_MRNet/{dataset}/{views_fullname[view]}/{subject}"
#             out_path = f"data/MRI_MRNet_{view}/{subject.replace('.npy', '')}/{view}_org.nii.gz"
#             # print(in_path, out_path)
#             # continue
#             if not os.path.exists(os.path.dirname(out_path)):
#                 os.makedirs(os.path.dirname(out_path))
#             data = np.load(in_path)
#             resized_data = resize_images(data)
#             converted_data = convert(resized_data, ref_data[view], view)
#             sitk.WriteImage(converted_data, out_path)
#         # exit()
