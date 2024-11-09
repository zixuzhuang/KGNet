from multiprocessing import Pool
from os import listdir, makedirs
from os import system as cmd
from os.path import exists, join
import argparse

from graph_construction.methods.edge import extractEdges
from graph_construction.methods.fov import adjustFOV
from graph_construction.methods.patch import extractPatch
from graph_construction.methods.save import saveData
from graph_construction.methods.vertex import extractVertices
from graph_construction.MRIData import MRIData


def constrct_graph(mri_folder, bone_index, major_view, save_path):
    # Process data
    subject = MRIData(mri_folder, major_view)
    subject.bones_idx = [int(i) for i in bone_index.split(",")]
    adjustFOV(subject)
    extractVertices(subject)
    extractEdges(subject)
    extractPatch(subject)
    saveData(subject, save_path)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Convert MRNet data to NIfTI format.")
    # Add arguments
    parser.add_argument(
        "--subject_folder",
        default="data/nifti_mrnet_example/0001_sag",
        type=str,
        help="Path to the nifti folder (include org and seg).",
    )
    parser.add_argument(
        "--bone_index",
        default="1,2,3",
        type=str,
        help="The index of the bones in segmentation.",
    )
    parser.add_argument(
        "--main_view",
        default="sag",
        type=str,
        choices=["sag", "cor", "axi"],
        help="The view to use ('sag', 'cor', 'axi').",
    )
    parser.add_argument(
        "--save_path",
        default="data/graph_mrnet_example/sag/0001.npz",
        type=str,
        help="Path to save the resulting knee graph.",
    )

    # Parse the arguments
    args = parser.parse_args()
    # Run the main function
    constrct_graph(args.subject_folder, args.bone_index, args.main_view, args.save_path)
