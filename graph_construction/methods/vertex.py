import numpy as np
import scipy.spatial

import graph_construction.config as cfg
from graph_construction.MRIData import MRIData


def extractVertices(subject: MRIData):
    """
    Extract vertices along bone surface for each bone index specified in the configuration.

    Parameters:
        data (MRIData): An object containing MR imaging data and structures for storing vertices.
    """
    for bone_index in range(len(subject.bones_idx)):
        vertices_2d, vertices_3d = process_bone(subject, bone_index)
        subject.v_2d.append(vertices_2d)
        subject.v_3d.append(vertices_3d)

    subject.v_idx = generate_vertex_indices(subject)
    subject.v_3d = np.vstack(subject.v_3d)
    # print(data.v_2d)

def process_bone(subject: MRIData, bone_index):
    """
    Process each slice for a given bone to extract and simplify vertex coordinates.

    Returns:
        tuple: A tuple containing 2D and 3D vertex arrays for the bone.
    """
    vertices_2d = []
    vertices_3d = []

    for slice_index in range(subject.shape[0]):
        vertex_data = process_slice(subject.surface, bone_index, slice_index)
        if vertex_data:
            vertices_2d.append(vertex_data['2d'])
            vertices_3d.append(vertex_data['3d'])
        else:
            vertices_2d.append(None)

    return vertices_2d, np.vstack(vertices_3d) if vertices_3d else []


def process_slice(surface: MRIData, bone_index, slice_index):
    """
    Identify non-zero vertices in the slice, sort them, and simplify the coordinates.

    Returns:
        dict: A dictionary containing 2D vertices and stacked 3D vertices.
    """
    vertices_xy = np.array(surface[bone_index][slice_index].nonzero(), dtype=np.int32)
    if vertices_xy.size == 0:
        return None

    sorted_vertices = sort_xy(vertices_xy)
    simplified_vertices = simplify_xy(sorted_vertices)
    slice_positions = np.zeros((simplified_vertices.shape[0], 1), dtype=np.int32) + slice_index

    return {
        '2d': simplified_vertices,
        '3d': np.hstack((slice_positions, simplified_vertices))
    }

def generate_vertex_indices(subject: MRIData):
    """
    Generate unique indices for all vertices across all bones and slices.

    Returns:
        list: A list of lists containing indices for each vertex.
    """
    vertex_indices = []
    index_counter = 0

    for bone_vertices in subject.v_2d:
        bone_indices = []
        for slice_vertices in bone_vertices:
            slice_indices = []
            if slice_vertices is not None:
                for _ in slice_vertices:
                    slice_indices.append(index_counter)
                    index_counter += 1
            bone_indices.append(slice_indices if slice_indices else None)
        vertex_indices.append(bone_indices)

    return vertex_indices


def sort_xy(points_list):

    _n = points_list.shape[0] - 1

    # find the first point that located at left top
    fisrt_idx = 0
    points_sort = [points_list[fisrt_idx]]
    points_list = np.delete(points_list, fisrt_idx, 0)

    # add point in sorted set and delete it from raw set
    for _i in range(_n):
        # find the point that closest to sorted set's first or latest points
        idx_l, idx_s = find_closest_point(points_list, points_sort)
        if idx_s == 0:
            points_sort.insert(0, points_list[idx_l])
        else:
            points_sort.append(points_list[idx_l])
        points_list = np.delete(points_list, idx_l, 0)
    points_sort = np.array(points_sort)

    # sort direction
    direction = False
    if len(points_sort) > 3:
        k1, k2, k3 = 0, len(points_sort) // 3, len(points_sort) // 3 * 2
        direction = find_direction(points_sort[k1], points_sort[k2], points_sort[k3])
    if direction:
        points_sort = np.flip(points_sort, axis=0)

    return np.array(points_sort)


def simplify_xy(points_sorted):

    # if the overlap between p_i and p_i+1 smaller than LAP_RATIO, add v_i+1 to vertex set
    _n = points_sorted.shape[0]
    vertex = []
    vertex.append(points_sorted[0])
    for _i in range(1, _n - 1):
        lap_ratio = cal_overlap(vertex[-1], points_sorted[_i])
        if lap_ratio < cfg.LAP_RATIO:
            vertex.append(points_sorted[_i])

    # add end point
    if cal_overlap(vertex[-1], points_sorted[-1]) > 0.5:
        vertex.pop()
    vertex.append(points_sorted[-1])

    if cal_overlap(vertex[0], vertex[-1]) > 0.5 and len(vertex) > 1:
        vertex.pop()

    if len(vertex) == 2:
        vertex.pop()

    return np.array(vertex)


def find_closest_point(points_list, points_sort):
    points_sort_sne = [points_sort[0], points_sort[-1]]
    mytree1 = scipy.spatial.cKDTree(points_sort_sne)  # the sorted point set's start point and end point
    dist, idx = mytree1.query(points_list)
    min_idx = np.argmin(dist)
    return min_idx, idx[min_idx]


def cal_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def cal_distance3d(p1, p2, z_scale=1.0):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + z_scale * (p1[2] - p2[2]) ** 2) ** 0.5


def cal_overlap(p1, p2):
    w = cfg.PATCH_SIZE - np.abs(p1[0] - p2[0])
    h = cfg.PATCH_SIZE - np.abs(p1[1] - p2[1])
    if w < 0 or h < 0:
        return 0.0
    else:
        return w * h / cfg.PATCH_SIZE ** 2


def find_direction(p1, p2, p3):
    if ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) != 0:
        return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < 0
    else:
        return None
