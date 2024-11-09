import dgl
import numpy as np
import scipy.spatial
import torch

import graph_construction.config as cfg
from graph_construction.methods.vertex import cal_distance
from graph_construction.MRIData import MRIData


def extractEdges(subject: MRIData):
    edges_Es = [[], []]
    edges_Ec = [[], []]
    edges_Ea = [[], []]

    # make edge along cartilage (Es)
    for bone_idx in range(len(subject.bones_idx)):
        for slice_idx in range(subject.shape[0]):
            if subject.v_2d[bone_idx][slice_idx] is None:
                continue
            edges_Es = add_to_edges(link_Es(subject, bone_idx, slice_idx), edges_Es)
            if slice_idx < subject.shape[0] - 1:
                edges_Ea = add_to_edges(link_Ea(subject, bone_idx, slice_idx), edges_Ea)
            if len(subject.bones_idx) > 1:
                edges_Ec = add_to_edges(link_Ec(subject, bone_idx, slice_idx), edges_Ec)

    # sort
    edges_Es = sort_edges(edges_Es)
    edges_Ec = sort_edges(edges_Ec)
    edges_Ea = sort_edges(edges_Ea)
    subject.edges = torch.cat([edges_Es, edges_Ec, edges_Ea], dim=1)
    # subject.edges = edges_Ea

    graph = dgl.graph((subject.edges[0], subject.edges[1]))
    subject.graph = dgl.add_self_loop(graph)
    return


def link_Es(subject: MRIData, bone_idx, slice_idx):
    vertex = subject.v_2d[bone_idx][slice_idx]
    idx = subject.v_idx[bone_idx][slice_idx]
    # add edge along bone surface
    lines = [idx[:-1], idx[1:]]
    # add addation edge if it is a loop
    if cal_distance(vertex[0], vertex[-1]) < subject.t_dist:
        lines[0].append(idx[-1])
        lines[1].append(idx[0])
    if bone_idx == 2:  # patella
        lines[0].append(idx[-1])
        lines[1].append(idx[0])
    return lines


def link_Ec(subject: MRIData, bone_idx, slice_idx):
    distance = cfg.PATCH_SIZE
    bone_num = len(subject.bones_idx)

    vertices_current = subject.v_2d[bone_idx][slice_idx]
    vertices_next = subject.v_2d[(bone_idx + 1) % bone_num][slice_idx]
    lines = [[], []]

    if vertices_next is None or vertices_current is None:
        return lines

    # Create KDTree for the next bone vertices
    tree_next = scipy.spatial.cKDTree(vertices_next)
    # Query the tree for each vertex in the current slice to find vertices in the next slice within the distance threshold
    for idx_current, vertex in enumerate(vertices_current):
        # Query all points within distance threshold
        indices_next = tree_next.query_ball_point(vertex, distance)
        # Store connections
        for idx_next in indices_next:
            lines[0].append(subject.v_idx[bone_idx][slice_idx][idx_current])
            lines[1].append(subject.v_idx[(bone_idx + 1) % bone_num][slice_idx][idx_next])
    return lines


def link_Ea(subject: MRIData, bone_idx, slice_idx):
    vertices_current = subject.v_2d[bone_idx][slice_idx]
    vertices_next = subject.v_2d[bone_idx][slice_idx + 1]
    lines = [[], []]

    if vertices_next is None or vertices_current is None:
        return lines

    # Create KDTree for the next slice vertices
    tree_next = scipy.spatial.cKDTree(vertices_next)
    # Query the tree for each vertex in the current slice to find vertices in the next slice within the distance threshold
    for idx_current, vertex in enumerate(vertices_current):
        # Query all points within distance threshold
        indices_next = tree_next.query_ball_point(vertex, subject.t_dist)
        # Store connections
        for idx_next in indices_next:
            lines[0].append(subject.v_idx[bone_idx][slice_idx][idx_current])
            lines[1].append(subject.v_idx[bone_idx][slice_idx + 1][idx_next])

    return lines


def add_to_edges(lines, edges):
    edges[0].extend(lines[0])
    edges[1].extend(lines[1])
    return edges


def sort_edges(edges):
    unique_edges = set()
    edges = np.array(edges).transpose().tolist()
    for edge in edges:
        unique_edges.add(tuple(sorted(edge)))
    sorted_unique_edges = sorted(unique_edges)
    sorted_unique_edges = torch.tensor(np.array(sorted_unique_edges).transpose(), dtype=torch.int32)
    return sorted_unique_edges
