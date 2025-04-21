import numpy as np
from zen_mapper.simplex import get_neighborhood, SimplexTree


def get_centroids(nodes: list, data):
    centroid_list = []
    for i in range(len(nodes)):
        indices = nodes[i]
        mean = np.mean(data[indices], axis=0)
        centroid_list.append(mean)
    return centroid_list


def get_central_points(nodes: list, data):
    centroid_list = get_centroids(nodes, data)
    central_point_indices = []

    for i in range(len(nodes)):
        indices = nodes[i]
        if len(indices) == 0:
            central_point_indices.append(None)
            continue

        cluster_points = data[indices]
        mean = centroid_list[i]

        dist_to_mean = np.linalg.norm(cluster_points - mean, axis=1)
        local_min_idx = np.argmin(dist_to_mean)
        min_idx = indices[local_min_idx]
        central_point_indices.append(min_idx)

    return central_point_indices


def get_nodes_containing_idx(nodes: list, datapoint_index: int):
    matching_nodes = []

    for i, node_indices in enumerate(nodes):
        if datapoint_index in node_indices:
            matching_nodes.append(i)

    return matching_nodes


# is redundant -- will eventually remove
def get_simplex_neighborhood(st: SimplexTree, node_indices):
    return get_neighborhood(st=st, simplex=node_indices)
