import numpy as np
from zen_mapper import MapperResult
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


def centroid_similarity(result1: MapperResult, result2: MapperResult, data):
    central_points1 = get_central_points(nodes=result1.nodes, data=data)
    central_points2 = get_central_points(nodes=result2.nodes, data=data)

    centroids = set([cp for cp in central_points1])
    centroids = centroids.union(set([cp for cp in central_points2]))

    total_similarity = 0.0
    total_valid_centroids = 0

    for centroid_id in centroids:
        containing_nodes1 = get_nodes_containing_idx(
            nodes=result1.nodes, datapoint_index=centroid_id
        )
        containing_nodes2 = get_nodes_containing_idx(
            nodes=result2.nodes, datapoint_index=centroid_id
        )
        if containing_nodes1 and containing_nodes2:
            nbhd1 = get_simplex_neighborhood(result1.nerve, containing_nodes1)
            nbhd2 = get_simplex_neighborhood(result2.nerve, containing_nodes2)

            data_indices1 = set()
            for node_idx in nbhd1:
                data_indices1.update(result1.nodes[node_idx])

            data_indices2 = set()
            for node_idx in nbhd2:
                data_indices2.update(result2.nodes[node_idx])

            intersection = len(data_indices1.intersection(data_indices2))
            union = len(data_indices1.union(data_indices2))

            if union > 0:
                jaccard_sim = intersection / union
                total_similarity += jaccard_sim
                total_valid_centroids += 1
    final_similarity = total_similarity / total_valid_centroids
    return final_similarity
