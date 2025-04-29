import numpy as np
import random
from typing import List, Optional
from zen_mapper import MapperResult
from zen_mapper.simplex import (
    get_nodes_containing_idx,
    get_neighborhood_data_indices,
)


def get_centroids(nodes: List[List[int]], data: np.ndarray) -> np.ndarray:
    centroids = []
    for indices in nodes:
        if len(indices) > 0:
            centroids.append(np.mean(data[indices], axis=0))
        else:
            centroids.append(np.zeros(data.shape[1]))
    return np.array(centroids)


def get_central_points(nodes, data) -> List[Optional[int]]:
    centroids = get_centroids(nodes, data)
    central_point_indices = []

    for i, indices in enumerate(nodes):
        if len(indices) == 0:
            central_point_indices.append(None)
            continue

        cluster_points = data[indices]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        central_point_indices.append(indices[np.argmin(distances)])

    return central_point_indices


def jaccard_similarity(set1, set2) -> float:
    if not set1 and not set2:
        return 1.0

    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    return intersection_size / union_size if union_size > 0 else 0.0


def weighted_centroid_similarity(
    result1: MapperResult,
    result2: MapperResult,
    data: np.ndarray,
    weight_method: str = "node size",
) -> float:
    """
    Compute weighted similarity between two mapper graphs via central points.

    Weight methods:
    - node size: Weight by the number of points in each neighborhood
    - importance: Weight by the proportion of points relative to total
    - inverse spread: Weight by inverse of standard deviation
    - unique proportion: Weight by proportion of unique points
    """
    EPSILON = 1e-15  # to avoid div by zero

    central_points1 = get_central_points(nodes=result1.nodes, data=data)
    central_points2 = get_central_points(nodes=result2.nodes, data=data)

    centroids = {cp for cp in central_points1 if cp is not None}
    centroids.update(cp for cp in central_points2 if cp is not None)

    total_similarity = 0.0
    total_weight = 0.0
    count = 0

    if weight_method == "importance":
        total_size1 = sum(len(n) for n in result1.nodes)
        total_size2 = sum(len(n) for n in result2.nodes)

    for centroid_id in centroids:
        containing_nodes1 = get_nodes_containing_idx(
            nodes=result1.nodes, datapoint_index=centroid_id
        )
        containing_nodes2 = get_nodes_containing_idx(
            nodes=result2.nodes, datapoint_index=centroid_id
        )

        if not containing_nodes1 or not containing_nodes2:
            continue

        data_indices1 = get_neighborhood_data_indices(
            mapper_result=result1, node_indices=containing_nodes1
        )
        data_indices2 = get_neighborhood_data_indices(
            mapper_result=result2, node_indices=containing_nodes2
        )

        data_indices1_list = list(data_indices1)
        data_indices2_list = list(data_indices2)

        if not data_indices1_list or not data_indices2_list:
            continue

        jaccard_sim = jaccard_similarity(data_indices1, data_indices2)

        if weight_method == "node size":
            weight = len(data_indices1) + len(data_indices2)

        elif weight_method == "importance":
            weight = (len(data_indices1) / max(total_size1, 1)) + (
                len(data_indices2) / max(total_size2, 1)
            )

        elif weight_method == "inverse spread":
            std_dev1 = np.std(data[data_indices1_list], axis=0)
            std_dev2 = np.std(data[data_indices2_list], axis=0)
            mean_std1 = np.mean(std_dev1)
            mean_std2 = np.mean(std_dev2)
            weight = 2.0 / (mean_std1 + mean_std2 + EPSILON)

        elif weight_method == "unique proportion":
            unique_points1 = set()
            for idx in data_indices1:
                if sum(1 for node in result1.nodes if idx in node) == 1:
                    unique_points1.add(idx)

            unique_points2 = set()
            for idx in data_indices2:
                if sum(1 for node in result2.nodes if idx in node) == 1:
                    unique_points2.add(idx)

            unique_ratio1 = len(unique_points1) / max(len(data_indices1), 1)
            unique_ratio2 = len(unique_points2) / max(len(data_indices2), 1)
            weight = unique_ratio1 + unique_ratio2

        else:
            weight = 1.0
            if count < 1:
                print("FALLBACK: using default weight = 1")
                count += 1
        total_similarity += jaccard_sim * weight
        total_weight += weight

    return total_similarity / max(total_weight, EPSILON)


def random_sample_similarity(
    result1: MapperResult, result2: MapperResult, data: np.ndarray
):
    """
    Compute similarity between two mapper graphs via a randomly sampled data
    point in each cluster. Uses a default weighting.
    """

    point_set = set()
    similarity_list = list()
    for node in result1.nerve.get_simplices(dim=0):
        point = random.choice(result1.nodes[node[0]])
        point_set.add(point)
    for node in result2.nerve.get_simplices(dim=0):
        point = random.choice(result2.nodes[node[0]])
        point_set.add(point)

    for point_id in point_set:
        nbhd1 = get_nodes_containing_idx(result1.nodes, point_id)
        nbhd2 = get_nodes_containing_idx(result2.nodes, point_id)
        set1 = get_neighborhood_data_indices(result1, nbhd1)
        set2 = get_neighborhood_data_indices(result2, nbhd2)

        sim = jaccard_similarity(set1, set2)
        similarity_list.append(sim)

    final_similarity = sum(similarity_list) / len(similarity_list)
    return final_similarity
