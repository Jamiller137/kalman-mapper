import numpy as np
from zen_mapper import mapper
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from similarity import get_central_points, get_nodes_containing_idx
from similarity import get_simplex_neighborhood


def generate_circle(num_points: int, noise_level: float):
    angles = 2*np.pi*np.random.random(num_points)
    x = np.cos(angles) + noise_level * np.random.normal(0, 1, num_points)
    y = np.sin(angles) + noise_level * np.random.normal(0, 1, num_points)

    data = np.column_stack((x, y))
    return data


pca = PCA(n_components=1)
data = generate_circle(10000, 0.05)
proj = data[:, 0]
proj = data
clusterer = sk_learn(DBSCAN(eps=0.25, min_samples=1))

result = mapper(
        data=data,
        clusterer=clusterer,
        cover_scheme=Width_Balanced_Cover(n_elements=5, percent_overlap=0.25),
        projection=proj,
        dim=2
        )

st = result.nerve
cluster_list = result.nodes


central_points = get_central_points(nodes=cluster_list, data=data)

central_point_nodes = get_nodes_containing_idx(
        nodes=cluster_list,
        datapoint_index=central_points[24])

print(st)
print(central_point_nodes)
print(f"{central_points[24]} = {data[central_points[24]]}")
print(get_simplex_neighborhood(st, central_point_nodes))
