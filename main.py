import numpy as np
from zen_mapper import mapper
from sklearn.decomposition import PCA
import time

# from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from similarity import weighted_centroid_similarity, random_sample_similarity


def generate_circle(num_points: int, noise_level: float):
    angles = 2 * np.pi * np.random.random(num_points)
    x = np.cos(angles) + noise_level * np.random.normal(0, 1, num_points)
    y = np.sin(angles) + noise_level * np.random.normal(0, 1, num_points)

    data = np.column_stack((x, y))
    return data


pca = PCA(n_components=1)
data = generate_circle(10000, 0.05)
proj1 = data[:, 0]
proj2 = pca.fit_transform(data)
clusterer = sk_learn(DBSCAN(eps=0.25, min_samples=1))

result1 = mapper(
    data=data,
    clusterer=clusterer,
    cover_scheme=Width_Balanced_Cover(n_elements=11, percent_overlap=0.25),
    projection=proj1,
    dim=2,
)

result2 = mapper(
    data=data,
    projection=proj2,
    clusterer=clusterer,
    cover_scheme=Width_Balanced_Cover(n_elements=10, percent_overlap=0.25),
    dim=2,
)

weight_methods = [
    "node size",
    "importance",
    "inverse spread",
    "unique" + " proportion",
    "literally anything",
]

for method in weight_methods:
    time.sleep(1)
    print(method)
    sim_index = weighted_centroid_similarity(result1, result2, data, method)
    print(sim_index)
    print("Random Sample Similarity")
    sim_index = random_sample_similarity(result1, result2, data)
    print(sim_index)
