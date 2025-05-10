import numpy as np
from zen_mapper import mapper
from zen_mapper.adapters import to_networkx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import networkx as nx
from sklearn.cluster import DBSCAN
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover, GMapperCover, CoverObj
from similarity import weighted_centroid_similarity, random_sample_similarity


# This is not really a main file -- just helper for kalman.py
class DataGenerator:
    @staticmethod
    def generate_circle(
        num_points: int,
        noise_level: float,
        radius: float,
        center_x: float,
        center_y: float,
    ) -> np.ndarray:
        """Generate a noisy circle."""
        angles = 2 * np.pi * np.random.random(num_points)
        x = (
            radius * np.cos(angles)
            + noise_level * np.random.normal(0, 1, num_points)
            + center_x
        )
        y = (
            radius * np.sin(angles)
            + noise_level * np.random.normal(0, 1, num_points)
            + center_y
        )

        return np.column_stack((x, y))

    @staticmethod
    def generate_test_data() -> np.ndarray:
        circle1 = DataGenerator.generate_circle(700, 0.05, 1, 0, 0)
        circle2 = DataGenerator.generate_circle(400, 0.025, 0.5, -1.5, 0)
        circle3 = DataGenerator.generate_circle(600, 0.1, 2, 0, 3)
        return np.vstack((circle1, circle2, circle3))


def plot_data_with_mapper_graph(data, projection, result, title, cover_scheme):
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(5, 1)
    ax_main = fig.add_subplot(gs[0:4, 0])
    scatter = ax_main.scatter(
        data[:, 0], data[:, 1], s=1, alpha=0.7, c=projection, cmap="viridis"
    )
    ax_main.set_title(f"Embedded Mapper Graph - {title}")
    ax_main.set_xlabel("X")
    ax_main.set_ylabel("Y")
    ax_main.grid(True, linestyle="--", alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label("Projection Value")

    proj_min, proj_max = np.min(projection), np.max(projection)

    if hasattr(cover_scheme, "cover") and hasattr(cover_scheme.cover, "intervals"):
        intervals = cover_scheme.cover.intervals
    elif hasattr(cover_scheme, "intervals"):
        intervals = cover_scheme.intervals
    elif isinstance(cover_scheme, Width_Balanced_Cover):

        centers = getattr(cover_scheme, "centers", None)
        if centers is None:
            if hasattr(cover_scheme, "intervals"):
                centers = [
                    (interval[0] + interval[1]) / 2
                    for interval in cover_scheme.intervals
                ]
            else:
                centers = []
        width = cover_scheme.width
        intervals = []
        for center in centers:
            left = center[0] - width[0] / 2
            right = center[0] + width[0] / 2
            intervals.append([left, right])
        intervals = np.array(intervals)
    else:
        if hasattr(result, "intervals"):
            intervals = result.intervals
        else:
            intervals = []
            for i, cover_element in enumerate(result.cover):
                if len(cover_element) > 0:
                    element_proj = projection[cover_element]
                    intervals.append([np.min(element_proj), np.max(element_proj)])
            intervals = np.array(intervals)

    colors = cm.rainbow(np.linspace(0, 1, len(intervals)))

    for i, interval in enumerate(intervals):
        mask = (projection >= interval[0]) & (projection <= interval[1])
        points_in_interval = data[mask]

        if len(points_in_interval) > 0:
            min_x, min_y = np.min(points_in_interval, axis=0)
            max_x, max_y = np.max(points_in_interval, axis=0)

            rect = Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                fill=True,
                alpha=0.2,
                edgecolor=colors[i],
                facecolor=colors[i],
                linewidth=2,
            )
            ax_main.add_patch(rect)

        rect_interval = Rectangle(
            (interval[0], 0),
            interval[1] - interval[0],
            1,
            alpha=0.7,
            color=colors[i],
        )

    G = to_networkx(result.nerve)
    nodelist = [n for n in G.nodes() if n != "root"]

    pos = {}
    for node_id in nodelist:
        if isinstance(node_id, int) and node_id < len(result.nodes):
            node_indices = result.nodes[node_id]
            if len(node_indices) > 0:
                centroid = np.mean(data[node_indices], axis=0)
                pos[node_id] = (centroid[0], centroid[1])
            else:
                pos[node_id] = (0, 0)
        else:
            continue
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax_main,
        node_size=200,
        node_color="blue",
        alpha=0.3,
    )
    edgelist = [(u, v) for u, v in G.edges() if u in nodelist and v in nodelist]
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax_main,
        edge_color="gray",
        width=2,
        alpha=0.8,
        edgelist=edgelist,
    )
    nx.draw_networkx_labels(
        G,
        pos=pos,
        ax=ax_main,
        font_size=8,
        font_color="black",
        font_weight="bold",
    )

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_with_graph.svg", format="svg")


def generate_circle(num_points: int, noise_level: float):
    angles = 2 * np.pi * np.random.random(num_points)
    x = np.cos(angles) + noise_level * np.random.normal(0, 1, num_points)
    y = np.sin(angles) + noise_level * np.random.normal(0, 1, num_points)

    data = np.column_stack((x, y))
    return data


pca = PCA(n_components=1)
data = DataGenerator.generate_test_data()
proj1 = data[:, 0]
proj2 = pca.fit_transform(data).flatten()
clusterer = sk_learn(DBSCAN(eps=0.25, min_samples=1))
gmeans_cover = GMapperCover(
    iterations=30, max_intervals=35, method="DFS", ad_threshold=4, g_overlap=0.2
)

cover_elements = gmeans_cover.__call__(data)
result1 = mapper(
    data=data,
    clusterer=clusterer,
    cover_scheme=gmeans_cover,
    projection=proj1,
    dim=1,
    precomputed_cover=None,
)

plot_data_with_mapper_graph(data, proj1, result1, "G-Mapper Cover", cover_elements)

balanced_cover = Width_Balanced_Cover(n_elements=10, percent_overlap=0.25)
result2 = mapper(
    data=data,
    projection=proj1,
    clusterer=clusterer,
    cover_scheme=balanced_cover,
    dim=1,
)

plot_data_with_mapper_graph(
    data, proj1, result2, "Width_Balanced_Cover", balanced_cover
)


print("\nNerve Structures:")
print("-" * 50)
print(f"GMeansCover Nerve:\n{result1.nerve}")
print(f"\nWidth_Balanced_Cover Nerve:\n{result2.nerve}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
G1 = to_networkx(result1.nerve)
nodelist1 = [n for n in G1.nodes() if n != "root"]

pos1 = {}
for node_id in nodelist1:
    if isinstance(node_id, int) and node_id < len(result1.nodes):
        node_indices = result1.nodes[node_id]
        if len(node_indices) > 0:
            centroid = np.mean(data[node_indices], axis=0)
            pos1[node_id] = (centroid[0], centroid[1])
        else:
            pos1[node_id] = (0, 0)

nx.draw_networkx_nodes(G1, pos=pos1, node_size=150, node_color="skyblue", alpha=0.8)

edgelist1 = [(u, v) for u, v in G1.edges() if u in nodelist1 and v in nodelist1]
nx.draw_networkx_edges(
    G1, pos=pos1, edge_color="gray", width=1.5, alpha=0.8, edgelist=edgelist1
)

nx.draw_networkx_labels(
    G1,
    pos=pos1,
    font_size=8,
    font_color="black",
    font_weight="bold",
)

plt.title("G-Mapper Graph")

plt.subplot(1, 2, 2)
G2 = to_networkx(result2.nerve)
nodelist2 = [n for n in G2.nodes() if n != "root"]

pos2 = {}
for node_id in nodelist2:
    if isinstance(node_id, int) and node_id < len(result2.nodes):
        node_indices = result2.nodes[node_id]
        if len(node_indices) > 0:
            centroid = np.mean(data[node_indices], axis=0)
            pos2[node_id] = (centroid[0], centroid[1])
        else:
            pos2[node_id] = (0, 0)

nx.draw_networkx_nodes(
    G2,
    pos=pos2,
    node_size=350,
    node_color="skyblue",
    alpha=0.8,
)

edgelist2 = [(u, v) for u, v in G2.edges() if u in nodelist2 and v in nodelist2]
nx.draw_networkx_edges(
    G2, pos=pos2, edge_color="gray", width=1.5, alpha=0.8, edgelist=edgelist2
)

nx.draw_networkx_labels(
    G2,
    pos=pos2,
    font_size=8,
    font_color="black",
    font_weight="bold",
)

plt.title("Width_Balanced_Cover Mapper Graph")

plt.tight_layout()
plt.savefig("Mapper_Graphs_Comparison.svg", format="svg")
