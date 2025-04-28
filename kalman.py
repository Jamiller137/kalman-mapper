import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from zen_mapper import mapper
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.cluster import sk_learn
import networkx as nx
from zen_mapper.adapters import to_networkx
from similarity import weighted_centroid_similarity
from tqdm import tqdm
import copy


# Constants
ENSEMBLE_SIZE = 10
EPS_MIN = 0.001
EPS_MAX = 2.0
N_ELEMENTS = 5
OVERLAP = 0.3
MAX_ITERATIONS = 20
OBSERVATION_NOISE = 0.15
K_NEIGHBORS = 10  # Number of neighbors to sample for each ensemble member
NEIGHBOR_STDDEV = 0.5


def ensemble_kalman_filter_optimizer(data):
    """
    Use Ensemble Kalman Filter to optimize eps parameter
    """
    eps_ensemble = np.random.uniform(EPS_MIN, EPS_MAX, ENSEMBLE_SIZE)

    mean_eps_path = [np.mean(eps_ensemble)]
    similarity_path = []

    all_eps_history = [copy.deepcopy(eps_ensemble)]

    for iteration in tqdm(range(MAX_ITERATIONS)):
        similarities = []

        for i, eps in enumerate(eps_ensemble):
            current_result = mapper(
                data=data,
                projection=PCA(n_components=1).fit_transform(data),
                clusterer=sk_learn(DBSCAN(eps=eps, min_samples=1)),
                cover_scheme=Width_Balanced_Cover(
                    n_elements=N_ELEMENTS, percent_overlap=OVERLAP
                ),
                dim=1,
            )

            # K neighboring eps values using normal distribution
            neighbor_eps_values = np.clip(
                np.random.normal(eps, NEIGHBOR_STDDEV, K_NEIGHBORS), EPS_MIN, EPS_MAX
            )

            # Calculate similarity with each neighbor
            neighbor_similarities = []
            for neighbor_eps in neighbor_eps_values:
                neighbor_result = mapper(
                    data=data,
                    projection=PCA(n_components=1).fit_transform(data),
                    clusterer=sk_learn(DBSCAN(eps=neighbor_eps, min_samples=1)),
                    cover_scheme=Width_Balanced_Cover(
                        n_elements=N_ELEMENTS, percent_overlap=OVERLAP
                    ),
                    dim=1,
                )

                similarity = weighted_centroid_similarity(
                    current_result, neighbor_result, data, "node_size"
                )
                neighbor_similarities.append(similarity)

            # Use average similarity as the observation
            similarities.append(np.mean(neighbor_similarities))

        similarities = np.array(similarities)
        similarity_path.append(np.mean(similarities))

        # update step
        # State
        X = eps_ensemble.reshape(-1, 1)
        # Observation
        Y = similarities.reshape(-1, 1)

        X_mean = np.mean(X, axis=0, keepdims=True)
        Y_mean = np.mean(Y, axis=0, keepdims=True)

        X_anomalies = X - X_mean
        Y_anomalies = Y - Y_mean

        # covariance matrices
        P_xy = (X_anomalies.T @ Y_anomalies) / (ENSEMBLE_SIZE - 1)
        P_yy = (Y_anomalies.T @ Y_anomalies) / (ENSEMBLE_SIZE - 1) + OBSERVATION_NOISE

        # Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)

        # The "observation" we're trying to match is maximum similarity (1.0)
        target_similarity = 1.0
        innovation = target_similarity - Y

        # Update ensemble
        X_updated = X + (K @ innovation.T).T

        # values remain within bounds
        eps_ensemble = np.clip(X_updated.flatten(), EPS_MIN, EPS_MAX)

        mean_eps_path.append(np.mean(eps_ensemble))
        all_eps_history.append(copy.deepcopy(eps_ensemble))

        # convergence check
        if iteration > 1 and abs(mean_eps_path[-1] - mean_eps_path[-2]) < 1e-7:
            print(f"Converged after {iteration+1} iterations")
            break

    return mean_eps_path, similarity_path, all_eps_history


def visualize_results(mean_eps_path, similarity_path, all_eps_history):
    """
    Visualize the optimization path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: eps vs iteration
    iterations = np.arange(len(mean_eps_path))
    ax1.plot(iterations, mean_eps_path, "b-", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("eps value")
    ax1.set_title("Path of eps parameter during optimization")
    ax1.grid(True)

    # Add ensemble members
    for i in range(ENSEMBLE_SIZE):
        member_path = [history[i] for history in all_eps_history]
        ax1.plot(iterations, member_path, "b-", alpha=0.2)

    # Plot 2: similarity vs eps
    ax2.plot(mean_eps_path[:-1], similarity_path, "r-", linewidth=2)
    ax2.set_xlabel("eps value")
    ax2.set_ylabel("Similarity")
    ax2.set_title("Similarity vs eps parameter")
    ax2.grid(True)

    # optimal point
    ax2.scatter(
        [mean_eps_path[-2]],
        [similarity_path[-1]],
        color="red",
        s=100,
        label=f"Optimal eps = {mean_eps_path[-2]:.3f}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("eps_optimization_path.png", dpi=600)

    fig, ax = plt.subplots(figsize=(8, 6))

    norm = plt.Normalize(min(similarity_path), max(similarity_path))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    colors = plt.cm.viridis(norm(similarity_path))

    for i in range(len(mean_eps_path) - 2):
        ax.arrow(
            mean_eps_path[i],
            similarity_path[i],
            mean_eps_path[i + 1] - mean_eps_path[i],
            similarity_path[i + 1] - similarity_path[i],
            head_width=0.0005,
            head_length=0.0005,
            width=0.00015,
            fc=colors[i],
            ec=colors[i],
            length_includes_head=True,
        )

    ax.set_xlabel("eps parameter")
    ax.set_ylabel("Similarity")
    ax.set_title("Gradient Path to Optimal eps Value")
    ax.grid(True)

    plt.colorbar(sm, ax=ax, label="Similarity")

    plt.tight_layout()
    plt.savefig("eps_gradient_path.png", dpi=300)


def main(data):
    mean_eps_path, similarity_path, all_eps_history = ensemble_kalman_filter_optimizer(
        data
    )
    visualize_results(mean_eps_path, similarity_path, all_eps_history)

    final_result = mapper(
        data=data,
        projection=PCA(1).fit_transform(data),
        cover_scheme=Width_Balanced_Cover(N_ELEMENTS, OVERLAP),
        clusterer=sk_learn(DBSCAN(float(mean_eps_path[-2]))),
        dim=1,
    )
    print(final_result.nerve)
    plt.scatter(data[0], data[1])
    graph = to_networkx(final_result.nerve)
    nx.draw(graph)
    plt.savefig("graph.png")


if __name__ == "__main__":

    def generate_circle(
        num_points: int, noise_level: float, r: float, center_x: float, center_y: float
    ):
        angles = 2 * np.pi * np.random.random(num_points)
        x = (
            r * np.cos(angles)
            + noise_level * np.random.normal(0, 1, num_points)
            + center_x
        )
        y = (
            r * np.sin(angles)
            + noise_level * np.random.normal(0, 1, num_points)
            + center_y
        )

        data = np.column_stack((x, y))
        return data

    circle1 = generate_circle(500, 0.05, 1, 0, 0)
    circle2 = generate_circle(500, 0.025, 0.5, -1, 0)
    circle3 = generate_circle(500, 0.1, 2, 0, 4)

    data = np.vstack((circle1, circle2, circle3))
    main(data)
