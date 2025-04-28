import numpy as np
import matplotlib.pyplot as plt
from zen_mapper import mapper, MapperResult
from zen_mapper.cover import Width_Balanced_Cover
from sklearn.decomposition import PCA
from zen_mapper.cluster import sk_learn
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from similarity import weighted_centroid_similarity


def generate_circle(
    num_points: int, noise_level: float, r: float, center_x: float, center_y: float
):
    angles = 2 * np.pi * np.random.random(num_points)
    x = r * np.cos(angles) + noise_level * np.random.normal(0, 1, num_points) + center_x
    y = r * np.sin(angles) + noise_level * np.random.normal(0, 1, num_points) + center_y
    return np.column_stack((x, y))


def generate_dataset():
    np.random.seed(42)
    data1 = generate_circle(100, 0.01, 1, 0, 0)
    data2 = generate_circle(100, 0.05, 2, 0, 2)
    return np.vstack((data1, data2))


def create_mapper_results(data, n_elements_list, overlap_list, eps_list):
    pca = PCA(n_components=1)
    proj = pca.fit_transform(data)

    results = []
    params = []

    for n_elements in n_elements_list:
        for overlap in overlap_list:
            for eps in eps_list:
                result = mapper(
                    data=data,
                    projection=proj,
                    clusterer=sk_learn(DBSCAN(eps=eps, min_samples=1)),
                    cover_scheme=Width_Balanced_Cover(
                        n_elements=n_elements, percent_overlap=overlap
                    ),
                    dim=1,
                )
                results.append(result)
                params.append(
                    {"n_elements": n_elements, "overlap": overlap, "eps": eps}
                )

    return results, params


def compare_similarity_methods(results, data):
    n_results = len(results)

    weight_methods = [
        "node_size",
        "importance",
        "inverse_spread",
        "unique_proportion",
    ]

    similarity_matrices = {}
    for method in weight_methods:
        similarity_matrices[method] = np.zeros((n_results, n_results))

    # Similarities
    for i in tqdm(range(n_results)):
        for j in range(i, n_results):
            for method in weight_methods:
                weighted_sim = weighted_centroid_similarity(
                    results[i], results[j], data, weight_method=method
                )
                similarity_matrices[method][i, j] = weighted_sim
                similarity_matrices[method][j, i] = weighted_sim

    return similarity_matrices


def plot_similarity_matrices(similarity_matrices, params):
    """Plot heatmaps"""
    n_methods = len(similarity_matrices)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    labels = [
        f"n={p['n_elements']}, o={p['overlap']:.1f}, ε={p['eps']:.2f}" for p in params
    ]

    for i, (method, matrix) in enumerate(similarity_matrices.items()):
        sns.heatmap(
            matrix,
            ax=axes[i],
            cmap="viridis",
            vmin=0,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
        )

        axes[i].set_xticklabels(labels, rotation=35, ha="right")
        axes[i].set_yticklabels(labels, rotation=0)

        method_title = method.replace("_", " ").title()
        axes[i].set_title(f"{method_title} Similarity", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig("similarity_comparison.png", dpi=600, bbox_inches="tight")


def analyze_similarities(similarity_matrices, params):
    avg_similarities = {
        method: np.mean(matrix) for method, matrix in similarity_matrices.items()
    }
    var_similarities = {
        method: np.var(matrix) for method, matrix in similarity_matrices.items()
    }

    # correlation
    method_names = list(similarity_matrices.keys())
    readable_method_names = [name.replace("_", " ").title() for name in method_names]

    correlation_matrix = np.zeros((len(method_names), len(method_names)))
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            flat1 = similarity_matrices[method1].flatten()
            flat2 = similarity_matrices[method2].flatten()
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            correlation_matrix[i, j] = correlation

    # correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".5f",
        cmap="coolwarm",
        xticklabels=readable_method_names,
        yticklabels=readable_method_names,
    )
    plt.title("Correlation Between Similarity Methods")
    plt.tight_layout()
    plt.savefig("method_correlations.png", dpi=600)

    summary = pd.DataFrame(
        {
            "Method": readable_method_names,
            "Average Similarity": [avg_similarities[m] for m in method_names],
            "Variance": [var_similarities[m] for m in method_names],
        }
    )

    plt.figure(figsize=(10, 5))
    summary.set_index("Method").plot(kind="bar", alpha=0.7)
    plt.title("Comparison of Similarity Methods")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig("similarity_summary.png", dpi=600)
    return summary


def main():
    print("Generating dataset...")
    data = generate_dataset()

    n_elements_list = [4, 5, 6]
    overlap_list = [0.2, 0.25, 0.3]
    eps_list = [0.1, 0.5, 1.0]
    results, params = create_mapper_results(
        data, n_elements_list, overlap_list, eps_list
    )

    print("Comparing similarity methods...")
    similarity_matrices = compare_similarity_methods(results, data)

    print("Plotting similarity matrices...")
    plot_similarity_matrices(similarity_matrices, params)

    print("Analyzing similarities...")
    summary = analyze_similarities(similarity_matrices, params)

    print("\nSummary of similarity methods:")
    print(summary)

    summary.to_csv("similarity_summary.csv", index=False)

    print("\nAnalysis complete. Results saved to:")
    print("- similarity_comparison.png")
    print("- method_correlations.png")
    print("- similarity_summary.csv")


if __name__ == "__main__":
    main()
