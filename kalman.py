import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from zen_mapper import mapper
from zen_mapper.cover import GMapperCover
from zen_mapper.cluster import sk_learn
import os
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import concurrent.futures
from tqdm import tqdm
import copy
import pandas as pd

from similarity import random_sample_similarity
from main import plot_data_with_mapper_graph


@dataclass
class Config:
    # Ensemble parameters
    ensemble_size: int = 50
    eps_min: float = 0.0001
    eps_max: float = 3
    max_iterations: int = 100
    observation_noise: float = 0.15
    k_neighbors: int = 5
    neighbor_stddev: float = 0.5

    # Mapper parameters
    n_elements: int = 7
    overlap: float = 0.2
    min_samples: int = 1

    # Optimization parameters
    learning_rate: float = 0.1
    gradient_delta: float = 0.1
    exploration_probability: float = 0.05
    exploration_stddev: float = 0.05
    convergence_threshold: float = 1e-5

    # Stable region identification parameters
    min_clusters: int = 2
    max_clusters: int = ensemble_size
    stability_delta: float = 0.05

    # Forecast Model Parameters
    density_window: float = 0.1
    density_samples: int = 5
    stability_threshold: float = 0.01
    stable_region_step: float = 0.005
    transition_noise: float = 0.01
    max_step_size: float = 0.1
    process_noise_std: float = 0.01

    # Visualization parameters
    graph_node_size: int = 75
    graph_node_color: str = "red"
    graph_edge_color: str = "lightblue"
    graph_edge_width: int = 2
    graph_layout_k: float = 0.5
    graph_layout_iterations: int = 100
    graph_layout_seed: int = 42
    dpi_standard: int = 800
    dpi_high: int = 1200

    def __post_init__(self):
        if self.max_clusters > self.ensemble_size:
            self.max_clusters = self.ensemble_size


@dataclass
class StableRegion:
    eps: float
    similarity: float
    stability: float
    score: float
    result: Any


class MapperAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = "stable_regions"
        self.precover = None
        os.makedirs(self.output_dir, exist_ok=True)

    def precompute_cover(self, data: np.ndarray) -> None:
        self.precover = GMapperCover(
            iterations=50, max_intervals=20, ad_threshold=3, method="BFS", g_overlap=0.2
        )(data)

    def run_mapper(self, data: np.ndarray, eps: float) -> Any:
        if self.precover is None:
            self.precompute_cover(data)

        gmeans_scheme = GMapperCover(
            iterations=50, max_intervals=20, ad_threshold=3, method="BFS", g_overlap=0.2
        )

        return mapper(
            data=data,
            projection=data[:, 0],
            clusterer=sk_learn(DBSCAN(eps=eps, min_samples=self.config.min_samples)),
            cover_scheme=gmeans_scheme,
            dim=1,
            precomputed_cover=self.precover,
        )

    def calculate_similarity(
        self, result1: Any, result2: Any, data: np.ndarray
    ) -> float:
        return random_sample_similarity(result1, result2, data)

    def _forecast_epsilon_evolution(self, data: np.ndarray, eps_ensemble: np.ndarray):
        """
        Returns:
            Tuple of (forecasted_eps, expected_stability_change)
        """
        forecasted_eps = np.zeros_like(eps_ensemble)
        expected_stability_change = np.zeros_like(eps_ensemble)

        # process ensemble member
        for i, eps in enumerate(eps_ensemble):
            # estimate around current epsilon
            nearby_epsilons = np.linspace(
                max(eps - self.config.density_window, self.config.eps_min),
                min(eps + self.config.density_window, self.config.eps_max),
                self.config.density_samples,
            )

            stabilities = []
            base_result = self.run_mapper(data, eps)
            for nearby_eps in nearby_epsilons:
                nearby_result = self.run_mapper(data, nearby_eps)
                similarity = self.calculate_similarity(base_result, nearby_result, data)
                stabilities.append(similarity)

            stabilities = np.array(stabilities)
            eps_values = nearby_epsilons - eps

            # fit quadratic model
            coeffs = np.polyfit(eps_values, stabilities, 2)
            a, b, c = coeffs

            # if in a stable region, take smaller steps
            # if in a transition region, take larger step
            curvature = abs(2 * a)

            if curvature < self.config.stability_threshold:
                # Stable region - small random walk
                step = np.random.normal(0, self.config.stable_region_step)
            else:
                # the step in a quadratic model is -b/(2c)
                optimal_step = -b / (2 * a) if a != 0 else 0
                # clip step size
                optimal_step = np.clip(
                    optimal_step, -self.config.max_step_size, self.config.max_step_size
                )
                step = optimal_step + np.random.normal(0, self.config.transition_noise)

            # Update forecast
            forecasted_eps[i] = eps + step

            expected_stability_change[i] = b * step + a * step * step

        forecasted_eps = np.clip(
            forecasted_eps, self.config.eps_min, self.config.eps_max
        )

        return forecasted_eps, expected_stability_change

    def optimize_epsilon(
        self, data: np.ndarray, best_guess: Optional[float] = None
    ) -> Tuple[List[float], List[float], List[np.ndarray], np.ndarray]:
        """Optimize epsilon parameter using EnKF with forecast model.

        Parameters:
        -----------
        data : np.ndarray
            The input data for mapper analysis
        best_guess : Optional[float]
            Initial best guess for epsilon. If provided, the ensemble will be
            initialized around this value. If None, uniform initialization.

        Returns:
        --------
        Tuple containing:
            - mean_eps_path: List of mean epsilon values at each iteration
            - similarity_path: List of mean similarity values at each iteration
            - all_eps_history: List of all epsilon ensembles at each iteration
            - eps_ensemble: Final epsilon ensemble
        """
        if best_guess is None:
            eps_ensemble = np.random.uniform(
                self.config.eps_min, self.config.eps_max, self.config.ensemble_size
            )
        else:
            std_dev = (self.config.eps_max - self.config.eps_min) / 10
            eps_ensemble = np.clip(
                np.random.normal(best_guess, std_dev, self.config.ensemble_size),
                self.config.eps_min,
                self.config.eps_max,
            )

        mean_eps_path = [np.mean(eps_ensemble)]
        similarity_path = []
        all_eps_history = [copy.deepcopy(eps_ensemble)]

        for iteration in tqdm(range(self.config.max_iterations)):
            results = self._process_ensemble_batch(data, eps_ensemble)

            similarities = np.array([r[0] for r in results])
            gradients = np.array([r[1] for r in results])
            similarity_path.append(np.mean(similarities))

            eps_ensemble = self._update_ensemble(eps_ensemble, similarities, data)

            mean_eps_path.append(np.mean(eps_ensemble))
            all_eps_history.append(copy.deepcopy(eps_ensemble))

            if self._check_convergence(mean_eps_path, iteration):
                print(f"Converged after {iteration+1} iterations")
                break

        return mean_eps_path, similarity_path, all_eps_history, eps_ensemble

    def _process_ensemble_batch(
        self, data: np.ndarray, eps_ensemble: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Process ensemble members in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_ensemble_member, data, eps)
                for eps in eps_ensemble
            ]
            return [future.result() for future in futures]

    def _process_ensemble_member(
        self, data: np.ndarray, eps: float
    ) -> Tuple[float, float]:
        """Process a single ensemble member."""
        current_result = self.run_mapper(data, eps)

        eps_plus = min(eps + self.config.gradient_delta, self.config.eps_max)
        plus_result = self.run_mapper(data, eps_plus)

        eps_minus = max(eps - self.config.gradient_delta, self.config.eps_min)
        minus_result = self.run_mapper(data, eps_minus)

        plus_similarity = self.calculate_similarity(current_result, plus_result, data)
        minus_similarity = self.calculate_similarity(current_result, minus_result, data)
        me_similarity = 1

        gradient = (plus_similarity - minus_similarity) / (
            2 * self.config.gradient_delta
        )

        return me_similarity, gradient

    def _update_ensemble(
        self, eps_ensemble: np.ndarray, similarities: np.ndarray, data: np.ndarray
    ):
        forecasted_eps, expected_stability_change = self._forecast_epsilon_evolution(
            data, eps_ensemble
        )

        X_forecast = forecasted_eps.reshape(-1, 1)
        Y_obs = similarities.reshape(-1, 1)

        # EnKF Update
        X_mean = np.mean(X_forecast, axis=0, keepdims=True)
        Y_mean = np.mean(Y_obs, axis=0, keepdims=True)

        X_anomalies = X_forecast - X_mean
        Y_anomalies = Y_obs - Y_mean

        # Covariance
        P_xy = (X_anomalies.T @ Y_anomalies) / (self.config.ensemble_size - 1)
        P_yy = (Y_anomalies.T @ Y_anomalies) / (
            self.config.ensemble_size - 1
        ) + self.config.observation_noise

        # Kalman Gain
        K = P_xy @ np.linalg.inv(P_yy)

        expected_stability = Y_obs + expected_stability_change.reshape(-1, 1)

        target_similarity = np.ones_like(Y_obs)

        innovation = target_similarity - expected_stability

        X_updated = X_forecast + (K @ innovation.T).T

        eps_updated = np.clip(
            X_updated.flatten(), self.config.eps_min, self.config.eps_max
        )

        # aid exploration
        if np.random.random() < self.config.exploration_probability:
            eps_updated += np.random.normal(
                0, self.config.exploration_stddev, self.config.ensemble_size
            )
            eps_updated = np.clip(eps_updated, self.config.eps_min, self.config.eps_max)

        return eps_updated

    def _check_convergence(self, mean_eps_path: List[float], iteration: int) -> bool:
        return (
            iteration > 3
            and np.std(mean_eps_path[-3:]) < self.config.convergence_threshold
        )

    def identify_stable_regions(
        self, eps_ensemble: np.ndarray, data: np.ndarray
    ) -> List[StableRegion]:
        """Identify stable regions in the epsilon parameter space."""
        X = eps_ensemble.reshape(-1, 1)

        cluster_centers = self._find_optimal_clusters(X)

        region_results = self._evaluate_regions_batch(cluster_centers, data)

        stable_regions = [
            StableRegion(
                eps=eps,
                similarity=similarity,
                stability=stability,
                score=stability,
                result=result,
            )
            for eps, similarity, stability, result in region_results
        ]

        stable_regions.sort(key=lambda x: x.score, reverse=True)

        return stable_regions

    def _evaluate_regions_batch(
        self, cluster_centers: np.ndarray, data: np.ndarray
    ) -> List[Tuple[float, float, float, Any]]:
        """Evaluate regions in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_region, eps, data)
                for eps in cluster_centers
            ]
            return [future.result() for future in futures]

    def _find_optimal_clusters(self, X: np.ndarray) -> np.ndarray:
        best_score = -1
        best_kmeans = None

        for n_clusters in range(
            self.config.min_clusters + 1, min(self.config.max_clusters + 1, len(X))
        ):
            if n_clusters <= 1:
                continue

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.graph_layout_seed,
                n_init=10,
            ).fit(X)

            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(X, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans

        # if silhouette scoring fails, use density
        if best_kmeans is None:
            return self._find_clusters_by_density(X)
        else:
            return best_kmeans.cluster_centers_.flatten()

    def _find_clusters_by_density(self, X: np.ndarray) -> np.ndarray:
        """Find clusters using kernel density estimation."""
        kde = gaussian_kde(X.flatten())
        x_grid = np.linspace(self.config.eps_min, self.config.eps_max, 1000)
        density = kde(x_grid)

        peaks, _ = find_peaks(density)

        if len(peaks) == 0:
            return np.array([np.min(X), np.median(X), np.max(X)])
        else:
            return x_grid[peaks]

    def _evaluate_region(
        self, eps: float, data: np.ndarray
    ) -> Tuple[float, float, float, Any]:
        result = self.run_mapper(data, eps)
        self_similarity = 1.0

        stability = self._calculate_region_stability(eps, result, data)

        return eps, self_similarity, stability, result

    def _calculate_region_stability(
        self, eps: float, result: Any, data: np.ndarray
    ) -> float:
        nearby_epsilons = np.random.normal(
            loc=eps, scale=self.config.neighbor_stddev, size=self.config.k_neighbors
        )

        nearby_epsilons = np.clip(
            nearby_epsilons, self.config.eps_min, self.config.eps_max
        )
        nearby_similarities = []

        for nearby_ep in nearby_epsilons:
            try:
                nearby_result = self.run_mapper(data, nearby_ep)
                nearby_sim = self.calculate_similarity(result, nearby_result, data)
                nearby_similarities.append(nearby_sim)
            except Exception as e:
                print(f"Error evaluating nearby epsilon {nearby_ep:.4f}: {e}")

        return np.mean(nearby_similarities) if nearby_similarities else 0

    def visualize_stable_regions(
        self, stable_regions: List[StableRegion], data: np.ndarray
    ) -> None:
        self._create_summary_plot(stable_regions)

        for i, region in enumerate(stable_regions):
            self._visualize_region(region, data, i)

    def _visualize_region(
        self, region: StableRegion, data: np.ndarray, index: int
    ) -> None:
        projection = data[:, 0]

        cover_scheme = self._get_cover_scheme(region, data)

        plot_data_with_mapper_graph(
            data=data,
            projection=projection,
            result=region.result,
            title=f"Region {index+1}: ε={region.eps:.3f}, Score={region.score:.3f}",
            cover_scheme=self.precover,
        )

        plt.savefig(
            f"{self.output_dir}/graph_eps_{region.eps:.3f}.svg",
            format="svg",
        )
        plt.close()

    def _get_cover_scheme(
        self, region: StableRegion, data: np.ndarray
    ) -> Optional[Any]:
        if self.precover is not None:
            return self.precover
        elif hasattr(region.result, "cover_scheme"):
            return region.result.cover_scheme
        elif hasattr(region.result, "cover") and hasattr(
            region.result.cover, "intervals"
        ):
            return region.result.cover
        elif hasattr(region.result, "intervals"):
            return region.result
        return None

    def visualize_optimization_path(
        self,
        mean_eps_path: List[float],
        similarity_path: List[float],
        all_eps_history: List[np.ndarray],
    ) -> None:
        self._create_optimization_plot(mean_eps_path, all_eps_history)

    def _create_optimization_plot(
        self, mean_eps_path: List[float], all_eps_history: List[np.ndarray]
    ) -> None:
        """Create plot showing eps vs iteration."""
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = np.arange(len(mean_eps_path))
        ax.plot(iterations, mean_eps_path, "b-", linewidth=2, label="Mean ε", alpha=0.9)

        for i in range(self.config.ensemble_size):
            member_path = [history[i] for history in all_eps_history]
            ax.plot(iterations, member_path, "b-", alpha=0.2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("ε value")
        ax.set_title("Optimization Path")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/eps_optimization_path.svg", format="svg")
        plt.close()


class DataGenerator:
    @staticmethod
    def generate_circle(
        num_points: int,
        noise_level: float,
        radius: float,
        center_x: float,
        center_y: float,
    ) -> np.ndarray:
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
        """Generate test data consisting of three circles."""
        circle1 = DataGenerator.generate_circle(400, 0.05, 1, 0, 0)
        circle2 = DataGenerator.generate_circle(200, 0.025, 0.5, -1.5, 0)
        circle3 = DataGenerator.generate_circle(800, 0.1, 2, 0, 3)
        return np.vstack((circle1, circle2, circle3))

    @staticmethod
    def load_csv_data(filepath: str, x_col: str = "x", y_col: str = "y") -> np.ndarray:
        """
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        x_col : str
            Name of the column to use as x coordinate
        y_col : str
            Name of the column to use as y coordinate

        Returns:
        --------
        np.ndarray
            2D array with x and y coordinates
        """
        try:
            df = pd.read_csv(filepath)
            return df[[x_col, y_col]].values
        except Exception as e:
            print(f"Error loading CSV data: {e}. Using default test data.")
            return DataGenerator.generate_test_data()


def main(data_source="synthetic", best_guess=None):
    """
    Parameters:
    -----------
    data_source : str
        Source data to analyze. Options: "synthetic", "csv"
    best_guess : Optional[float]
        Initial best guess for epsilon. If None, uniform initialization is used.
    """
    config = Config()

    if data_source == "synthetic":
        data = DataGenerator.generate_test_data()
    elif data_source == "csv":
        data = DataGenerator.load_csv_data(
            filepath="your_data.csv", x_col="x", y_col="y"
        )
    else:
        print(f"Unknown data source: {data_source}. Using default synthetic test data.")
        data = DataGenerator.generate_test_data()

    analyzer = MapperAnalysis(config)

    analyzer.precompute_cover(data)

    mean_eps_path, similarity_path, all_eps_history, final_ensemble = (
        analyzer.optimize_epsilon(data, best_guess)
    )

    analyzer.visualize_optimization_path(
        mean_eps_path, similarity_path, all_eps_history
    )

    stable_regions = analyzer.identify_stable_regions(final_ensemble, data)

    print("\n=== STABLE REGIONS IDENTIFIED ===")
    for i, region in enumerate(stable_regions):
        print(f"Region {i+1}:")
        print(f"  Epsilon: {region.eps:.4f}")
        print(f"  Stability: {region.stability:.4f}")
        print(f"  Combined Score: {region.score:.4f}")

    analyzer.visualize_stable_regions(stable_regions, data)


if __name__ == "__main__":

    # main(data_source="synthetic", best_guess=0.24)
    main(data_source="synthetic")
