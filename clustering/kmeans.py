import numpy as np

from clustering.__cluster_base import __ClusterBase
from utils.metrics import LpMetric


class KMeans(__ClusterBase):
    """
    The K-Means clustering algorithm.
    """

    def __init__(self, n_clusters: int = 2, init: str = "kmeans++", metric=LpMetric(p=2)):
        """
        :param n_clusters: (int) the number of clusters
        :param init: (str) the method to initialize the cluster centers
        :param metric: the metric to calculate the distance between two vectors
        """
        super(KMeans, self).__init__()
        self.init_mode = init
        """(str) the method to initialize the cluster centers"""
        self.n_clusters = n_clusters
        """(int) the number of clusters"""
        self.metric = metric
        """the metric to calculate the distance between two vectors"""

    # Clustering given input data using K-Means algorithm.
    def fit(self, X: np.ndarray, iter_max: int = 100, eps: float = 1e-5) -> np.ndarray:
        """Clustering given input data using K-Means algorithm.

        :param X: (np.ndarray) input points, shape (n_samples, n_features)
        :param iter_max: (int) the maximum number of iterations
        :param eps: (float) the threshold to determine convergence
        :return: (np.ndarray) the cluster labels of each sample, shape (n_samples,)
        """
        # Initialize the cluster centers.
        self.__init_centers(X)
        # Iterate until convergence.
        for iter_num in range(iter_max):
            old_centers = self._centers
            z = self.__cluster_assign(X)
            self.__update_centers(X, z)
            if self.metric.distance(old_centers, self._centers) < eps:
                # if converged, break.
                break
        # Calculate labels according to label assignment matrix
        labels = np.argmax(z, axis=1)
        print(f'KMeans Fitting finished in {iter_num} iters.')
        return labels

    # Predict the cluster labels for the given data.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the cluster labels for the given data.

        :param X: (np.ndarray) input points, shape (n_samples, n_features)
        :return: (np.ndarray) the cluster labels of each sample, shape (n_samples,)
        """
        z = self.__cluster_assign(X)
        return np.argmax(z, axis=1)

    # Assign samples to clusters
    def __cluster_assign(self, X) -> np.ndarray:
        """
        Assign each sample to the nearest cluster center.
        :param X: (np.ndarray) input points, shape (n_samples, n_features)
        :return: (np.ndarray) cluster assignment matrix, shape (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        # Initialize the cluster assignment matrix.
        z = np.zeros((n_samples, self.n_clusters))

        # for each sample, assign it to the nearest cluster.
        for i in range(n_samples):
            distances = self.metric.distance(X[i], self._centers)
            z[i, np.argmin(distances)] = 1
        return z

    # Update the cluster centers.
    def __update_centers(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Update the cluster centers.

        :param X: (np.ndarray) input points, shape (n_samples, n_features)
        :param z: (np.ndarray) cluster assignment matrix, shape (n_samples, n_clusters)
        :return: (np.ndarray) updated cluster centers, shape (n_clusters, n_features)
        """
        # Update the cluster centers.
        for k in range(self.n_clusters):
            self._centers[k] = np.sum(X * z[:, k].reshape(-1, 1), axis=0) / np.sum(z[:, k])
        return self._centers

    # Initialize the cluster centers.
    def __init_centers(self, X: np.ndarray = None) -> np.ndarray:
        """
        Initialize the cluster centers using given init mode.
        :param X: (np.ndarray) inputs, shape (n_samples, n_features)
        :return: (np.ndarray) centers, shape (n_clusters, n_features)
        """
        if self.init_mode == "uniform":
            # Initialize the cluster centers uniformly at random.
            ranges = np.hstack((np.min(X, axis=0).reshape(-1, 1), np.max(X, axis=0).reshape(-1, 1)))
            centers = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(self.n_clusters, X.shape[1]))
            self._centers = centers
        elif self.init_mode == "kmeans++":
            # Initialize the cluster centers using the k-means++ algorithm.
            raise NotImplementedError(f"The initialization method {self.init_mode} is not implemented yet.")
        return self._centers
