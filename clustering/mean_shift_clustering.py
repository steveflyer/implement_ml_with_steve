import numpy as np
import warnings

from clustering.__cluster_base import __ClusterBase
from clustering.mean_shift import MeanShift
from utils.metrics import L1Metric


class MeanShiftClustering(__ClusterBase):
    """
    The Mean Shift clustering algorithm.

    Mean Shift clustering is like K Nearest Neighbors clustering:

    - The number of clusters cannot be specified in advance. We adjust the bandwidth of the kernel to get the desired number of clusters. The larger bandwidth, the fewer clusters.
    - The model has no explicit fitting stage, it just gives predictions of labels of a given dataset.

    In this implementation, `predict` just calls ``

    """

    def __init__(self, bandwidth: float = 1.0, metric=L1Metric):
        """
        :param bandwidth: (float) the bandwidth of the kernel
        :param metric: the metric used for calculating the distance between two points
        """
        super(MeanShiftClustering, self).__init__()
        self.bandwidth = bandwidth
        """(float) the bandwidth of the kernel"""
        self.metric = metric
        """the metric used for calculating the distance between two points"""
        self._mean_shift = MeanShift(bandwidth=bandwidth, metric=metric)
        """(MeanShift) the inner mean shift model"""

    def fit(self, X: np.ndarray, iter_max: int = 100, iter_eps: float = 1e-5, merge_eps: float = 1e-2) -> np.ndarray:
        """
        Performing Mean Shift Clustering on given dataset.
        Calling `fit` on MeanShiftClustering is not recommended.
        Please call `predict` predict directly.

        :param X: (np.ndarray) shape (n_samples, n_features) input data
        :param iter_max:  (int) the maximum number of iterations
        :param iter_eps:  (float) the convergence threshold for mean shift
        :param merge_eps: (float) the threshold for merging two centers
        :return: (np.ndarray) shape (n_samples,) the cluster labels of each sample
        """
        warnings.warn("Calling `fit` on MeanShiftClustering is not recommended. Please call `predict` predict directly.")
        return self.predict(X, iter_max=iter_max, iter_eps=iter_eps, merge_eps=merge_eps)

    def predict(self, X: np.ndarray, iter_max: int = 100, iter_eps: float = 1e-5, merge_eps: float = 1e-2) -> np.ndarray:
        """
        Predict the cluster labels of given dataset.

        :param X: (np.ndarray) shape (n_samples, n_features) input data
        :param iter_max: (int) the maximum number of iterations
        :param iter_eps: (float) the convergence threshold for mean shift
        :param merge_eps: (float) the threshold for merging two centers
        :return: (np.ndarray) shape (n_samples,) the cluster labels of each sample
        """
        # take every point as an initial center, running mean shift algorithm
        n_samples, n_features = X.shape
        centers = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            centers[i] = self._mean_shift.fit(X, X[i], iter_max=iter_max, eps=iter_eps)

        # merge centers, assign labels
        self._centers, labels = self._merge_centers(centers, merge_eps)
        return labels

    def _merge_centers(self, centers: np.ndarray, eps: float) -> np.ndarray:
        """
        Merge centers that are close enough.

        :param centers: (np.ndarray) shape (n_centers, n_features) the centers
        :param eps: (float) the threshold for merging two centers
        :return: (np.ndarray, np.ndarray) the merged centers(shape (n_merged_centers, n_features))
                 and labels of each center(shape (n_merged_centers,))
        """
        # `cores` is the list of merged centers, we name it `core` for convenient differentiation
        cores = [centers[0]]
        # `labels` is the assigned labels for all centers
        labels = np.zeros(len(centers))
        for i, center in enumerate(centers[1:]):
            center = np.tile(center.reshape(1, -1), (len(cores), 1))
            # calculate the distance between the current center and all cores
            dists = self.metric.distance(cores, center)  # shape (num_merged_centers, )
            min_dist, min_core_idx = np.min(dists)
            # if it has a close enough core, then assign the center to that core cluster
            if min_dist < eps:
                labels[i+1] = min_core_idx
            # else, add the current center to the core list `cores`, assign the center to its own cluster
            else:
                labels[i+1] = len(cores)
                cores.append(center)
        return cores, labels
