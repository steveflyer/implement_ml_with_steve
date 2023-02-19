import numpy as np
from abc import abstractmethod


class __ClusterBase:
    """
    The base class for clustering algorithms.
    """

    def __init__(self):
        self.n_clusters = None
        """(int) the number of clusters"""
        self._centers = None
        """(np.ndarray) shape (n_clusters, n_features) the cluster centers"""

    @abstractmethod
    def fit(self, X: np.ndarray, iter_max: int = 100) -> None:
        """Fit the model to the given data.

        For most algorithms with an explicit fitting process, this method
        returns nothing and just updates the inner parameters ready for
        prediction.

        But some KNN-like algorithm has no explicit fitting process.
        For compatibility, we still keep this method, but just call
        `predict` method in it and gives a warning.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :param iter_max: (int) the maximum number of iterations
        :return: (None)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the cluster labels for the given data.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :return: (np.ndarray) shape (n_samples,) the cluster labels of each sample
        """
        pass
