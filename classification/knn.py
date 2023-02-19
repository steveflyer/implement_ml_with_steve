import numpy as np
from scipy import stats

from utils.metrics import LpMetric


class KNearestNeighbors:
    def __init__(self, k: int, metric = LpMetric(p=2)):
        self.n_neighbors = k
        """(int) the number of neighbors"""
        self._X = None
        """(np.ndarray) shape (n_train_samples, n_features) the training data"""
        self._labels = None
        """(np.ndarray) shape (n_train_samples,) the labels of the training data"""
        self._metric = metric
        """the metric to calculate the distance between two vectors"""

    def fit(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        KNearestNeighbors is a lazy learner, so we just store the training data
        and labels.

        :param X: (np.ndarray) shape (n_train_samples, n_features) the training data
        :param labels: (np.ndarray) shape (n_train_samples,) the labels of the training data
        :return: None
        """
        self._X = X
        self._labels = labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data.

        For naive implementation, we just calculate `n_samples` x `n_train_samples`
        distances and argmin over axis 1.

        :param X: (np.ndarray) shape (n_samples, n_features) the evaluate data
        :return: (np.ndarray) shape (n_samples,) the predicted labels
        """
        dists = self._metric.distance(X, self._X)
        neighbors = dists.argsort()[:, :self.n_neighbors]   # the default axis is -1
        return np.array([stats.mode(self._labels[ns], keepdims=False)[0] for ns in neighbors])
