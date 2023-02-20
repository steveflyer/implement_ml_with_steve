import numpy as np
from scipy import stats

from utils.metrics import LpMetric


class KNearestNeighbors:
    """
    The base KNearestNeighbor class.

    KNearestNeighbor is a non-linear model, which make predictions
    based on neighborhood.

    It has a lazy fitting period, which just stores the training
    data.

    For prediction, it just gives result by:

      1. Calculate the distance between the eval point and the training set
      2. Retrieve the neighborhood
      3. Use the neighborhood to vote for the prediction
    """
    def __init__(self, k: int, metric=LpMetric(p=2), classification: bool = True):
        """
        :param k: (int) the number of neighbors
        :param metric: the metric to calculate the distance between two vectors
        :param classification: (bool) true if used for classification
        """
        self.n_neighbors = k
        """(int) the number of neighbors"""
        self._X = None
        """(np.ndarray) shape (n_train_samples, n_features) the training data"""
        self._labels = None
        """(np.ndarray) shape (n_train_samples,) the labels of the training data"""
        self._metric = metric
        """the metric to calculate the distance between two vectors"""
        self.classification = classification

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
        if self.classification:
            # classification
            result = self.__majority_vote(neighbors)
        else:
            # regression
            result = self.__average_vote(neighbors)
        return result

    def __majority_vote(self, neighbors: np.ndarray):
        """
        Output vote based on neighborhood majority vote.

        :param neighbors: (np.ndarray) shape (n_samples, n_neighbors)
        :return: (np.array) prediction, shape (n_samples,)
        """
        return np.array([stats.mode(self._labels[ns], keepdims=False)[0] for ns in neighbors])

    def __average_vote(self, neighbors: np.ndarray):
        """
        Output the neighborhood average.

        :param neighbors: (np.ndarray) shape (n_samples, n_neighbors)
        :return: (np.ndarray) prediction, shape (n_samples,)
        """
        return np.array([np.mean(self._labels[ns]) for ns in neighbors])



