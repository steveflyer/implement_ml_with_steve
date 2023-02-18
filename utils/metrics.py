import numpy as np


class LpMetric:
    """
    The Lp metric.
    """
    def __init__(self, p: int = 2):
        """
        :param p: (int) the order of the Lp metric
        """
        self.p = p

    # Calculate the Lp distance between two vectors.
    def distance(self, u: np.ndarray, v: np.ndarray):
        """
        Calculate the Lp distance between two vectors.

        :param u: (np.ndarray) shape (n_features,) or (n_samples, n_features)
        :param v: (np.ndarray) shape (n_features,) or (n_samples, n_features)
        :return: (float) or (np.ndarray) shape (n_samples,)
        """
        return self.norm(u - v)

    # Calculate the Lp norm of a vector.
    def norm(self, u: np.ndarray):
        """
        Calculate the Lp norm of a vector.

        :param u: (np.ndarray) vector, shape (n_features,) or (n_samples, n_features)
        :return: (float) or (np.ndarray) shape (n_samples,)
        """
        if len(u.shape < 2):
            return np.sum(np.abs(u) ** self.p) ** (1 / self.p)
        else:
            return np.sum(np.abs(u) ** self.p, axis=1) ** (1 / self.p)


euclidean_distance = LpMetric(p=2).norm
