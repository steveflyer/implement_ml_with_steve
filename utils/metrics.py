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
        if len(u.shape) < 2:
            u = u.reshape(1, -1)
        if len(v.shape) < 2:
            v = v.reshape(1, -1)
        dist = self.__distance(u, v)
        return dist

    def __distance(self, u, v):
        """
        Calculate distances between `n_a` vectors and `n_b` vectors

        :param u: shape (n_a, n_feat)
        :param v: shape (n_b, n_feat)
        :return: distances, shape (n_a, n_b)
        """
        u_tile = np.tile(np.expand_dims(u, 0), (len(v), 1, 1,)).swapaxes(0, 1)
        v_tile = np.tile(np.expand_dims(v, 0), (len(u), 1, 1,))
        return self.norm(u_tile - v_tile)

    # Calculate the Lp norm of a vector.
    def norm(self, u: np.ndarray):
        """
        Calculate the Lp norm of a vector.

        :param u: (np.ndarray) vector, shape (n_features,) or (n_samples, n_features)
        :return: (float) or (np.ndarray) shape (n_samples,)
        """
        if len(u.shape) < 2:
            return np.sum(np.abs(u) ** self.p) ** (1 / self.p)
        else:
            return np.sum(np.abs(u) ** self.p, axis=-1) ** (1 / self.p)


# common metric shortcuts
euclidean_norm = LpMetric(p=2).norm
mean_square_error = LpMetric(p=2).distance
mean_absolute_error = LpMetric(p=1).distance

L1Metric = LpMetric(p=1)
L2Metric = LpMetric(p=2)