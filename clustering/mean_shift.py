import numpy as np
from scipy.stats import multivariate_normal

from utils.metrics import L1Metric


class MeanShift:
    """
    The Mean Shift clustering algorithm.
    """

    def __init__(self, bandwidth: float = 1.0, metric=L1Metric):
        """
        :param bandwidth: (float) the bandwidth of the kernel
        :param metric: the metric used for calculating the distance between two points
        """
        assert bandwidth > 0, "bandwidth must be positive"

        super(MeanShift, self).__init__()
        self.bandwidth = bandwidth
        """(float) the bandwidth of the kernel"""
        self.metric = metric
        """(LpMetric) the metric used for calculating the distance between two points"""

    def fit(self, X: np.ndarray, x_0: np.ndarray, iter_max: int = 100, eps: float = 1e-5) -> np.ndarray:
        """
        Mean Shift fitting the dataset, initialized at center `x_0`

        :param X: (np.ndarray) shape (n_samples, n_features) input data
        :param x_0: (np.ndarray) shape (n_features,) the initial center
        :param iter_max: (int) the maximum number of iterations
        :param eps: (float) the convergence threshold
        :return: x_t_updated: (np.ndarray) shape (n_features,) the final center
        """
        assert iter_max > 0, "iter_max must be positive"
        assert eps > 0, "eps must be positive"

        x_t = x_0
        for i in range(iter_max):
            x_t_updated = self._update_x_t(x_t, X)
            if self.metric.distance(x_t_updated, x_t) < eps:
                break
            x_t = x_t_updated
        print(f'Mean Shift Converged in {i} iterations.')
        return x_t_updated

    def _update_x_t(self, x_t: np.ndarray, X: np.ndarray) -> np.ndarray:
        _, n_features = X.shape
        pdf_arr = multivariate_normal.pdf(X, mean=x_t, cov=np.eye(n_features) * self.bandwidth)
        nominator = np.sum(np.hstack([pdf_arr.reshape(-1, 1) for i in range(n_features)]) * X, axis=0)
        denominator = np.sum(pdf_arr)
        return nominator / denominator
