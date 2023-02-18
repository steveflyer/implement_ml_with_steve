import numpy as np

from clustering.__cluster_base import __ClusterBase


class GMMClustering(__ClusterBase):
    """
    The Gaussian Mixture Model clustering algorithm.
    """

    def __init__(self, n_clusters: int = 2, init: str = "kmeans++"):
        super(GMMClustering, self).__init__()


    def fit(self, X: np.ndarray, iter_max: int = 100, eps : float = 1e-5) -> None:


    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
