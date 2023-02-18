import numpy as np

from clustering.__cluster_base import __ClusterBase


class MeanShift(__ClusterBase):
    """
    The Mean Shift clustering algorithm.
    """

    def __init__(self):
        super(MeanShift, self).__init__()
        self.

    def fit(self, X: np.ndarray, iter_max: int = 100) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
