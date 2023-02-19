import numpy as np

from clustering.__cluster_base import __ClusterBase
from clustering.gmm import GaussianMixture


class GMMClustering(__ClusterBase):
    """
    The Gaussian Mixture Model clustering algorithm.
    """

    def __init__(self, n_clusters: int = 2, init_mode: str = "uniform"):
        """
        :param n_clusters: (int) the number of clusters
        :param init_mode: (str) the initialization mode, default is "uniform", set to "manual" if you'd like to set parameters manually
        """
        super(GMMClustering, self).__init__()
        self.n_clusters = n_clusters
        """(int) the number of clusters"""
        self.init_mode = init_mode
        """(str) the initialization mode"""
        self._gmm_model = GaussianMixture(n_components=n_clusters, init_mode=init_mode)
        """(GaussianMixtureModel) the underlying Gaussian Mixture Model"""
        self._centers = self._gmm_model.means
        """(np.ndarray) shape (n_clusters, n_features) the centers of each cluster"""

    # Fitting a clustering model use EM-GMM
    def fit(self, X: np.ndarray, iter_max: int = 100, eps : float = 1e-5) -> None:
        """
        Fitting a clustering model use EM-GMM. The training in done on the inner GaussianMixtureModel.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :param iter_max: (int) the maximum number of iterations
        :param eps: (float) the threshold of convergence
        :return: (None)
        """
        self._gmm_model.fit(X, iter_max, eps)
        self._centers = self._gmm_model.means

    # Predict the cluster index for each sample based on the maximum pdf component
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster index for each sample.
        The prediction is based on argmax over the pdf of each component.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :return: (np.ndarray) shape (n_samples,) the cluster index for each sample
        """
        return np.argmax(self._gmm_model.pdf_components(X), axis=1)

    # Set Gaussian parameters manually: means and covariances.
    def set_params(self, means: np.ndarray, covariances: np.ndarray) -> None:
        """
        Set Gaussian parameters manually: means and covariances.
        This function will be called successfully only when `self.init_mode` is `manual`

        :param means: (np.ndarray) means
        :param covariances: (np.ndarray) covariances
        :return: (None)
        """
        self._gmm_model.set_params(means, covariances)
        self._centers = self._gmm_model.means

