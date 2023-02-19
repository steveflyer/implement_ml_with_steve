import numpy as np
from scipy.stats import multivariate_normal

from utils.metrics import euclidean_norm


class GaussianMixture:
    """
    The Gaussian Mixture Model.

    Gaussian Mixture is a probabilistic model that assumes all the data points are generated
    from a mixture of a finite number of Gaussian distributions with unknown parameters.

        $$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

    Where:
    - $pi_k$ is the weight of the k-th Gaussian distribution, i.e. self._weights in the impl
    - $\mu_k$ is the mean of the k-th Gaussian distribution, i.e. self._means in the impl
    - $\Sigma_k$ is the covariance of the k-th Gaussian distribution, i.e. self._covariances in the impl
    """
    def __init__(self, n_components: int = 1, init_mode: str = "uniform"):
        """
        :param n_components: (int) the number of Gaussian components
        :param init_mode: (str) the initialization mode, default is "uniform", set to "manual" if you'd like to set parameters manually
        """
        self.n_components = n_components
        """(int) the number of components"""
        self._weights = None
        """(np.ndarray) shape (n_components,) the weights of each component"""
        self.means = None
        """(np.ndarray) shape (n_components, n_features) the means of each component"""
        self._covariances = None
        """(np.ndarray) shape (n_components, n_features, n_features) the covariances of each component"""
        self.init_mode = init_mode
        """(str) the initialization mode"""

    # Set Gaussian parameters manually: means and covariances.
    def set_params(self, means: np.ndarray, covariances: np.ndarray) -> None:
        """
        Set Gaussian parameters manually: means and covariances.
        This function will be called successfully only when `self.init_mode` is `manual`

        :param means: (np.ndarray) means
        :param covariances: (np.ndarray) covariances
        :return: (None)
        """
        if self.init_mode == "manual":
            self.means = means
            self._covariances = covariances
        else:
            raise ValueError(f"The initialization mode is not manually.")

    # Set the initialization mode.
    def set_init_mode(self, init_mode: str) -> None:
        """
        Set the initialization mode.

        :param init_mode: (str) the initialization mode
        :return: (None)
        """
        self.init_mode = init_mode

    # Fit the model to the given data.
    def fit(self, X: np.ndarray, iter_max: int = 100, eps: float = 1e-5) -> None:
        """
        Fit a Gaussian Mixture to the given data using EM Algorithm.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :param iter_max: (int) the maximum number of iterations
        :param eps: (float) the threshold of convergence
        :return: (None)
        """
        # Initialize the parameters.
        if self.init_mode != "manual":
            self.__init_params(X)

        # Expectation-Maximization algorithm.
        for iter_num in range(iter_max):
            # E-step: calculate the responsibility matrix.
            z = self.__e_step(X)
            # M-step: update the parameters.
            self.__m_step(X, z)
            # If converged, break.
            if euclidean_norm(self.means, self._means_old) < eps:
                break
        print(f'GMM Converged in {iter_num} iterations.')

    # Calculate the probability density of the given data.
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density of the given data.

        :param X: (np.ndarray) input data, shape (n_samples, n_features)
        :return: (np.ndarray) the probability density, shape (n_samples,)
        """
        return np.sum(self.pdf_components(X), axis=1)

    def pdf_components(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density of the given data on each Gaussian Component

        :param X: (np.ndarray) input data, shape (n_samples, n_features)
        :return: (np.ndarray) the component probability density list, shape (n_samples, n_components)
        """
        return np.vstack([self._weights[j] * multivariate_normal(X, self.means[j], self.covariances[j]) for j in range(self.n_components)])


    # Initialize the parameters: weights, means and covariances.
    def __init_params(self, X: np.ndarray = None) -> None:
        """
        Initialize the parameters: weights, means and covariances.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :return: (None)
        """
        _, n_features = X.shape
        # Initialize the parameters uniformly.
        if self.init_mode == "uniform":
            #   1. initialize weights
            self._weights = np.ones(self.n_components) / self.n_components
            #   2. initialize means
            ranges = np.hstack(np.min(X, axis=0).reshape(-1, 1), np.max(X, axis=0).reshape(-1, 1))
            self.means = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(self.n_components, n_features))
            #  3. initialize covariances
            covs = []
            for _ in range(self.n_components):
                covs.append(np.eye(n_features))
            self._covariances = np.stack(covs)
        # Other initialization modes.
        else:
            raise NotImplementedError(f"The initialization mode {self.init_mode} is not implemented.")

    # The E-step of the Expectation-Maximization algorithm.
    def __e_step(self, X: np.ndarray) -> np.ndarray:
        """
        The E-step of the Expectation-Maximization algorithm.
        Use the model parameters to calculate `z`, i.e. responsibility matrix.
        `z[i, j]` = 1 means the i-th sample belongs to the j-th component.

        :param X: (np.ndarray) input data, shape (n_samples, n_features)
        :return: (np.ndarray) the resp matrix, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        # initialize the resp matrix
        z = np.zeros((n_samples, self.n_components))
        # calculate the resp matrix
        for i in range(n_samples):
            for j in range(self.n_components):
                z[i, j] = self._weights[j] * self.pdf(X[i], self.means[j], self._covariances[j])
            z[i] /= np.sum(z[i])
        return z

    # The M-step of the Expectation-Maximization algorithm.
    def __m_step(self, X: np.ndarray, z: np.ndarray) -> None:
        """
        The M-step of the Expectation-Maximization algorithm.
        Use the resp matrix to update the model parameters.

        :param X: (np.ndarray) input data, shape (n_samples, n_features)
        :param z: (np.ndarray) the resp matrix, shape (n_samples, n_components)
        :return: (None)
        """
        n_samples, n_features = X.shape
        # update the weights
        self._weights = np.sum(z, axis=0) / n_samples
        # update the means
        for j in range(self.n_components):
            self.means[j] = np.sum([z[i, j] * X[i] for i in range(n_samples)], axis=0) / np.sum(z[:, j])
        # update the covariances
        for j in range(self.n_components):
            self._covariances[j] = np.sum([z[i, j] * np.outer(X[i] - self.means[j], X[i] - self.means[j])
                                           for i in range(n_samples)], axis=0) / np.sum(z[:, j])
