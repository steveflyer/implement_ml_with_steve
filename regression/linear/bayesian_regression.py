import numpy as np
from numpy.random import multivariate_normal

from regression.linear.__linear_regression_base import __LinearRegressionBase


class BayesianRegression(__LinearRegressionBase):
    """
    Bayesian regression model.

    Bayesian regression update the posterior distribution of w after each training.
    The posterior distribution of w is a Gaussian distribution with mean `self.mu` and
    variance `self.cov`.
    """

    def __init__(self, alpha: float = 1.0, noise_cov: float = 1.,
                 sigma_2: float = 1., mu: float = 0.):
        """
        :param alpha: (float) the initialized precision of the prior distribution of w
        :param noise_cov: (float) the variance of the observation noise
        :param sigma_2: (float) the variance of the posterior distribution of w
        :param mu: (float) the mean of the posterior distribution of w
        """
        super().__init__()
        self.alpha = alpha
        """(float) the initialized precision of the prior distribution of w"""
        self.noise_sigma_2 = noise_cov
        """(float) the variance of the observation noise"""
        self.cov = sigma_2
        """(float) the variance of the posterior distribution of w"""
        self.mu = mu
        """(float) the mean of the posterior distribution of w"""

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # calculate the posterior distribution of w
        self.cov = 1 / (self.alpha + n_samples / self.noise_sigma_2)
        self.mu = self.cov * (self.alpha * self.mu + X.T @ y / self.noise_sigma_2)
        # sample from the posterior distribution of w
        self.w = np.random.normal(self.mu, self.cov, (n_features, 1))

    def predict(self, X: np.ndarray, mode: str = 'mean'):
        """Predict the target values for the given input data using Bayesian Regression.

        Bayesian regression prediction is rather special. Since we get a distribution
        rather than a precise fixed value of weights after training. So we can either
        sample from the posterior distribution of w or use the mean of it to make the
        prediction.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :param mode: (str) 'mean' or 'sample', 'mean' means to return the mean of the posterior distribution of w,
        :return: (np.ndarray) shape (n_samples, n_targets)
        """
        if mode == 'mean':
            return X @ self.mu
        elif mode == 'sample':
            w = self.__sample()
            return X @ w
        else:
            raise NotImplemented("mode should be 'mean' or 'sample'")

    def __sample(self) -> np.ndarray:
        """Sample from the posterior distribution of w.

        :return: sampled_w: (np.ndarray) shape (n_features, n_targets)
        """
        return multivariate_normal(self.mu, self.cov)
