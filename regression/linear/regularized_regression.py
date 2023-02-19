import numpy as np

from regression.linear.__linear_regression_base import __LinearRegressionBase


class RegularizedRegression(__LinearRegressionBase):
    """
    Regularized regression model.

    The fitting process is to solve the following optimization problem.

             $$minimize_{w} ||Xw - y||_2^2 + \lambda ||w||_2^2$$

    It can be solved analytically by calculating the modified pseudo-inverse
    of X and then multiplying it with y:

             $$w = (X^T X + \lambda I)^{-1} X^T y$$

    The mathematical derivation can refer to [this blog](google.com).
    """

    def __init__(self, labda: float = 1.0):
        """
        :param labda: (float) the regularization parameter, should be non-negative
        """
        super().__init__()
        self.labda = labda
        """(float) the regularization parameter"""

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # calculate the modified pseudo-inverse of X
        regularized_pinv = np.linalg.inv(X.T @ X + self.labda * np.eye(n_features)) @ X.T
        self.w = regularized_pinv @ y
