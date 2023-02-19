import numpy as np

from regression.linear.__linear_regression_base import __LinearRegressionBase


class LeastSquareRegression(__LinearRegressionBase):
    """
    Least square regression model.
    
    The fitting process is to calculate the pseudo-inverse of the input data(`X`), and thenmultiply it with the target data(`y`).
    
             $w = (X^T X X^T) Y$
    
    The mathematical derivation can refer to [this blog](google.com).
    the following two lines are equivalent
    self.w = np.linalg.pinv(X) @ y
    """
    def fit(self, X, y):
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
