import cvxpy as cp

from regression.linear.__linear_regression_base import __LinearRegressionBase


class LassoRegression(__LinearRegressionBase):
    """
    Lasso regression model.

    The fitting process is to solve the following optimization problem.

             $$minimize_{w} ||Xw - y||_2^2 + \lambda ||w||_1$$

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
        w = cp.Variable((n_features, 1))
        objective = cp.Minimize(cp.sum_squares(X @ w - y) + self.lam * cp.norm(w, 1))
        prob = cp.Problem(objective)
        prob.solve()
        self.w = w.value
