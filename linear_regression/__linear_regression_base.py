import numpy as np
from abc import abstractmethod


class __LinearRegressionBase:
    """
    Base class for linear regression models.
    """
    def __init__(self):
        self.w = None
        """(np.ndarray) the weight, shape (n_features, n_targets)"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the given input data.

        :param X: (np.ndarray) shape (n_samples, n_features)
        :param y: (np.ndarray) shape (n_samples, n_targets)
        :return:
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values for the given input data.For
        this linear regression model, the prediction is simply the
        dot product of the input data(`X`) and the weight(`self.w`).

        :param X: (np.ndarray) shape (n_samples, n_features)
        :return: (np.ndarray) shape (n_samples, n_targets)
        """
        if self.w is None:
            raise Exception("the model has not been trained yet, self.w is None")
        else:
            return X @ self.w
