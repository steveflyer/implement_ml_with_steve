import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_data(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot the cluster data

    :param points: (np.ndarray) shape (n, 2)
    :param labels: (np.ndarray) shape (n,)
    :return: None
    """
    plt.figure()
    label_set = np.unique(labels)
    for label in label_set:
        subset = points[labels == label]
        plt.scatter(subset[:, 0], subset[:, 1], label=label)
    plt.legend()
