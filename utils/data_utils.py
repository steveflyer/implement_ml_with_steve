import numpy as np
from pathlib import Path

# configurations
__DATA_DIR_PATH = Path(__file__).parent.parent / 'data'

TOY_POLYDATA_PATH = __DATA_DIR_PATH / 'linear_regression/poly_data'
TOY_CLUSTER_DATA_DIR = __DATA_DIR_PATH / 'clustering/toy_cluster'


# get toy clustering data
def get_cluster_toy_data():
    """
    Get the toy clustering data.

    :return: (list, np.ndarray) the data list and the label set
    """
    CLUSTER_DATA_A_X_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataA_X.txt'
    CLUSTER_DATA_A_Y_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataA_Y.txt'
    CLUSTER_DATA_B_X_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataB_X.txt'
    CLUSTER_DATA_B_Y_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataB_Y.txt'
    CLUSTER_DATA_C_X_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataC_X.txt'
    CLUSTER_DATA_C_Y_PATH = TOY_CLUSTER_DATA_DIR / 'cluster_data_dataC_Y.txt'
    CLUSTER_A_X = load_mat(CLUSTER_DATA_A_X_PATH)
    CLUSTER_B_X = load_mat(CLUSTER_DATA_B_X_PATH)
    CLUSTER_C_X = load_mat(CLUSTER_DATA_C_X_PATH)
    CLUSTER_A_Y = load_mat(CLUSTER_DATA_A_Y_PATH).reshape(-1)
    CLUSTER_B_Y = load_mat(CLUSTER_DATA_B_Y_PATH).reshape(-1)
    CLUSTER_C_Y = load_mat(CLUSTER_DATA_C_Y_PATH).reshape(-1)
    CLUSTER_DATA_LIST = [[CLUSTER_A_X, CLUSTER_A_Y],
                         [CLUSTER_B_X, CLUSTER_B_Y],
                         [CLUSTER_C_X, CLUSTER_C_Y]]
    return CLUSTER_DATA_LIST, np.array([0, 1, 2, 3])


# get toy polynomial data
def get_poly_observation_data():
    """
    Get the toy polynomial data.
    :return: (np.ndarray, np.ndarray) the feature and target
    """
    # read polynomial data
    sampx_path = TOY_POLYDATA_PATH / 'polydata_data_sampx.txt'
    sampy_path = TOY_POLYDATA_PATH / 'polydata_data_sampy.txt'
    sampx = load_mat(sampx_path)
    sampy = load_mat(sampy_path).reshape(-1, 1)
    return sampx, sampy


def get_poly_plot_data():
    """
    Get the toy polynomial data.
    :return: (np.ndarray, np.ndarray) the feature and target
    """
    # read polynomial data
    plotx_path = TOY_POLYDATA_PATH / 'polydata_data_polyx.txt'
    ploty_path = TOY_POLYDATA_PATH / 'polydata_data_polyy.txt'
    plotx = load_mat(plotx_path)
    ploty = load_mat(ploty_path).reshape(-1, 1)
    return plotx, ploty


# load matrix from txt file
def load_mat(path: str):
    """
    read matrix from txt file, return a numpy.ndarray
    @param
      filepath: (str) the path txt file containing matrix information
    @return
      mat: (numpy.ndarray)
    """
    with open(path, 'r') as f:
        vec_list = []
        for line in f.readlines():
            numstr_list = line.split('\t')
            num_list = [float(numstr.strip()) for numstr in numstr_list if numstr.strip() != '']
            vec_list.append(np.array(num_list).reshape(-1, 1))
        mat = np.hstack(vec_list)
    return mat
