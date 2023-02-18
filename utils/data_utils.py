import numpy as np
from pathlib import Path

# configurations
__DATA_DIR_PATH = Path(__file__).parent.parent / 'data'

TOY_POLYDATA_PATH = __DATA_DIR_PATH / 'linear_regression/PA-1-data-text'


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
