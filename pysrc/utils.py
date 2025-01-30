import numpy as np


def np_remove_from(arr_2d: np.ndarray, arr_1d: np.ndarray):
    n, d = arr_2d.shape
    assert (d,) == arr_1d.shape
    return arr_2d[~np.all(arr_2d == arr_1d, axis=1)]
