import numpy as np

from pysrc.agents import respond
from pysrc.models import NumpyModel


def get_mse(yh: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Args:
        yh (np.ndarray): of size (n,).
        y (np.ndarray): of size (n,).
    Returns:
        (np.float64)
    """
    return np.square(yh - y).mean()


def evaluate(xb: np.ndarray, cf: np.ndarray, u: np.ndarray, xr: np.ndarray,
             true_model: NumpyModel, pred_model: NumpyModel) -> np.float64:
    """
    To evaluate 'pred_model' and 'xr', given a fixed set of agents.

    Args:
        xb (np.ndarray):
        cf (np.ndarray):
        u (np.ndarray):
        xr (np.ndarray):
        true_model (NumpyModel):
        pred_model (NumpyModel):

    Returns:
        (np.float64): the mse for a given pair of 'pred_model' and 'xr'.
    """

    n, d = xb.shape
    assert n >= d, "There should be more data points than number of features"

    # deploy g and xr to observe agents' responses.
    g = pred_model
    x = respond(xb=xb, yhb=g(xb), cf=cf, xr=xr, yhr=g(xr))
    xu = np.concatenate((x, u.reshape(n, 1)), axis=1)

    # observe agents' outcomes
    h = true_model
    return get_mse(yh=g(x), y=h(xu))
