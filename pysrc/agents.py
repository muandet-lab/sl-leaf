from typing import Tuple

import numpy as np


def gen_agents(n: int, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        n (int): number of data points.
        d (int): length of covariate vector.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): covariates of size (n,d), cost factors of size (n,),
                                              and unobserved variables of size (n,).
    """
    assert d > 0

    # unobserved confounders
    u = np.random.randint(low=0, high=4, size=n)  # in {0,1,2,3}

    # shift x and c, according to u.
    x = (
            np.random.multivariate_normal(mean=[10] * d, cov=np.diag([2] * d), size=n)
            + np.tile(u.reshape(n, 1), (1, d))
    )

    cf = 0.01 * np.abs(np.random.normal(loc=2, scale=1, size=n) + u)  # cost factors

    return x, cf, u


def does_comply(xb: np.ndarray, yhb: np.ndarray, cf: np.ndarray, xr: np.ndarray, yhr: np.ndarray) \
        -> np.ndarray[bool]:
    """
    In the minimisation problem.
    """
    # agents' evaluating the hypothetical moves
    delta = xr - xb
    c = cf * np.square(delta).sum(axis=1)  # quadratic cost
    z = (yhr + c < yhb)  # compliance behaviour
    return z


def respond(xb: np.ndarray, yhb: np.ndarray, cf: np.ndarray, xr: np.ndarray, yhr: np.ndarray) \
        -> np.ndarray:
    """
    Args:
        xb (np.ndarray): base covariates of size (n,d).
        yhb (np.ndarray): base predictions of size (n,).
        cf (np.ndarray): cost factors of size (n,).
        xr (np.ndarray): recommended covariates of size (n,d).
        yhr (np.ndarray): predictions w.r.t. recommended covariates, of size (n,)

    Returns:
        (np.ndarray): the updated covariates, of size (n,d).
    """
    n, d = xb.shape
    assert n >= d, "There should be more data points than number of features"

    # agents' evaluating the hypothetical moves
    delta = xr - xb
    z = does_comply(xb=xb, yhb=yhb, cf=cf, xr=xr, yhr=yhr)  # compliance behaviour

    # agents' best responses
    delta = z.reshape(n, 1) * delta
    x = xb + delta

    return x
