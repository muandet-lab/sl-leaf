from typing import Tuple, Union, Callable

import numpy as np
import torch


class AgentsPool:
    """
    To manage a fixed set of agents.
    """

    def __init__(self, n: int, d: int, h: Callable = None):
        """
        Supplying `h` implies computing a specific normalisation constant.
        """

        xb, cf, u = AgentsPool.gen_continuous_agents(n=n, d=d)

        # assignment
        self.xb = xb
        self.cf = cf
        self.u = u
        self.norm_const = 1.0

        # update norm_const if necessary
        if h is not None:
            self.set_norm_const(h=h)

        return

    def best_respond(self, g: torch.nn.Module, xr: Union[np.ndarray, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reaction model defined for only 1D outputs: yhb and yhr.
        """
        # init
        xb = torch.from_numpy(self.xb)
        cf = torch.from_numpy(self.cf)

        if isinstance(xr, np.ndarray):
            xr = torch.from_numpy(xr)
        elif not isinstance(xr, torch.Tensor):
            raise ValueError("xr must be either a tensor or a numpy array.")

        yhb = g(xb).squeeze(-1)  # because 1D output
        yhr = g(xr).squeeze(-1)

        n, d = xb.shape
        assert (n, d) == xr.shape
        assert (n,) == yhb.shape
        assert (n,) == yhr.shape

        x, w = AgentsPool.do_best_respond(xb=xb, yhb=yhb, cf=cf, xr=xr, yhr=yhr)

        return x, w

    def set_norm_const(self, h: Callable):
        xb = torch.from_numpy(self.xb)
        u = torch.from_numpy(self.u)

        y = h(torch.cat((xb, u.unsqueeze(1)), dim=1))
        self.norm_const: float = torch.mean(y ** 2).item()

        return

    @staticmethod
    def gen_continuous_agents(n: int, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert d > 0

        # unobserved confounders
        u = np.random.randint(low=0, high=4, size=n)  # in {0,1,2,3}

        # shift x and c, according to u.
        x = (
                np.random.multivariate_normal(mean=[10] * d, cov=np.diag([2] * d), size=n)
                + np.tile(u.reshape(n, 1), (1, d))
        )

        # draw cost factors, conditioned on u
        cf = (
                0.01 * np.abs(np.random.normal(loc=2, scale=1, size=n)) +
                0.1 * u
        )

        return x, cf, u

    @staticmethod
    def do_best_respond(xb: torch.Tensor, yhb: torch.Tensor, cf: torch.Tensor,
                        xr: torch.Tensor, yhr: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """

        n, d = xb.shape
        assert (n,) == cf.shape

        delta = xr - xb
        c: torch.Tensor = cf * torch.sum(delta ** 2, dim=1)  # (n,)
        w: torch.Tensor = ((yhr + c) <= yhb)  # (n,)
        x = xb + torch.reshape(w, (-1, 1)) * delta
        return x, w
