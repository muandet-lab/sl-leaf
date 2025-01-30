from typing import Callable

import torch
import torch.nn as nn

from experiments.operationalising_ar.agents import AgentsPool


def query_strategic_nmse(ap: AgentsPool, g: torch.nn.Module, xr: torch.Tensor, h: Callable):
    """
    Deploy g and xr to observe agents' reaction and their outcomes,
    then compute the prediction errors.
    """
    # init
    u = torch.from_numpy(ap.u)
    x, w = ap.best_respond(g=g, xr=xr)

    # compute normalised MSE loss
    yh = g(x).squeeze(-1)
    y = h(torch.cat((x, u.unsqueeze(1)), dim=1))
    nmse = nn.MSELoss()(yh, y) / ap.norm_const

    compliance_ratio = w.detach().numpy().mean()

    return nmse, compliance_ratio
