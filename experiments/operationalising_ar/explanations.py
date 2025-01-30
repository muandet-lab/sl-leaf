import torch
from torch import optim
from tqdm import tqdm


def gen_ce(xb: torch.Tensor, g: torch.nn.Module, n_epochs: int, progress_bar: tqdm = None) \
        -> torch.Tensor:
    """
    To generate counterfactual explanations.
    """

    # init with xr=xb
    xr = xb.detach().clone().requires_grad_()

    optimizer = optim.Adam(params=[xr], lr=0.001)
    for epoch in range(n_epochs):
        yhr = g(xr)
        ce_loss = torch.mean(yhr) + torch.mean((xb - xr) ** 2)

        # Backpropagation and optimization
        optimizer.zero_grad()  # Zero the gradients for all parameters
        ce_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        # Update the progress bar with the current loss
        if progress_bar is not None:
            progress_bar.set_postfix({"CE loss": ce_loss.item()})
            progress_bar.update(1)

    return xr
