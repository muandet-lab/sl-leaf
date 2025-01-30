from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from experiments.operationalising_ar.agents import AgentsPool
from experiments.operationalising_ar.dm import TrainedAgentModel, ARSampler
from experiments.operationalising_ar.eval import query_strategic_nmse
from experiments.operationalising_ar.explanations import gen_ce
from pysrc.models import ThreeLayerReLUNet


def pretrain_ar_policy_offline(sigma: ThreeLayerReLUNet, xb: torch.Tensor, n_epochs: int):
    """
    Try to match sigma(xb) and xb.
    """
    norm_const: float = torch.mean(xb ** 2).item()

    optimizer = optim.Adam(params=list(sigma.parameters()), lr=0.001)
    with tqdm(total=n_epochs, desc="Pre-training continuous sigma", unit="epoch",
              ncols=100) as pbar:
        for epoch in range(n_epochs):
            xr = sigma(xb)
            nmse = nn.MSELoss()(xb, xr) / norm_const

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients for all parameters
            nmse.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Update the progress bar with the current loss
            pbar.set_postfix({"xr loss": nmse.item()})
            pbar.update(1)
    return


def run_offline_training(ap: AgentsPool, g: torch.nn.Module, h: Callable,
                         n_epochs: int, stop_at: float = 0.0):
    """
    This does not consider strategic behaviour, it is meant to pretrain the model.
    """
    x = torch.from_numpy(ap.xb)
    u = torch.from_numpy(ap.u)

    # print(x.dtype)

    optimizer = optim.Adam(params=list(g.parameters()), lr=0.001)
    mse_fn = nn.MSELoss()

    with tqdm(total=n_epochs, desc="Training g", unit="epochs") as pbar:
        for epoch in range(n_epochs):
            yh = g(x).squeeze(-1)  # (n,1) to (n,)
            y = h(torch.cat((x, u.unsqueeze(1)), dim=1))

            # Compute loss
            nmse = mse_fn(yh, y) / ap.norm_const

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients for all parameters
            nmse.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Update the progress bar with the current loss
            pbar.set_postfix({"Loss": nmse.item()})
            pbar.update(1)

            if nmse.item() <= stop_at:
                break
    return


def run_rrm_with_ar(training_set: list[AgentsPool], test_pool: AgentsPool,
                    h: Callable, g: torch.nn.Module,
                    psi: TrainedAgentModel, sigma: torch.nn.Module,
                    n_epochs: int, xr_frequency: float = 0.1):
    """
    Agents do NOT best respond in the training set.
    This procedure only trains `g` & `sigma`, not `psi`.

    Steps:
    - Observe xb
    - Deploy g, sigma, xr
    - Observe x, y
    - (optional) Minimise (psi(xb,x)-1)^2 and (psi(xb,xr)-0)^2
    - Minimise (g(psi(xb, sigma(xb)))-y)^2
    """

    def _observe_outcomes(ap: AgentsPool, g: torch.nn.Module, xr: torch.Tensor) -> torch.Tensor:
        # init
        u = torch.from_numpy(ap.u)
        x, _ = ap.best_respond(g=g, xr=xr)

        # compute agents' outcomes
        y = h(torch.cat((x, u.unsqueeze(1)), dim=1))
        return y

    def _query_test_nmse(ap: AgentsPool, g: torch.nn.Module, sigma: torch.nn.Module):
        xr = sigma(torch.from_numpy(ap.xb))
        test_nmse, compliance_ratio = query_strategic_nmse(ap=test_pool, g=g, xr=xr, h=h)
        return test_nmse, compliance_ratio

    # before training, deploy `g` and `sigma`
    test_nmse, compliance_ratio = _query_test_nmse(ap=test_pool, g=g, sigma=sigma)
    print(f"Compliance ratio: {compliance_ratio}")
    print(f"Initial (test) nmse: {test_nmse}")

    # `joint` training
    g.train()
    sigma.train()

    # the outer loop (for the `repeated` part in RRM)
    n_rounds = len(training_set)
    training_losses = []
    test_losses = []
    with tqdm(total=n_rounds * n_epochs, desc="RRM", ncols=160) as pbar:
        for i in range(n_rounds):
            # init
            ap = training_set[i]
            xb = torch.from_numpy(ap.xb)

            # deploy `g` & `sigma` to observe y
            # IMPORTANT: `detach` is a must.
            xr = sigma(xb).detach()
            y = _observe_outcomes(ap=ap, g=g, xr=xr).detach()

            optimizer = optim.Adam(params=list(sigma.parameters()) + list(g.parameters()), lr=0.001)

            # the inner loop (to train `g` & `sigma` with ERM)
            for epoch in range(n_epochs):

                xr = xr.detach()  # detach when re-using `xr` in previous epochs.

                # how frequent a new `xr` is generated.
                if epoch == 0 or epoch % int(1 / xr_frequency) == 0:
                    xr = sigma(xb)

                # simulate agents' responses
                gain = (g(xr) - g(xb))
                joint_input = torch.concatenate((xb, xr, gain), dim=1)
                wh = psi.classify(joint_input)
                xh = xb * (1 - wh) + xr * wh

                # compute offline nmse
                yh = g(xh).squeeze()
                nmse = nn.MSELoss()(yh, y) / ap.norm_const

                # Backpropagation and optimization
                optimizer.zero_grad()  # Zero the gradients for all parameters
                nmse.backward()  # Compute gradients
                optimizer.step()  # Update weights

                # Update the progress bar with the current loss
                postfix = OrderedDict([
                    ("round", i), ("epoch", epoch),
                    ("training_loss", f"{nmse.item():.6f}"),
                    ("training_compliance_ratio", f"{wh.mean():.6f}"),
                    ("test_loss", f"{test_nmse.item():.6f}")
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)

                training_losses.append(nmse.item())

            # update test nmse after each round (of ERM)
            test_nmse, _ = _query_test_nmse(ap=test_pool, g=g, sigma=sigma)
            test_losses.append(test_nmse.item())

    # after training, deploy sigma
    test_nmse, compliance_ratio = _query_test_nmse(ap=test_pool, g=g, sigma=sigma)
    print(f"Compliance ratio: {compliance_ratio}")
    print(f"Final (test) nmse: {test_nmse}")

    return training_losses, test_losses


def run_rrm_with_ce(training_set: list[AgentsPool], test_pool: AgentsPool,
                    h: Callable, g: torch.nn.Module,
                    psi: TrainedAgentModel,
                    n_rrm_epochs: int, n_ce_epochs: int, xr_frequency: float = 0.1):
    """
    A copy of `run_rrm()` for the case of counterfactual explanations.
    """

    def _observe_outcomes(ap: AgentsPool, g: torch.nn.Module, xr: torch.Tensor) -> torch.Tensor:
        # init
        u = torch.from_numpy(ap.u)
        x, _ = ap.best_respond(g=g, xr=xr)

        # compute agents' outcomes
        y = h(torch.cat((x, u.unsqueeze(1)), dim=1))
        return y

    def _query_test_nmse(ap: AgentsPool, g: torch.nn.Module):
        xr = gen_ce(xb=torch.from_numpy(ap.xb), g=g, n_epochs=n_ce_epochs)
        test_nmse, compliance_ratio = query_strategic_nmse(ap=test_pool, g=g, xr=xr, h=h)
        return test_nmse, compliance_ratio

    # before training, deploy `g` and `sigma`
    test_nmse, compliance_ratio = _query_test_nmse(ap=test_pool, g=g)
    print(f"Compliance ratio: {compliance_ratio}")
    print(f"Initial (test) nmse: {test_nmse}")

    # `joint` training
    g.train()

    # the outer loop (for the `repeated` part in RRM)
    n_rounds = len(training_set)
    training_losses = []
    test_losses = []
    with tqdm(total=(n_rounds * n_rrm_epochs), desc="RRM", ncols=160) as pbar:
        for i in range(n_rounds):
            # init
            ap = training_set[i]
            xb = torch.from_numpy(ap.xb)

            # deploy `g` & `sigma` to observe y
            # IMPORTANT: `detach` is a must.
            xr = gen_ce(xb=xb, g=g, n_epochs=n_ce_epochs).detach()
            y = _observe_outcomes(ap=ap, g=g, xr=xr).detach()

            optimizer = optim.Adam(params=list(g.parameters()), lr=0.001)

            # the inner loop (to train `g` & `sigma` with ERM)
            for epoch in range(n_rrm_epochs):

                xr = xr.detach()  # detach when re-using `xr` in previous epochs.

                # how frequent a new `xr` is generated.
                if epoch == 0 or epoch % int(1 / xr_frequency) == 0:
                    xr = gen_ce(xb=xb, g=g, n_epochs=n_ce_epochs)

                # simulate agents' responses
                gain = (g(xr) - g(xb))
                joint_input = torch.concatenate((xb, xr, gain), dim=1)
                wh = psi.classify(joint_input)
                xh = xb * (1 - wh) + xr * wh

                # compute offline nmse
                yh = g(xh).squeeze()
                nmse = nn.MSELoss()(yh, y) / ap.norm_const

                # Backpropagation and optimization
                optimizer.zero_grad()  # Zero the gradients for all parameters
                nmse.backward()  # Compute gradients
                optimizer.step()  # Update weights

                # Update the progress bar with the current loss
                postfix = OrderedDict([
                    ("round", i), ("epoch", epoch),
                    ("training_loss", f"{nmse.item():.6f}"),
                    ("training_compliance_ratio", f"{wh.mean():.6f}"),
                    ("test_loss", f"{test_nmse.item():.6f}")
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)

                training_losses.append(nmse.item())

            # update test nmse after each round (of ERM)
            test_nmse, _ = _query_test_nmse(ap=test_pool, g=g)
            test_losses.append(test_nmse.item())

    # after training, deploy sigma
    test_nmse, compliance_ratio = _query_test_nmse(ap=test_pool, g=g)
    print(f"Compliance ratio: {compliance_ratio}")
    print(f"Final (test) nmse: {test_nmse}")

    return training_losses, test_losses


def train_ar_response_model(training_pool: AgentsPool, test_pool: AgentsPool,
                            g: torch.nn.Module, pi: ARSampler, psi: TrainedAgentModel,
                            n_epochs: int):
    """
    """

    def _gen_joint_input(xb: np.ndarray, xr: np.ndarray, g: torch.nn.Module) -> torch.Tensor:
        """ For `psi` """
        xb = torch.from_numpy(xb)
        xr = torch.from_numpy(xr)
        gain = (g(xr) - g(xb))

        joint_input = torch.concatenate((xb, xr, gain), dim=1)  # (n,2d+1)
        return joint_input

    def _query_compliance(ap: AgentsPool, xr: np.ndarray) -> torch.Tensor:
        _, w = ap.best_respond(g=g, xr=xr)
        assert w.ndim == 1
        return w.type(torch.double)  # (n,)

    def _safe_np_mean(arr: np.ndarray):
        return np.nan if arr.size == 0 else np.mean(arr)

    def _get_accuracies(pred: np.ndarray, truth: np.ndarray):
        corrects: np.ndarray[bool] = np.equal(pred, truth)
        overall_acc = corrects.mean()
        acc_0 = _safe_np_mean(corrects[(truth == 0)])
        acc_1 = _safe_np_mean(corrects[(truth == 1)])
        compliance_ratio = truth.mean()
        return overall_acc, acc_0, acc_1, compliance_ratio

    # generate test data
    test_xr = pi.gen_mixed_recommendations(xb=test_pool.xb, g=g)
    test_joint_input = _gen_joint_input(xb=test_pool.xb, xr=test_xr, g=g)
    test_compliance = _query_compliance(ap=test_pool, xr=test_xr).detach().numpy()

    # before training
    pred_compliance = psi.classify(test_joint_input).squeeze().detach().numpy()
    overall_acc, acc_0, acc_1, compliance_ratio = _get_accuracies(pred=pred_compliance,
                                                                  truth=test_compliance)
    print(
        f"""Initial (test) accuracies: 
            overall = {overall_acc:.4f}, 
            acc at `0` = {acc_0:.4f}, 
            acc at `1` = {acc_1:.4f},
            label ratio = {compliance_ratio:.2f}"""
    )

    # training
    psi.model.train()
    optimizer = optim.Adam(params=list(psi.model.parameters()), lr=0.001)

    loss_histories = []
    with tqdm(total=n_epochs, desc="Training psi", unit="epochs", ncols=110) as pbar:
        xr = pi.gen_mixed_recommendations(xb=training_pool.xb, g=g)
        joint_input = _gen_joint_input(xb=training_pool.xb, xr=xr, g=g).detach()
        true_compliance = _query_compliance(ap=training_pool, xr=xr).detach()

        for epoch in range(n_epochs):
            pred_probs = psi.predict_probs(joint_input).squeeze()
            bce_loss = nn.BCELoss()(pred_probs, true_compliance)

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients for all parameters
            bce_loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Update the progress bar with the current loss
            pred_compliance = psi.to_classes(pred_probs)
            acc, _, _, label_ratio = _get_accuracies(pred_compliance.detach().numpy(),
                                                     true_compliance.detach().numpy())
            pbar.set_postfix(loss=f"{bce_loss.item():.4f}",
                             accuracy=f"{acc:.4f}",
                             label_ratio=f"{label_ratio:.2f}")
            pbar.update(1)

            loss_histories.append(bce_loss.item())

    # after training
    pred_compliance = psi.classify(test_joint_input).squeeze().detach().numpy()
    overall_acc, acc_0, acc_1, compliance_ratio = _get_accuracies(pred=pred_compliance,
                                                                  truth=test_compliance)
    print(
        f"""Final (test) accuracies: 
            overall = {overall_acc:.4f}, 
            acc at `0` = {acc_0:.4f}, 
            acc at `1` = {acc_1:.4f},
            label ratio = {compliance_ratio:.2f}"""
    )

    return
