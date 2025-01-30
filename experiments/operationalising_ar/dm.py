from typing import Callable

import numpy as np
import torch.nn

from experiments.operationalising_ar.explanations import gen_ce


class ARSampler:
    def __init__(self, local_sigma: torch.nn.Module = None, global_sigma: torch.nn.Module = None):
        self.local_sigma = local_sigma
        self.global_sigma = global_sigma
        return

    def gen_mixed_recommendations(self, xb: np.ndarray, g: torch.nn.Module) -> np.ndarray:
        """
        `g` is only needed for generating counterfactual explanations.
        """

        # generate different types of recommendations
        xr1 = self.gen_base_recommendations(xb=xb)
        xr2 = self.gen_local_recommendations(xb=xb)
        xr3 = self.gen_global_recommendations(xb=xb)
        xr4 = self.gen_counterfactual_explanations(xb=xb, g=g, n_epochs=10)

        n, _ = xb.shape
        indices = [0, int(n / 4), int(2 * n / 4), int(3 * n / 4), n]

        # mix the recommendations together.
        xr = xr1
        xr[indices[1]:indices[2]] = xr2[indices[1]:indices[2]]
        xr[indices[2]:indices[3]] = xr3[indices[2]:indices[3]]
        xr[indices[3]:indices[4]] = xr4[indices[3]:indices[4]]

        return xr

    def gen_global_recommendations(self, xb: np.ndarray) -> np.ndarray:
        # init
        sigma = self.global_sigma
        assert sigma is not None

        n, d = xb.shape
        delta = np.random.normal(loc=0, scale=2, size=n * d).reshape(n, d)
        xr = sigma(torch.from_numpy(xb)).detach().numpy() + delta

        return xr

    def gen_local_recommendations(self, xb: np.ndarray) -> np.ndarray:
        # init
        sigma = self.local_sigma
        assert sigma is not None

        n, d = xb.shape
        delta = np.random.normal(loc=0, scale=2, size=n * d).reshape(n, d)
        xr = sigma(torch.from_numpy(xb)).detach().numpy() + delta

        return xr

    @staticmethod
    def gen_base_recommendations(xb: np.ndarray) -> np.ndarray:
        # generate xr around xb
        n, d = xb.shape
        delta = np.random.normal(loc=0, scale=2, size=n * d).reshape(n, d)
        xr = xb + delta
        return xr

    @staticmethod
    def gen_counterfactual_explanations(xb: np.ndarray, g: torch.nn.Module, n_epochs: int) \
            -> np.ndarray:
        xb = torch.from_numpy(xb)
        xr = gen_ce(xb=xb, g=g, n_epochs=n_epochs).detach().numpy()
        return xr


# def gen_nearby_recommendations(xb: np.ndarray, sigma: torch.nn.Module) -> np.ndarray:
#     # generate xr around xb
#     n, d = xb.shape
#     delta = np.random.normal(loc=0, scale=2, size=n * d).reshape(n, d)
#     xr = xb + delta
#
#     # generate xr around sigma(xb)
#     if sigma is not None:
#         xr_2 = sigma(torch.from_numpy(xb)).detach().numpy() + delta
#         idx = int(n / 2)
#         xr[idx:-1] = xr_2[idx:-1]
#
#     return xr


class TrainedAgentModel:
    """
    A DM's model of the agents' behaviour.
    A binary-classification wrapper for torch models.
    """

    def __init__(self, probabilistic_model: torch.nn.Module, binary_threshold: float = 0.5):
        self.model = probabilistic_model
        self.threshold = binary_threshold
        return

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        predicted_probs = self.predict_probs(x)
        predicted_classes = self.to_classes(predicted_probs)
        return predicted_classes

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def to_classes(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 2:
            n, d = probs.shape
            assert d == 1

        predicted_classes: torch.Tensor = (probs >= self.threshold).double()
        return predicted_classes
