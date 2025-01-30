from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class NumpyModel(nn.Module, ABC):  # TODO: remove this?
    """
    A wrapper to switch between numpy.ndarray and torch.Tensor.
    """

    def vectorise(self) -> np.ndarray:
        vec = torch.cat([p.view(-1) for p in self.parameters()]).detach().cpu().numpy()
        return vec

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).to(torch.float32)
        y = self._do_forward(x).detach().cpu().numpy().flatten()
        return y

    @abstractmethod
    def _do_forward(self, x: torch.Tensor) -> torch.Tensor: pass


class UsefulModel(nn.Module, ABC):
    """
    Useful helper functions for torch-based models.
    """

    def reset_parameters(self):
        counts = [0, 0]
        for layer in self.children():
            counts[0] += 1
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                counts[1] += 1

        print(f"Reset {counts[1]} out of all {counts[0]} layers")
        return self

    def serialise(self):
        """
        Because torch complains about pickle.
        """
        m_class = self.__class__
        m_args = self.get_init_args()
        m_simplified_dict = {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in self.state_dict().items()
        }
        return m_class, m_args, m_simplified_dict

    @staticmethod
    def deserialise(m_class, m_args, m_simplified_dict):
        model: UsefulModel = m_class(*m_args)
        m_state_dict = {
            key: torch.tensor(value) if isinstance(value, np.ndarray) else value
            for key, value in m_simplified_dict.items()
        }
        model.load_state_dict(m_state_dict)
        return model

    @abstractmethod
    def get_init_args(self):
        pass


# Define the three-layer ReLU neural network
class ThreeLayerReLUNet(UsefulModel):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 is_classification: bool = False):
        super(ThreeLayerReLUNet, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.relu1 = nn.ReLU()  # ReLU activation for first layer

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.relu2 = nn.ReLU()  # ReLU activation for second layer

        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer

        if is_classification:
            self.sigmoid = nn.Sigmoid()

        # useful stuff
        self._init_args = (input_size, hidden_size1, hidden_size2, output_size, is_classification)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        out = self.fc1(x)  # First layer
        out = self.relu1(out)  # ReLU activation
        out = self.fc2(out)  # Second layer
        out = self.relu2(out)  # ReLU activation
        out = self.fc3(out)  # Output layer

        if hasattr(self, 'sigmoid'):
            out = self.sigmoid(out)

        return out

    def get_init_args(self):
        return self._init_args


class NLayerReLUNet(NumpyModel):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super(NLayerReLUNet, self).__init__()

        n = len(hidden_sizes)
        assert n >= 1

        # List of layers:
        # Input layer (input_size -> hidden_size)
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        # Hidden layers (hidden_size -> hidden_size)
        for i in range(1, n, 1):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())  # Activation function

        # Output layer (hidden_size -> output_size)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Use nn.Sequential to stack all the layers together
        self.model = nn.Sequential(*layers)

    def _do_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        return self.model(x)


class TrainableConstantNet(nn.Module):
    def __init__(self, initial_value: torch.Tensor):
        super(TrainableConstantNet, self).__init__()

        assert initial_value.ndim == 1

        self.trainable_const = nn.Parameter(initial_value.clone().detach())
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        out = self.trainable_const.repeat(n, 1)
        return out

    # class PolynomialFunction(NumpyModel):
#     def __init__(self, degree: int = None, coefficients: list = None):
#         super(PolynomialFunction, self).__init__()
#
#         # Initialize learnable parameters (coefficients) for the polynomial
#         if coefficients is not None:
#             self.coefficients = nn.Parameter(torch.tensor(coefficients, dtype=torch.float32))
#         elif degree is not None:
#             self.coefficients = nn.Parameter(torch.randn(degree + 1, dtype=torch.float32))
#         else:
#             raise ValueError("Either 'degree' or 'coefficients' must be provided.")
#
#     def _do_forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Compute the polynomial output
#         # Sum(coefficient * x^i for each degree i)
#         out: torch.Tensor = sum(c * x ** i for i, c in enumerate(self.coefficients))
#         return out
