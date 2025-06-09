import torch
from torch.utils.data import Dataset

from ..environment.environment import Environment

class CustomFunctionDataset(Dataset):
    """
    Generates (temperature, curve) pairs by calling a user‐provided function:
        curve_fn(time_tensor, temperature) → Tensor of shape (T,)
    """

    def __init__(
        self,
        T: int,
        num_examples: int,
        temp_min: float,
        temp_max: float,
        curve_fn,
    ):
        """
        Args:
          - T            : number of time steps per curve.
          - num_examples : size of dataset.
          - temp_min, temp_max : range of random temperatures.
          - curve_fn     : user‐supplied function f(time_tensor, temperature) → Tensor(T,)
        """
        super().__init__()
        self.T = T
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.curve_fn = curve_fn

        # Randomly sample temperatures in [temp_min, temp_max]
        self.temperatures = temp_min + (temp_max - temp_min) * torch.rand(num_examples)
        
        # Precompute the time grid once
        self.t = torch.arange(0, T, dtype=torch.float32)  # shape (T,)

        # Precompute all curves so __getitem__ is cheap
        self.curves = []
        print("Building Dataset")
        for i in range(num_examples):
            temp_val = float(self.temperatures[i].item())
            curve = self.curve_fn(self.t, temp_val)
            if not (isinstance(curve, torch.Tensor) and curve.shape == (T,)):
                raise ValueError(
                    f"curve_fn must return a torch.Tensor of shape ({T},). "
                    f"Got {type(curve)} with shape {curve.shape}."
                )
            self.curves.append(curve)

    def __len__(self):
        return len(self.temperatures)

    def __getitem__(self, idx):
        env_tensor = float(self.temperatures[idx].item())
        curve = self.curves[idx]  # Tensor of shape (T,)
        return env_tensor, curve
    

import numpy as np

class OfflineRBFInterpolationDataset(Dataset):
    """
    Loads precomputed (temperature, curve) pairs from an offline .npz dataset file.
    
    Expects the .npz file to contain:
      - 'times': 1D array of time steps, shape (T,)
      - 'temperatures': 1D array of sampled temperatures, shape (num_examples,)
      - 'curves': 2D array of curves, shape (num_examples, T)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.times = data['times']        # shape (T,)
        self.temperatures = data['temperatures']  # shape (num_examples,)
        self.curves = data['curves']      # shape (num_examples, T)

        # Convert to torch tensors for convenience
        self.temperatures = torch.from_numpy(self.temperatures).float()
        self.curves = torch.from_numpy(self.curves).float()

    def __len__(self):
        return len(self.temperatures)

    def __getitem__(self, idx):
        temp = self.temperatures[idx]  # scalar tensor
        curve = self.curves[idx]       # tensor shape (T,)
        return temp, curve

