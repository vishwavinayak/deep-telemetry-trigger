from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    def __init__(self, windows: np.ndarray) -> None:
        if windows is None:
            raise ValueError("windows cannot be None")
        if windows.ndim != 3:
            raise ValueError(f"windows must be 3D, got shape {windows.shape}")
        if windows.shape[0] == 0:
            raise ValueError("windows must contain at least one sample")

        self.windows = torch.from_numpy(windows.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        return self.windows[index]


# Alias to align with training pipeline terminology
TimeSeriesDataset = WindowedDataset
