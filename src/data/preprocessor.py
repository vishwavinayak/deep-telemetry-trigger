from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        if train_data is None:
            raise ValueError("train_data cannot be None")
        if train_data.ndim != 2:
            raise ValueError(f"train_data must be 2D, got shape {train_data.shape}")

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(train_data)
        logger.info("Fitted scaler on train data with shape %s", train_data.shape)
        return scaled

    def transform(self, test_data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit_transform first.")
        if test_data is None:
            raise ValueError("test_data cannot be None")
        if test_data.ndim != 2:
            raise ValueError(f"test_data must be 2D, got shape {test_data.shape}")

        transformed = self.scaler.transform(test_data)
        logger.info("Transformed test data with shape %s", test_data.shape)
        return transformed

    @staticmethod
    def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
        if data is None:
            raise ValueError("data cannot be None")
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {data.shape}")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if data.shape[0] < window_size:
            raise ValueError(
                f"data length {data.shape[0]} is smaller than window_size {window_size}"
            )

        n_samples = data.shape[0] - window_size + 1
        windows = np.empty((n_samples, window_size, data.shape[1]), dtype=data.dtype)

        for start_idx in range(n_samples):
            end_idx = start_idx + window_size
            windows[start_idx] = data[start_idx:end_idx]

        logger.info(
            "Created %s windows of size %s from input shape %s",
            n_samples,
            window_size,
            data.shape,
        )
        return windows
