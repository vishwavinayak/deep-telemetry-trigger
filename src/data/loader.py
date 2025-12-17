from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration for model/data settings."""
    if not config_path.exists():
        logger.warning(
            "Config file %s not found; using defaults where possible", config_path
        )
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read config %s", config_path, exc_info=exc)
        return {}

    if not isinstance(data, dict):
        logger.error("Config file %s is malformed; expected a mapping", config_path)
        return {}

    return data


class SMAPLoader:
    def __init__(
        self,
        base_path: Path | str,
        channel_id: str = "T-1",
        config_path: Path | str = "config/model.yaml",
    ) -> None:
        self.config = load_model_config(Path(config_path))
        self.base_path = Path(base_path)
        self.channel_id = channel_id

        if not self.base_path.exists():
            logger.warning("Base path %s does not exist yet", self.base_path)

    def _load_split(self, split: str) -> np.ndarray:
        file_path = self.base_path / split / f"{self.channel_id}.npy"
        logger.debug("Attempting to load %s", file_path)

        if not file_path.exists():
            message = f"Data file not found: {file_path}"
            logger.error(message)
            raise FileNotFoundError(message)

        try:
            data = np.load(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load %s", file_path)
            raise

        if data.ndim < 2:
            message = (
                f"Expected at least 2D array for {file_path}, got shape {data.shape}"
            )
            logger.error(message)
            raise ValueError(message)

        # Keep only the telemetry signal (first column) and preserve 2D shape
        data = data[:, 0:1]
        logger.info(
            "Reducing features to 1 (Telemetry only). New shape: %s", data.shape
        )

        logger.info("Loaded %s with shape %s", file_path, data.shape)
        return data

    def load_train(self) -> np.ndarray:
        return self._load_split("train")

    def load_test(self) -> np.ndarray:
        return self._load_split("test")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config_path = Path("config/model.yaml")
    config = load_model_config(config_path)
    data_config: Dict[str, Any] = (
        config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    )

    base_path: Path = Path(data_config.get("raw_path", "data/raw"))
    channel: str = data_config.get("channel_id", "T-1")

    loader = SMAPLoader(
        base_path=base_path, channel_id=channel, config_path=config_path
    )

    try:
        train_array = loader.load_train()
        logger.info("Train array shape: %s", train_array.shape)
        logger.info("First 5 rows:\n%s", train_array[:5])
    except Exception as exc:  # noqa: BLE001
        logger.error("Verification run failed", exc_info=exc)
