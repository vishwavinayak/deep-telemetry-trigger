from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TimeSeriesDataset
from src.data.loader import SMAPLoader
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    return config


def prepare_data(config: Dict[str, Any]) -> Tuple[DataLoader, DataPreprocessor, int]:
    data_cfg: Dict[str, Any] = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    training_cfg: Dict[str, Any] = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}

    base_path = Path(data_cfg.get("raw_path", "data/raw"))
    channel_id = str(data_cfg.get("channel_id", "T-1"))
    window_size = int(data_cfg.get("window_size", 100))
    batch_size = int(training_cfg.get("batch_size", 64))

    loader = SMAPLoader(base_path=base_path, channel_id=channel_id)
    train_data = loader.load_train()

    preprocessor = DataPreprocessor()
    scaled_train = preprocessor.fit_transform(train_data)
    train_windows = preprocessor.create_windows(scaled_train, window_size)

    dataset = TimeSeriesDataset(train_windows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader, preprocessor, train_data.shape[1]


def prepare_model(config: Dict[str, Any], input_dim: int) -> LSTMAutoencoder:
    model_cfg: Dict[str, Any] = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}

    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    latent_dim = int(model_cfg.get("latent_dim", 10))
    dropout = float(model_cfg.get("dropout", 0.0))

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    logger.info(
        "Initialized LSTMAutoencoder with input_dim=%s, hidden_dim=%s, latent_dim=%s, dropout=%s",
        input_dim,
        hidden_dim,
        latent_dim,
        dropout,
    )
    return model


def train(config: Dict[str, Any]) -> None:
    training_cfg: Dict[str, Any] = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}

    dataloader, preprocessor, input_dim = prepare_data(config)
    model = prepare_model(config, input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    epochs = int(training_cfg.get("epochs", 50))

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting training for %s epochs on device %s", epochs, device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        logger.info("Epoch %d/%d - Loss: %.6f", epoch, epochs, avg_loss)

    save_cfg: Dict[str, Any] = training_cfg
    save_dir = Path(save_cfg.get("save_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "lstm_autoencoder.pth"
    scaler_path = save_dir / "scaler.pkl"

    torch.save(model.state_dict(), model_path)
    joblib.dump(preprocessor.scaler, scaler_path)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved scaler to %s", scaler_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for SMAP data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/model.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()

    try:
        cfg = load_config(args.config)
        train(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        raise SystemExit(1) from exc
