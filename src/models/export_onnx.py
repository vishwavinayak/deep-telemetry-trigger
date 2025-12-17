from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

from src.models.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def export() -> None:
    config_path = Path("config/model.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = yaml.safe_load(config_path.read_text())

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    input_dim = int(model_cfg.get("input_dim", 1))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    latent_dim = int(model_cfg.get("latent_dim", 20))
    dropout = float(model_cfg.get("dropout", 0.0))
    window_size = int(data_cfg.get("window_size", 100))

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    model_path = Path(training_cfg.get("save_dir", "models")) / "lstm_autoencoder.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, window_size, input_dim, dtype=torch.float32)
    out_path = Path(training_cfg.get("save_dir", "models")) / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes=None,
    )

    logger.info(
        "Success: Model exported to %s. Ready for FPGA synthesis tools like hls4ml.",
        out_path,
    )


if __name__ == "__main__":
    export()
