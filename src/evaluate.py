from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def build_ground_truth(
    label_path: Path, channel_id: str, series_length: int
) -> np.ndarray:
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    df = pd.read_csv(label_path)
    if df.empty:
        raise ValueError("Label CSV is empty")

    # Accept common column names
    chan_col = None
    for candidate in ("chan_id", "channel_id", "channel"):
        if candidate in df.columns:
            chan_col = candidate
            break
    if chan_col is None:
        raise ValueError(
            "Label CSV must contain a channel identifier column (chan_id/channel_id/channel)"
        )

    seq_col = "anomaly_sequences"
    if seq_col not in df.columns:
        raise ValueError("Label CSV must contain an 'anomaly_sequences' column")

    row = df[df[chan_col] == channel_id]
    if row.empty:
        raise ValueError(f"No label row found for channel {channel_id}")

    seq_str = row.iloc[0][seq_col]
    try:
        sequences: Sequence[Sequence[int]] = ast.literal_eval(seq_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed to parse anomaly_sequences for {channel_id}: {seq_str}"
        ) from exc

    y_true = np.zeros(series_length, dtype=int)
    for interval in sequences:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            logger.warning("Skipping malformed interval %s", interval)
            continue
        start, end = int(interval[0]), int(interval[1])
        start = max(0, start)
        end = min(series_length - 1, end)
        if start > end:
            logger.warning("Skipping interval with start > end: (%s, %s)", start, end)
            continue
        y_true[start : end + 1] = 1

    return y_true


def load_artifacts(config: Dict[str, Any]) -> Tuple[DataPreprocessor, Path]:
    model_cfg: Dict[str, Any] = (
        config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    )
    training_cfg: Dict[str, Any] = (
        config.get("training", {})
        if isinstance(config.get("training", {}), dict)
        else {}
    )

    save_dir = Path(training_cfg.get("save_dir", "models"))
    model_path = save_dir / "lstm_autoencoder.pth"
    scaler_path = save_dir / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    preprocessor = DataPreprocessor(scaler=scaler)

    return preprocessor, model_path


def compute_reconstruction_errors(
    dataloader: DataLoader, model: LSTMAutoencoder, device: torch.device
) -> np.ndarray:
    criterion = nn.MSELoss(reduction="none")
    all_errors: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss_per_sample = criterion(output, batch).mean(dim=(1, 2))
            all_errors.extend(loss_per_sample.detach().cpu().numpy().tolist())

    errors = np.asarray(all_errors)
    if errors.size == 0:
        raise ValueError(
            "No reconstruction errors computed; check input data and windowing"
        )
    return errors


def evaluate(config: Dict[str, Any]) -> None:
    data_cfg: Dict[str, Any] = (
        config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    )
    model_cfg: Dict[str, Any] = (
        config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    )
    training_cfg: Dict[str, Any] = (
        config.get("training", {})
        if isinstance(config.get("training", {}), dict)
        else {}
    )

    base_path = Path(data_cfg.get("raw_path", "data/raw"))
    channel_id = str(data_cfg.get("channel_id", "T-1"))
    window_size = int(data_cfg.get("window_size", 100))
    batch_size = int(training_cfg.get("batch_size", 64))

    loader = SMAPLoader(base_path=base_path, channel_id=channel_id)
    train_data = loader.load_train()
    test_data = loader.load_test()

    preprocessor, model_path = load_artifacts(config)

    # Build model and load weights (input_dim derived from sliced data, should be 1)
    input_dim = train_data.shape[1]
    if input_dim != 1:
        logger.warning("Expected telemetry-only input_dim=1, found %s", input_dim)
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        latent_dim=int(model_cfg.get("latent_dim", 10)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # TRAIN errors for threshold
    scaled_train = preprocessor.transform(train_data)
    train_windows = preprocessor.create_windows(scaled_train, window_size)
    train_loader = DataLoader(
        TimeSeriesDataset(train_windows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    train_mse = compute_reconstruction_errors(train_loader, model, device)
    train_mse_smooth = pd.Series(train_mse).ewm(span=50).mean().values
    threshold = float(np.mean(train_mse_smooth) + 4.0 * np.std(train_mse_smooth))
    if threshold < 0.005:
        logger.warning("Threshold %.6f is too low; clamping to 0.005", threshold)
        threshold = 0.005

    # TEST errors for detection
    scaled_test = preprocessor.transform(test_data)
    test_windows = preprocessor.create_windows(scaled_test, window_size)
    test_loader = DataLoader(
        TimeSeriesDataset(test_windows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_mse = compute_reconstruction_errors(test_loader, model, device)
    test_mse_smooth = pd.Series(test_mse).ewm(span=50).mean().values
    logger.info("Computed reconstruction errors for %s test windows", len(test_mse))

    label_path = base_path / "labeled_anomalies.csv"
    y_true_full = build_ground_truth(
        label_path, channel_id, series_length=test_data.shape[0]
    )

    # Align lengths: each window prediction maps to the last timestamp of that window
    y_true_aligned = y_true_full[window_size - 1 :]
    if len(test_mse_smooth) != len(y_true_aligned):
        raise ValueError(
            f"Length mismatch after alignment: errors={len(test_mse_smooth)} vs y_true={len(y_true_aligned)}"
        )

    y_pred = (test_mse_smooth >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_aligned, y_pred, average="binary", zero_division=0
    )

    try:
        auc_score = (
            roc_auc_score(y_true_aligned, test_mse_smooth)
            if len(np.unique(y_true_aligned)) > 1
            else None
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("AUC computation failed: %s", exc)
        auc_score = None

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.csv"

    metrics_df = pd.DataFrame(
        [
            {
                "channel_id": channel_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc_score,
                "threshold": threshold,
                "window_size": window_size,
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        test_mse_smooth,
        label="Smoothed Reconstruction Error",
        color="steelblue",
        linewidth=1,
    )
    ax.axhline(threshold, color="tomato", linestyle="--", label="Threshold (Z-score)")

    anomaly_indices = np.where(y_true_aligned == 1)[0]
    if anomaly_indices.size > 0:
        ax.scatter(
            anomaly_indices,
            test_mse_smooth[anomaly_indices],
            color="darkred",
            marker="o",
            s=12,
            label="True Anomaly",
            alpha=0.8,
        )

    ax.set_title(f"Smoothed Reconstruction Error - Channel {channel_id}")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("MSE Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = results_dir / "anomaly_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Evaluation complete. Precision=%.4f Recall=%.4f F1=%.4f AUC=%s Threshold=%.6f",
        precision,
        recall,
        f1,
        f"{auc_score:.4f}" if auc_score is not None else "n/a",
        threshold,
    )
    logger.info("Saved metrics to %s and plot to %s", metrics_path, plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Autoencoder on SMAP test data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/model.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    args = parse_args()

    try:
        cfg = load_config(args.config)
        evaluate(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed")
        raise SystemExit(1) from exc
