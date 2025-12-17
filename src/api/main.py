from __future__ import annotations

import io
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from torch import nn

# Ensure project root on path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_autoencoder import LSTMAutoencoder  # noqa: E402
from src.api.schema import PredictRequest, PredictResponse  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="SMAP LSTM Autoencoder API", version="1.0")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "total_requests",
    "Total inference requests",
    labelnames=["model_type"],
)
ANOMALY_COUNT = Counter(
    "anomalies_detected",
    "Total anomalies detected",
    labelnames=["model_type"],
)
LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency",
    labelnames=["model_type"],
)


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def load_model(
    model_path: Path, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float
) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def quantize_model(model: LSTMAutoencoder) -> LSTMAutoencoder:
    # Ensure a quantization backend is selected (needed on macOS / some CPU builds)
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
        logger.info("Quantization backend set to qnnpack")

    try:
        quantized = torch.quantization.quantize_dynamic(
            deepcopy(model), {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        quantized.eval()
        return quantized
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Dynamic quantization with LSTM failed (%s); falling back to Linear-only quantization",
            exc,
        )
        quantized = torch.quantization.quantize_dynamic(
            deepcopy(model), {nn.Linear}, dtype=torch.qint8
        )
        quantized.eval()
        return quantized


def model_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def to_tensor(payload: PredictRequest) -> torch.Tensor:
    arr = np.array(payload.data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Input data must be 2D: [seq_len, input_dim]")
    return torch.from_numpy(arr).unsqueeze(0)  # (1, seq, feat)


# Globals initialized on startup
standard_model: LSTMAutoencoder | None = None
quantized_model: LSTMAutoencoder | None = None
window_size: int = 50
input_dim_cfg: int = 1
std_size_bytes: int | None = None
q_size_bytes: int | None = None


@app.on_event("startup")
def startup_event() -> None:
    global \
        standard_model, \
        quantized_model, \
        window_size, \
        input_dim_cfg, \
        std_size_bytes, \
        q_size_bytes

    config_path = PROJECT_ROOT / "config" / "model.yaml"
    config = load_config(config_path)
    model_cfg = (
        config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    )
    training_cfg = (
        config.get("training", {})
        if isinstance(config.get("training", {}), dict)
        else {}
    )
    data_cfg = (
        config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    )

    input_dim_cfg = int(model_cfg.get("input_dim", 1))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    latent_dim = int(model_cfg.get("latent_dim", 20))
    dropout = float(model_cfg.get("dropout", 0.0))
    window_size = int(data_cfg.get("window_size", 100))

    save_dir = Path(training_cfg.get("save_dir", "models"))
    model_path = PROJECT_ROOT / save_dir / "lstm_autoencoder.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    standard_model = load_model(
        model_path, input_dim_cfg, hidden_dim, latent_dim, dropout
    )
    quantized_model = quantize_model(standard_model)

    std_size_bytes = model_size_bytes(standard_model)
    q_size_bytes = model_size_bytes(quantized_model)
    logger.info("Standard model size: %.2f KB", std_size_bytes / 1024)
    logger.info("Quantized model size: %.2f KB", q_size_bytes / 1024)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Deep Telemetry Trigger API",
        "status": "running",
        "version": "1.0",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # type: ignore[override]
    return JSONResponse(
        status_code=400,
        content={"error": "Input must be a list of 100 time steps"},
    )


@torch.no_grad()
def run_inference(tensor: torch.Tensor, use_quantized: bool) -> np.ndarray:
    if standard_model is None or quantized_model is None:
        raise RuntimeError("Models are not initialized")
    model = quantized_model if use_quantized else standard_model
    output = model(tensor)
    return output.squeeze(0).cpu().numpy()


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest, use_quantized: bool = Query(False)
) -> Dict[str, Any]:
    model_label = "quantized" if use_quantized else "standard"

    try:
        if len(payload.data) != window_size:
            raise ValueError("Input must be a list of 100 time steps")
        tensor = to_tensor(payload)
        if tensor.shape[-1] != input_dim_cfg:
            raise ValueError(
                f"Expected input_dim={input_dim_cfg}, got {tensor.shape[-1]}"
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    REQUEST_COUNT.labels(model_type=model_label).inc()

    with LATENCY.labels(model_type=model_label).time():
        with torch.no_grad():
            recon = run_inference(tensor, use_quantized=use_quantized)

    # Mean squared reconstruction error over the window
    mse = float(((tensor.numpy().squeeze(0) - recon) ** 2).mean())
    threshold = 0.005
    is_anomaly = mse > threshold
    if is_anomaly:
        ANOMALY_COUNT.labels(model_type=model_label).inc()

    return {
        "use_quantized": use_quantized,
        "reconstruction": recon.tolist(),
        "mse": mse,
        "is_anomaly": is_anomaly,
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def benchmark_model(
    model: nn.Module, sample: torch.Tensor, runs: int = 100, warmup: int = 10
) -> float:
    # Warm-up to stabilize kernels
    for _ in range(warmup):
        _ = model(sample)

    times: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(sample)
        end = time.perf_counter()
        times.append(end - start)
    return percentile(times, 95) * 1000.0  # ms


@app.get("/benchmark")
def benchmark() -> Dict[str, Any]:
    if standard_model is None or quantized_model is None:
        raise HTTPException(status_code=500, detail="Models are not initialized")

    sample = torch.zeros((1, window_size, input_dim_cfg), dtype=torch.float32)

    p95_standard = benchmark_model(standard_model, sample, runs=100, warmup=10)
    p95_quantized = benchmark_model(quantized_model, sample, runs=100, warmup=10)
    speedup_val = p95_standard / p95_quantized if p95_quantized > 0 else float("inf")
    speedup = f"{speedup_val:.2f}x" if np.isfinite(speedup_val) else "inf"

    if std_size_bytes is not None and q_size_bytes is not None and q_size_bytes > 0:
        size_reduction_val = std_size_bytes / q_size_bytes
        size_reduction = f"{size_reduction_val:.2f}x"
    else:
        size_reduction = "n/a"

    response = {
        "standard_latency_p95": p95_standard,
        "quantized_latency_p95": p95_quantized,
        "speedup_ratio": speedup,
        "size_reduction_ratio": size_reduction,
    }
    if np.isfinite(speedup_val) and speedup_val < 1.0:
        response["note"] = (
            "Quantization overhead exceeds benefit on this CPU architecture for small batches, "
            "but memory reduction is confirmed."
        )
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
