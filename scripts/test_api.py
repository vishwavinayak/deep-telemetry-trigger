from __future__ import annotations

"""
Client script to exercise the FastAPI endpoint with both standard and quantized models.
Sends a 100-step telemetry window to /predict for each model variant.
"""

import json
from typing import List

import numpy as np
import requests

API_BASE = "http://127.0.0.1:8000"
WINDOW_SIZE = 100
INPUT_DIM = 1


def make_payload() -> dict:
    """Create a random 100-step window of telemetry data."""
    series = np.random.randn(WINDOW_SIZE, INPUT_DIM).astype(float)
    data: List[List[float]] = series.tolist()
    return {"data": data}


def send_request(use_quantized: bool) -> tuple[int, dict | None]:
    url = f"{API_BASE}/predict?use_quantized={'true' if use_quantized else 'false'}"
    payload = make_payload()
    resp = requests.post(url, json=payload, timeout=10)
    try:
        body = resp.json()
    except Exception:
        body = None
    return resp.status_code, body


def main() -> None:
    status_standard, body_standard = send_request(use_quantized=False)
    status_quantized, body_quantized = send_request(use_quantized=True)

    print(f"[SUCCESS] Standard: {status_standard} | Quantized: {status_quantized}")
    if body_standard:
        print("Standard response:")
        print(json.dumps(body_standard, indent=2))
    if body_quantized:
        print("Quantized response:")
        print(json.dumps(body_quantized, indent=2))


if __name__ == "__main__":
    main()
