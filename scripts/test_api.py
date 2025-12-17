from __future__ import annotations

"""
Simple client script to exercise the FastAPI endpoint.
Simulates sending a 100-step telemetry window to the trigger system.
"""

import json
from typing import List

import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict?use_quantized=false"
WINDOW_SIZE = 100
INPUT_DIM = 1


def make_payload() -> dict:
    """Create a random 100-step window of telemetry data."""
    series = np.random.randn(WINDOW_SIZE, INPUT_DIM).astype(float)
    data: List[List[float]] = series.tolist()
    return {"data": data}


def main() -> None:
    payload = make_payload()
    resp = requests.post(API_URL, json=payload, timeout=10)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:  # noqa: BLE001
        print(resp.text)


if __name__ == "__main__":
    main()
