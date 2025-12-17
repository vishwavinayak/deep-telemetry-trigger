from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    data: List[List[float]] = Field(..., description="2D array: [seq_len, input_dim]")

    @validator("data")
    def validate_data(cls, v: List[List[float]]):  # noqa: N805
        if not isinstance(v, list) or not v:
            raise ValueError("Input must be a list of 100 time steps")
        inner_lengths = {len(row) for row in v if isinstance(row, list)}
        if len(inner_lengths) != 1:
            raise ValueError("All rows must have the same length")
        return v


class PredictResponse(BaseModel):
    reconstruction: List[List[float]]
    mse: float
    is_anomaly: bool
