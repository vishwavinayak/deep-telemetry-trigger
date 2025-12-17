# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Install uv for dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project metadata first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies globally inside the container
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy source and configs
COPY src ./src
COPY config ./config
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
