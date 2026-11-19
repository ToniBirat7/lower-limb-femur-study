# Runs the unit tests on synthetic data; no GPU or dataset needed.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-q"]
