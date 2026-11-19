FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY results ./results
COPY figures ./figures
COPY docs ./docs
COPY splits.json hfvalid_audit.csv ./

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-q"]
