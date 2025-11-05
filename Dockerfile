FROM python:3.11-slim

# System build basics (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install backend-only requirements
COPY requirements.backend.txt /app/requirements.backend.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.backend.txt

# Copy the app
COPY . /app

# Default command (compose overrides ok)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]