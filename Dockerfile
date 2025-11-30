# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements early to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configuration
COPY src ./src
COPY config ./config
COPY models ./models
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]






