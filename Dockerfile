# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PORT=5050

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libglib2.0-0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (including model weights)
COPY . .

# Expose the port the app runs on
EXPOSE 5050

# Run gunicorn with 4 worker processes
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "4", "--timeout", "120", "api:app"]