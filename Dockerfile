# Use NVIDIA CUDA 12.8 runtime base image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, git, and ffmpeg
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY rp_handler.py . 
COPY requirements.txt .

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Expose port if needed
EXPOSE 8080

# Default command
CMD python3.11 -u rp_handler.py
