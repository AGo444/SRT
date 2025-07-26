# Stage 1: GPU-accelerated ffmpeg
FROM jrottenberg/ffmpeg:6.0-nvidia AS ffmpeg-nv

# Stage 2: PyTorch + WhisperX environment
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory and create user
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser && \
    mkdir -p /data && \
    chown -R appuser:appgroup /app /data

# Copy ffmpeg with GPU support
COPY --from=ffmpeg-nv /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-nv /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY --chown=appuser:appgroup requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir debugpy==1.8.0

# Copy application files
COPY --chown=appuser:appgroup createSrt.py .

# Switch to non-root user
USER appuser

# Volume and environment configuration
VOLUME /data
ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    USE_CUDA=1

# Create entrypoint script with proper variable handling
COPY --chown=appuser:appgroup <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Check if health check
if [ "$1" = "--health-check" ]; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
    exit 0
fi

# Check CUDA availability at runtime
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" || exit 1

# Start the application with all arguments
if [ $# -eq 0 ]; then
    exec python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client createSrt.py --language nl
else
    exec python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client createSrt.py "$@"
fi
EOF

RUN chmod +x /app/entrypoint.sh

# Update ENTRYPOINT and remove CMD since it's handled in script
ENTRYPOINT ["/app/entrypoint.sh"]

# Update healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/app/entrypoint.sh", "--health-check"]