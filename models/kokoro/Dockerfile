# =============================================================================
# Kokoro TTS – RunPod Serverless Docker Image
# =============================================================================
# Builds a production-ready image for Kokoro v1.0 text-to-speech inference on
# RunPod's serverless GPU platform.
#
# Build:
#   docker build -t kokoro-tts-runpod .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 kokoro-tts-runpod
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libespeak-ng1 \
        libsndfile1 \
        ffmpeg \
        git \
        curl \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY handler.py .

# ---------------------------------------------------------------------------
# Pre-download model weights at build time so cold starts are fast.
# We use huggingface_hub to download weights without requiring GPU.
# ---------------------------------------------------------------------------
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Kokoro model weights...'); \
snapshot_download(repo_id='hexgrad/Kokoro-82M', local_dir_use_symlinks=False); \
print('Model weights cached successfully.') \
"

# ---------------------------------------------------------------------------
# RunPod serverless entry-point
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
