# =============================================================================
# Kokoro TTS v2 – RunPod Serverless Docker Image
# =============================================================================
# Optimized handler with forced alignment timestamps, crossfade word
# separation, and word boundary analysis.
#
# Build:
#   docker build -t kokoro-tts-v2-runpod .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 kokoro-tts-v2-runpod
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
# Skip ffmpeg backend in torchaudio (we use soundfile for I/O)
ENV TORIO_USE_FFMPEG=0

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
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

# Install torchaudio from the CUDA 12.4 index to match base image torch
RUN pip install --no-cache-dir torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code (v2 handler with timestamps + crossfade)
# ---------------------------------------------------------------------------
COPY handler_v2.py handler.py

# ---------------------------------------------------------------------------
# Pre-download model weights at build time so cold starts are fast.
# Downloads both Kokoro TTS weights and MMS forced alignment model.
# ---------------------------------------------------------------------------
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Kokoro model weights...'); \
snapshot_download(repo_id='hexgrad/Kokoro-82M'); \
print('Kokoro weights cached.'); \
\
import torchaudio; \
print('Downloading MMS forced alignment model...'); \
bundle = torchaudio.pipelines.MMS_FA; \
model = bundle.get_model(with_star=False); \
print('MMS FA model cached.'); \
"

# ---------------------------------------------------------------------------
# RunPod serverless entry-point
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
