# =============================================================================
# CosyVoice 2.0 — RunPod Serverless Docker Image
# =============================================================================
# Build:  docker build -t cosyvoice2-runpod .
# Run:    docker run --gpus all -p 8080:8080 cosyvoice2-runpod
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL maintainer="TTS-Deployment" \
      description="CosyVoice 2.0 (0.5B) — RunPod Serverless" \
      version="1.0.0"

# ── Prevent interactive prompts during apt installs ──────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        sox \
        libsox-dev \
        libsox-fmt-all \
        git \
        git-lfs \
        ffmpeg \
        libsndfile1 \
        build-essential \
        wget \
        curl \
        ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Clone CosyVoice repository (with submodules for Matcha-TTS, etc.) ───────
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

# ── Install Python dependencies ─────────────────────────────────────────────
# Install repo-level requirements first (filter out proprietary ttsfrd)
RUN grep -v "ttsfrd" /app/CosyVoice/requirements.txt > /tmp/cosyvoice_req.txt \
    && pip install --no-cache-dir -r /tmp/cosyvoice_req.txt 2>/dev/null || true

# Install our pinned requirements (ensures runpod + any overrides)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install CosyVoice package itself (if setup.py / pyproject.toml exists)
RUN cd /app/CosyVoice && pip install --no-cache-dir -e . 2>/dev/null || true

# ── Set PYTHONPATH so Matcha-TTS and CosyVoice internals are importable ─────
ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# ── Download CosyVoice2-0.5B model from HuggingFace ────────────────────────
# Uses huggingface_hub snapshot_download; cached under /app/models
ENV HF_HOME=/app/hf_cache
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='FunAudioLLM/CosyVoice2-0.5B', \
    local_dir='/app/models/CosyVoice2-0.5B', \
    local_dir_use_symlinks=False \
)"

# ── Download SFT speaker profiles (spk2info.pt) ─────────────────────────────
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='FunAudioLLM/CosyVoice-300M-SFT', \
    filename='spk2info.pt', \
    local_dir='/app/models/CosyVoice2-0.5B' \
)"


# ── Copy handler LAST so edits don't invalidate model cache ─────────────────
COPY handler.py /app/handler.py

# ── Expose default RunPod health-check port ─────────────────────────────────
EXPOSE 8080

# ── Launch handler ──────────────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
