# =============================================================================
# Kokoro TTS v2 – RunPod Serverless Docker Image (ONNX Optimized)
# =============================================================================
# Uses ONNX Runtime FP16 + CUDAExecutionProvider for 25x-40x RTF on L4 GPU.
# Includes torchaudio MMS_FA for word-level forced alignment timestamps.
#
# Build:
#   docker build -t kokoro-tts-v2-runpod .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 kokoro-tts-v2-runpod
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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

# Install torchaudio from CUDA 12.4 index (for forced alignment only)
RUN pip install --no-cache-dir torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install ONNX Runtime GPU (replaces PyTorch for TTS inference)
RUN pip install --no-cache-dir onnxruntime-gpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY handler_v2.py handler.py

# ---------------------------------------------------------------------------
# Download model files at build time
# ---------------------------------------------------------------------------
RUN mkdir -p /app/model

# Download Kokoro ONNX FP16 model + voices
RUN curl -L -o /app/model/kokoro-v1.0.fp16.onnx \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx" \
    && curl -L -o /app/model/voices-v1.0.bin \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Pre-download MMS forced alignment model (torchaudio caches to ~/.cache)
RUN python -c "\
import torchaudio; \
print('Downloading MMS FA model...'); \
bundle = torchaudio.pipelines.MMS_FA; \
model = bundle.get_model(with_star=False); \
print('MMS FA cached.'); \
"

# ---------------------------------------------------------------------------
# RunPod serverless entry-point
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
