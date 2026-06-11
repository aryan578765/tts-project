# TTS Evaluation & Deployment Project

Production-ready text-to-speech evaluation, benchmarking, and RunPod serverless deployment pipeline. Compares open-source TTS models (Kokoro, CosyVoice, StyleTTS 2) across 30 quality dimensions using 13 standardized test texts.

**Latest: Kokoro V2 (ONNX FP16 + CUDA) deployed - 14-15x real-time on L4 GPU**

## Directory Structure

```
TTS/
├── README.md                          # This file
├── V2/                                # Kokoro V2 - ONNX optimized (production)
│   ├── handler.py                     # RunPod handler (ONNX FP16 + CUDA + FA)
│   ├── Dockerfile                     # Docker build with GPU support
│   ├── test_handler.py                # 10-test suite
│   ├── requirements.txt               # Python dependencies
│   ├── README.md                      # V2 API docs + test results
│   └── output/                        # Test audio samples + results JSON
├── test_texts/
│   └── test_texts.json                # 13 standardized evaluation texts
├── benchmark/
│   ├── scoring_template.csv           # 30-dimension quality scoring (with StyleTTS2 baseline)
│   ├── run_benchmark.py               # RTF & VRAM measurement tool
│   └── cost_calculator.py             # Cost-per-audio-hour calculator
├── scripts/
│   ├── generate_all_audio.py          # Generate audio for all 13 test texts
│   ├── test_ssml.py                   # SSML <break> tag support tester
│   └── test_multilingual.py           # Multi-language support tester
├── models/
│   ├── kokoro/                        # Kokoro TTS v1 model files & handler
│   │   ├── Dockerfile
│   │   ├── handler.py                 # RunPod serverless handler
│   │   └── ...
│   ├── cosyvoice/                     # CosyVoice 2 model files & handler
│   │   ├── Dockerfile
│   │   ├── handler.py
│   │   └── ...
│   └── styletts2/                     # StyleTTS 2 baseline
│       └── ...
├── audio_output/                      # Generated audio (git-ignored)
│   ├── kokoro/
│   ├── cosyvoice/
│   └── styletts2/
└── results/                           # Benchmark results & analysis
```

## Quick Start

### Prerequisites

- Python 3.10+
- `requests` library: `pip install requests`
- NVIDIA GPU + `nvidia-smi` (for VRAM monitoring)
- Docker (for building deployment images)
- RunPod account (for serverless deployment)

### 1. Generate Audio for All Test Texts

```bash
# Generate audio using a local or remote TTS endpoint
python scripts/generate_all_audio.py --model kokoro --endpoint http://localhost:8000/generate

# Output is saved to audio_output/kokoro/
```

### 2. Run Benchmarks

```bash
# Measure RTF (Real-Time Factor) and VRAM usage
python benchmark/run_benchmark.py --model kokoro --endpoint http://localhost:8000/generate

# With custom iterations
python benchmark/run_benchmark.py --model kokoro --endpoint http://localhost:8000/generate --iterations 10
```

### 3. Calculate Costs

```bash
# Using a known RTF value
python benchmark/cost_calculator.py --rtf 0.05 --gpu all --model kokoro

# Using benchmark results
python benchmark/cost_calculator.py --benchmark-csv benchmark/benchmark_results.csv --gpu L4

# Custom GPU pricing
python benchmark/cost_calculator.py --rtf 0.05 --gpu custom --gpu-cost 0.50
```

### 4. Test SSML Support

```bash
python scripts/test_ssml.py --endpoint http://localhost:8000/generate --model kokoro
```

### 5. Test Multilingual Support

```bash
# Test all 15 languages
python scripts/test_multilingual.py --endpoint http://localhost:8000/generate --model kokoro

# Test specific languages only
python scripts/test_multilingual.py --endpoint http://localhost:8000/generate --model kokoro --languages en es fr ja zh
```

## Model Setup

### Kokoro TTS

```bash
cd models/kokoro

# Build Docker image
docker build -t kokoro-tts .

# Run locally
docker run --gpus all -p 8000:8000 kokoro-tts

# Deploy to RunPod
# Push to Docker Hub, then create serverless endpoint on RunPod
docker tag kokoro-tts your-registry/kokoro-tts:latest
docker push your-registry/kokoro-tts:latest
```

### Kokoro TTS V2 (ONNX - Production)

Optimized ONNX FP16 handler with word timestamps, micro-pause insertion, and multilingual forced alignment.

```bash
cd V2

# Build Docker image
docker build -t kokoro-tts-v2 .

# Push to DockerHub
docker tag kokoro-tts-v2 patelaryan777/kokoro-tts-v2:final
docker push patelaryan777/kokoro-tts-v2:final
```

**V2 Features:**
- ONNX Runtime FP16 + CUDA (14-15x real-time on L4)
- Word-level timestamps via MMS forced alignment
- Word boundary analysis (clean/tight gap detection)
- Micro-pause insertion with equal-power crossfade
- Multilingual FA support (English, French, Spanish, Italian, Portuguese)
- 8 language codes supported

**V2 Production Test Results (L4 GPU):**

| Metric | Result |
|--------|--------|
| RTF (25s audio) | 0.07 (14x real-time) |
| Synthesis time (25s audio) | 1.73s |
| Long text (158s audio) | 10.7s |
| Word timestamps | 23 words, accurate |
| Word boundaries | 15 clean, 7 tight |
| All 13 test texts | Pass |
| Edge cases | All handled |

### CosyVoice 2

```bash
cd models/cosyvoice

# Build Docker image
docker build -t cosyvoice-tts .

# Run locally
docker run --gpus all -p 8000:8000 cosyvoice-tts

# Deploy to RunPod
docker tag cosyvoice-tts your-registry/cosyvoice-tts:latest
docker push your-registry/cosyvoice-tts:latest
```

## RunPod Serverless Deployment

### Handler Convention

All RunPod handlers follow this structure:

```python
import runpod

def handler(event):
    text = event["input"]["text"]
    # ... generate audio ...
    return {"audio": base64_audio, "duration": duration_seconds}

runpod.serverless.start({"handler": handler})
```

### Deployment Steps

1. **Build** the Docker image with model weights baked in
2. **Push** to Docker Hub or another container registry
3. **Create** a RunPod serverless endpoint pointing to your image
4. **Configure** GPU type (L4 recommended for cost efficiency)
5. **Test** using the benchmark scripts with the RunPod endpoint URL

### RunPod Endpoint Format

```
https://api.runpod.ai/v2/{endpoint_id}/runsync
```

Set the `RUNPOD_API_KEY` environment variable or pass it as a header:

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello world"}}'
```

## Evaluation Framework

### 30 Quality Dimensions

The scoring template (`benchmark/scoring_template.csv`) covers:

| Category | Dimensions |
|----------|-----------|
| **Voice Quality** (1-5) | Naturalness, Prosody, Intonation, Pauses, Rhythm |
| **Emotion** (6-7) | Subtlety, Content alignment |
| **Stability** (8-10) | Voice stability, Long-form coherence, Listener comfort |
| **Structure** (11-15) | Structural awareness, Questions, Emphasis, Negation, Quoted speech |
| **Multilingual** (16-17) | Language prosody, Code-switch smoothness |
| **Technical** (18-22) | Phonetic robustness, Audio cleanliness, Silence, Sibilance, Start/end |
| **Realism** (23-24) | AI audibility, Overall realism ceiling |
| **Formatting** (25-30) | Acronyms, Numbers, Currency, Dates, URLs, Technical strings |

### Scoring Scale

- **0** — Fails or produces unusable output
- **1** — Functional but clearly flawed
- **2** — Good, minor issues only
- **3** — Excellent, near-human quality

### 13 Test Texts

Each test text targets specific quality dimensions. See `test_texts/test_texts.json` for the full list with coverage mappings.

## GPU Pricing Reference

| GPU | RunPod $/hr | Best For |
|-----|------------|----------|
| L4 | $0.69 | Cost-efficient inference |
| RTX 4090 | $0.74 | Development/testing |
| A40 | $1.22 | Balanced performance |
| L40S | $1.59 | High throughput |
| A100 | $2.72 | Maximum performance |
| H100 | $4.49 | Fastest inference |

## Commercial Baseline

The cost calculator compares against a commercial baseline of **$1.28/hr** of generated audio (typical cloud TTS API pricing).

## Contributing

1. Add model handlers in `models/{model_name}/`
2. Follow the RunPod handler convention
3. Run all 13 test texts and fill in the scoring template
4. Submit benchmark results to `results/`
