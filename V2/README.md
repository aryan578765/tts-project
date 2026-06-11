# Kokoro TTS v2 - ONNX Optimized

Optimized Kokoro handler for RunPod serverless deployment.

## What's New in v2

| Feature | v1 (PyTorch) | v2 (ONNX) |
|---------|-------------|-----------|
| Inference | PyTorch KPipeline | ONNX Runtime FP16 + CUDA |
| RTF on L4 | ~0.23 (4.3x) | ~0.07 (14-15x) |
| Timestamps | None | Word-level (MMS_FA) |
| Word separation | None | Crossfade + boundary analysis |
| Languages (FA) | N/A | English, French, Spanish, Italian, Portuguese |

## Files

```
V2/
├── handler.py          # Main RunPod handler (ONNX + FA + crossfade)
├── Dockerfile          # Docker build (downloads FP16 model + MMS_FA)
├── requirements.txt    # Python dependencies
├── test_handler.py     # Test script (RTF benchmark + all features)
└── README.md           # This file
```

## Build & Deploy

```bash
# Build Docker image
docker build -t kokoro-tts-v2 .

# Push to DockerHub
docker tag kokoro-tts-v2 patelaryan777/kokoro-tts-v2:final
docker push patelaryan777/kokoro-tts-v2:final

# Deploy to RunPod (create new serverless endpoint with this image)
```

## API

### Request
```json
{
    "input": {
        "text": "Hello, world!",
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "a",
        "timestamps": true,
        "word_boundaries": true,
        "micro_pause_ms": 10,
        "crossfade_ms": 5.0
    }
}
```

### Response
```json
{
    "audio_base64": "<base64 WAV>",
    "sample_rate": 24000,
    "duration_seconds": 2.35,
    "rtf": 0.07,
    "synth_rtf": 0.07,
    "word_timestamps": [
        {"word": "Hello", "start": 0.12, "end": 0.45},
        {"word": "world", "start": 0.52, "end": 0.89}
    ],
    "word_boundaries": [
        {"pair": "Hello|world", "gap_ms": 70.0, "status": "clean", "can_separate": true}
    ]
}
```

## Language Codes

| Code | Language | Voice Example |
|------|----------|--------------|
| `a` | American English | `af_heart`, `am_adam` |
| `b` | British English | `bf_emma`, `bm_george` |
| `e` | Spanish | `ef_dora`, `em_alex` |
| `f` | French | `ff_siwis` |
| `i` | Italian | `if_sara`, `im_nicola` |
| `j` | Japanese | `jf_alpha`, `jm_kumo` |
| `p` | Portuguese | `pf_dora`, `pm_alex` |
| `z` | Chinese | `zf_xiaobei`, `zm_yunjian` |

## Word Boundary Status

| Status | Gap | Meaning |
|--------|-----|---------|
| `clean` | ≥ 50ms | Safe to cut, clear word boundary |
| `tight` | 20-50ms | Can separate, but boundary is narrow |
| `coarticulated` | 0-20ms | Words blend together, needs crossfade |
| `overlapping` | < 0ms | Words overlap in time, cannot separate cleanly |

## Testing

```bash
# Set environment variables
set RUNPOD_API_KEY=rpa_xxx
set KOKORO_ENDPOINT_ID=xxx

# Run tests
python test_handler.py
```

## Production Test Results (L4 GPU)

| Test | Result |
|------|--------|
| Basic synthesis (13 texts) | All pass |
| Word timestamps | 23 words, ordered |
| Word boundaries | 15 clean, 7 tight |
| Micro-pause 10ms | Working |
| Micro-pause 50ms | Working |
| French (ff_siwis) | 11 word timestamps |
| Spanish (ef_dora) | 13 word timestamps |
| RTF benchmark (3 runs) | 0.07 avg (14x real-time) |
| Long text (362 words) | 157.9s audio in 10.7s |
| Edge cases | All handled |
