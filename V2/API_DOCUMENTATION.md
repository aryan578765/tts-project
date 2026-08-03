# Kokoro TTS V2 — API Documentation

## Endpoint

**RunPod Serverless** — POST to your RunPod endpoint URL.

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
Headers:
  Authorization: Bearer {RUNPOD_API_KEY}
  Content-Type: application/json
```

---

## Request Format

```json
{
  "input": {
    "text": "Your text here.",
    "voice": "af_heart",
    "speed": 1.0,
    "lang_code": "a",
    "timestamps": true,
    "word_boundaries": true,
    "pause_after": [4, 9, 15],
    "micro_pause_ms": 2000
  }
}
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | — | The text to synthesize. |
| `voice` | string | No | `"af_heart"` | Voice name. See Available Voices. |
| `speed` | float | No | `1.0` | Speaking speed. Range: `0.1` to `5.0`. |
| `lang_code` | string | No | `"a"` | Language code. See Language Codes. |
| `timestamps` | boolean | No | `false` | Return word-level timestamps. |
| `word_boundaries` | boolean | No | `false` | Return word boundary gap analysis. |
| `pause_after` | int[] | No | `null` | Word indices (0-based) after which to insert phrase pauses. Automatically enables `timestamps` and `word_boundaries`. |
| `micro_pause_ms` | float | No | `0` | Duration of silence to insert at each `pause_after` position (in milliseconds). Set `0` to get timestamps without inserted pauses. |

---

## Response Format

```json
{
  "audio_base64": "<base64-encoded WAV>",
  "sample_rate": 24000,
  "duration_seconds": 5.234,
  "rtf": 0.048,
  "synth_rtf": 0.021,
  "word_timestamps": [...],
  "word_boundaries": [...],
  "phrase_cut_points": [...]
}
```

### Response Fields

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `audio_base64` | string | Yes | Base64-encoded WAV audio (24kHz, 16-bit PCM, mono). |
| `sample_rate` | int | Yes | Always `24000` Hz. |
| `duration_seconds` | float | Yes | Total audio duration in seconds. |
| `rtf` | float | Yes | Real-time factor (total processing time / audio duration). Lower = faster. |
| `synth_rtf` | float | Yes | Synthesis-only RTF (excludes alignment and post-processing). |
| `word_timestamps` | array | When `timestamps=true` | Word-level timing. See Word Timestamps. |
| `word_boundaries` | array | When `word_boundaries=true` | Gap analysis between words. See Word Boundaries. |
| `phrase_cut_points` | array | When `pause_after` is used | Phrase boundary positions. See Phrase Cut Points. |

---

## Word Timestamps

Each entry maps a word to its exact time position in the audio.

```json
{
  "word": "Hello",
  "start": 0.12,
  "end": 0.45
}
```

| Field | Type | Description |
|-------|------|-------------|
| `word` | string | The original word from the input text (with punctuation). |
| `start` | float | Start time in seconds. |
| `end` | float | End time in seconds. |

Notes:
- The number of timestamps always matches the number of words in the input text.
- Times are relative to the start of the audio (0.0).
- If `micro_pause_ms > 0`, timestamps are adjusted to account for inserted silences.

---

## Word Boundaries

Analyzes the gap between every consecutive word pair.

```json
{
  "pair": "Hello|world",
  "gap_ms": 62.0,
  "status": "clean",
  "can_separate": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pair` | string | The two words separated by `|`. |
| `gap_ms` | float | Gap in milliseconds between end of word 1 and start of word 2. |
| `status` | string | One of: `clean` (>=50ms gap), `tight` (20-50ms), `coarticulated` (0-20ms), `overlapping` (<0ms). |
| `can_separate` | boolean | `true` if the words can be cleanly cut apart without audio artifacts. |

---

## Phrase Cut Points

When `pause_after` is provided, returns the exact position and duration of each phrase boundary.

```json
{
  "time": 2.1727,
  "duration_ms": 2000,
  "after_word_idx": 1,
  "after_word": "Compute,",
  "trough_energy": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Center of the pause in seconds (relative to audio start). |
| `duration_ms` | float | Duration of the inserted silence in milliseconds. |
| `after_word_idx` | int | 0-based index of the word after which this pause occurs. |
| `after_word` | string | The word after which this pause is placed. |
| `trough_energy` | float | Energy level at the cut point (0 = perfect silence). |

---

## Usage Modes

### Mode 1: Basic TTS (no timestamps)

Just generate audio.

```json
{
  "input": {
    "text": "Hello, world!",
    "voice": "af_heart"
  }
}
```

Response: `audio_base64`, `sample_rate`, `duration_seconds`, `rtf`

---

### Mode 2: TTS + Word Timestamps

Generate audio with word-level timing.

```json
{
  "input": {
    "text": "Hello, world!",
    "voice": "af_heart",
    "timestamps": true,
    "word_boundaries": true
  }
}
```

Response: Same as Mode 1 + `word_timestamps` + `word_boundaries`

---

### Mode 3: TTS + Phrase Pauses (2-second pauses)

Generate audio with 2-second silences inserted at specified word positions.

```json
{
  "input": {
    "text": "General Compute, an AI inference cloud startup, has landed a deal.",
    "voice": "af_heart",
    "pause_after": [1, 6],
    "micro_pause_ms": 2000
  }
}
```

This inserts a 2000ms pause after word index 1 ("Compute,") and index 6 ("startup,").

Response: Same as Mode 2 + `phrase_cut_points`

---

### Mode 4: TTS + Markers Only (no pauses)

Generate natural audio without pauses, but return word timestamps and phrase boundary positions. Your team can use the timestamps to cut and insert pauses manually.

```json
{
  "input": {
    "text": "General Compute, an AI inference cloud startup, has landed a deal.",
    "voice": "af_heart",
    "pause_after": [1, 6],
    "micro_pause_ms": 0
  }
}
```

Response: Same as Mode 2 + `phrase_cut_points` (with `duration_ms: 0`)

---

## Language Codes

| Code | Language |
|------|----------|
| `a` | American English |
| `b` | British English |
| `e` | Spanish |
| `f` | French |
| `i` | Italian |
| `j` | Japanese |
| `p` | Portuguese |
| `z` | Chinese |

---

## Available Voices

Default voice: `af_heart`. For a full list, see: https://huggingface.co/hexgrad/Kokoro-82M

Common voices:
- `af_heart` — Female, warm
- `af_bella` — Female, clear
- `am_michael` — Male, neutral
- `am_adam` — Male, deep

---

## Decoding Audio (Python)

```python
import base64
import io
import soundfile as sf

# From API response
audio_b64 = response["audio_base64"]

# Decode to WAV file
wav_bytes = base64.b64decode(audio_b64)
with open("output.wav", "wb") as f:
    f.write(wav_bytes)

# Or decode to numpy array
audio, sr = sf.read(io.BytesIO(wav_bytes))
```

---

## Full Example (Python)

```python
import requests
import base64

ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "your_runpod_api_key"

text = "General Compute, an AI inference cloud startup, has landed a $400 million loan from Upper90, a tech investment firm."

words = text.split()
pause_after = [1, 6, 14, 18]

response = requests.post(
    ENDPOINT_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "input": {
            "text": text,
            "voice": "af_heart",
            "speed": 1.0,
            "lang_code": "a",
            "timestamps": True,
            "word_boundaries": True,
            "pause_after": pause_after,
            "micro_pause_ms": 2000,
        }
    },
)

result = response.json()["output"]

# Save audio
wav_bytes = base64.b64decode(result["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(wav_bytes)

# Print word timestamps
for ts in result["word_timestamps"]:
    print(f'{ts["word"]:20s} {ts["start"]:.3f}s - {ts["end"]:.3f}s')

# Print phrase cut points
for cp in result["phrase_cut_points"]:
    print(f'Pause after "{cp["after_word"]}" at {cp["time"]:.3f}s ({cp["duration_ms"]}ms)')
```

---

## Error Handling

On error, the response contains an `error` field:

```json
{
  "error": "Missing or empty 'text' field in input."
}
```

Common errors:
- `Missing or empty 'text' field` — No text provided.
- `Invalid lang_code` — Use one of: a, b, e, f, i, j, p, z.
- `Speed must be between 0.1 and 5.0` — Speed out of range.
- `Synthesis failed: <details>` — Internal synthesis error.
