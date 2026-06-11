"""
Kokoro TTS v2 - RunPod Serverless Handler (ONNX Optimized)
============================================================

Optimized handler with:
  - ONNX Runtime FP16 inference via CUDAExecutionProvider (25x-40x RTF on L4)
  - Word-level timestamps via torchaudio MMS forced alignment
  - Crossfade-based micro-pause insertion (preserves intonation)
  - Word spacing analysis (reports which word pairs can be cleanly separated)

API Response:
  {
      "audio_base64": "<WAV>",
      "sample_rate": 24000,
      "duration_seconds": 2.35,
      "rtf": 0.04,
      "word_timestamps": [
          {"word": "Hello", "start": 0.12, "end": 0.45},
          ...
      ],
      "word_boundaries": [
          {"pair": "Hello|world", "gap_ms": 62, "can_separate": true},
          ...
      ]
  }

Supported language codes:
    a - American English    b - British English
    e - Spanish (es)        f - French (fr)
    i - Italian (it)        j - Japanese (ja)
    p - Portuguese (pt)     z - Chinese (zh)
"""

import base64
import io
import logging
import os
import re
import time
import threading
import unicodedata
from typing import Any

import runpod
import soundfile as sf
import torch
import torchaudio
import numpy as np

from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kokoro-handler-v2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map single-letter lang codes (user-facing) → kokoro-onnx locale strings
LANG_CODE_MAP: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}

VALID_LANG_CODES: dict[str, str] = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "z": "Chinese",
}

# Languages where forced alignment works (Latin-script based)
FA_SUPPORTED_LANG_CODES = {"a", "b", "e", "f", "i", "p"}

SAMPLE_RATE: int = 24000  # Kokoro outputs 24 kHz audio
FA_SAMPLE_RATE: int = 16000  # MMS_FA expects 16 kHz

# Threshold (ms) for classifying word boundaries
CLEAN_BOUNDARY_THRESHOLD_MS = 50
COARTICULATED_THRESHOLD_MS = 20

# Model file paths (downloaded in Dockerfile)
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/app/model/kokoro-v1.0.fp16.onnx")
VOICES_PATH = os.environ.get("VOICES_PATH", "/app/model/voices-v1.0.bin")

# ---------------------------------------------------------------------------
# Global model caches (thread-safe)
# ---------------------------------------------------------------------------
_kokoro_model = None
_kokoro_lock = threading.Lock()

_fa_model = None
_fa_tokenizer = None
_fa_aligner = None
_fa_dict = None
_fa_lock = threading.Lock()


def _get_kokoro() -> Kokoro:
    """Return cached Kokoro ONNX model (thread-safe)."""
    global _kokoro_model
    if _kokoro_model is None:
        with _kokoro_lock:
            if _kokoro_model is None:
                logger.info("Loading Kokoro ONNX model (FP16 + CUDA)...")
                start = time.perf_counter()

                import onnxruntime as ort

                # Log available providers before creating session
                available = ort.get_available_providers()
                logger.info("ONNX available providers: %s", available)

                # Set CUDA provider options
                provider_options = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "DEFAULT",
                    }),
                    "CPUExecutionProvider",
                ]

                session = ort.InferenceSession(
                    ONNX_MODEL_PATH,
                    providers=[p if isinstance(p, str) else p[0] for p in provider_options],
                    provider_options=[p[1] if isinstance(p, tuple) else {} for p in provider_options],
                )
                active = session.get_providers()
                logger.info("ONNX active providers: %s", active)

                if "CUDAExecutionProvider" in active:
                    logger.info("✅ ONNX using GPU (CUDAExecutionProvider)")
                else:
                    logger.warning("⚠️ ONNX fell back to CPU! Available: %s, Active: %s", available, active)

                _kokoro_model = Kokoro.from_session(session, VOICES_PATH)

                elapsed = time.perf_counter() - start
                logger.info("Kokoro ONNX ready in %.2fs", elapsed)
    return _kokoro_model


def _get_fa_components():
    """Load forced alignment model, tokenizer, aligner (thread-safe).

    Uses torchaudio MMS_FA high-level API. Runs on CPU to save GPU for ONNX.
    """
    global _fa_model, _fa_tokenizer, _fa_aligner, _fa_dict

    if _fa_model is None:
        with _fa_lock:
            if _fa_model is None:
                logger.info("Loading MMS forced alignment model...")
                start = time.perf_counter()

                bundle = torchaudio.pipelines.MMS_FA
                _fa_model = bundle.get_model(with_star=False).to("cpu")
                _fa_model.eval()
                _fa_tokenizer = bundle.get_tokenizer()
                _fa_aligner = bundle.get_aligner()
                _fa_dict = bundle.get_dict()

                elapsed = time.perf_counter() - start
                logger.info("MMS FA model loaded in %.2fs", elapsed)

    return _fa_model, _fa_tokenizer, _fa_aligner, _fa_dict


# Pre-warm models at import time
try:
    logger.info("Pre-warming Kokoro ONNX model...")
    _get_kokoro()
except Exception:
    logger.exception("Failed to pre-warm Kokoro ONNX model")

try:
    logger.info("Pre-warming FA model...")
    _get_fa_components()
except Exception:
    logger.exception("Failed to pre-warm FA model")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _audio_to_base64_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode a numpy audio array as a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _find_zero_crossing(audio: np.ndarray, index: int, search_range: int = 120) -> int:
    """Find nearest zero-crossing to prevent clicks at cut points."""
    for offset in range(1, search_range):
        fwd = index + offset
        if fwd < len(audio) - 1:
            if audio[fwd] * audio[fwd + 1] <= 0:
                return fwd
        bwd = index - offset
        if bwd >= 0 and bwd < len(audio) - 1:
            if audio[bwd] * audio[bwd + 1] <= 0:
                return bwd
    return index


# ---------------------------------------------------------------------------
# Forced Alignment — Word-level timestamps
# ---------------------------------------------------------------------------

def _clean_word(word: str) -> str:
    """Strip punctuation and accents from a word for alignment matching.

    MMS_FA tokenizer only supports basic Latin a-z, so we normalize
    accented characters: é→e, ñ→n, ü→u, etc.
    """
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE).strip()
    # Normalize accented characters to ASCII (NFD decomposes, then strip combining marks)
    normalized = unicodedata.normalize('NFD', cleaned)
    ascii_only = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return ascii_only


def _get_word_timestamps(
    audio: np.ndarray,
    transcript: str,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """Get word-level timestamps using torchaudio MMS_FA high-level API."""
    model, tokenizer, aligner, fa_dict = _get_fa_components()
    start_time = time.perf_counter()

    # Convert numpy to torch tensor [1, T]
    waveform = torch.from_numpy(audio).float().unsqueeze(0)

    # Resample 24kHz -> 16kHz
    if sample_rate != FA_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, FA_SAMPLE_RATE
        )

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 1: Generate emissions
    with torch.inference_mode():
        emission, _ = model(waveform)

    # CRITICAL: Convert to log probabilities
    emission = torch.log_softmax(emission, dim=-1)

    # Step 2: Prepare word list (MMS_FA requires lowercase)
    words_raw = transcript.split()
    words_clean = []
    words_display = []
    for w in words_raw:
        cleaned = _clean_word(w).lower()
        if cleaned:
            words_clean.append(cleaned)
            words_display.append(w)

    if not words_clean:
        logger.error("No valid words for alignment in: %s", transcript[:80])
        return []

    logger.info("FA: %d words to align, first 5: %s", len(words_clean), words_clean[:5])

    # Step 3: Tokenize
    try:
        token_spans = tokenizer(words_clean)
        logger.info("FA: tokenized %d words into %d token spans", len(words_clean), len(token_spans))
    except Exception as e:
        logger.error("Tokenization failed: %s", e, exc_info=True)
        return []

    # Step 4: Align
    try:
        alignment = aligner(emission[0], token_spans)
        logger.info("FA: alignment returned %d word spans", len(alignment))
    except Exception as e:
        logger.error("Alignment failed: %s", e, exc_info=True)
        return []

    # Step 5: Convert frames to timestamps
    ratio = waveform.shape[1] / emission.shape[1]

    word_timestamps = []
    for word_idx, word_spans in enumerate(alignment):
        if word_idx >= len(words_display):
            break
        if not word_spans:
            continue

        start_frame = word_spans[0].start
        end_frame = word_spans[-1].end

        start_sec = (start_frame * ratio) / FA_SAMPLE_RATE
        end_sec = (end_frame * ratio) / FA_SAMPLE_RATE

        word_timestamps.append({
            "word": words_display[word_idx],
            "start": round(start_sec, 4),
            "end": round(end_sec, 4),
        })

    elapsed = time.perf_counter() - start_time
    logger.info("FA completed in %.3fs for %d words", elapsed, len(word_timestamps))

    return word_timestamps


# ---------------------------------------------------------------------------
# Word Boundary Analysis
# ---------------------------------------------------------------------------

def _analyze_word_boundaries(word_timestamps: list[dict]) -> list[dict]:
    """Analyze gaps between consecutive words."""
    boundaries = []
    for i in range(len(word_timestamps) - 1):
        w1 = word_timestamps[i]
        w2 = word_timestamps[i + 1]
        gap_sec = w2["start"] - w1["end"]
        gap_ms = round(gap_sec * 1000, 1)

        if gap_ms >= CLEAN_BOUNDARY_THRESHOLD_MS:
            status = "clean"
            can_separate = True
        elif gap_ms >= COARTICULATED_THRESHOLD_MS:
            status = "tight"
            can_separate = True
        elif gap_ms >= 0:
            status = "coarticulated"
            can_separate = False
        else:
            status = "overlapping"
            can_separate = False

        boundaries.append({
            "pair": f"{_clean_word(w1['word'])}|{_clean_word(w2['word'])}",
            "gap_ms": gap_ms,
            "status": status,
            "can_separate": can_separate,
        })

    return boundaries


# ---------------------------------------------------------------------------
# Crossfade-based micro-pause insertion
# ---------------------------------------------------------------------------

def _equal_power_fades(fade_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate equal-power (cos/sin) fade curves."""
    t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)
    return np.cos(t).astype(np.float32), np.sin(t).astype(np.float32)


def _insert_micro_pauses(
    audio: np.ndarray,
    word_timestamps: list[dict],
    pause_ms: float = 10.0,
    crossfade_ms: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[dict]]:
    """Insert micro-pauses between words using equal-power crossfade.

    Preserves original intonation by working on the already-synthesised audio.
    Returns (modified_audio, updated_timestamps).
    """
    if not word_timestamps or len(word_timestamps) < 2:
        return audio, word_timestamps

    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)  # DC offset removal

    pause_samples = int(sample_rate * pause_ms / 1000)
    crossfade_samples = max(int(sample_rate * crossfade_ms / 1000), 4)
    fade_out, fade_in = _equal_power_fades(crossfade_samples)

    parts = []
    updated_timestamps = []
    cumulative_offset = 0.0

    for i, wt in enumerate(word_timestamps):
        start_sample = int(wt["start"] * sample_rate)
        end_sample = int(wt["end"] * sample_rate)
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(start_sample, min(end_sample, len(audio)))

        if i == 0:
            chunk = audio[:end_sample].copy()
            if len(chunk) > crossfade_samples:
                chunk[-crossfade_samples:] *= fade_out
            parts.append(chunk)
            updated_timestamps.append({
                "word": wt["word"],
                "start": round(wt["start"] + cumulative_offset, 4),
                "end": round(wt["end"] + cumulative_offset, 4),
            })
        else:
            silence = np.zeros(pause_samples, dtype=np.float32)
            parts.append(silence)
            cumulative_offset += pause_ms / 1000.0

            cut_start = _find_zero_crossing(audio, start_sample)
            chunk = audio[cut_start:end_sample].copy()

            if len(chunk) > crossfade_samples:
                chunk[:crossfade_samples] *= fade_in
                if i < len(word_timestamps) - 1:
                    chunk[-crossfade_samples:] *= fade_out

            parts.append(chunk)
            updated_timestamps.append({
                "word": wt["word"],
                "start": round(wt["start"] + cumulative_offset, 4),
                "end": round(wt["end"] + cumulative_offset, 4),
            })

    last_end = int(word_timestamps[-1]["end"] * sample_rate)
    if last_end < len(audio):
        parts.append(audio[last_end:])

    return np.concatenate(parts), updated_timestamps


# ---------------------------------------------------------------------------
# Core synthesis (ONNX)
# ---------------------------------------------------------------------------

def _synthesise(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
) -> np.ndarray:
    """Run Kokoro ONNX TTS and return numpy audio array."""
    kokoro = _get_kokoro()
    onnx_lang = LANG_CODE_MAP.get(lang_code, "en-us")

    samples, sr = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        lang=onnx_lang,
    )

    return samples.astype(np.float32)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless handler entry-point.

    Expected input schema::

        {
            "input": {
                "text": "Hello, world!",              # required
                "voice": "af_heart",                   # optional, default "af_heart"
                "speed": 1.0,                          # optional, default 1.0
                "lang_code": "a",                      # optional, default "a"
                "timestamps": true,                    # optional, return word timestamps
                "word_boundaries": true,               # optional, return boundary analysis
                "micro_pause_ms": 0,                   # optional, insert pauses (0 = off)
                "crossfade_ms": 5.0                    # optional, crossfade duration
            }
        }
    """
    start_time = time.perf_counter()

    try:
        job_input: dict[str, Any] = job.get("input", {})

        # ---- Parse inputs ----
        text: str = job_input.get("text", "").strip()
        if not text:
            return {"error": "Missing or empty 'text' field in input."}

        voice: str = job_input.get("voice", "af_heart")
        speed: float = float(job_input.get("speed", 1.0))
        lang_code: str = job_input.get("lang_code", "a").lower()
        want_timestamps: bool = bool(job_input.get("timestamps", False))
        want_boundaries: bool = bool(job_input.get("word_boundaries", False))
        micro_pause_ms: float = float(job_input.get("micro_pause_ms", 0))
        crossfade_ms: float = float(job_input.get("crossfade_ms", 5.0))

        if micro_pause_ms > 0:
            want_timestamps = True
            want_boundaries = True

        # ---- Validate ----
        if lang_code not in VALID_LANG_CODES:
            return {
                "error": f"Invalid lang_code '{lang_code}'. Valid: {list(VALID_LANG_CODES.keys())}",
            }

        if not (0.1 <= speed <= 5.0):
            return {"error": f"Speed must be between 0.1 and 5.0, got {speed}."}

        # Guard: FA not supported for non-Latin scripts
        if (want_timestamps or want_boundaries) and lang_code not in FA_SUPPORTED_LANG_CODES:
            logger.warning(
                "FA not supported for lang_code='%s'. Timestamps disabled.", lang_code
            )
            want_timestamps = False
            want_boundaries = False
            micro_pause_ms = 0

        logger.info(
            "Job: text_len=%d, voice=%s, speed=%.2f, lang=%s, ts=%s, bounds=%s, pause=%.1fms",
            len(text), voice, speed, lang_code,
            want_timestamps, want_boundaries, micro_pause_ms,
        )

        # ---- Step 1: Synthesise (ONNX FP16 + CUDA) ----
        synth_start = time.perf_counter()
        audio = _synthesise(text, voice, speed, lang_code)
        synth_elapsed = time.perf_counter() - synth_start
        logger.info("ONNX synthesis complete in %.3fs", synth_elapsed)

        # ---- Step 2: Forced alignment for timestamps ----
        word_ts: list[dict] = []
        boundaries: list[dict] = []

        if want_timestamps or want_boundaries:
            word_ts = _get_word_timestamps(audio, text, SAMPLE_RATE)
            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts)

        # ---- Step 3: Insert micro-pauses if requested ----
        if micro_pause_ms > 0 and word_ts:
            logger.info("Inserting %.1fms pauses with %.1fms crossfade", micro_pause_ms, crossfade_ms)
            audio, word_ts = _insert_micro_pauses(
                audio, word_ts,
                pause_ms=micro_pause_ms,
                crossfade_ms=crossfade_ms,
                sample_rate=SAMPLE_RATE,
            )
            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts)

        # ---- Encode result ----
        audio_b64 = _audio_to_base64_wav(audio)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.perf_counter() - start_time
        rtf = elapsed / max(duration, 0.001)

        synth_rtf = synth_elapsed / max(duration, 0.001)
        logger.info("Done: duration=%.2fs, processing=%.2fs, RTF=%.4f, synth_RTF=%.4f", duration, elapsed, rtf, synth_rtf)

        result: dict[str, Any] = {
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": round(duration, 3),
            "rtf": round(rtf, 4),
            "synth_rtf": round(synth_rtf, 4),
        }

        if want_timestamps:
            result["word_timestamps"] = word_ts
        if want_boundaries:
            result["word_boundaries"] = boundaries

        return result

    except Exception as exc:
        logger.exception("Handler error")
        return {"error": f"Synthesis failed: {exc}"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
