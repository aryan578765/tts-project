"""
Kokoro TTS v2 - RunPod Serverless Handler
==========================================

Optimized handler with:
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
from typing import Any, Optional

import runpod
import soundfile as sf
import torch
import torchaudio
import numpy as np

from kokoro import KPipeline

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
CLEAN_BOUNDARY_THRESHOLD_MS = 50   # gap >= 50ms -> clean, safe to cut
COARTICULATED_THRESHOLD_MS = 20    # gap < 20ms -> coarticulated, needs crossfade

# ---------------------------------------------------------------------------
# Thread-safe global caches
# ---------------------------------------------------------------------------
_pipelines: dict[str, KPipeline] = {}
_pipeline_lock = threading.Lock()

_fa_model = None
_fa_tokenizer = None
_fa_aligner = None
_fa_dict = None
_fa_lock = threading.Lock()


def _get_pipeline(lang_code: str) -> KPipeline:
    """Return a cached KPipeline for the given language code (thread-safe)."""
    if lang_code not in _pipelines:
        with _pipeline_lock:
            if lang_code not in _pipelines:  # double-check after lock
                logger.info("Initialising KPipeline for lang_code='%s' ...", lang_code)
                start = time.perf_counter()
                _pipelines[lang_code] = KPipeline(lang_code=lang_code)
                elapsed = time.perf_counter() - start
                logger.info("KPipeline for '%s' ready in %.2fs", lang_code, elapsed)
    return _pipelines[lang_code]


def _get_fa_components():
    """Load forced alignment model, tokenizer, aligner, and dict (thread-safe).

    Uses torchaudio MMS_FA high-level API:
      - get_model(with_star=False): Wav2Vec2 model for emission generation
      - get_tokenizer(): converts word lists to token sequences
      - get_aligner(): aligns emissions with tokens, returns TokenSpan lists
      - get_dict(): {char: index} mapping for manual tokenization if needed

    Model runs on CPU (fast enough, saves GPU VRAM for Kokoro).
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
    logger.info("Pre-warming default pipeline (lang_code='a') ...")
    _get_pipeline("a")
except Exception:
    logger.exception("Failed to pre-warm default pipeline")

try:
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
    """Find nearest zero-crossing to index within search_range samples.
    Cuts at zero-crossings prevent clicks/pops in audio splicing.
    """
    best = index
    for offset in range(1, search_range):
        # Search forward
        fwd = index + offset
        if fwd < len(audio) - 1:
            if audio[fwd] * audio[fwd + 1] <= 0:
                return fwd
        # Search backward
        bwd = index - offset
        if bwd >= 0 and bwd < len(audio) - 1:
            if audio[bwd] * audio[bwd + 1] <= 0:
                return bwd
    return best  # fallback to original index


# ---------------------------------------------------------------------------
# Forced Alignment — Word-level timestamps
# ---------------------------------------------------------------------------

def _clean_word(word: str) -> str:
    """Strip punctuation from a word for alignment matching."""
    # Remove common punctuation (including unicode variants)
    return re.sub(r'[^\w\s]', '', word, flags=re.UNICODE).strip()


def _get_word_timestamps(
    audio: np.ndarray,
    transcript: str,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """Get word-level timestamps using torchaudio MMS_FA high-level API.

    Uses get_tokenizer() + get_aligner() for robust alignment instead of
    manual tokenization + raw forced_align().

    Args:
        audio: numpy audio array (24kHz from Kokoro)
        transcript: the known text transcript
        sample_rate: audio sample rate

    Returns:
        List of {"word": str, "start": float, "end": float} dicts
    """
    model, tokenizer, aligner, fa_dict = _get_fa_components()
    start_time = time.perf_counter()

    # Convert numpy to torch tensor [1, T]
    waveform = torch.from_numpy(audio).float().unsqueeze(0)

    # Resample from 24kHz -> 16kHz (required by MMS_FA)
    if sample_rate != FA_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, FA_SAMPLE_RATE
        )

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 1: Generate emission probabilities from wav2vec2
    with torch.inference_mode():
        emission, _ = model(waveform)

    # CRITICAL: Convert to log probabilities (required by forced_align)
    emission = torch.log_softmax(emission, dim=-1)

    # Step 2: Prepare word list for tokenization
    words_raw = transcript.split()
    words_clean = []
    words_display = []
    for w in words_raw:
        cleaned = _clean_word(w)
        if cleaned:
            words_clean.append(cleaned)
            words_display.append(w)  # preserve original for display

    if not words_clean:
        logger.warning("No valid words found for alignment in: %s", transcript[:50])
        return []

    # Step 3: Tokenize using the high-level tokenizer
    try:
        token_spans = tokenizer(words_clean)
    except Exception as e:
        logger.warning("Tokenization failed: %s", e)
        return []

    # Step 4: Align using the high-level aligner
    try:
        alignment = aligner(emission[0], token_spans)
    except Exception as e:
        logger.warning("Alignment failed: %s", e)
        return []

    # Step 5: Convert frame indices to timestamps
    # ratio = samples per emission frame
    ratio = waveform.shape[1] / emission.shape[1]

    word_timestamps = []
    for word_idx, word_spans in enumerate(alignment):
        if word_idx >= len(words_display):
            break
        if not word_spans:
            continue

        # word_spans is a list of TokenSpan for each character in the word
        start_frame = word_spans[0].start
        end_frame = word_spans[-1].end  # end is exclusive

        start_sec = (start_frame * ratio) / FA_SAMPLE_RATE
        end_sec = (end_frame * ratio) / FA_SAMPLE_RATE

        word_timestamps.append({
            "word": words_display[word_idx],
            "start": round(start_sec, 4),
            "end": round(end_sec, 4),
        })

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Forced alignment completed in %.3fs for %d words",
        elapsed, len(word_timestamps),
    )

    return word_timestamps


# ---------------------------------------------------------------------------
# Word Boundary Analysis
# ---------------------------------------------------------------------------

def _analyze_word_boundaries(
    word_timestamps: list[dict],
) -> list[dict]:
    """Analyze gaps between consecutive words.

    Returns a list of boundary reports indicating which word pairs
    can be cleanly separated and which are coarticulated.
    """
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
            # Negative gap means alignment overlap (words overlap in time)
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
    """Generate equal-power (cos/sin) fade curves.

    Equal-power crossfades maintain constant energy across the transition,
    preventing the ~3dB volume dip that linear fades produce.
    """
    t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)
    fade_out = np.cos(t).astype(np.float32)
    fade_in = np.sin(t).astype(np.float32)
    return fade_out, fade_in


def _insert_micro_pauses(
    audio: np.ndarray,
    word_timestamps: list[dict],
    pause_ms: float = 10.0,
    crossfade_ms: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[dict]]:
    """Insert micro-pauses between words using crossfade blending.

    Generates the full sentence with natural intonation first (already done),
    then inserts tiny silences at word boundaries using equal-power crossfade
    to prevent abrupt starts/ends.

    Args:
        audio: original full-sentence audio (natural intonation preserved)
        word_timestamps: word-level timestamps from forced alignment
        pause_ms: duration of silence to insert between words
        crossfade_ms: duration of crossfade at cut points
        sample_rate: audio sample rate

    Returns:
        Tuple of (modified_audio, updated_word_timestamps)
    """
    if not word_timestamps or len(word_timestamps) < 2:
        return audio, word_timestamps

    # Ensure float32
    audio = audio.astype(np.float32)

    # Remove DC offset
    audio = audio - np.mean(audio)

    pause_samples = int(sample_rate * pause_ms / 1000)
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    crossfade_samples = max(crossfade_samples, 4)  # minimum 4 samples

    # Generate equal-power fade curves
    fade_out, fade_in = _equal_power_fades(crossfade_samples)

    # Build output by cutting between words and inserting pauses
    parts = []
    updated_timestamps = []
    cumulative_offset = 0.0  # track time shift from inserted pauses

    for i, wt in enumerate(word_timestamps):
        start_sample = int(wt["start"] * sample_rate)
        end_sample = int(wt["end"] * sample_rate)

        # Clamp to audio bounds
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(start_sample, min(end_sample, len(audio)))

        if i == 0:
            # First word: include everything from start of audio to end of word
            chunk = audio[:end_sample].copy()
            # Apply fade-out at the end
            if len(chunk) > crossfade_samples:
                chunk[-crossfade_samples:] *= fade_out
            parts.append(chunk)

            updated_timestamps.append({
                "word": wt["word"],
                "start": round(wt["start"] + cumulative_offset, 4),
                "end": round(wt["end"] + cumulative_offset, 4),
            })
        else:
            # Insert silence between previous word and this word
            silence = np.zeros(pause_samples, dtype=np.float32)
            parts.append(silence)
            cumulative_offset += pause_ms / 1000.0

            # Extract this word's audio with fade-in at start
            # Find best cut point near word start using zero-crossing
            cut_start = _find_zero_crossing(audio, start_sample)
            chunk = audio[cut_start:end_sample].copy()

            if len(chunk) > crossfade_samples:
                chunk[:crossfade_samples] *= fade_in
                # Apply fade-out at end (for next pause insertion)
                if i < len(word_timestamps) - 1:
                    chunk[-crossfade_samples:] *= fade_out

            parts.append(chunk)

            updated_timestamps.append({
                "word": wt["word"],
                "start": round(wt["start"] + cumulative_offset, 4),
                "end": round(wt["end"] + cumulative_offset, 4),
            })

    # Include trailing audio after the last word
    last_end = int(word_timestamps[-1]["end"] * sample_rate)
    if last_end < len(audio):
        trailing = audio[last_end:].copy()
        parts.append(trailing)

    result = np.concatenate(parts)
    return result, updated_timestamps


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def _synthesise(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
) -> np.ndarray:
    """Run Kokoro TTS and return the concatenated audio numpy array."""
    pipeline = _get_pipeline(lang_code)

    audio_chunks: list[np.ndarray] = []
    for _gs, _ps, audio_chunk in pipeline(
        text,
        voice=voice,
        speed=speed,
    ):
        if audio_chunk is not None:
            if isinstance(audio_chunk, torch.Tensor):
                audio_chunk = audio_chunk.cpu().numpy()
            audio_chunks.append(audio_chunk.astype(np.float32))

    if not audio_chunks:
        raise ValueError("Kokoro produced no audio for the given input.")

    return np.concatenate(audio_chunks)


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

    Returns::

        {
            "audio_base64": "<base64 WAV string>",
            "sample_rate": 24000,
            "duration_seconds": 2.35,
            "rtf": 0.04,
            "word_timestamps": [...],                  # if requested
            "word_boundaries": [...]                   # if requested
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

        # If micro_pause requested, force timestamps and boundaries on
        if micro_pause_ms > 0:
            want_timestamps = True
            want_boundaries = True

        # ---- Validate inputs ----
        if lang_code not in VALID_LANG_CODES:
            return {
                "error": (
                    f"Invalid lang_code '{lang_code}'. "
                    f"Valid options: {list(VALID_LANG_CODES.keys())}"
                ),
            }

        if not (0.1 <= speed <= 5.0):
            return {"error": f"Speed must be between 0.1 and 5.0, got {speed}."}

        # Check if timestamps are supported for this language
        if (want_timestamps or want_boundaries) and lang_code not in FA_SUPPORTED_LANG_CODES:
            logger.warning(
                "Forced alignment not supported for lang_code='%s' (%s). "
                "Timestamps will not be returned. Supported: %s",
                lang_code, VALID_LANG_CODES[lang_code],
                list(FA_SUPPORTED_LANG_CODES),
            )
            want_timestamps = False
            want_boundaries = False
            micro_pause_ms = 0

        logger.info(
            "Job received: text_len=%d, voice=%s, speed=%.2f, lang=%s, "
            "timestamps=%s, boundaries=%s, micro_pause=%.1fms",
            len(text), voice, speed, lang_code,
            want_timestamps, want_boundaries, micro_pause_ms,
        )

        # ---- Step 1: Synthesise full sentence (preserves natural intonation) ----
        synth_start = time.perf_counter()
        audio = _synthesise(text, voice, speed, lang_code)
        synth_elapsed = time.perf_counter() - synth_start
        logger.info("Synthesis complete in %.3fs", synth_elapsed)

        # ---- Step 2: Forced alignment for timestamps ----
        word_ts: list[dict] = []
        boundaries: list[dict] = []

        if want_timestamps or want_boundaries:
            word_ts = _get_word_timestamps(audio, text, SAMPLE_RATE)

            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts)

        # ---- Step 3: Insert micro-pauses if requested ----
        if micro_pause_ms > 0 and word_ts:
            logger.info(
                "Inserting %.1fms micro-pauses with %.1fms crossfade",
                micro_pause_ms, crossfade_ms,
            )
            audio, word_ts = _insert_micro_pauses(
                audio, word_ts,
                pause_ms=micro_pause_ms,
                crossfade_ms=crossfade_ms,
                sample_rate=SAMPLE_RATE,
            )
            # Recalculate boundaries with updated timestamps
            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts)

        # ---- Encode result ----
        audio_b64 = _audio_to_base64_wav(audio)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.perf_counter() - start_time
        rtf = elapsed / max(duration, 0.001)

        logger.info(
            "Complete: duration=%.2fs, processing=%.2fs, RTF=%.4f",
            duration, elapsed, rtf,
        )

        result: dict[str, Any] = {
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": round(duration, 3),
            "rtf": round(rtf, 4),
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
