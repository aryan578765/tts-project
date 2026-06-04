"""
Kokoro TTS - RunPod Serverless Handler
=======================================

Production-ready serverless handler for Kokoro v1.0 text-to-speech inference.
Loads the KPipeline globally for warm starts, accepts text input, and returns
base64-encoded WAV audio.

Supported language codes:
    a - American English
    b - British English
    e - Spanish (es)
    f - French (fr)
    i - Italian (it)
    j - Japanese (ja)
    p - Portuguese (pt)
    z - Chinese (zh)
"""

import base64
import io
import logging
import os
import re
import time
from typing import Any

import runpod
import soundfile as sf
import torch
import numpy as np

from kokoro import KPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kokoro-handler")

# ---------------------------------------------------------------------------
# Valid language codes supported by Kokoro
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

# ---------------------------------------------------------------------------
# Global pipeline cache – one pipeline per language code, lazily initialised.
# Keeping them global means they survive across warm invocations on RunPod.
# ---------------------------------------------------------------------------
_pipelines: dict[str, KPipeline] = {}

SAMPLE_RATE: int = 24000  # Kokoro outputs 24 kHz audio


def _get_pipeline(lang_code: str) -> KPipeline:
    """Return a cached KPipeline for the given language code, creating it if needed."""
    if lang_code not in _pipelines:
        logger.info("Initialising KPipeline for lang_code='%s' ...", lang_code)
        start = time.perf_counter()
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
        elapsed = time.perf_counter() - start
        logger.info("KPipeline for '%s' ready in %.2fs", lang_code, elapsed)
    return _pipelines[lang_code]


# Pre-warm the default (American English) pipeline at import time so the
# first request doesn't pay the full cold-start cost.
try:
    logger.info("Pre-warming default pipeline (lang_code='a') ...")
    _get_pipeline("a")
except Exception:
    logger.exception("Failed to pre-warm default pipeline – will retry on first request")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _audio_to_base64_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode a numpy audio array as a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _split_ssml_breaks(text: str) -> list[tuple[str, float]]:
    """Split text on SSML ``<break>`` tags.

    Returns a list of ``(segment_text, pause_seconds)`` tuples.  The pause
    value comes from the ``time`` attribute of the **preceding** ``<break>``
    tag (default 0.5 s).  The first segment always has pause = 0.
    """
    # Match <break time="500ms"/> or <break time="1.5s"/> or just <break/>
    pattern = re.compile(
        r'<break\s*(?:time\s*=\s*"([\d.]+)(ms|s)")?\s*/?>',
        re.IGNORECASE,
    )
    segments: list[tuple[str, float]] = []
    last_end = 0

    for m in pattern.finditer(text):
        chunk = text[last_end : m.start()].strip()
        if chunk:
            segments.append((chunk, 0.0))

        # Parse pause duration
        if m.group(1) is not None:
            value = float(m.group(1))
            unit = m.group(2).lower()
            pause = value / 1000.0 if unit == "ms" else value
        else:
            pause = 0.5  # default pause

        # Attach pause to *preceding* segment or store as empty marker
        if segments:
            prev_text, _ = segments[-1]
            segments[-1] = (prev_text, pause)
        else:
            segments.append(("", pause))

        last_end = m.end()

    # Trailing text after the last <break>
    remaining = text[last_end:].strip()
    if remaining:
        segments.append((remaining, 0.0))

    # If no <break> tags were found, return the whole text as one segment
    if not segments:
        segments.append((text.strip(), 0.0))

    return segments


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def _synthesise(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
    split_pattern: str | None = None,
) -> np.ndarray:
    """Run Kokoro TTS and return the concatenated audio numpy array."""
    pipeline = _get_pipeline(lang_code)

    audio_chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=split_pattern,
    ):
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("Kokoro produced no audio for the given input.")

    return np.concatenate(audio_chunks)


def _synthesise_ssml(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
) -> np.ndarray:
    """Synthesise text that contains SSML <break> tags.

    Each segment between breaks is synthesised individually, and silence is
    inserted according to the ``time`` attribute of each ``<break>`` tag.
    """
    segments = _split_ssml_breaks(text)
    audio_parts: list[np.ndarray] = []

    for segment_text, pause_after in segments:
        if segment_text:
            audio_parts.append(_synthesise(segment_text, voice, speed, lang_code))
        if pause_after > 0:
            silence = np.zeros(int(SAMPLE_RATE * pause_after), dtype=np.float32)
            audio_parts.append(silence)

    if not audio_parts:
        raise ValueError("No audio segments produced from SSML input.")

    return np.concatenate(audio_parts)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless handler entry-point.

    Expected input schema::

        {
            "input": {
                "text": "Hello, world!",          # required
                "voice": "af_heart",               # optional, default "af_heart"
                "speed": 1.0,                      # optional, default 1.0
                "lang_code": "a",                  # optional, default "a"
                "ssml": false                      # optional, default false
            }
        }

    Returns::

        {
            "audio_base64": "<base64 WAV string>",
            "sample_rate": 24000,
            "duration_seconds": 2.35
        }
    """
    start_time = time.perf_counter()

    try:
        job_input: dict[str, Any] = job.get("input", {})

        # ---- Validate required fields ----
        text: str = job_input.get("text", "").strip()
        if not text:
            return {"error": "Missing or empty 'text' field in input."}

        voice: str = job_input.get("voice", "af_heart")
        speed: float = float(job_input.get("speed", 1.0))
        lang_code: str = job_input.get("lang_code", "a").lower()
        use_ssml: bool = bool(job_input.get("ssml", False))

        # ---- Validate language code ----
        if lang_code not in VALID_LANG_CODES:
            return {
                "error": (
                    f"Invalid lang_code '{lang_code}'. "
                    f"Valid options: {list(VALID_LANG_CODES.keys())}"
                ),
            }

        # ---- Validate speed ----
        if not (0.1 <= speed <= 5.0):
            return {"error": f"Speed must be between 0.1 and 5.0, got {speed}."}

        logger.info(
            "Job received – text length=%d, voice=%s, speed=%.2f, lang=%s, ssml=%s",
            len(text), voice, speed, lang_code, use_ssml,
        )

        # ---- Synthesise ----
        if use_ssml or "<break" in text.lower():
            audio = _synthesise_ssml(text, voice, speed, lang_code)
        else:
            audio = _synthesise(text, voice, speed, lang_code)

        # ---- Encode result ----
        audio_b64 = _audio_to_base64_wav(audio)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.perf_counter() - start_time

        logger.info(
            "Synthesis complete – duration=%.2fs, processing_time=%.2fs, RTF=%.2f",
            duration, elapsed, elapsed / max(duration, 0.001),
        )

        return {
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": round(duration, 3),
        }

    except Exception as exc:
        logger.exception("Handler error")
        return {"error": f"Synthesis failed: {exc}"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
