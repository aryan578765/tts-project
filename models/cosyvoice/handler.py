"""
CosyVoice 2.0 — RunPod Serverless Handler
==========================================

Production handler for CosyVoice2-0.5B on RunPod serverless infrastructure.

Input schema
------------
{
    "input": {
        "text":        str   — Text to synthesise (required).
        "voice":       str   — Speaker ID / name for SFT mode (default: "中文女").
        "speed":       float — Playback speed multiplier (default: 1.0).
        "language":    str   — ISO 639-1 language code (default: "zh").
                                Supported: zh, en, ja, ko, de, es, fr, it, ru.
        "mode":        str   — Inference mode (default: "sft").
                                "sft"       → standard speaker-fine-tuned voice.
                                "zero_shot" → clone voice from reference audio.
                                "instruct"  → control emotion / style via text
                                              instruction.
        "instruction": str   — Natural-language instruction for instruct mode
                                (e.g. "Speak with excitement"). Ignored in
                                other modes.
        "ref_audio":   str   — Base64-encoded WAV of reference speaker for
                                zero_shot mode. Ignored in other modes.
        "ref_text":    str   — Transcript of reference audio for zero_shot
                                mode. Ignored in other modes.
    }
}

Output
------
{
    "audio":       str — Base64-encoded WAV.
    "sample_rate": int — Sample rate of the output audio.
    "duration":    float — Duration in seconds.
}
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Generator, Optional

import numpy as np
import runpod
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cosyvoice-handler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "/app/models/CosyVoice2-0.5B")
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/app/CosyVoice")

SUPPORTED_LANGUAGES = {"zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"}
SUPPORTED_MODES = {"sft", "zero_shot", "instruct"}
DEFAULT_VOICE = "中文女"
DEFAULT_SAMPLE_RATE = 22050  # CosyVoice default output rate

# ---------------------------------------------------------------------------
# Model singleton — loaded once, reused across warm invocations
# ---------------------------------------------------------------------------
_model: Optional[Any] = None


def _load_model() -> Any:
    """Load CosyVoice2 model into GPU memory (singleton)."""
    global _model
    if _model is not None:
        return _model

    logger.info("Loading CosyVoice2 model from %s …", MODEL_DIR)
    start = time.perf_counter()

    # Ensure CosyVoice source is importable
    if COSYVOICE_ROOT not in sys.path:
        sys.path.insert(0, COSYVOICE_ROOT)
    matcha_path = os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS")
    if matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=True)
    except ImportError:
        # Fallback: some repo versions export as CosyVoice with a version flag
        from cosyvoice.cli.cosyvoice import CosyVoice
        _model = CosyVoice(MODEL_DIR, load_jit=False, load_trt=False, fp16=True)

    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2f s", elapsed)
    return _model


def _collect_audio_chunks(generator: Generator) -> np.ndarray:
    """Concatenate streamed audio chunks from the CosyVoice generator.

    CosyVoice inference methods yield dicts with a ``tts_speech`` tensor.
    We concatenate all chunks and return a single numpy array.
    """
    chunks: list[np.ndarray] = []
    for chunk in generator:
        if isinstance(chunk, dict) and "tts_speech" in chunk:
            audio_tensor = chunk["tts_speech"]
        elif isinstance(chunk, torch.Tensor):
            audio_tensor = chunk
        else:
            # Unexpected type — try to use directly
            audio_tensor = chunk

        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.squeeze().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor).squeeze()

        chunks.append(audio_np)

    if not chunks:
        raise RuntimeError("Model produced no audio output.")

    return np.concatenate(chunks)


def _apply_speed(audio: np.ndarray, speed: float, sr: int) -> tuple[np.ndarray, int]:
    """Time-stretch *audio* by *speed* factor without changing pitch.

    Uses simple resampling for speed changes — adequate for TTS output.
    Falls back to no-op on import errors.
    """
    if abs(speed - 1.0) < 0.01:
        return audio, sr

    try:
        import librosa

        audio = librosa.effects.time_stretch(audio, rate=speed)
    except Exception:
        # Naive fallback: resample to emulate speed change
        target_len = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(np.int64)
        audio = audio[indices]

    return audio, sr


def _audio_to_base64_wav(audio: np.ndarray, sr: int) -> str:
    """Encode a numpy audio array as a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_ref_audio(b64_audio: str) -> tuple[np.ndarray, int]:
    """Decode a base64-encoded WAV into numpy array + sample rate."""
    raw = base64.b64decode(b64_audio)
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    return audio, sr


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler for CosyVoice 2.0 TTS inference.

    Parameters
    ----------
    job : dict
        RunPod job payload. Must contain an ``input`` key with at least a
        ``text`` field.

    Returns
    -------
    dict
        ``audio`` (base64 WAV), ``sample_rate``, ``duration``.
    """
    try:
        payload: Dict[str, Any] = job.get("input", {})

        # -- Validate required fields ----------------------------------------
        text: str = payload.get("text", "").strip()
        if not text:
            return {"error": "Field 'text' is required and must be non-empty."}

        mode: str = payload.get("mode", "sft").lower()
        if mode not in SUPPORTED_MODES:
            return {
                "error": f"Unsupported mode '{mode}'. Choose from: {SUPPORTED_MODES}."
            }

        language: str = payload.get("language", "zh").lower()
        if language not in SUPPORTED_LANGUAGES:
            return {
                "error": (
                    f"Unsupported language '{language}'. "
                    f"Supported: {sorted(SUPPORTED_LANGUAGES)}."
                )
            }

        voice: str = payload.get("voice", DEFAULT_VOICE)
        speed: float = float(payload.get("speed", 1.0))
        instruction: str = payload.get("instruction", "")
        ref_audio_b64: str = payload.get("ref_audio", "")
        ref_text: str = payload.get("ref_text", "")

        if speed <= 0 or speed > 3.0:
            return {"error": "Speed must be between 0.01 and 3.0."}

        # -- Load model -------------------------------------------------------
        model = _load_model()

        # -- Determine sample rate from model if available --------------------
        sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)

        # -- Run inference based on mode --------------------------------------
        logger.info(
            "Generating | mode=%s lang=%s voice=%s speed=%.2f len(text)=%d",
            mode, language, voice, speed, len(text),
        )
        start = time.perf_counter()

        if mode == "sft":
            # Standard speaker-fine-tuned synthesis
            audio_gen = model.inference_sft(
                tts_text=text,
                spk_id=voice,
            )

        elif mode == "zero_shot":
            # Voice cloning from reference audio
            if not ref_audio_b64:
                return {
                    "error": (
                        "zero_shot mode requires 'ref_audio' (base64 WAV) "
                        "and 'ref_text' fields."
                    )
                }
            # Decode reference audio and save to a temp WAV for the model
            ref_audio_np, ref_sr = _decode_ref_audio(ref_audio_b64)
            ref_wav_path = "/tmp/_cosyvoice_ref.wav"
            sf.write(ref_wav_path, ref_audio_np, ref_sr, format="WAV")

            audio_gen = model.inference_zero_shot(
                tts_text=text,
                prompt_text=ref_text,
                prompt_speech_16k=ref_wav_path,
            )

        elif mode == "instruct":
            # Instruction-controlled synthesis (emotion, style, etc.)
            if not instruction:
                return {
                    "error": (
                        "instruct mode requires a non-empty 'instruction' "
                        "field (e.g. 'Speak with excitement')."
                    )
                }
            audio_gen = model.inference_instruct2(
                tts_text=text,
                instruct_text=instruction,
                spk_id=voice,
            )
        else:
            return {"error": f"Unhandled mode '{mode}'."}

        # -- Collect audio chunks ---------------------------------------------
        audio = _collect_audio_chunks(audio_gen)

        # -- Apply speed adjustment -------------------------------------------
        audio, sample_rate = _apply_speed(audio, speed, sample_rate)

        elapsed = time.perf_counter() - start
        duration = len(audio) / sample_rate
        logger.info(
            "Done | duration=%.2f s  inference_time=%.2f s  rtf=%.3f",
            duration, elapsed, elapsed / max(duration, 0.01),
        )

        # -- Encode output ----------------------------------------------------
        audio_b64 = _audio_to_base64_wav(audio, sample_rate)

        return {
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "duration": round(duration, 4),
        }

    except Exception as exc:
        logger.error("Handler error: %s\n%s", exc, traceback.format_exc())
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting CosyVoice 2.0 RunPod serverless handler …")
    # Model will be loaded lazily on first request to avoid crashing the worker
    try:
        _load_model()
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning("Model pre-warm failed (will retry on first request): %s", e)
    runpod.serverless.start({"handler": handler})
