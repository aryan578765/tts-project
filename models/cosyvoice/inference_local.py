#!/usr/bin/env python3
"""
CosyVoice 2.0 — Local Inference Script
=======================================

Standalone script for running CosyVoice2-0.5B inference without RunPod.
Generates speech from text and saves it as a WAV file.

Usage
-----
    # Basic SFT mode (default speaker)
    python inference_local.py --text "Hello world" --language en --output output.wav

    # Instruct mode with emotion control
    python inference_local.py \\
        --text "This is amazing news!" \\
        --mode instruct \\
        --instruction "Speak with excitement and joy" \\
        --output excited.wav

    # Zero-shot voice cloning
    python inference_local.py \\
        --text "Cloned voice speaking" \\
        --mode zero_shot \\
        --ref-audio reference.wav \\
        --ref-text "Transcript of the reference audio" \\
        --output cloned.wav

    # Custom speed
    python inference_local.py --text "Slow speech" --speed 0.8 --output slow.wav

Environment
-----------
Set ``COSYVOICE_MODEL_DIR`` to override the default model path.
Set ``COSYVOICE_ROOT`` to override the CosyVoice source directory.

Requirements
------------
See requirements.txt (torch, torchaudio, soundfile, numpy, etc.).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cosyvoice-local")

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    # Sensible defaults — adjust to your local setup
    str(Path(__file__).resolve().parent / "pretrained_models" / "CosyVoice2-0.5B"),
)
COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT",
    str(Path(__file__).resolve().parent / "CosyVoice"),
)

SUPPORTED_LANGUAGES = {"zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"}
SUPPORTED_MODES = {"sft", "zero_shot", "instruct"}
DEFAULT_VOICE = "中文女"
DEFAULT_SAMPLE_RATE = 22050


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pythonpath() -> None:
    """Add CosyVoice source and Matcha-TTS to sys.path if not present."""
    for extra in [
        COSYVOICE_ROOT,
        os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    ]:
        if extra not in sys.path and os.path.isdir(extra):
            sys.path.insert(0, extra)
            logger.debug("Added to sys.path: %s", extra)


def _load_model(model_dir: str) -> Any:
    """Load CosyVoice2 model from *model_dir*.

    Attempts to import ``CosyVoice2`` first (CosyVoice v2 API), falling back
    to ``CosyVoice`` for older repo versions.

    Parameters
    ----------
    model_dir : str
        Absolute path to the downloaded model directory.

    Returns
    -------
    model
        Loaded CosyVoice model instance.
    """
    _ensure_pythonpath()

    logger.info("Loading model from %s …", model_dir)
    start = time.perf_counter()

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)
    except ImportError:
        logger.warning("CosyVoice2 class not found — falling back to CosyVoice.")
        from cosyvoice.cli.cosyvoice import CosyVoice
        model = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=True)

    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2f s", elapsed)
    return model


def _collect_audio_chunks(generator: Generator) -> np.ndarray:
    """Concatenate all audio chunks yielded by the CosyVoice generator.

    Each chunk is typically a dict with key ``tts_speech`` holding a torch
    tensor, or a raw tensor.

    Parameters
    ----------
    generator : Generator
        The inference generator from CosyVoice.

    Returns
    -------
    np.ndarray
        Concatenated 1-D float32 audio array.

    Raises
    ------
    RuntimeError
        If the generator yields no audio chunks.
    """
    chunks: list[np.ndarray] = []
    for chunk in generator:
        if isinstance(chunk, dict) and "tts_speech" in chunk:
            audio_tensor = chunk["tts_speech"]
        elif isinstance(chunk, torch.Tensor):
            audio_tensor = chunk
        else:
            audio_tensor = chunk

        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.squeeze().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor, dtype=np.float32).squeeze()

        chunks.append(audio_np)

    if not chunks:
        raise RuntimeError("Model produced no audio output.")

    return np.concatenate(chunks)


def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Apply time-stretch to *audio* by *speed* factor (no pitch change).

    Parameters
    ----------
    audio : np.ndarray
        1-D audio signal.
    speed : float
        Speed multiplier (>1 = faster, <1 = slower).

    Returns
    -------
    np.ndarray
        Time-stretched audio array.
    """
    if abs(speed - 1.0) < 0.01:
        return audio

    try:
        import librosa

        return librosa.effects.time_stretch(audio, rate=speed)
    except ImportError:
        logger.warning("librosa not installed — using naive resampling for speed.")
        target_len = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(np.int64)
        return audio[indices]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def synthesize(
    model: Any,
    text: str,
    *,
    mode: str = "sft",
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    language: str = "zh",
    instruction: str = "",
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
) -> tuple[np.ndarray, int]:
    """Run TTS inference and return (audio_array, sample_rate).

    Parameters
    ----------
    model : Any
        Loaded CosyVoice model.
    text : str
        Text to synthesise.
    mode : str
        One of ``sft``, ``zero_shot``, ``instruct``.
    voice : str
        Speaker ID for SFT / instruct modes.
    speed : float
        Playback speed multiplier.
    language : str
        ISO 639-1 language code.
    instruction : str
        Natural-language style instruction (instruct mode only).
    ref_audio_path : str or None
        Path to reference WAV for zero_shot mode.
    ref_text : str
        Transcript of the reference audio for zero_shot mode.

    Returns
    -------
    tuple[np.ndarray, int]
        Audio samples and sample rate.
    """
    sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)

    logger.info(
        "Synthesising | mode=%s  lang=%s  voice=%s  speed=%.2f  chars=%d",
        mode, language, voice, speed, len(text),
    )
    start = time.perf_counter()

    if mode == "sft":
        audio_gen = model.inference_sft(tts_text=text, spk_id=voice)

    elif mode == "zero_shot":
        if not ref_audio_path or not os.path.isfile(ref_audio_path):
            raise FileNotFoundError(
                f"Reference audio not found: {ref_audio_path}. "
                "Provide a valid WAV file with --ref-audio."
            )
        audio_gen = model.inference_zero_shot(
            tts_text=text,
            prompt_text=ref_text,
            prompt_speech_16k=ref_audio_path,
        )

    elif mode == "instruct":
        if not instruction:
            raise ValueError(
                "Instruct mode requires --instruction (e.g. 'Speak angrily')."
            )
        audio_gen = model.inference_instruct2(
            tts_text=text,
            instruct_text=instruction,
            spk_id=voice,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {SUPPORTED_MODES}")

    audio = _collect_audio_chunks(audio_gen)
    audio = _apply_speed(audio, speed)

    elapsed = time.perf_counter() - start
    duration = len(audio) / sample_rate
    logger.info(
        "Done | duration=%.2f s  time=%.2f s  RTF=%.3f",
        duration, elapsed, elapsed / max(duration, 0.01),
    )

    return audio, sample_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CosyVoice 2.0 — Local TTS Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python inference_local.py --text "Hello world" --language en\n'
            '  python inference_local.py --text "你好世界" --language zh '
            "--voice 中文女\n"
            '  python inference_local.py --text "Exciting!" --mode instruct '
            '--instruction "Speak with joy"\n'
        ),
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text to synthesise."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code (default: zh).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sft",
        choices=sorted(SUPPORTED_MODES),
        help="Inference mode (default: sft).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Speaker ID for SFT/instruct modes (default: {DEFAULT_VOICE}).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier, 0.5–2.0 (default: 1.0).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Style/emotion instruction for instruct mode.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default="",
        help="Path to reference WAV for zero_shot mode.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Transcript of reference audio for zero_shot mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to CosyVoice2-0.5B model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Validate
    if args.speed <= 0 or args.speed > 3.0:
        parser.error("--speed must be between 0.01 and 3.0.")

    if args.mode == "zero_shot" and not args.ref_audio:
        parser.error("--ref-audio is required for zero_shot mode.")

    if args.mode == "instruct" and not args.instruction:
        parser.error("--instruction is required for instruct mode.")

    # Load model
    model = _load_model(args.model_dir)

    # Synthesise
    audio, sr = synthesize(
        model,
        args.text,
        mode=args.mode,
        voice=args.voice,
        speed=args.speed,
        language=args.language,
        instruction=args.instruction,
        ref_audio_path=args.ref_audio or None,
        ref_text=args.ref_text,
    )

    # Save
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, format="WAV", subtype="PCM_16")
    logger.info("Saved %s (%.2f s, %d Hz)", output_path, len(audio) / sr, sr)


if __name__ == "__main__":
    main()
