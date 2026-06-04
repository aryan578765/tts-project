#!/usr/bin/env python3
"""
Kokoro TTS – Local Inference Script
=====================================

Standalone script for testing Kokoro text-to-speech locally without RunPod.
Generates audio from text and saves it as a WAV file.

Usage examples:

    # Basic usage
    python inference_local.py --text "Hello, world!" --output hello.wav

    # Custom voice and speed
    python inference_local.py --text "Good morning!" --voice af_heart --speed 1.2 --output morning.wav

    # British English
    python inference_local.py --text "Cheerio!" --voice bf_emma --lang b --output cheerio.wav

    # Japanese
    python inference_local.py --text "こんにちは世界" --voice jf_alpha --lang j --output konnichiwa.wav

    # Read text from a file
    python inference_local.py --file input.txt --voice af_heart --output speech.wav

    # With SSML break tags
    python inference_local.py --text 'Hello.<break time="500ms"/>How are you?' --ssml --output greeting.wav

Supported language codes:
    a - American English (default)
    b - British English
    e - Spanish
    f - French
    i - Italian
    j - Japanese
    p - Portuguese
    z - Chinese
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("kokoro-local")

SAMPLE_RATE: int = 24000

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS – Local inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesise.",
    )
    text_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file to read input from.",
    )

    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="af_heart",
        help="Voice name (default: af_heart).",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speech speed multiplier, 0.1–5.0 (default: 1.0).",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="a",
        choices=list(VALID_LANG_CODES.keys()),
        help="Language code (default: a = American English).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "--ssml",
        action="store_true",
        help="Enable SSML <break> tag processing.",
    )

    return parser.parse_args()


def synthesise(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
) -> np.ndarray:
    """Synthesise speech from text using Kokoro KPipeline.

    Args:
        text: The input text to convert to speech.
        voice: The voice model name.
        speed: Speed multiplier (0.1 to 5.0).
        lang_code: One of the supported language codes.

    Returns:
        Numpy array of audio samples at 24 kHz.

    Raises:
        ValueError: If no audio was generated.
    """
    from kokoro import KPipeline  # deferred import so --help is fast

    logger.info("Loading KPipeline (lang_code='%s') ...", lang_code)
    pipeline = KPipeline(lang_code=lang_code)

    logger.info(
        "Synthesising %d chars with voice='%s', speed=%.2f ...",
        len(text), voice, speed,
    )

    audio_chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("Kokoro produced no audio for the given input.")

    return np.concatenate(audio_chunks)


def synthesise_ssml(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
) -> np.ndarray:
    """Synthesise text containing SSML <break> tags.

    Segments between ``<break>`` tags are synthesised individually and
    silence is inserted according to the ``time`` attribute.

    Args:
        text: Input text with optional SSML break tags.
        voice: The voice model name.
        speed: Speed multiplier.
        lang_code: Language code.

    Returns:
        Concatenated numpy audio array.
    """
    import re

    pattern = re.compile(
        r'<break\s*(?:time\s*=\s*"([\d.]+)(ms|s)")?\s*/?>',
        re.IGNORECASE,
    )
    parts: list[np.ndarray] = []
    last_end = 0

    for m in pattern.finditer(text):
        chunk = text[last_end : m.start()].strip()
        if chunk:
            parts.append(synthesise(chunk, voice, speed, lang_code))

        # Parse pause duration
        if m.group(1) is not None:
            value = float(m.group(1))
            unit = m.group(2).lower()
            pause = value / 1000.0 if unit == "ms" else value
        else:
            pause = 0.5

        silence = np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32)
        parts.append(silence)
        last_end = m.end()

    # Trailing text
    remaining = text[last_end:].strip()
    if remaining:
        parts.append(synthesise(remaining, voice, speed, lang_code))

    if not parts:
        raise ValueError("No audio segments produced from SSML input.")

    return np.concatenate(parts)


def main() -> None:
    """Entry-point for local inference."""
    args = parse_args()

    # ---- Resolve input text ----
    if args.file:
        filepath = Path(args.file)
        if not filepath.is_file():
            logger.error("Input file not found: %s", filepath)
            sys.exit(1)
        text = filepath.read_text(encoding="utf-8").strip()
        logger.info("Read %d chars from %s", len(text), filepath)
    else:
        text = args.text.strip()

    if not text:
        logger.error("Input text is empty.")
        sys.exit(1)

    # ---- Validate speed ----
    if not (0.1 <= args.speed <= 5.0):
        logger.error("Speed must be between 0.1 and 5.0, got %.2f", args.speed)
        sys.exit(1)

    # ---- Synthesise ----
    start = time.perf_counter()

    if args.ssml or "<break" in text.lower():
        audio = synthesise_ssml(text, args.voice, args.speed, args.lang)
    else:
        audio = synthesise(text, args.voice, args.speed, args.lang)

    elapsed = time.perf_counter() - start
    duration = len(audio) / SAMPLE_RATE

    # ---- Save output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")

    logger.info("Audio saved to: %s", output_path.resolve())
    logger.info(
        "Duration: %.2fs | Processing time: %.2fs | RTF: %.2f",
        duration, elapsed, elapsed / max(duration, 0.001),
    )


if __name__ == "__main__":
    main()
