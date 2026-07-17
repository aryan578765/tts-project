"""
Kokoro TTS v2 - RunPod Serverless Handler (ONNX Optimized)
============================================================

Optimized handler with:
  - ONNX Runtime FP16 inference via CUDAExecutionProvider (25x-40x RTF on L4)
  - Word-level timestamps via torchaudio MMS forced alignment
  - Zero-modification splicing for phrase pauses (preserves intonation bit-for-bit)
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

# Languages where forced alignment works (all supported via uroman + segmentation)
FA_SUPPORTED_LANG_CODES = {"a", "b", "e", "f", "i", "j", "p", "z"}

# CJK language codes that need special segmentation + romanization
CJK_LANG_CODES = {"j", "z"}

SAMPLE_RATE: int = 24000  # Kokoro outputs 24 kHz audio
FA_SAMPLE_RATE: int = 16000  # MMS_FA expects 16 kHz

# Default threshold (ms) for classifying word boundaries
DEFAULT_CLEAN_BOUNDARY_THRESHOLD_MS = 50
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

# Lazy-loaded CJK segmentation and romanization tools
_jieba_loaded = False
_uroman_loaded = False
_uroman = None


def _ensure_jieba():
    """Lazy-load jieba for Chinese word segmentation."""
    global _jieba_loaded
    if not _jieba_loaded:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba_loaded = True


def _ensure_uroman():
    """Lazy-load uroman for CJK romanization."""
    global _uroman_loaded, _uroman
    if not _uroman_loaded:
        try:
            from uroman import Uroman
            _uroman = Uroman()
            logger.info("uroman loaded successfully")
        except ImportError:
            logger.warning("uroman not installed, CJK alignment will be limited")
            _uroman = None
        _uroman_loaded = True


def _segment_cjk_text(text: str, lang_code: str) -> list[str]:
    """Segment CJK text into words (these languages have no spaces).

    Returns list of word strings.
    """
    if lang_code == "z":  # Chinese
        _ensure_jieba()
        import jieba
        words = list(jieba.cut(text))
        # Filter out whitespace-only tokens
        return [w.strip() for w in words if w.strip()]
    elif lang_code == "j":  # Japanese
        # Try fugashi/mecab first, fall back to character-level
        try:
            import fugashi
            tagger = fugashi.Tagger()
            words = [word.surface for word in tagger(text) if word.surface.strip()]
            return words
        except (ImportError, RuntimeError):
            logger.warning("fugashi not available, using character-level segmentation for Japanese")
            # Character-level fallback: each character is a "word"
            return [ch for ch in text if ch.strip() and not re.match(r'[\s\p{P}]', ch)]
    else:
        return text.split()


def _romanize_text(text: str) -> str:
    """Romanize non-Latin text using uroman."""
    _ensure_uroman()
    if _uroman is None:
        return text
    try:
        romanized = _uroman.romanize_string(text)
        return romanized
    except Exception as e:
        logger.warning("uroman romanization failed: %s", e)
        return text


def _clean_word(word: str, lang_code: str = "a") -> str:
    """Strip punctuation and normalize a word for alignment matching.

    MMS_FA tokenizer only supports basic Latin a-z, so we:
    - For Latin scripts: normalize accents (é→e, ñ→n, ü→u)
    - For CJK scripts: romanize via uroman first, then normalize
    - Strip digits (the tokenizer crashes on 0-9)
    """
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE).strip()
    if not cleaned:
        return ''

    # Check if text contains CJK characters
    has_cjk = any(ord(c) > 0x2E80 for c in cleaned)

    if has_cjk and lang_code in CJK_LANG_CODES:
        # Romanize CJK characters
        cleaned = _romanize_text(cleaned)
        # Remove any remaining non-Latin characters
        cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned).strip()
    else:
        # Normalize accented characters to ASCII (NFD decomposes, then strip combining marks)
        normalized = unicodedata.normalize('NFD', cleaned)
        cleaned = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

    # Strip digits — MMS_FA tokenizer only supports a-z
    cleaned = re.sub(r'[0-9]', '', cleaned).strip()

    # If word was all-digits (e.g. "18"), use placeholder to preserve index position
    if not cleaned:
        return 'num'

    return cleaned


def _get_word_timestamps(
    audio: np.ndarray,
    transcript: str,
    sample_rate: int = SAMPLE_RATE,
    lang_code: str = "a",
) -> list[dict]:
    """Get word-level timestamps using torchaudio MMS_FA high-level API.

    For CJK languages, performs word segmentation and romanization first.
    """
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
    # For CJK, segment first since there are no spaces
    if lang_code in CJK_LANG_CODES:
        words_raw = _segment_cjk_text(transcript, lang_code)
        logger.info("CJK segmentation: %d words from '%s...'", len(words_raw), transcript[:30])
    else:
        words_raw = transcript.split()

    words_clean = []
    words_display = []
    for w in words_raw:
        cleaned = _clean_word(w, lang_code).lower()
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

def _analyze_word_boundaries(word_timestamps: list[dict], clean_threshold_ms: float = DEFAULT_CLEAN_BOUNDARY_THRESHOLD_MS) -> list[dict]:
    """Analyze gaps between consecutive words."""
    boundaries = []
    for i in range(len(word_timestamps) - 1):
        w1 = word_timestamps[i]
        w2 = word_timestamps[i + 1]
        gap_sec = w2["start"] - w1["end"]
        gap_ms = round(gap_sec * 1000, 1)

        if gap_ms >= clean_threshold_ms:
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
# Silence-based phrase cut point detection
# ---------------------------------------------------------------------------

def _detect_silence_cut_points(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_silence_ms: float = 20.0,
    energy_threshold_ratio: float = 0.01,
    window_ms: float = 10.0,
) -> list[dict]:
    """Detect silence regions in audio and return safe cut points.

    After micro-pauses are inserted, this scans the actual waveform for silence
    regions and returns the center of each as a guaranteed-safe cut point.
    Much more reliable than alignment timestamps for phrase cutting.

    Args:
        energy_threshold_ratio: Fraction of peak amplitude below which audio
            is considered silence (default 0.01 = 1% of peak).

    Returns list of {"time": float, "duration_ms": float}.
    """
    window_samples = max(int(sample_rate * window_ms / 1000), 1)
    min_silence_samples = int(sample_rate * min_silence_ms / 1000)

    # Adaptive threshold: 1% of peak amplitude
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return []
    energy_threshold = peak * energy_threshold_ratio

    # Compute RMS energy per window
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return []

    energies = np.array([
        np.sqrt(np.mean(audio[i * window_samples:(i + 1) * window_samples] ** 2))
        for i in range(n_windows)
    ])

    # Find silence regions (consecutive low-energy windows)
    is_silent = energies < energy_threshold
    cut_points = []
    in_silence = False
    silence_start_win = 0

    for i in range(n_windows):
        if is_silent[i]:
            if not in_silence:
                silence_start_win = i
                in_silence = True
        else:
            if in_silence:
                silence_end_win = i
                silence_len = (silence_end_win - silence_start_win) * window_samples
                if silence_len >= min_silence_samples:
                    # Center of silence region = safest cut point
                    center_sample = ((silence_start_win + silence_end_win) // 2) * window_samples
                    center_time = center_sample / sample_rate
                    duration_ms = round(silence_len / sample_rate * 1000, 1)
                    cut_points.append({
                        "time": round(center_time, 4),
                        "duration_ms": duration_ms,
                    })
                in_silence = False

    # Handle silence at end
    if in_silence:
        silence_end_win = n_windows
        silence_len = (silence_end_win - silence_start_win) * window_samples
        if silence_len >= min_silence_samples:
            center_sample = ((silence_start_win + silence_end_win) // 2) * window_samples
            center_time = center_sample / sample_rate
            duration_ms = round(silence_len / sample_rate * 1000, 1)
            cut_points.append({
                "time": round(center_time, 4),
                "duration_ms": duration_ms,
            })

    return cut_points


# ---------------------------------------------------------------------------
# Phoneme-level comma injection
# ---------------------------------------------------------------------------

# Lazy-loaded Misaki G2P
_g2p_instance = None
_g2p_lock = threading.Lock()


def _get_g2p():
    """Return cached Misaki English G2P instance (thread-safe)."""
    global _g2p_instance
    if _g2p_instance is None:
        with _g2p_lock:
            if _g2p_instance is None:
                from misaki import en
                _g2p_instance = en.G2P()
                logger.info("Misaki G2P loaded")
    return _g2p_instance


def _phonemize_with_boundary_commas(
    text: str,
    pause_after: list[int],
    lang_code: str = "a",
) -> tuple[str, list[int]]:
    """Phonemize full text and inject commas at specific word boundaries.

    This makes Kokoro generate natural word endings at those positions
    (the model sees the comma during generation and produces falling
    intonation / energy decay) WITHOUT modifying the original text.

    Returns (phoneme_string, comma_word_indices).
    """
    g2p = _get_g2p()

    # Phonemize the full text for natural cross-word pronunciation
    full_phonemes, _ = g2p(text)
    logger.info("Full-text phonemes: %s...", full_phonemes[:80])

    # Also phonemize word-by-word to find word boundary positions
    words = text.split()
    pause_set = set(pause_after)

    # Strategy: phonemize the full text, then find where to insert commas.
    # We phonemize each word individually to identify its phoneme span,
    # then insert commas at the matching positions in the full phoneme string.
    word_phonemes = []
    for w in words:
        wp, _ = g2p(w)
        # Strip leading/trailing whitespace from individual phonemes
        word_phonemes.append(wp.strip())

    # Build the phoneme string with commas at pause_after positions
    parts = []
    comma_indices = []
    for i, wp in enumerate(word_phonemes):
        parts.append(wp)
        if i in pause_set and i < len(words) - 1:
            parts.append(",")
            comma_indices.append(i)

    result = " ".join(parts)
    logger.info("Phonemized with commas at %d positions: %s...",
                len(comma_indices), result[:80])
    return result, comma_indices


# ---------------------------------------------------------------------------
# Zero-modification micro-pause insertion
# ---------------------------------------------------------------------------


def _insert_micro_pauses(
    audio: np.ndarray,
    word_timestamps: list[dict],
    pause_ms: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
    pause_after: list[int] | None = None,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Insert silence at phrase boundaries using zero-modification splicing.

    CRITICAL DESIGN PRINCIPLE: We never modify any audio sample.
    No fades, no DC offset removal, no amplitude changes.
    The original Kokoro audio is preserved bit-for-bit.

    Algorithm:
      1. For each boundary, find the energy trough between the two words.
      2. At the trough, find the nearest zero-crossing for a click-free cut.
      3. Splice: [original audio up to zero-crossing] + [silence] + [rest of audio]

    Returns (spliced_audio, updated_timestamps, pause_positions).
    """
    if not word_timestamps or len(word_timestamps) < 2 or pause_after is None:
        return audio, word_timestamps, []

    # Work with float32 copy for energy calculations, but splice the ORIGINAL
    audio_f = audio.astype(np.float32)
    pause_samples = int(sample_rate * pause_ms / 1000)

    valid_indices = sorted(set(
        idx for idx in pause_after
        if 0 <= idx < len(word_timestamps) - 1
    ))

    boundaries = []

    for idx in valid_indices:
        word = word_timestamps[idx]
        word_end_s = word["end"]
        word_end_sample = int(word_end_s * sample_rate)

        next_start_s = word_timestamps[idx + 1]["start"]
        next_start_sample = min(len(audio_f), int(next_start_s * sample_rate))

        # --- Step 1: Find the energy trough ---
        # Search the gap between the two words for the quietest point.
        # Use a 10ms RMS window, stepping every 2.5ms.
        window = int(sample_rate * 0.01)  # 10ms = 240 samples
        step = max(1, window // 4)

        # Search range: from word end to next word start
        search_start = max(0, word_end_sample)
        search_end = min(len(audio_f) - window, next_start_sample)

        # If the gap is very small (<20ms), extend search slightly beyond both edges
        if search_end - search_start < int(sample_rate * 0.02):
            search_start = max(0, word_end_sample - int(sample_rate * 0.01))
            search_end = min(len(audio_f) - window, next_start_sample + int(sample_rate * 0.01))

        best_pos = word_end_sample
        best_energy = float("inf")

        if search_end > search_start:
            for pos in range(search_start, search_end, step):
                seg = audio_f[pos:pos + window]
                e = float(np.sqrt(np.mean(seg ** 2)))
                if e < best_energy:
                    best_energy = e
                    best_pos = pos + window // 2

        # --- Step 2: Find the nearest zero-crossing at the trough ---
        # A zero-crossing is where the waveform crosses zero amplitude.
        # Cutting at a zero-crossing prevents any audible click/pop.
        cut_pos = _find_zero_crossing(audio_f, best_pos, search_range=240)

        boundaries.append({
            "idx": idx,
            "cut_sample": cut_pos,
            "trough_energy": round(best_energy, 1),
            "after_word": word["word"],
        })

    # --- Step 3: Splice original audio with silence ---
    # We NEVER modify audio samples. We just concatenate:
    #   original[0:cut1] + silence + original[cut1:cut2] + silence + ... + original[cutN:]
    silence = np.zeros(pause_samples, dtype=audio.dtype)
    pieces = []
    cursor = 0
    pause_positions = []
    total_samples_before = 0

    for b in boundaries:
        cs = max(0, min(b["cut_sample"], len(audio)))

        # Take the original audio chunk — UNMODIFIED
        chunk = audio[cursor:cs]
        pieces.append(chunk)
        total_samples_before += len(chunk)

        # Record pause position (center of silence)
        pause_center_sample = total_samples_before + pause_samples // 2
        pause_positions.append({
            "time": round(pause_center_sample / sample_rate, 4),
            "duration_ms": pause_ms,
            "after_word_idx": b["idx"],
            "after_word": b["after_word"],
            "trough_energy": b["trough_energy"],
        })

        pieces.append(silence)
        total_samples_before += len(silence)
        cursor = cs

    # Append the remaining audio — UNMODIFIED
    pieces.append(audio[cursor:])
    final_audio = np.concatenate(pieces)

    # --- Step 4: Update timestamps ---
    current_pause_idx = 0
    cumulative_offset = 0.0
    updated_timestamps = []

    for i, wt in enumerate(word_timestamps):
        if current_pause_idx < len(boundaries) and i > boundaries[current_pause_idx]["idx"]:
            cumulative_offset += pause_ms / 1000.0
            current_pause_idx += 1

        updated_timestamps.append({
            "word": wt["word"],
            "start": round(wt["start"] + cumulative_offset, 4),
            "end": round(wt["end"] + cumulative_offset, 4),
        })

    return final_audio, updated_timestamps, pause_positions



# ---------------------------------------------------------------------------
# Core synthesis (ONNX)
# ---------------------------------------------------------------------------

def _synthesise(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
    is_phonemes: bool = False,
) -> np.ndarray:
    """Run Kokoro ONNX TTS and return numpy audio array.

    If is_phonemes=True, `text` is treated as a pre-phonemized string
    and the internal G2P is bypassed.
    """
    kokoro = _get_kokoro()
    onnx_lang = LANG_CODE_MAP.get(lang_code, "en-us")

    samples, sr = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        lang=onnx_lang,
        is_phonemes=is_phonemes,
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
                "crossfade_ms": 5.0,                   # optional, crossfade duration
                "smart_pause": false                   # optional, only pause at tight boundaries
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
        smart_pause: bool = bool(job_input.get("smart_pause", False))
        smart_pause_threshold_ms: float = float(job_input.get("smart_pause_threshold_ms", DEFAULT_CLEAN_BOUNDARY_THRESHOLD_MS))
        pause_after: list[int] | None = job_input.get("pause_after", None)

        if micro_pause_ms > 0:
            want_timestamps = True
            want_boundaries = True
        if pause_after is not None:
            want_timestamps = True
            want_boundaries = True
            if micro_pause_ms <= 0:
                micro_pause_ms = 10.0  # default pause for pause_after mode

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

        # ---- Step 0: Phoneme-level comma injection ----
        # For pause_after mode: phonemize the text and inject commas at
        # the specified word boundaries. This makes Kokoro generate natural
        # word endings at those positions without modifying the original text.
        synth_text = text
        is_phonemes = False
        comma_indices: list[int] = []

        if pause_after is not None and micro_pause_ms > 0:
            try:
                synth_text, comma_indices = _phonemize_with_boundary_commas(
                    text, pause_after, lang_code
                )
                is_phonemes = True
                logger.info("Using phoneme-level commas at %d positions",
                            len(comma_indices))
            except Exception as e:
                logger.warning("Phonemization failed, falling back to plain text: %s", e)
                synth_text = text
                is_phonemes = False

        # ---- Step 1: Synthesise (ONNX FP16 + CUDA) ----
        synth_start = time.perf_counter()
        audio = _synthesise(synth_text, voice, speed, lang_code,
                            is_phonemes=is_phonemes)
        synth_elapsed = time.perf_counter() - synth_start
        logger.info("ONNX synthesis complete in %.3fs", synth_elapsed)

        # ---- Step 2: Forced alignment for timestamps ----
        word_ts: list[dict] = []
        boundaries: list[dict] = []
        pause_positions: list[dict] = []

        if want_timestamps or want_boundaries:
            # Always align against the ORIGINAL text (not phonemes)
            # so word indices match the user's original text
            align_text = text
            word_ts = _get_word_timestamps(audio, align_text, SAMPLE_RATE, lang_code)
            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts, smart_pause_threshold_ms)

        # ---- Step 3: Insert micro-pauses at specific boundaries ----
        if pause_after is not None and micro_pause_ms > 0 and word_ts:
            logger.info("Inserting %.1fms pauses at %d positions (zero-modification splice)", micro_pause_ms, len(pause_after))
            audio, word_ts, pause_positions = _insert_micro_pauses(
                audio, word_ts,
                pause_ms=micro_pause_ms,
                sample_rate=SAMPLE_RATE,
                pause_after=pause_after,
            )
            if want_boundaries and word_ts:
                boundaries = _analyze_word_boundaries(word_ts, smart_pause_threshold_ms)

        # ---- Step 4: Determine phrase cut points ----
        cut_points: list[dict] = []
        if pause_positions:
            cut_points = pause_positions
            logger.info("Returning %d deterministic pause positions", len(cut_points))
        elif (pause_after is not None or micro_pause_ms > 0) and word_ts:
            cut_points = _detect_silence_cut_points(audio, SAMPLE_RATE, min_silence_ms=max(micro_pause_ms * 0.5, 15))
            logger.info("Detected %d silence-based cut points", len(cut_points))
        elif pause_after is not None and not word_ts and comma_indices:
            # Fallback: forced alignment failed but phoneme commas created
            # natural pauses in the audio. Detect those silence regions.
            logger.warning("FA failed — using silence detection on comma-injected audio")
            cut_points = _detect_silence_cut_points(audio, SAMPLE_RATE, min_silence_ms=100)
            logger.info("Detected %d silence-based cut points (fallback)", len(cut_points))

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
        if cut_points:
            result["phrase_cut_points"] = cut_points

        return result

    except Exception as exc:
        logger.exception("Handler error")
        return {"error": f"Synthesis failed: {exc}"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
