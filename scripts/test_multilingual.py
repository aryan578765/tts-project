"""
Multilingual TTS Tester
========================
Tests TTS model support for multiple languages by generating audio samples
in each language, saving results, and reporting success/failure.

Usage:
    python test_multilingual.py --endpoint http://localhost:8000
    python test_multilingual.py --endpoint http://localhost:8000 --model kokoro
    python test_multilingual.py --endpoint http://localhost:8000 --model kokoro --languages en es fr
"""

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "audio_output"


# ---------------------------------------------------------------------------
# Language test samples
# ---------------------------------------------------------------------------
LANGUAGE_SAMPLES: dict[str, dict] = {
    "en": {
        "name": "English",
        "text": "The quick brown fox jumps over the lazy dog. This is a natural English sentence with varied phonemes.",
        "script": "Latin",
    },
    "es": {
        "name": "Spanish",
        "text": "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración natural en español.",
        "script": "Latin",
    },
    "fr": {
        "name": "French",
        "text": "Le renard brun rapide saute par-dessus le chien paresseux. C'est une phrase naturelle en français.",
        "script": "Latin",
    },
    "de": {
        "name": "German",
        "text": "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein natürlicher deutscher Satz.",
        "script": "Latin",
    },
    "it": {
        "name": "Italian",
        "text": "La volpe marrone veloce salta sopra il cane pigro. Questa è una frase naturale in italiano.",
        "script": "Latin",
    },
    "pt": {
        "name": "Portuguese",
        "text": "A rápida raposa marrom pula sobre o cão preguiçoso. Esta é uma frase natural em português.",
        "script": "Latin",
    },
    "ja": {
        "name": "Japanese",
        "text": "速い茶色の狐が怠けた犬を飛び越えます。これは自然な日本語の文章です。",
        "script": "CJK",
    },
    "zh": {
        "name": "Chinese (Mandarin)",
        "text": "快速的棕色狐狸跳过了懒惰的狗。这是一个自然的中文句子。",
        "script": "CJK",
    },
    "ko": {
        "name": "Korean",
        "text": "빠른 갈색 여우가 게으른 개를 뛰어넘습니다. 이것은 자연스러운 한국어 문장입니다.",
        "script": "Hangul",
    },
    "hi": {
        "name": "Hindi",
        "text": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। यह एक स्वाभाविक हिंदी वाक्य है।",
        "script": "Devanagari",
    },
    "ar": {
        "name": "Arabic",
        "text": "الثعلب البني السريع يقفز فوق الكلب الكسول. هذه جملة طبيعية باللغة العربية.",
        "script": "Arabic",
    },
    "ru": {
        "name": "Russian",
        "text": "Быстрая коричневая лиса перепрыгивает через ленивую собаку. Это естественное русское предложение.",
        "script": "Cyrillic",
    },
    "tr": {
        "name": "Turkish",
        "text": "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar. Bu doğal bir Türkçe cümledir.",
        "script": "Latin",
    },
    "nl": {
        "name": "Dutch",
        "text": "De snelle bruine vos springt over de luie hond. Dit is een natuurlijke Nederlandse zin.",
        "script": "Latin",
    },
    "pl": {
        "name": "Polish",
        "text": "Szybki brązowy lis przeskakuje nad leniwym psem. To jest naturalne polskie zdanie.",
        "script": "Latin",
    },
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def get_wav_duration(audio_bytes: bytes) -> float:
    """Parse WAV header for duration."""
    if len(audio_bytes) < 44:
        return 0.0
    try:
        sample_rate = struct.unpack_from("<I", audio_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", audio_bytes, 34)[0]
        num_channels = struct.unpack_from("<H", audio_bytes, 22)[0]
        idx = 12
        while idx < len(audio_bytes) - 8:
            chunk_id = audio_bytes[idx : idx + 4]
            chunk_size = struct.unpack_from("<I", audio_bytes, idx + 4)[0]
            if chunk_id == b"data":
                bps = bits_per_sample // 8
                if sample_rate > 0 and bps > 0 and num_channels > 0:
                    return (chunk_size // (bps * num_channels)) / sample_rate
                break
            idx += 8 + chunk_size
    except (struct.error, ZeroDivisionError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# TTS request
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RunPod / TTS request helpers
# ---------------------------------------------------------------------------
RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


def get_payload(model: str, text: str, lang: str) -> dict:
    """Build payload for Kokoro or CosyVoice based on target language."""
    if model == "kokoro":
        # Map ISO 639-1 language code to Kokoro language code and appropriate voice
        # Supported: a (US EN), b (UK EN), e (ES), f (FR), i (IT), j (JA), p (PT), z (ZH)
        lang_map = {
            "en": ("a", "af_heart"),
            "es": ("e", "ef_dora"),
            "fr": ("f", "ff_sixtine"),
            "it": ("i", "if_sara"),
            "ja": ("j", "jf_mika"),
            "pt": ("p", "pf_daniele"),
            "zh": ("z", "zf_xiaobei"),
        }
        
        lang_code, voice = lang_map.get(lang, (lang, "af_heart"))
        return {
            "input": {
                "text": text,
                "voice": voice,
                "speed": 1.0,
                "lang_code": lang_code,
                "ssml": False,
            }
        }
    elif model == "cosyvoice":
        # CosyVoice supported languages: zh, en, ja, ko, de, es, fr, it, ru
        # SFT voices can speak cross-lingually. We use '英文女' (English Female) as base speaker.
        return {
            "input": {
                "text": text,
                "voice": "英文女",
                "speed": 1.0,
                "language": lang,
                "mode": "sft",
            }
        }
    else:
        return {
            "input": {
                "text": text,
            }
        }


def submit_job(endpoint_id: str, api_key: str, payload: dict, timeout: int = 300) -> dict:
    """Submit job to RunPod serverless and wait for completion."""
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}

    if resp.status_code == 401:
        return {"error": "Unauthorized – check your RunPod API key"}
    if resp.status_code == 404:
        return {"error": f"Endpoint {endpoint_id} not found"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

    status = data.get("status", "")

    if status == "COMPLETED":
        return data

    job_id = data.get("id")
    if not job_id:
        return {"error": f"No job ID in response: {data}"}

    if status in ("IN_QUEUE", "IN_PROGRESS"):
        print(f"  [INFO] Job {job_id} is {status}. Polling...")
        return poll_job(endpoint_id, api_key, job_id, timeout)

    if status == "FAILED":
        error = data.get("error", "Unknown error")
        return {"error": f"Job failed: {error}", "raw": data}

    return {"error": f"Unknown status '{status}': {data}"}


def poll_job(endpoint_id: str, api_key: str, job_id: str, timeout: int = 300) -> dict:
    """Poll RunPod job status."""
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    poll_interval = 2.0

    while time.time() - start < timeout:
        time.sleep(poll_interval)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            data = resp.json()
        except Exception as exc:
            print(f"  [WARN] Poll error: {exc}")
            continue

        status = data.get("status", "")

        if status == "COMPLETED":
            return data
        elif status == "FAILED":
            error = data.get("error", "Unknown error")
            return {"error": f"Job failed: {error}", "raw": data}
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            poll_interval = min(poll_interval * 1.2, 10.0)
        else:
            return {"error": f"Unknown status '{status}'"}

    return {"error": f"Job timed out after {timeout}s"}


def extract_audio(result: dict) -> bytes | None:
    """Extract audio bytes from RunPod job result."""
    output = result.get("output", result)
    if isinstance(output, dict):
        if "error" in output:
            print(f"  [ERROR] Handler error: {output['error']}", file=sys.stderr)
            return None
        for key in ("audio_base64", "audio", "audio_data"):
            if key in output and isinstance(output[key], str):
                try:
                    return base64.b64decode(output[key])
                except Exception:
                    continue
    return None


def generate_audio_runpod(
    endpoint_id: str,
    api_key: str,
    model: str,
    text: str,
    language: str,
    timeout: int = 300,
) -> tuple[bytes | None, float, str]:
    """Send TTS request to RunPod serverless. Returns (audio_bytes, generation_time, error)."""
    payload = get_payload(model, text, language)
    start_time = time.perf_counter()
    result = submit_job(endpoint_id, api_key, payload, timeout)
    elapsed = time.perf_counter() - start_time

    if "error" in result:
        return None, elapsed, result["error"]

    audio_bytes = extract_audio(result)
    if not audio_bytes:
        return None, elapsed, "No audio in response"

    return audio_bytes, elapsed, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_multilingual_test(
    model: str,
    endpoint_id: str,
    api_key: str,
    languages: list[str] | None = None,
    timeout: int = 300,
) -> list[dict]:
    """Run multilingual tests and return results."""
    output_dir = AUDIO_OUTPUT_DIR / model / "multilingual"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter languages
    if languages:
        test_langs = {k: v for k, v in LANGUAGE_SAMPLES.items() if k in languages}
        unknown = set(languages) - set(LANGUAGE_SAMPLES.keys())
        if unknown:
            print(f"[WARN] Unknown languages skipped: {unknown}")
    else:
        test_langs = LANGUAGE_SAMPLES

    print(f"{'=' * 65}")
    print(f"Multilingual TTS Test: {model}")
    print(f"Endpoint ID:  {endpoint_id}")
    print(f"Output:       {output_dir}")
    print(f"Languages:    {len(test_langs)}")
    print(f"{'=' * 65}\n")

    results = []

    for lang_code, lang_info in test_langs.items():
        lang_name = lang_info["name"]
        text = lang_info["text"]
        script = lang_info["script"]

        print(f"[{lang_code}] {lang_name} ({script})")
        print(f"  Text: {text[:60]}{'...' if len(text) > 60 else ''}")

        audio_bytes, gen_time, error = generate_audio_runpod(
            endpoint_id=endpoint_id,
            api_key=api_key,
            model=model,
            text=text,
            language=lang_code,
            timeout=timeout,
        )

        result = {
            "language_code": lang_code,
            "language_name": lang_name,
            "script": script,
            "text": text,
            "success": False,
            "generation_time_s": round(gen_time, 4),
            "audio_duration_s": 0.0,
            "file_size_bytes": 0,
            "error": error,
        }

        if audio_bytes:
            audio_path = output_dir / f"{lang_code}_{lang_name.lower().replace(' ', '_')}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            audio_dur = get_wav_duration(audio_bytes)
            result["success"] = True
            result["audio_duration_s"] = round(audio_dur, 4)
            result["file_size_bytes"] = len(audio_bytes)

            print(f"  ✓ Generated ({audio_dur:.3f}s, {len(audio_bytes)/1024:.1f}KB, {gen_time:.3f}s)")
        else:
            print(f"  ✗ Failed: {error or 'Unknown error'}")

        results.append(result)
        print()

    return results


def print_summary(results: list[dict]):
    """Print multilingual test summary."""
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"{'=' * 65}")
    print(f"MULTILINGUAL TEST SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Language':<25} {'Code':<6} {'Script':<12} {'Status':<10} {'Duration':>10}")
    print(f"{'-' * 25} {'-' * 6} {'-' * 12} {'-' * 10} {'-' * 10}")

    for r in results:
        status = "✓ OK" if r["success"] else "✗ FAIL"
        dur_str = f"{r['audio_duration_s']:.3f}s" if r["success"] else "—"
        print(
            f"{r['language_name']:<25} {r['language_code']:<6} "
            f"{r['script']:<12} {status:<10} {dur_str:>10}"
        )

    print(f"\n{'Succeeded:':<25} {len(succeeded)}/{len(results)}")
    print(f"{'Failed:':<25} {len(failed)}/{len(results)}")

    if succeeded:
        scripts = {}
        for r in results:
            scripts.setdefault(r["script"], {"ok": 0, "fail": 0})
            if r["success"]:
                scripts[r["script"]]["ok"] += 1
            else:
                scripts[r["script"]]["fail"] += 1

        print(f"\nBy script system:")
        for script, counts in sorted(scripts.items()):
            total = counts["ok"] + counts["fail"]
            print(f"  {script:<15} {counts['ok']}/{total} succeeded")

    if failed:
        print(f"\nFailed languages:")
        for r in failed:
            print(f"  - {r['language_name']} ({r['language_code']}): {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test multilingual TTS support on RunPod serverless"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (kokoro or cosyvoice)"
    )
    parser.add_argument(
        "--endpoint-id", required=True, help="RunPod endpoint ID"
    )
    parser.add_argument(
        "--languages", nargs="*", default=None,
        help=f"Languages to test (codes: {', '.join(LANGUAGE_SAMPLES.keys())}). Default: all"
    )
    parser.add_argument(
        "--api-key", default=None, help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[ERROR] RunPod API key required. Use --api-key or set RUNPOD_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    results = run_multilingual_test(
        model=args.model,
        endpoint_id=args.endpoint_id,
        api_key=api_key,
        languages=args.languages,
        timeout=args.timeout,
    )

    print_summary(results)

    # Save results
    output_dir = AUDIO_OUTPUT_DIR / args.model / "multilingual"
    results_path = output_dir / "multilingual_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "endpoint_id": args.endpoint_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_languages": len(results),
                "succeeded": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
