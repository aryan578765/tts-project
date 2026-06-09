"""
Test script for Kokoro handler v2 features:
  - Word-level timestamps
  - Word boundary analysis
  - Micro-pause insertion with crossfade
  - RTF benchmarking

Usage:
  RUNPOD_API_KEY=rpa_xxx python scripts/test_handler_v2.py
"""

import requests
import json
import base64
import time
import os

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("KOKORO_ENDPOINT_ID", "3z59kpduf0vkil")

if not API_KEY:
    print("ERROR: Set RUNPOD_API_KEY environment variable")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def submit_job(payload, timeout=300):
    """Submit a runsync job and handle polling."""
    url = f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/runsync"
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    data = resp.json()

    if data.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = data.get("id")
        print(f"  Polling job {job_id}...")
        for i in range(60):
            time.sleep(3)
            poll = requests.get(
                f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/status/{job_id}",
                headers={"Authorization": f"Bearer {API_KEY}"},
            ).json()
            if poll.get("status") in ("COMPLETED", "FAILED"):
                return poll
        return {"status": "TIMEOUT"}

    return data


def save_audio(b64_data, path):
    """Save base64 audio to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_bytes = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    print(f"  Saved {len(audio_bytes)} bytes → {path}")


# =========================================================================
# Test 1: Basic synthesis with timestamps
# =========================================================================
print("=" * 60)
print("TEST 1: Word-level timestamps")
print("=" * 60)

test_text = "I didn't rush this decision. I took my time, listened carefully, and chose what felt right."

payload = {
    "input": {
        "text": test_text,
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "a",
        "timestamps": True,
        "word_boundaries": True,
    }
}

result = submit_job(payload)
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"\n  Duration: {output.get('duration_seconds')}s")

    if "word_timestamps" in output:
        print(f"\n  Word Timestamps ({len(output['word_timestamps'])} words):")
        for wt in output["word_timestamps"]:
            print(f"    {wt['start']:.3f}s - {wt['end']:.3f}s  '{wt['word']}'")

    if "word_boundaries" in output:
        print(f"\n  Word Boundaries ({len(output['word_boundaries'])} pairs):")
        for wb in output["word_boundaries"]:
            status_icon = "✅" if wb["can_separate"] else "⚠️"
            print(f"    {status_icon} {wb['pair']:30s} gap={wb['gap_ms']:6.1f}ms  {wb['status']}")

    if "audio_base64" in output:
        save_audio(output["audio_base64"], "audio_output/kokoro/v2_tests/test1_timestamps.wav")
else:
    print(f"  FAILED: {result.get('error', result.get('status'))}")


# =========================================================================
# Test 2: Micro-pause insertion (10ms with crossfade)
# =========================================================================
print("\n" + "=" * 60)
print("TEST 2: Micro-pause insertion (10ms gaps, 5ms crossfade)")
print("=" * 60)

payload = {
    "input": {
        "text": test_text,
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "a",
        "timestamps": True,
        "word_boundaries": True,
        "micro_pause_ms": 10,
        "crossfade_ms": 5,
    }
}

result = submit_job(payload)
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"\n  Duration with pauses: {output.get('duration_seconds')}s")
    print(f"  (Original was ~8s, with 10ms×16 words should be ~8.16s)")

    if "audio_base64" in output:
        save_audio(output["audio_base64"], "audio_output/kokoro/v2_tests/test2_micro_pause_10ms.wav")
else:
    print(f"  FAILED: {result.get('error', result.get('status'))}")


# =========================================================================
# Test 3: Larger pause (50ms) to hear the effect
# =========================================================================
print("\n" + "=" * 60)
print("TEST 3: Micro-pause insertion (50ms gaps, 10ms crossfade)")
print("=" * 60)

payload = {
    "input": {
        "text": test_text,
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "a",
        "timestamps": True,
        "word_boundaries": True,
        "micro_pause_ms": 50,
        "crossfade_ms": 10,
    }
}

result = submit_job(payload)
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"\n  Duration with pauses: {output.get('duration_seconds')}s")

    if "audio_base64" in output:
        save_audio(output["audio_base64"], "audio_output/kokoro/v2_tests/test3_micro_pause_50ms.wav")
else:
    print(f"  FAILED: {result.get('error', result.get('status'))}")


# =========================================================================
# Test 4: French language
# =========================================================================
print("\n" + "=" * 60)
print("TEST 4: French generation with timestamps")
print("=" * 60)

payload = {
    "input": {
        "text": "Bonjour, comment allez-vous aujourd'hui?",
        "voice": "ff_sixtine",
        "speed": 1.0,
        "lang_code": "f",
        "timestamps": True,
    }
}

result = submit_job(payload)
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"\n  Duration: {output.get('duration_seconds')}s")
    if "word_timestamps" in output:
        for wt in output["word_timestamps"]:
            print(f"    {wt['start']:.3f}s - {wt['end']:.3f}s  '{wt['word']}'")
    if "audio_base64" in output:
        save_audio(output["audio_base64"], "audio_output/kokoro/v2_tests/test4_french.wav")
else:
    print(f"  FAILED: {result.get('error', result.get('status'))}")


print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
