"""
Timestamp Accuracy Demo
========================
Proves word timestamps are accurate by splitting audio at word boundaries
and inserting 500ms silence between every word pair.

If timestamps are correct: each word chunk sounds clean.
If timestamps are wrong: words will be cut mid-syllable.

Usage:
  set RUNPOD_API_KEY=rpa_xxx
  set KOKORO_ENDPOINT_ID=xxx
  python generate_timestamp_demo.py
"""

import requests
import json
import base64
import struct
import os
import time

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("KOKORO_ENDPOINT_ID", "")
SAMPLE_RATE = 24000
SILENCE_MS = 500

if not API_KEY or not ENDPOINT_ID:
    print("ERROR: Set RUNPOD_API_KEY and KOKORO_ENDPOINT_ID")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_TEXTS_PATH = os.path.join(SCRIPT_DIR, "..", "test_texts", "test_texts.json")

with open(TEST_TEXTS_PATH, "r", encoding="utf-8") as f:
    TEST_TEXTS = json.load(f)


def submit_job(payload, timeout=300):
    url = f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/runsync"
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    data = resp.json()
    if data.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = data.get("id")
        print(f"  Polling job {job_id}...")
        for _ in range(60):
            time.sleep(3)
            poll = requests.get(
                f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/status/{job_id}",
                headers={"Authorization": f"Bearer {API_KEY}"},
            ).json()
            if poll.get("status") in ("COMPLETED", "FAILED"):
                return poll
        return {"status": "TIMEOUT"}
    return data


def decode_wav_samples(b64_data):
    """Decode base64 WAV to raw float32 samples."""
    wav_bytes = base64.b64decode(b64_data)
    # Skip WAV header (44 bytes standard)
    header_size = 44
    # Find 'data' chunk for robustness
    data_pos = wav_bytes.find(b'data')
    if data_pos >= 0:
        header_size = data_pos + 8  # 'data' + 4 bytes size

    raw = wav_bytes[header_size:]
    # 16-bit PCM samples
    num_samples = len(raw) // 2
    samples = []
    for i in range(num_samples):
        sample = struct.unpack_from('<h', raw, i * 2)[0]
        samples.append(sample / 32768.0)
    return samples


def encode_wav(samples, sample_rate=SAMPLE_RATE):
    """Encode float32 samples to WAV bytes (16-bit PCM)."""
    num_samples = len(samples)
    data_size = num_samples * 2
    file_size = 36 + data_size

    header = struct.pack('<4sI4s', b'RIFF', file_size, b'WAVE')
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_header = struct.pack('<4sI', b'data', data_size)

    pcm_data = b''
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        pcm_data += struct.pack('<h', int(clamped * 32767))

    return header + fmt + data_header + pcm_data


def generate_timestamp_demo(test_text, test_id, test_title):
    """Generate demo audio with 500ms silence between every word."""
    print(f"\n{'='*60}")
    print(f"TIMESTAMP DEMO: {test_id} - {test_title}")
    print(f"{'='*60}")

    # Step 1: Get audio + timestamps
    print("  Step 1: Synthesizing with timestamps...")
    result = submit_job({
        "input": {
            "text": test_text,
            "voice": "af_heart",
            "lang_code": "a",
            "timestamps": True,
            "word_boundaries": True,
        }
    })

    if result.get("status") != "COMPLETED":
        print(f"  FAILED: {result.get('status')}")
        return None

    output = result["output"]
    timestamps = output.get("word_timestamps", [])
    boundaries = output.get("word_boundaries", [])
    audio_b64 = output["audio_base64"]
    duration = output["duration_seconds"]

    print(f"  Audio: {duration}s, {len(timestamps)} words detected")

    if not timestamps:
        print("  ERROR: No timestamps returned")
        return None

    # Step 2: Decode audio
    print("  Step 2: Decoding audio...")
    samples = decode_wav_samples(audio_b64)
    print(f"  Decoded {len(samples)} samples ({len(samples)/SAMPLE_RATE:.2f}s)")

    # Step 3: Split at word boundaries and insert 500ms silence
    print(f"  Step 3: Splitting at {len(timestamps)} word boundaries, inserting {SILENCE_MS}ms silence...")
    silence_samples = [0.0] * int(SAMPLE_RATE * SILENCE_MS / 1000)
    demo_audio = []

    for i, wt in enumerate(timestamps):
        start_idx = int(wt["start"] * SAMPLE_RATE)
        end_idx = int(wt["end"] * SAMPLE_RATE)
        start_idx = max(0, min(start_idx, len(samples)))
        end_idx = max(start_idx, min(end_idx, len(samples)))

        word_chunk = samples[start_idx:end_idx]
        demo_audio.extend(word_chunk)

        # Add 500ms silence after every word except the last
        if i < len(timestamps) - 1:
            demo_audio.extend(silence_samples)

        chunk_duration = len(word_chunk) / SAMPLE_RATE
        print(f"    [{wt['start']:.3f}s - {wt['end']:.3f}s] \"{wt['word']}\" ({chunk_duration:.3f}s)")

    # Step 4: Save demo audio
    demo_wav = encode_wav(demo_audio)
    demo_path = os.path.join(OUTPUT_DIR, f"{test_id}_timestamp_demo.wav")
    with open(demo_path, "wb") as f:
        f.write(demo_wav)

    demo_duration = len(demo_audio) / SAMPLE_RATE
    print(f"\n  Original: {duration}s")
    print(f"  Demo (with {SILENCE_MS}ms gaps): {demo_duration:.2f}s")
    print(f"  Saved -> {demo_path} ({len(demo_wav)} bytes)")

    # Step 5: Also save the boundary analysis
    if boundaries:
        print(f"\n  Word Boundary Analysis:")
        for b in boundaries:
            icon = "OK" if b["can_separate"] else "!!"
            print(f"    [{icon}] {b['pair']:30s} {b['gap_ms']:6.1f}ms  {b['status']}")

    # Step 6: Save smart pause version
    print(f"\n  Step 6: Generating smart pause version (10ms at tight boundaries only)...")
    smart_result = submit_job({
        "input": {
            "text": test_text,
            "voice": "af_heart",
            "lang_code": "a",
            "timestamps": True,
            "word_boundaries": True,
            "micro_pause_ms": 10,
            "crossfade_ms": 5,
            "smart_pause": True,
        }
    })

    if smart_result.get("status") == "COMPLETED":
        smart_output = smart_result["output"]
        smart_b64 = smart_output["audio_base64"]
        smart_wav = base64.b64decode(smart_b64)
        smart_path = os.path.join(OUTPUT_DIR, f"{test_id}_smart_pause.wav")
        with open(smart_path, "wb") as f:
            f.write(smart_wav)
        print(f"  Saved -> {smart_path} ({len(smart_wav)} bytes)")
        smart_bounds = smart_output.get("word_boundaries", [])
        if smart_bounds:
            print(f"  Smart pause boundaries:")
            for b in smart_bounds:
                icon = "OK" if b["can_separate"] else "!!"
                print(f"    [{icon}] {b['pair']:30s} {b['gap_ms']:6.1f}ms  {b['status']}")
    else:
        print(f"  Smart pause FAILED: {smart_result.get('status')}")

    return {
        "test_id": test_id,
        "title": test_title,
        "duration": duration,
        "demo_duration": round(demo_duration, 2),
        "word_count": len(timestamps),
        "timestamps": timestamps,
        "boundaries": boundaries,
    }


# =========================================================================
# Run demos for test_1 and test_3
# =========================================================================
print("=" * 60)
print("TIMESTAMP ACCURACY DEMO GENERATOR")
print("=" * 60)

results = []

# test_1: Short text (good for listening test)
r1 = generate_timestamp_demo(
    TEST_TEXTS[0]["text"],
    TEST_TEXTS[0]["id"],
    TEST_TEXTS[0]["title"],
)
if r1:
    results.append(r1)

# test_3: Long text (stability test)
r3 = generate_timestamp_demo(
    TEST_TEXTS[2]["text"],
    TEST_TEXTS[2]["id"],
    TEST_TEXTS[2]["title"],
)
if r3:
    results.append(r3)

# Save results JSON
results_path = os.path.join(OUTPUT_DIR, "timestamp_demo_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print("DONE - Listen to the *_timestamp_demo.wav files.")
print("If words sound clean at boundaries = timestamps are accurate.")
print("If words are cut mid-syllable = timestamps are off.")
print(f"{'='*60}")
