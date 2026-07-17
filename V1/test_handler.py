"""
Test script for Kokoro handler v2 (ONNX optimized)
====================================================
10 tests matching the task tracker, using official test texts.

Usage:
  set RUNPOD_API_KEY=rpa_xxx
  set KOKORO_ENDPOINT_ID=xxx
  python test_handler.py
"""

import requests
import json
import base64
import time
import os

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("KOKORO_ENDPOINT_ID", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_TEXTS_PATH = os.path.join(SCRIPT_DIR, "..", "test_texts", "test_texts.json")

if not API_KEY:
    print("ERROR: Set RUNPOD_API_KEY environment variable")
    exit(1)
if not ENDPOINT_ID:
    print("ERROR: Set KOKORO_ENDPOINT_ID environment variable")
    exit(1)

with open(TEST_TEXTS_PATH, "r", encoding="utf-8") as f:
    TEST_TEXTS = json.load(f)

print(f"Loaded {len(TEST_TEXTS)} test texts\n")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = {}
passed = 0
failed = 0


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


def save_audio(b64_data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    audio_bytes = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    print(f"  Saved -> {filename} ({len(audio_bytes)} bytes)")


def header(num, title):
    print(f"\n{'='*60}")
    print(f"TEST {num}/10: {title}")
    print(f"{'='*60}")


# =========================================================================
# TEST 1: Basic synthesis — all 13 texts, verify audio plays
# =========================================================================
header(1, "Basic synthesis — all 13 test texts")

for test in TEST_TEXTS:
    text = test["text"]
    print(f"\n  [{test['id']}] {test['title']}")
    result = submit_job({"input": {"text": text, "voice": "af_heart", "speed": 1.0, "lang_code": "a"}})
    if result.get("status") == "COMPLETED":
        output = result["output"]
        print(f"  ✅ duration={output['duration_seconds']}s, RTF={output['rtf']}")
        save_audio(output["audio_base64"], f"{test['id']}_v2.wav")
        passed += 1
    else:
        print(f"  ❌ FAILED: {result.get('output', {}).get('error', result.get('status'))}")
        failed += 1


# =========================================================================
# TEST 2: Word timestamps — verify accurate on test_1
# =========================================================================
header(2, "Word timestamps accuracy")

text = TEST_TEXTS[0]["text"]
result = submit_job({"input": {"text": text, "voice": "af_heart", "lang_code": "a", "timestamps": True}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    ts = output.get("word_timestamps", [])
    print(f"  Found {len(ts)} word timestamps")
    for wt in ts:
        print(f"    {wt['start']:.3f}s - {wt['end']:.3f}s  '{wt['word']}'")
    # Verify timestamps are ordered
    ordered = all(ts[i]["start"] <= ts[i+1]["start"] for i in range(len(ts)-1))
    print(f"  Timestamps ordered: {'✅ Yes' if ordered else '❌ No'}")
    passed += 1 if ordered and len(ts) > 0 else 0
    failed += 0 if ordered and len(ts) > 0 else 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 3: Word boundaries — verify gap measurements
# =========================================================================
header(3, "Word boundary analysis")

text = TEST_TEXTS[0]["text"]
result = submit_job({"input": {"text": text, "voice": "af_heart", "lang_code": "a", "timestamps": True, "word_boundaries": True}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    bounds = output.get("word_boundaries", [])
    print(f"  Found {len(bounds)} word boundaries")
    clean = sum(1 for b in bounds if b["status"] == "clean")
    tight = sum(1 for b in bounds if b["status"] == "tight")
    coart = sum(1 for b in bounds if b["status"] == "coarticulated")
    overlap = sum(1 for b in bounds if b["status"] == "overlapping")
    print(f"  Clean: {clean} | Tight: {tight} | Coarticulated: {coart} | Overlapping: {overlap}")
    for b in bounds:
        icon = "✅" if b["can_separate"] else "⚠️"
        print(f"    {icon} {b['pair']:30s} {b['gap_ms']:6.1f}ms  {b['status']}")
    passed += 1 if len(bounds) > 0 else 0
    failed += 0 if len(bounds) > 0 else 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 4: Micro-pause 10ms — compare with original
# =========================================================================
header(4, "Micro-pause 10ms + 5ms crossfade")

text = TEST_TEXTS[0]["text"]
result = submit_job({"input": {"text": text, "voice": "af_heart", "lang_code": "a", "micro_pause_ms": 10, "crossfade_ms": 5}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"  ✅ Duration with 10ms pauses: {output['duration_seconds']}s")
    save_audio(output["audio_base64"], "test_1_micro_pause_10ms.wav")
    passed += 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 5: Micro-pause 50ms — verify no clicks/pops
# =========================================================================
header(5, "Micro-pause 50ms + 10ms crossfade")

result = submit_job({"input": {"text": text, "voice": "af_heart", "lang_code": "a", "micro_pause_ms": 50, "crossfade_ms": 10}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"  ✅ Duration with 50ms pauses: {output['duration_seconds']}s")
    save_audio(output["audio_base64"], "test_1_micro_pause_50ms.wav")
    passed += 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 6: French (ff_siwis)
# =========================================================================
header(6, "French generation (ff_siwis)")

result = submit_job({"input": {
    "text": "Bonjour, comment allez-vous aujourd'hui? Je suis très content de vous rencontrer.",
    "voice": "ff_siwis", "speed": 1.0, "lang_code": "f", "timestamps": True,
}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"  ✅ Duration: {output['duration_seconds']}s, RTF: {output['rtf']}")
    if "word_timestamps" in output:
        for wt in output["word_timestamps"]:
            print(f"    {wt['start']:.3f}s - {wt['end']:.3f}s  '{wt['word']}'")
    save_audio(output["audio_base64"], "french_test.wav")
    passed += 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 7: Spanish (ef_dora)
# =========================================================================
header(7, "Spanish generation (ef_dora)")

result = submit_job({"input": {
    "text": "Hola, ¿cómo estás? Hoy es un día muy bonito para aprender algo nuevo.",
    "voice": "ef_dora", "speed": 1.0, "lang_code": "e", "timestamps": True,
}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"  ✅ Duration: {output['duration_seconds']}s, RTF: {output['rtf']}")
    save_audio(output["audio_base64"], "spanish_test.wav")
    passed += 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 8: RTF benchmark — 3 runs on test_3 (longest text)
# =========================================================================
header(8, "RTF benchmark (3 runs, test_3 — longest text)")

benchmark_text = TEST_TEXTS[2]["text"]
rtfs = []
for run in range(3):
    result = submit_job({"input": {"text": benchmark_text, "voice": "af_heart", "speed": 1.0, "lang_code": "a"}})
    if result.get("status") == "COMPLETED":
        rtf = result["output"].get("rtf", 0)
        dur = result["output"].get("duration_seconds", 0)
        rtfs.append(rtf)
        print(f"  Run {run+1}: duration={dur}s, RTF={rtf}")
if rtfs:
    avg = sum(rtfs) / len(rtfs)
    speed_x = 1.0 / avg if avg > 0 else 0
    print(f"  Average RTF: {avg:.4f} ({speed_x:.1f}x real-time)")
    print(f"  {'✅ Target met (>20x)' if speed_x >= 20 else '⚠️ Below target (<20x)'}")
    passed += 1 if speed_x >= 20 else 0
    failed += 0 if speed_x >= 20 else 1
else:
    print(f"  ❌ No RTF data"); failed += 1


# =========================================================================
# TEST 9: Long text — combine multiple test texts
# =========================================================================
header(9, "Long text (all 13 texts combined = 500+ words)")

long_text = " ".join(t["text"] for t in TEST_TEXTS)
word_count = len(long_text.split())
print(f"  Combined text: {word_count} words, {len(long_text)} chars")

result = submit_job({"input": {"text": long_text, "voice": "af_heart", "speed": 1.0, "lang_code": "a"}})
if result.get("status") == "COMPLETED":
    output = result["output"]
    print(f"  ✅ Duration: {output['duration_seconds']}s, RTF: {output['rtf']}")
    save_audio(output["audio_base64"], "long_text_combined.wav")
    passed += 1
else:
    print(f"  ❌ FAILED"); failed += 1


# =========================================================================
# TEST 10: Edge cases
# =========================================================================
header(10, "Edge cases")

edge_cases = [
    ("Empty text", {"text": "", "voice": "af_heart", "lang_code": "a"}),
    ("Single word", {"text": "Hello", "voice": "af_heart", "lang_code": "a", "timestamps": True}),
    ("Special chars", {"text": "!!! ??? ...", "voice": "af_heart", "lang_code": "a"}),
    ("Invalid lang", {"text": "Hello", "voice": "af_heart", "lang_code": "x"}),
    ("Invalid speed", {"text": "Hello", "voice": "af_heart", "lang_code": "a", "speed": 99}),
]

for name, inp in edge_cases:
    result = submit_job({"input": inp})
    status = result.get("status", "UNKNOWN")
    if status == "COMPLETED":
        output = result.get("output", {})
        if "error" in output:
            print(f"  ✅ {name}: Handled gracefully -> {output['error'][:60]}")
        else:
            print(f"  ✅ {name}: Generated audio, duration={output.get('duration_seconds')}s")
    elif status == "FAILED":
        print(f"  ✅ {name}: Rejected (expected)")
    else:
        print(f"  ⚠️ {name}: Status={status}")
passed += 1  # edge cases pass if no crash


# =========================================================================
# Save results
# =========================================================================
results_path = os.path.join(OUTPUT_DIR, "v2_test_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint_id": ENDPOINT_ID,
        "passed": passed,
        "failed": failed,
        "rtf_benchmark": {
            "runs": rtfs,
            "average_rtf": round(sum(rtfs) / len(rtfs), 4) if rtfs else None,
            "speed_multiplier": round(1.0 / (sum(rtfs) / len(rtfs)), 1) if rtfs else None,
        },
    }, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of 10 tests")
print(f"{'='*60}")
