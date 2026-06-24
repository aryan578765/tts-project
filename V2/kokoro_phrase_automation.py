"""
Kokoro TTS - Phrase Cutting Automation
======================================
This script shows how to:
1. Generate long audio with pauses at specific word positions
2. Use phrase_cut_points to reliably cut audio into phrases
3. Save individual phrase audio files

Usage:
    pip install requests
    python kokoro_phrase_automation.py
"""
import requests
import base64
import wave
import numpy as np
import os

# ---- Configuration ----
API_KEY = "YOUR_RUNPOD_API_KEY"  # Replace with your API key
ENDPOINT_ID = "54td14oe86jexh"   # Replace with your endpoint ID
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ---- Your text and phrase boundaries ----
TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software."""

# Word indices (0-based) where you want phrase breaks.
# These define where pauses will be inserted.
# Example: pause after word 2 ("overhaul"), word 8 ("announcement"), etc.
PAUSE_AFTER = [2, 8, 13, 16, 21]

# ---- Step 1: Generate audio with pauses ----
print("Step 1: Generating audio with pauses...")
payload = {
    "input": {
        "text": TEXT,
        "voice": "af_heart",       # Voice name
        "lang_code": "a",          # a=American English, b=British, e=Spanish, etc.
        "speed": 1.0,
        "timestamps": True,        # Get word-level timestamps
        "pause_after": PAUSE_AFTER, # Where to insert pauses
        "micro_pause_ms": 50       # Pause duration in ms (50-200 recommended)
    }
}

response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
result = response.json()

if "error" in result.get("output", {}):
    print(f"Error: {result['output']['error']}")
    exit(1)

output = result["output"]
print(f"  Audio duration: {output['duration_seconds']}s")
print(f"  Words: {len(output.get('word_timestamps', []))}")

# ---- Step 2: Save full audio ----
audio_bytes = base64.b64decode(output["audio_base64"])
os.makedirs("output", exist_ok=True)
with open("output/full_audio.wav", "wb") as f:
    f.write(audio_bytes)
print("  Saved: output/full_audio.wav")

# ---- Step 3: Get phrase cut points ----
cut_points = output.get("phrase_cut_points", [])
word_timestamps = output.get("word_timestamps", [])

print(f"\nStep 2: Found {len(cut_points)} phrase cut points:")
for i, cp in enumerate(cut_points):
    print(f"  Cut {i+1}: after '{cp['after_word']}' (idx {cp['after_word_idx']}) at {cp['time']:.4f}s")

# ---- Step 4: Cut audio into phrases ----
print("\nStep 3: Cutting audio into phrases...")

# Read the WAV file
with wave.open("output/full_audio.wav", "rb") as wf:
    sample_rate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)

# Cut at each phrase_cut_point
cut_times = [cp["time"] for cp in cut_points]
phrases = []
cursor = 0

for i, t in enumerate(cut_times):
    cut_sample = int(round(t * sample_rate))
    cut_sample = max(0, min(cut_sample, len(audio)))
    phrase_audio = audio[cursor:cut_sample]
    phrases.append(phrase_audio)
    cursor = cut_sample

# Last phrase (after final cut point)
phrases.append(audio[cursor:])

# Save individual phrases
for i, phrase in enumerate(phrases):
    filename = f"output/phrase_{i+1:02d}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(phrase.tobytes())
    duration = len(phrase) / sample_rate
    print(f"  Saved: {filename} ({duration:.2f}s)")

# ---- Step 5: Create version with 1s silence between phrases ----
print("\nStep 4: Creating audio with 1s silence between phrases...")
silence = np.zeros(sample_rate, dtype=np.int16)  # 1 second of silence
pieces = []
for i, phrase in enumerate(phrases):
    pieces.append(phrase)
    if i < len(phrases) - 1:
        pieces.append(silence)

combined = np.concatenate(pieces)
with wave.open("output/phrases_with_silence.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(combined.tobytes())
print(f"  Saved: output/phrases_with_silence.wav ({len(combined)/sample_rate:.1f}s)")

print("\nDone!")
