"""Reproduce Mariam's Colab flow and verify timestamp accuracy.

Test 1: Use API returned timestamps (correct way)
Test 2: Use hardcoded probe timestamps (Mariam's bug)
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import re, base64, requests, json, unicodedata
import numpy as np
import wave, struct, io

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things.
Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

# Same indices Mariam used
PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107,
               115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

PAUSE_MS = 50
VOICE = "af_heart"
LANG_CODE = "a"
SPEED = 1.0

def save_wav_from_b64(b64_data, path):
    audio_bytes = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(audio_bytes)

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        if wf.getsampwidth() == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        else:
            audio = np.frombuffer(raw, dtype=np.float32)
    return sr, audio

def insert_silence(audio, sr, cut_times, silence_sec=1.0):
    """Insert silence_sec seconds at each cut_time."""
    silence = np.zeros(int(sr * silence_sec), dtype=audio.dtype)
    sorted_times = sorted(cut_times)
    pieces = []
    cursor = 0
    for t in sorted_times:
        sample = int(round(t * sr))
        sample = max(0, min(sample, len(audio)))
        pieces.append(audio[cursor:sample])
        pieces.append(silence)
        cursor = sample
    pieces.append(audio[cursor:])
    return np.concatenate(pieces)

def save_wav(path, audio, sr):
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

# =============================================
# STEP 1: Probe run (NO pauses) - like Mariam might have done
# =============================================
print("=" * 60)
print("STEP 1: Probe run (no pauses) for timestamps")
print("=" * 60)
probe_payload = {
    "input": {
        "text": TEXT, "voice": VOICE, "lang_code": LANG_CODE, "speed": SPEED,
        "timestamps": True, "word_boundaries": True
    }
}
probe_resp = requests.post(URL, headers=HEADERS, json=probe_payload, timeout=300)
probe_out = probe_resp.json().get("output", {})
probe_ts = probe_out.get("word_timestamps", [])
print("Probe: {} words, duration={}s".format(len(probe_ts), probe_out.get("duration_seconds")))

# =============================================
# STEP 2: Generate with pauses (Mariam's Part 1)
# =============================================
print("\n" + "=" * 60)
print("STEP 2: Generate WITH 50ms pauses")
print("=" * 60)
valid_indices = [i for i in PAUSE_AFTER if 0 <= i < len(probe_ts) - 1]
gen_payload = {
    "input": {
        "text": TEXT, "voice": VOICE, "lang_code": LANG_CODE, "speed": SPEED,
        "timestamps": True, "word_boundaries": True,
        "pause_after": valid_indices, "micro_pause_ms": PAUSE_MS, "crossfade_ms": 5.0
    }
}
gen_resp = requests.post(URL, headers=HEADERS, json=gen_payload, timeout=300)
gen_out = gen_resp.json().get("output", {})
gen_ts = gen_out.get("word_timestamps", [])
print("Generated: {} words, duration={}s".format(len(gen_ts), gen_out.get("duration_seconds")))

# Save the paused audio
save_wav_from_b64(gen_out["audio_base64"], "output/test_verify_paused.wav")

# =============================================
# STEP 3: Compare timestamps
# =============================================
print("\n" + "=" * 60)
print("STEP 3: Compare probe vs generated timestamps at pause points")
print("=" * 60)
print("{:>4}  {:>15}  {:>12}  {:>12}  {:>10}".format("idx", "word", "probe_end", "gen_end", "diff_ms"))
for idx in valid_indices[:15]:
    if idx < len(probe_ts) and idx < len(gen_ts):
        p_end = probe_ts[idx]["end"]
        g_end = gen_ts[idx]["end"]
        diff = (g_end - p_end) * 1000
        print("{:>4}  {:>15}  {:>12.3f}  {:>12.3f}  {:>+10.1f}".format(
            idx, probe_ts[idx]["word"], p_end, g_end, diff))

# =============================================
# STEP 4: Insert 1s silence using CORRECT timestamps (API returned)
# =============================================
print("\n" + "=" * 60)
print("STEP 4: Insert 1s silence with CORRECT (API) timestamps")
print("=" * 60)
sr, audio = read_wav("output/test_verify_paused.wav")
correct_cut_times = []
for idx in valid_indices:
    if idx < len(gen_ts):
        correct_cut_times.append(gen_ts[idx]["end"])

result_correct = insert_silence(audio, sr, correct_cut_times, 1.0)
save_wav("output/test_phrases_correct_ts.wav", result_correct, sr)
print("Saved: output/test_phrases_correct_ts.wav ({:.1f}s)".format(len(result_correct) / sr))

# =============================================
# STEP 5: Insert 1s silence using WRONG timestamps (probe, no pauses)
# =============================================
print("\n" + "=" * 60)
print("STEP 5: Insert 1s silence with WRONG (probe) timestamps")
print("=" * 60)
wrong_cut_times = []
for idx in valid_indices:
    if idx < len(probe_ts):
        wrong_cut_times.append(probe_ts[idx]["end"])

result_wrong = insert_silence(audio, sr, wrong_cut_times, 1.0)
save_wav("output/test_phrases_wrong_ts.wav", result_wrong, sr)
print("Saved: output/test_phrases_wrong_ts.wav ({:.1f}s)".format(len(result_wrong) / sr))

print("\n" + "=" * 60)
print("DONE - Compare the two files:")
print("  CORRECT: output/test_phrases_correct_ts.wav")
print("  WRONG:   output/test_phrases_wrong_ts.wav")
print("=" * 60)
