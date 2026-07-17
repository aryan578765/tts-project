"""Definitive timestamp analysis: check energy at exact cut points on ORIGINAL paused audio.

If CORRECT timestamps are accurate, they should land in/near silence (post-word boundary).
If WRONG timestamps are shifted, they should land mid-word (high energy).
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import wave
import numpy as np

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return sr, audio

# Read the ORIGINAL paused audio (before 1s silence insertion)
sr, audio = read_wav("output/test_verify_paused.wav")
print("Original paused audio: {:.2f}s, {} samples, {}Hz".format(len(audio)/sr, len(audio), sr))

# These are the timestamps from the verification test
# CORRECT = API returned (with pause offsets)
# WRONG = Probe (without pause offsets)
# Regenerate from the test output

# Read both timestamp sets from the verification test log
import json, requests, base64

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things.
Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107,
               115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

print("\nFetching probe timestamps (no pauses)...")
probe_resp = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
probe_ts = probe_resp.json().get("output", {}).get("word_timestamps", [])

print("Fetching generation timestamps (with pauses)...")
valid = [i for i in PAUSE_AFTER if 0 <= i < len(probe_ts) - 1]
gen_resp = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "word_boundaries": True,
    "pause_after": valid, "micro_pause_ms": 50, "crossfade_ms": 5.0
}}, timeout=300)
gen_ts = gen_resp.json().get("output", {}).get("word_timestamps", [])

print("Probe words: {}, Gen words: {}".format(len(probe_ts), len(gen_ts)))

def get_energy_at(audio, sr, timestamp, window_ms=30):
    """Get RMS energy in a window centered on timestamp."""
    center = int(timestamp * sr)
    half_win = int(sr * window_ms / 2000)
    start = max(0, center - half_win)
    end = min(len(audio), center + half_win)
    if end <= start:
        return 0
    chunk = audio[start:end]
    return np.sqrt(np.mean(chunk ** 2))

def get_max_energy_at(audio, sr, timestamp, window_ms=30):
    """Get max absolute amplitude in a window centered on timestamp."""
    center = int(timestamp * sr)
    half_win = int(sr * window_ms / 2000)
    start = max(0, center - half_win)
    end = min(len(audio), center + half_win)
    if end <= start:
        return 0
    return float(np.max(np.abs(audio[start:end])))

# Analyze energy at each cut point on the ORIGINAL paused audio
print("\n" + "=" * 90)
print("ENERGY AT CUT POINTS ON ORIGINAL PAUSED AUDIO (30ms window)")
print("=" * 90)
print("{:>4}  {:>15}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}".format(
    "idx", "word", "correct_t", "wrong_t", "correct_E", "wrong_E", "winner"))

correct_total_energy = 0
wrong_total_energy = 0
correct_wins = 0
wrong_wins = 0
ties = 0

for idx in valid:
    if idx >= len(gen_ts) or idx >= len(probe_ts):
        continue
    
    correct_time = gen_ts[idx]["end"]
    wrong_time = probe_ts[idx]["end"]
    
    correct_energy = get_energy_at(audio, sr, correct_time)
    wrong_energy = get_energy_at(audio, sr, wrong_time)
    
    correct_total_energy += correct_energy
    wrong_total_energy += wrong_energy
    
    if correct_energy < wrong_energy - 50:
        winner = "CORRECT"
        correct_wins += 1
    elif wrong_energy < correct_energy - 50:
        winner = "WRONG"
        wrong_wins += 1
    else:
        winner = "tie"
        ties += 1
    
    print("{:>4}  {:>15}  {:>10.3f}  {:>10.3f}  {:>10.0f}  {:>10.0f}  {:>8}".format(
        idx, gen_ts[idx]["word"][:15], correct_time, wrong_time, correct_energy, wrong_energy, winner))

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
print("CORRECT timestamps win: {} / {}".format(correct_wins, len(valid)))
print("WRONG timestamps win:   {} / {}".format(wrong_wins, len(valid)))
print("Ties:                   {} / {}".format(ties, len(valid)))
print("Avg energy at CORRECT cuts: {:.0f}".format(correct_total_energy / max(len(valid), 1)))
print("Avg energy at WRONG cuts:   {:.0f}".format(wrong_total_energy / max(len(valid), 1)))

if correct_total_energy < wrong_total_energy:
    ratio = wrong_total_energy / max(correct_total_energy, 1)
    print("\n=> CORRECT timestamps have {:.1f}x LOWER energy at cut points.".format(ratio))
    print("=> This confirms API timestamps are MORE ACCURATE for phrase cutting.")
elif wrong_total_energy < correct_total_energy:
    ratio = correct_total_energy / max(wrong_total_energy, 1)
    print("\n=> WRONG timestamps have {:.1f}x LOWER energy at cut points.".format(ratio))
    print("=> WARNING: Probe timestamps might actually be better!")
else:
    print("\n=> Both are equivalent in energy at cut points.")
