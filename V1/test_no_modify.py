"""
Test: Generate ORIGINAL text (no punctuation changes) and compare intonation.
Goal: Prove that generating without any text modification preserves natural flow.
Then cut using timestamps + silence detection (no text modification).
"""
import sys, os, requests, json, base64, wave, numpy as np
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

URL = "https://api.runpod.ai/v2/54td14oe86jexh/runsync"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software. Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things. Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable. Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107,
               115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

os.makedirs("output/no_modify_test", exist_ok=True)

def save_wav(path, audio, sr):
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

# ============================================================
# TEST 1: Generate with NO text modification (pure original)
# Use pause_mode="none" by NOT sending pause_after at all
# ============================================================
print("=" * 60)
print("TEST 1: Original text, NO modifications")
print("=" * 60)

r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out = r.json().get("output", {})
if "error" in out:
    print("ERROR:", out["error"]); sys.exit(1)

word_ts = out.get("word_timestamps", [])
dur = out.get("duration_seconds", 0)
print("Duration: {:.2f}s | Words: {}".format(dur, len(word_ts)))

audio_bytes = base64.b64decode(out["audio_base64"])
with open("output/no_modify_test/original_full.wav", "wb") as f:
    f.write(audio_bytes)

with wave.open("output/no_modify_test/original_full.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

print("Saved original_full.wav")

# Now cut at pause_after positions using timestamps only
# Find the best cut point near each boundary
silence_threshold = float(np.max(np.abs(audio))) * 0.02
window_samples = int(sr * 0.02)

print("\nFinding cut points at phrase boundaries:")
cut_times = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts):
        continue
    word = word_ts[idx]
    word_end = word["end"]
    
    # Search for lowest energy region between this word end and next word start
    if idx + 1 < len(word_ts):
        next_start = word_ts[idx + 1]["start"]
    else:
        next_start = word_end + 0.5
    
    # Search window: word_end - 20ms to next_start + 20ms
    search_start = max(0, int((word_end - 0.02) * sr))
    search_end = min(len(audio), int((next_start + 0.02) * sr))
    
    # Find minimum energy point in this window
    best_pos = int(word_end * sr)
    best_energy = float("inf")
    
    for pos in range(search_start, search_end - window_samples, window_samples // 4):
        seg = audio[pos:pos + window_samples]
        energy = np.sqrt(np.mean(seg ** 2))
        if energy < best_energy:
            best_energy = energy
            best_pos = pos + window_samples // 2
    
    cut_time = best_pos / sr
    gap_ms = (next_start - word_end) * 1000 if idx + 1 < len(word_ts) else 0
    cut_times.append(cut_time)
    print("  idx {:>3} {:>15} end={:.3f}s next={:.3f}s gap={:.0f}ms -> cut={:.3f}s energy={:.0f}".format(
        idx, word["word"], word_end, next_start, gap_ms, cut_time, best_energy))

# Insert 1s silence at each cut point
silence = np.zeros(int(sr * 1.0), dtype=audio.dtype)
cut_times_sorted = sorted(cut_times)

pieces = []
cursor = 0
for t in cut_times_sorted:
    s = int(round(t * sr))
    s = max(0, min(s, len(audio)))
    pieces.append(audio[cursor:s])
    pieces.append(silence)
    cursor = s
pieces.append(audio[cursor:])

result = np.concatenate(pieces)
save_wav("output/no_modify_test/original_phrases.wav", result, sr)
print("\nSaved original_phrases.wav ({:.1f}s)".format(len(result) / sr))

# ============================================================
# TEST 2: Generate with comma injection (current v16 approach)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Comma-injected text")
print("=" * 60)

r2 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "pause_after": PAUSE_AFTER, "pause_mode": "punctuation", "pause_char": ","
}}, timeout=300)
out2 = r2.json().get("output", {})
if "error" in out2:
    print("ERROR:", out2["error"]); sys.exit(1)

dur2 = out2.get("duration_seconds", 0)
word_ts2 = out2.get("word_timestamps", [])
print("Duration: {:.2f}s | Words: {}".format(dur2, len(word_ts2)))

audio_bytes2 = base64.b64decode(out2["audio_base64"])
with open("output/no_modify_test/comma_full.wav", "wb") as f:
    f.write(audio_bytes2)
print("Saved comma_full.wav")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print("Original duration: {:.2f}s".format(dur))
print("Comma duration:    {:.2f}s".format(dur2))
print("\nListen to both and compare intonation:")
print("  1. original_full.wav     -> natural intonation, no modifications")
print("  2. original_phrases.wav  -> same audio, cut + 1s silence at boundaries")
print("  3. comma_full.wav        -> comma-injected version")
