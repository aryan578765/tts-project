"""Test punctuation mode with a new 35s text to verify quality."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64, wave, numpy as np

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

TEXT = """The rapid evolution of artificial intelligence has transformed how businesses operate across every industry. Companies that once relied on manual data entry now use machine learning algorithms to process millions of records in seconds. Healthcare providers are leveraging deep learning models to detect diseases earlier than ever before, while financial institutions deploy neural networks to identify fraudulent transactions in real time. Education is also being reshaped, with personalized tutoring systems adapting to each student's unique learning pace and style."""

# Split into ~8 phrases for a 35s clip
# Word count: ~80 words
# Phrase boundaries at natural points
PAUSE_AFTER = [11, 21, 31, 41, 53, 63, 73]

os.makedirs("output/new_test", exist_ok=True)

# Generate with punctuation mode (default)
print("=" * 70)
print("Generating with PUNCTUATION mode (new default)")
print("=" * 70)
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "pause_after": PAUSE_AFTER
}}, timeout=300)
out = r.json().get("output", {})
if "error" in out:
    print("ERROR:", out["error"])
    sys.exit(1)

word_ts = out.get("word_timestamps", [])
cut_points = out.get("phrase_cut_points", [])
print("Duration: {:.2f}s | Words: {} | Cut points: {}".format(
    out.get("duration_seconds", 0), len(word_ts), len(cut_points)))

# Print words with phrase boundary markers
for i, w in enumerate(word_ts):
    marker = " <-- PAUSE" if i in PAUSE_AFTER else ""
    print("  {:>3} {:>20} {:.3f}s - {:.3f}s{}".format(i, w["word"], w["start"], w["end"], marker))

# Save audio
audio_bytes = base64.b64decode(out["audio_base64"])
with open("output/new_test/full_audio.wav", "wb") as f:
    f.write(audio_bytes)
print("\nSaved: output/new_test/full_audio.wav")

# Match cut points
print("\n" + "=" * 70)
print("Matching cut points to pause_after positions")
print("=" * 70)

matched = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts):
        continue
    word = word_ts[idx]["word"]
    word_end = word_ts[idx]["end"]
    
    # Try after_word_idx match first
    match = None
    for cp in cut_points:
        if cp.get("after_word_idx") == idx:
            match = cp
            break
    
    if not match:
        # Proximity match
        best_cp, best_dist = None, float("inf")
        for cp in cut_points:
            dist = abs(cp["time"] - word_end)
            if dist < best_dist:
                best_dist = dist
                best_cp = cp
        if best_cp and best_dist < 0.6:
            match = best_cp

    if match:
        matched.append({"idx": idx, "word": word, "cut_time": match["time"]})
        print("  idx {:>3} {:>20} end={:.3f}s -> cut={:.4f}s MATCHED".format(
            idx, word, word_end, match["time"]))
    else:
        print("  idx {:>3} {:>20} end={:.3f}s -> NO MATCH".format(idx, word, word_end))

print("\nMatched: {}/{}".format(len(matched), len(PAUSE_AFTER)))

# Cut and insert 1s silence
print("\n" + "=" * 70)
print("Cutting audio with 1s silence at phrase boundaries")
print("=" * 70)

with wave.open("output/new_test/full_audio.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

silence = np.zeros(int(sr * 1.0), dtype=audio.dtype)
cut_times = sorted([m["cut_time"] for m in matched])

pieces = []
cursor = 0
for t in cut_times:
    sample = int(round(t * sr))
    sample = max(0, min(sample, len(audio)))
    pieces.append(audio[cursor:sample])
    pieces.append(silence)
    cursor = sample
pieces.append(audio[cursor:])

result = np.concatenate(pieces)
result_int16 = np.clip(result, -32768, 32767).astype(np.int16)
with wave.open("output/new_test/phrases.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(result_int16.tobytes())
print("Saved: output/new_test/phrases.wav ({:.1f}s)".format(len(result) / sr))

# Verify cuts
print("\nVerifying cuts:")
clean = 0
for m in matched:
    center = int(m["cut_time"] * sr)
    s = max(0, center - int(sr * 0.015))
    e = min(len(audio), center + int(sr * 0.015))
    energy = np.sqrt(np.mean(audio[s:e] ** 2))
    status = "CLEAN" if energy < 200 else "WARNING"
    if energy < 200: clean += 1
    print("  {:>20} at {:.3f}s: energy={:.0f} -> {}".format(m["word"], m["cut_time"], energy, status))

print("\nRESULT: {}/{} cuts CLEAN".format(clean, len(matched)))
