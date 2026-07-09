"""Test with a completely new sentence to check if 'apps' issue is isolated."""
import sys, os, requests, json, base64, wave, numpy as np
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

URL = "https://api.runpod.ai/v2/54td14oe86jexh/runsync"
HEADERS = {"Authorization": "Bearer " + os.environ.get("RUNPOD_API_KEY", ""), "Content-Type": "application/json"}

TEXT = "The new update brings significant improvements to how users interact with their devices and applications. Battery life has been extended through smarter background processing and optimized network connections. Privacy settings now give users more control over which apps can access their photos and location data. The camera system includes advanced computational photography features that work seamlessly across all supported devices."

# Phrase boundaries
# Phrase 1: "The new update brings significant improvements to how users interact with their devices and applications." (idx 0-14)
# Phrase 2: "Battery life has been extended through smarter background processing and optimized network connections." (idx 15-24)
# Phrase 3: "Privacy settings now give users more control over which apps can access their photos and location data." (idx 25-39)
# Phrase 4: "The camera system includes advanced computational photography features that work seamlessly across all supported devices." (idx 40-52)
PAUSE_AFTER = [14, 24, 39]

os.makedirs("output/new_sentence_test", exist_ok=True)

r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "pause_after": PAUSE_AFTER
}}, timeout=300)
out = r.json().get("output", {})
if "error" in out:
    print("ERROR:", out["error"]); sys.exit(1)

word_ts = out.get("word_timestamps", [])
cut_points = out.get("phrase_cut_points", [])
dur = out.get("duration_seconds", 0)
print("Duration: {:.2f}s | Words: {} | Cut points: {}".format(dur, len(word_ts), len(cut_points)))

for i, w in enumerate(word_ts):
    marker = " <-- PAUSE" if i in PAUSE_AFTER else ""
    print("  {:>3} {:>20} {:.3f}s - {:.3f}s{}".format(i, w["word"], w["start"], w["end"], marker))

audio_bytes = base64.b64decode(out["audio_base64"])
with open("output/new_sentence_test/full_audio.wav", "wb") as f:
    f.write(audio_bytes)

with wave.open("output/new_sentence_test/full_audio.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

matched = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts): continue
    word_end = word_ts[idx]["end"]
    best_cp, best_dist = None, float("inf")
    for cp in cut_points:
        dist = abs(cp["time"] - word_end)
        if dist < best_dist: best_dist = dist; best_cp = cp
    if best_cp and best_dist < 0.6:
        matched.append({"idx": idx, "word": word_ts[idx]["word"], "cut_time": best_cp["time"]})
        print("  MATCH idx {} {} -> cut={:.3f}s".format(idx, word_ts[idx]["word"], best_cp["time"]))
    else:
        print("  NO MATCH idx {} {}".format(idx, word_ts[idx]["word"]))

silence = np.zeros(int(sr * 1.0), dtype=audio.dtype)
cut_times = sorted([m["cut_time"] for m in matched])
pieces = []
cursor = 0
for t in cut_times:
    s = int(round(t * sr)); s = max(0, min(s, len(audio)))
    pieces.append(audio[cursor:s]); pieces.append(silence); cursor = s
pieces.append(audio[cursor:])
result = np.clip(np.concatenate(pieces), -32768, 32767).astype(np.int16)
with wave.open("output/new_sentence_test/phrases.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(result.tobytes())
print("Saved phrases.wav ({:.1f}s)".format(len(result) / sr))

print("\nVerifying cuts:")
clean = 0
for m in matched:
    c = int(m["cut_time"] * sr)
    s, e = max(0, c - int(sr * 0.015)), min(len(audio), c + int(sr * 0.015))
    energy = np.sqrt(np.mean(audio[s:e] ** 2))
    status = "CLEAN" if energy < 200 else "WARNING"
    if energy < 200: clean += 1
    print("  {:>15} at {:.3f}s: energy={:.0f} -> {}".format(m["word"], m["cut_time"], energy, status))

print("\nRESULT: {}/{} matched, {}/{} CLEAN".format(len(matched), len(PAUSE_AFTER), clean, len(matched)))
