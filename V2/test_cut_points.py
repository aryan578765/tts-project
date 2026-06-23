"""Test phrase_cut_points: compare silence-based cuts vs alignment timestamps."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64, wave, numpy as np

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

print("Calling API with pause_after ({} positions)...".format(len(PAUSE_AFTER)))
payload = {
    "input": {
        "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
        "timestamps": True, "word_boundaries": True,
        "pause_after": PAUSE_AFTER, "micro_pause_ms": 50, "crossfade_ms": 5.0
    }
}
r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
out = r.json().get("output", {})

if "error" in out:
    print("ERROR: {}".format(out["error"]))
    sys.exit(1)

print("Duration: {}s | RTF: {}".format(out.get("duration_seconds"), out.get("rtf")))
print("Word timestamps: {} words".format(len(out.get("word_timestamps", []))))

# Check phrase_cut_points
cut_points = out.get("phrase_cut_points", [])
print("\n" + "=" * 60)
print("PHRASE CUT POINTS (silence-detected): {} found".format(len(cut_points)))
print("=" * 60)

if not cut_points:
    print("NO CUT POINTS RETURNED - feature may not be deployed")
    sys.exit(1)

for i, cp in enumerate(cut_points):
    print("  Cut {:>2}: time={:.4f}s  silence_duration={:.1f}ms".format(i+1, cp["time"], cp["duration_ms"]))

# Save audio
audio_bytes = base64.b64decode(out["audio_base64"])
with open("output/test_cut_points.wav", "wb") as f:
    f.write(audio_bytes)

# Now cut the audio using phrase_cut_points and insert 1s silence
with wave.open("output/test_cut_points.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

silence = np.zeros(int(sr * 1.0), dtype=audio.dtype)
cut_times = [cp["time"] for cp in cut_points]

# Filter: only use cuts that align with our PAUSE_AFTER positions (within 200ms of a word end)
word_ts = out.get("word_timestamps", [])
phrase_cuts = []
for cp in cut_points:
    ct = cp["time"]
    for idx in PAUSE_AFTER:
        if idx < len(word_ts):
            wend = word_ts[idx]["end"]
            if abs(ct - wend) < 0.2:
                phrase_cuts.append(ct)
                break

print("\nFiltered to {} phrase-boundary cuts (matching pause_after positions)".format(len(phrase_cuts)))

pieces = []
cursor = 0
for t in sorted(phrase_cuts):
    sample = int(round(t * sr))
    sample = max(0, min(sample, len(audio)))
    pieces.append(audio[cursor:sample])
    pieces.append(silence)
    cursor = sample
pieces.append(audio[cursor:])

result = np.concatenate(pieces)
result_int16 = np.clip(result, -32768, 32767).astype(np.int16)
with wave.open("output/test_phrases_silence_detected.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(result_int16.tobytes())

print("Saved: output/test_phrases_silence_detected.wav ({:.1f}s)".format(len(result) / sr))

# Energy check at each cut point
print("\n" + "=" * 60)
print("ENERGY AT CUT POINTS (should all be ~0 = silence)")
print("=" * 60)
all_clean = True
for i, ct in enumerate(phrase_cuts):
    center = int(ct * sr)
    start = max(0, center - int(sr * 0.015))
    end = min(len(audio), center + int(sr * 0.015))
    energy = np.sqrt(np.mean(audio[start:end] ** 2))
    status = "CLEAN" if energy < 200 else "WARNING"
    if energy >= 200:
        all_clean = False
    print("  Cut {:>2} at {:.3f}s: energy={:.0f} -> {}".format(i+1, ct, energy, status))

print("\n" + "=" * 60)
if all_clean:
    print("ALL CUTS LAND IN SILENCE - phrase cutting is reliable!")
else:
    print("Some cuts have energy - review those positions")
print("=" * 60)
