"""End-to-end test for phrase cutting workflow:
1. Generate long audio with pause_after
2. Get phrase_cut_points (silence-detected)
3. Use cut_points to insert 1s silence at phrase boundaries
4. Verify all cuts land in silence
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64, wave, numpy as np

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

# Test text
TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things.
Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

# Phrase boundary indices
PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107,
               115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

print("=" * 70)
print("STEP 1: Generate audio with 50ms pauses at {} positions".format(len(PAUSE_AFTER)))
print("=" * 70)
payload = {
    "input": {
        "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
        "timestamps": True, "word_boundaries": True,
        "pause_after": PAUSE_AFTER, "micro_pause_ms": 50
    }
}
r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
resp = r.json()
if "output" not in resp or "audio_base64" not in resp.get("output", {}):
    print("ERROR: Bad API response")
    print(json.dumps(resp, indent=2)[:500])
    sys.exit(1)
out = resp["output"]

word_ts = out.get("word_timestamps", [])
cut_points = out.get("phrase_cut_points", [])
print("Duration: {}s | Words: {} | Cut points: {}".format(
    out.get("duration_seconds"), len(word_ts), len(cut_points)))

# Save original audio
audio_bytes = base64.b64decode(out["audio_base64"])
os.makedirs("output", exist_ok=True)
with open("output/test_step1_paused.wav", "wb") as f:
    f.write(audio_bytes)
print("Saved: output/test_step1_paused.wav")

print("\n" + "=" * 70)
print("STEP 2: Match cut_points to pause_after word boundaries")
print("=" * 70)

# For each pause_after index, find the matching cut_point by after_word_idx
matched_cuts = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts):
        continue
    word = word_ts[idx]["word"]
    word_end = word_ts[idx]["end"]
    
    # Find cut_point with matching after_word_idx
    match = None
    for cp in cut_points:
        if cp.get("after_word_idx") == idx:
            match = cp
            break
    
    if match:
        matched_cuts.append({
            "word_idx": idx,
            "word": word,
            "word_end": word_end,
            "cut_time": match["time"],
            "silence_ms": match["duration_ms"],
        })
        print("  idx {:>3} {:>15} end={:.3f}s -> cut={:.4f}s (silence={:.0f}ms)".format(
            idx, word, word_end, match["time"], match["duration_ms"]))
    else:
        # Fallback: proximity match
        best_cp = None
        best_dist = float("inf")
        for cp in cut_points:
            dist = abs(cp["time"] - word_end)
            if dist < best_dist:
                best_dist = dist
                best_cp = cp
        if best_cp and best_dist < 0.6:
            matched_cuts.append({
                "word_idx": idx,
                "word": word,
                "word_end": word_end,
                "cut_time": best_cp["time"],
                "silence_ms": best_cp["duration_ms"],
            })
            print("  idx {:>3} {:>15} end={:.3f}s -> cut={:.4f}s (proximity, offset={:.1f}ms)".format(
                idx, word, word_end, best_cp["time"], best_dist*1000))
        else:
            print("  idx {:>3} {:>15} end={:.3f}s -> NO MATCH".format(idx, word, word_end))

print("\nMatched: {}/{} phrase boundaries".format(len(matched_cuts), len(PAUSE_AFTER)))

print("\n" + "=" * 70)
print("STEP 3: Insert 1s silence at matched cut points")
print("=" * 70)

with wave.open("output/test_step1_paused.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

silence = np.zeros(int(sr * 1.0), dtype=audio.dtype)
cut_times = sorted([mc["cut_time"] for mc in matched_cuts])

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
with wave.open("output/test_step3_phrases.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(result_int16.tobytes())
print("Saved: output/test_step3_phrases.wav ({:.1f}s)".format(len(result) / sr))

print("\n" + "=" * 70)
print("STEP 4: Verify all cuts land in silence")
print("=" * 70)

clean = 0
warning = 0
for mc in matched_cuts:
    ct = mc["cut_time"]
    center = int(ct * sr)
    start = max(0, center - int(sr * 0.015))
    end = min(len(audio), center + int(sr * 0.015))
    energy = np.sqrt(np.mean(audio[start:end] ** 2))
    status = "CLEAN" if energy < 200 else "WARNING"
    if energy < 200:
        clean += 1
    else:
        warning += 1
    print("  {:>15} at {:.3f}s: energy={:.0f} -> {}".format(mc["word"], ct, energy, status))

print("\n" + "=" * 70)
print("RESULT: {}/{} cuts are CLEAN ({:.0f}%)".format(clean, len(matched_cuts), clean/max(len(matched_cuts),1)*100))
print("Audio: output/test_step3_phrases.wav")
print("=" * 70)
