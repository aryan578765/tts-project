"""
Test: Generate from ORIGINAL text + apply fade-out at cut points.
No text modification = natural intonation preserved.
Fade-out at tight boundaries = clean cuts without abruptness.
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

os.makedirs("output/fadeout_test", exist_ok=True)

def save_wav(path, audio_data, sr):
    audio_int16 = np.clip(audio_data, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

# Generate original text - NO modifications
print("Generating original text (no modifications)...")
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
with open("output/fadeout_test/original_full.wav", "wb") as f:
    f.write(audio_bytes)

with wave.open("output/fadeout_test/original_full.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

# For each phrase boundary, find best cut point and apply fade-out
FADE_MS = 25  # 25ms fade-out at each boundary
SILENCE_MS = 1000  # 1s silence between phrases
fade_samples = int(sr * FADE_MS / 1000)
silence_samples = int(sr * SILENCE_MS / 1000)

print("\nFinding cut points and applying fade-out:")
boundaries = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts):
        continue
    
    word = word_ts[idx]
    word_end_s = word["end"]
    word_end = int(word_end_s * sr)
    
    # Next word start
    if idx + 1 < len(word_ts):
        next_start_s = word_ts[idx + 1]["start"]
        next_start = int(next_start_s * sr)
    else:
        next_start_s = word_end_s + 0.5
        next_start = int(next_start_s * sr)
    
    gap_ms = (next_start_s - word_end_s) * 1000
    
    # Search for minimum energy point between word_end and next_start
    search_start = max(0, word_end - int(sr * 0.01))  # 10ms before word end
    search_end = min(len(audio), next_start + int(sr * 0.01))  # 10ms after next start
    
    window = int(sr * 0.01)  # 10ms analysis window
    best_pos = word_end
    best_energy = float("inf")
    
    for pos in range(search_start, search_end - window, window // 4):
        seg = audio[pos:pos + window]
        energy = np.sqrt(np.mean(seg ** 2))
        if energy < best_energy:
            best_energy = energy
            best_pos = pos + window // 2
    
    # Determine if this needs fade-out (tight boundary with high energy)
    needs_fade = best_energy > 50
    
    boundaries.append({
        "idx": idx,
        "word": word["word"],
        "cut_sample": best_pos,
        "cut_time": best_pos / sr,
        "energy": best_energy,
        "gap_ms": gap_ms,
        "needs_fade": needs_fade
    })
    
    status = "FADE-OUT" if needs_fade else "CLEAN"
    print("  idx {:>3} {:>15} gap={:>5.0f}ms energy={:>6.0f} -> {}".format(
        idx, word["word"], gap_ms, best_energy, status))

# Build final audio: cut at each boundary, apply fade-out if needed, insert silence
audio_copy = audio.copy()

# Apply fade-out at each boundary BEFORE cutting
for b in boundaries:
    if b["needs_fade"]:
        cut = b["cut_sample"]
        fade_start = max(0, cut - fade_samples)
        fade_end = cut
        fade_len = fade_end - fade_start
        if fade_len > 0:
            fade_curve = np.cos(np.linspace(0, np.pi / 2, fade_len)).astype(np.float32)
            audio_copy[fade_start:fade_end] *= fade_curve
            # Zero out a tiny bit after cut to ensure clean silence
            zero_end = min(len(audio_copy), cut + int(sr * 0.005))  # 5ms zero
            audio_copy[cut:zero_end] = 0

# Now cut and insert silence
silence = np.zeros(silence_samples, dtype=np.float32)
cut_times = sorted([b["cut_sample"] for b in boundaries])

pieces = []
cursor = 0
for cs in cut_times:
    cs = max(0, min(cs, len(audio_copy)))
    pieces.append(audio_copy[cursor:cs])
    pieces.append(silence)
    cursor = cs
pieces.append(audio_copy[cursor:])

result = np.concatenate(pieces)
save_wav("output/fadeout_test/fadeout_phrases.wav", result, sr)
print("\nSaved fadeout_phrases.wav ({:.1f}s)".format(len(result) / sr))

# Verify all cuts
print("\nVerifying cut cleanliness:")
clean_count = 0
for b in boundaries:
    cs = b["cut_sample"]
    # Check energy right at cut point after fade-out was applied
    check_start = max(0, cs - int(sr * 0.01))
    check_end = min(len(audio_copy), cs + int(sr * 0.01))
    post_energy = np.sqrt(np.mean(audio_copy[check_start:check_end] ** 2))
    status = "CLEAN" if post_energy < 200 else "WARNING"
    if post_energy < 200:
        clean_count += 1
    print("  {:>15} at {:.3f}s: pre_energy={:.0f} post_energy={:.0f} -> {}".format(
        b["word"], b["cut_time"], b["energy"], post_energy, status))

print("\nRESULT: {}/{} CLEAN after fade-out".format(clean_count, len(boundaries)))
print("\nFiles:")
print("  original_full.wav    -> original audio, natural intonation")
print("  fadeout_phrases.wav  -> cut with fade-out at boundaries + 1s silence")
