"""
Test v2: Aggressive fade-out for ultra-tight boundaries.
- 40ms fade-out for tight gaps (<150ms)
- 15ms zero-pad after cut point
- Original text, no modifications
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

os.makedirs("output/fadeout_v2", exist_ok=True)

def save_wav(path, audio_data, sr):
    audio_int16 = np.clip(audio_data, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

# Generate original text
print("Generating original text (no modifications)...")
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out = r.json().get("output", {})
if "error" in out:
    print("ERROR:", out["error"]); sys.exit(1)

word_ts = out.get("word_timestamps", [])
print("Duration: {:.2f}s | Words: {}".format(out.get("duration_seconds", 0), len(word_ts)))

audio_bytes = base64.b64decode(out["audio_base64"])
with open("output/fadeout_v2/original_full.wav", "wb") as f:
    f.write(audio_bytes)

with wave.open("output/fadeout_v2/original_full.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

# Parameters
FADE_TIGHT_MS = 40   # Fade for tight gaps (<150ms)
FADE_NORMAL_MS = 15  # Fade for wider gaps
ZERO_PAD_MS = 15     # Zero-pad after cut
SILENCE_MS = 1000

silence_samples = int(sr * SILENCE_MS / 1000)

print("\nAnalyzing boundaries:")
boundaries = []
for idx in PAUSE_AFTER:
    if idx >= len(word_ts):
        continue
    
    word = word_ts[idx]
    word_end_s = word["end"]
    word_end = int(word_end_s * sr)
    
    if idx + 1 < len(word_ts):
        next_start_s = word_ts[idx + 1]["start"]
        next_start = int(next_start_s * sr)
    else:
        next_start_s = word_end_s + 0.5
        next_start = min(len(audio), int(next_start_s * sr))
    
    gap_ms = (next_start_s - word_end_s) * 1000
    
    # For tight gaps: cut right at word end (include full word)
    # For wide gaps: find minimum energy in the gap
    if gap_ms < 150:
        # Tight gap: cut at word end, will need aggressive fade
        cut_pos = word_end
        window = int(sr * 0.01)
        seg = audio[max(0, cut_pos - window):min(len(audio), cut_pos + window)]
        energy = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0
        fade_ms = FADE_TIGHT_MS
    else:
        # Wide gap: find silence in gap
        window = int(sr * 0.01)
        best_pos = word_end
        best_energy = float("inf")
        for pos in range(max(0, word_end), min(len(audio) - window, next_start), window // 4):
            seg = audio[pos:pos + window]
            e = np.sqrt(np.mean(seg ** 2))
            if e < best_energy:
                best_energy = e
                best_pos = pos + window // 2
        cut_pos = best_pos
        energy = best_energy
        fade_ms = FADE_NORMAL_MS
    
    boundaries.append({
        "idx": idx, "word": word["word"],
        "cut_sample": cut_pos, "cut_time": cut_pos / sr,
        "energy": energy, "gap_ms": gap_ms, "fade_ms": fade_ms
    })
    
    print("  idx {:>3} {:>15} gap={:>5.0f}ms energy={:>6.0f} fade={:>2.0f}ms".format(
        idx, word["word"], gap_ms, energy, fade_ms))

# Apply fade-out and zero-pad at EVERY boundary
audio_out = audio.copy()
zero_pad = int(sr * ZERO_PAD_MS / 1000)

for b in boundaries:
    cut = b["cut_sample"]
    fade_len = int(sr * b["fade_ms"] / 1000)
    
    # Apply cosine fade-out BEFORE the cut
    fade_start = max(0, cut - fade_len)
    actual_fade = cut - fade_start
    if actual_fade > 0:
        fade_curve = np.cos(np.linspace(0, np.pi / 2, actual_fade)).astype(np.float32)
        audio_out[fade_start:cut] *= fade_curve
    
    # Zero-pad AFTER the cut (kills any bleed from next word)
    zero_end = min(len(audio_out), cut + zero_pad)
    audio_out[cut:zero_end] = 0

# Cut and insert silence
cut_samples = sorted([b["cut_sample"] for b in boundaries])
silence = np.zeros(silence_samples, dtype=np.float32)

pieces = []
cursor = 0
for cs in cut_samples:
    cs = max(0, min(cs, len(audio_out)))
    pieces.append(audio_out[cursor:cs])
    pieces.append(silence)
    cursor = cs
pieces.append(audio_out[cursor:])

result = np.concatenate(pieces)
save_wav("output/fadeout_v2/phrases.wav", result, sr)
print("\nSaved phrases.wav ({:.1f}s)".format(len(result) / sr))

# Verify
print("\nVerifying:")
clean = 0
for b in boundaries:
    cs = b["cut_sample"]
    # Check 5ms window right at cut
    hw = int(sr * 0.005)
    s = max(0, cs - hw)
    e = min(len(audio_out), cs + hw)
    post_e = np.sqrt(np.mean(audio_out[s:e] ** 2))
    ok = post_e < 200
    if ok: clean += 1
    print("  {:>15} energy: {:.0f} -> {:.0f} {}".format(
        b["word"], b["energy"], post_e, "CLEAN" if ok else "WARNING"))

print("\nRESULT: {}/{} CLEAN".format(clean, len(boundaries)))
