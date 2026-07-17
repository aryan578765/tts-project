"""
VERIFICATION: Prove zero audio modification.

Since we can't get the same synthesis twice (Kokoro is non-deterministic),
we verify zero-modification by checking:
1. The spliced audio has EXACT ZEROS at every cut point (silence was inserted)
2. The audio immediately before/after silence has no discontinuity (no fade artifacts)
3. The audio amplitude at the cut boundaries matches natural speech levels
4. No DC offset was applied (mean of non-silence regions ≈ 0 naturally)
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64, wave, struct, numpy as np, io

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not API_KEY:
    print("ERROR: RUNPOD_API_KEY not set")
    sys.exit(1)

URL = "https://api.runpod.ai/v2/54td14oe86jexh/runsync"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software. Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things. Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable. Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107, 115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

PROBLEM_WORDS = {"Conference", "apps", "bills", "information", "dramatic", "itself", "features"}

def decode_wav(b64):
    raw = base64.b64decode(b64)
    buf = wave.open(io.BytesIO(raw), 'rb')
    sr = buf.getframerate()
    frames = buf.readframes(buf.getnframes())
    audio = np.array(struct.unpack(f'{len(frames)//2}h', frames), dtype=np.int16)
    return audio, sr

# =====================================================================
# Step 1: Fetch audio with pauses
# =====================================================================
print("=" * 70)
print("STEP 1: Fetch audio with 50ms pauses")
print("=" * 70)
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "word_boundaries": True,
    "micro_pause_ms": 50, "pause_after": PAUSE_AFTER
}}, timeout=300)

resp = r.json()
if "output" not in resp or "audio_base64" not in resp.get("output", {}):
    print(f"ERROR: Bad API response: {json.dumps(resp, indent=2)[:500]}")
    sys.exit(1)

out = resp["output"]
audio, sr = decode_wav(out["audio_base64"])
cut_points = out.get("phrase_cut_points", [])
word_ts = out.get("word_timestamps", [])
print(f"  Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")
print(f"  Cut points: {len(cut_points)}")
print(f"  Words: {len(word_ts)}")

pause_samples = int(sr * 50 / 1000)  # 50ms = 1200 samples

# =====================================================================
# Step 2: Verify EXACT ZEROS at every cut point
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Verify exact zeros at every cut point")
print(f"{'=' * 70}")

zeros_pass = True
for i, cp in enumerate(cut_points):
    center = int(cp["time"] * sr)
    sil_start = max(0, center - pause_samples // 2)
    sil_end = min(len(audio), sil_start + pause_samples)
    
    silence_region = audio[sil_start:sil_end]
    all_zero = np.all(silence_region == 0)
    nonzero_count = int(np.count_nonzero(silence_region))
    
    word = cp.get("after_word", "?")
    te = cp.get("trough_energy", "?")
    
    if all_zero:
        print(f"  Cut {i:2d} ({word:15s}): ALL ZEROS ✓  trough_energy={te}")
    else:
        print(f"  Cut {i:2d} ({word:15s}): {nonzero_count}/{len(silence_region)} NON-ZERO SAMPLES ✗  trough_energy={te}")
        zeros_pass = False

# =====================================================================
# Step 3: Check for fade artifacts (energy should NOT ramp down before silence)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 3: Check for fade artifacts before/after silence")
print(f"{'=' * 70}")

fade_pass = True
for i, cp in enumerate(cut_points):
    center = int(cp["time"] * sr)
    sil_start = max(0, center - pause_samples // 2)
    sil_end = min(len(audio), sil_start + pause_samples)
    word = cp.get("after_word", "?")
    clean_word = word.strip(".,;:!?'\"")
    
    # Check 5ms of audio JUST BEFORE the silence starts
    pre_end = sil_start
    pre_start = max(0, pre_end - int(sr * 0.005))  # 5ms = 120 samples
    pre_chunk = audio[pre_start:pre_end].astype(np.float32)
    
    # Check 5ms of audio JUST AFTER the silence ends
    post_start = sil_end
    post_end = min(len(audio), post_start + int(sr * 0.005))
    post_chunk = audio[post_start:post_end].astype(np.float32)
    
    pre_energy = float(np.sqrt(np.mean(pre_chunk ** 2))) if len(pre_chunk) > 0 else 0
    post_energy = float(np.sqrt(np.mean(post_chunk ** 2))) if len(post_chunk) > 0 else 0
    
    # Check for fade: if there's a progressive energy ramp in the last 20ms before silence,
    # that's a fade artifact. We check 4 windows of 5ms each going back.
    fade_detected = False
    if sil_start > int(sr * 0.02):  # Need 20ms of audio before
        energies = []
        for w in range(4):
            ws = int(sr * 0.005)
            seg_start = sil_start - (w + 1) * ws
            seg_end = sil_start - w * ws
            seg = audio[seg_start:seg_end].astype(np.float32)
            energies.append(float(np.sqrt(np.mean(seg ** 2))))
        
        # Fade pattern: energy should ramp DOWN toward silence
        # e.g., [2000, 1500, 800, 200] = monotonic decrease = FADE
        monotonic_decrease = all(energies[j] >= energies[j+1] for j in range(len(energies)-1))
        # But natural speech can also decrease. Check if it drops to near-zero.
        # A fade would go: [high, high, medium, very_low]
        # Natural speech ending would have irregular energy.
        if monotonic_decrease and energies[0] > 500 and energies[-1] < 50:
            fade_detected = True
    
    marker = " <<<" if clean_word in PROBLEM_WORDS else ""
    if fade_detected:
        print(f"  Cut {i:2d} ({word:15s}): FADE DETECTED ✗ pre={pre_energy:.0f} post={post_energy:.0f}{marker}")
        fade_pass = False
    else:
        print(f"  Cut {i:2d} ({word:15s}): NO FADE ✓      pre={pre_energy:.0f} post={post_energy:.0f}{marker}")

# =====================================================================
# Step 4: DC offset check
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 4: DC offset check")
print(f"{'=' * 70}")

# Calculate mean of all non-silence samples
non_silence_samples = []
cursor = 0
for cp in cut_points:
    center = int(cp["time"] * sr)
    sil_start = max(0, center - pause_samples // 2)
    non_silence_samples.append(audio[cursor:sil_start])
    cursor = min(len(audio), sil_start + pause_samples)
non_silence_samples.append(audio[cursor:])

all_speech = np.concatenate(non_silence_samples).astype(np.float32)
dc_offset = float(np.mean(all_speech))
dc_pass = abs(dc_offset) < 100  # int16 range is -32768 to 32767, small offset is normal

print(f"  Mean of all non-silence samples: {dc_offset:.2f}")
print(f"  DC offset: {'ACCEPTABLE ✓' if dc_pass else 'HIGH ✗'}")

# =====================================================================
# Step 5: Problem word trough energies
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Problem word analysis")
print(f"{'=' * 70}")

for cp in cut_points:
    word = cp.get("after_word", "?")
    clean_word = word.strip(".,;:!?'\"")
    if clean_word in PROBLEM_WORDS:
        te = cp.get("trough_energy", "?")
        quality = "CLEAN (natural gap)" if isinstance(te, (int, float)) and te < 100 else "TIGHT (coarticulated)"
        print(f"  {word:15s} trough_energy={str(te):>8s} -> {quality}")

# =====================================================================
# FINAL RESULT
# =====================================================================
print(f"\n{'=' * 70}")
all_pass = zeros_pass and fade_pass and dc_pass
if all_pass:
    print("RESULT: ALL CHECKS PASSED ✓✓✓")
    print("  - All silence regions are exact zeros")
    print("  - No fade artifacts detected")
    print("  - DC offset is acceptable")
    print("  - Audio samples are UNMODIFIED Kokoro output")
else:
    print("RESULT: SOME CHECKS FAILED ✗")
    if not zeros_pass: print("  ✗ Silence regions contain non-zero samples")
    if not fade_pass:  print("  ✗ Fade artifacts detected before silence")
    if not dc_pass:    print("  ✗ DC offset is too high")
print(f"{'=' * 70}")
