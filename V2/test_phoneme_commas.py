"""Test phoneme-level comma injection and compare energy at 5 problem words.

Calls the API with the same text/pause_after, saves audio, and generates
a comparison report at the 5 words Mariam identified:
  conference, apps, information, dramatic, itself
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64, wave, numpy as np, struct

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

# Same test text from all previous tests
TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things.
Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

# Same pause_after indices
PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107,
               115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]

# The 5 problem words Mariam identified
PROBLEM_WORDS = ["conference", "apps", "information", "dramatic", "itself"]

SR = 24000


def save_wav(path, audio_i16, sr=SR):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def rms_energy(samples, window_ms=5, sr=SR):
    """Return RMS energy in sliding windows."""
    win = int(sr * window_ms / 1000)
    if len(samples) < win:
        return [float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))]
    energies = []
    for i in range(0, len(samples) - win + 1, win):
        chunk = samples[i:i + win].astype(np.float64)
        energies.append(float(np.sqrt(np.mean(chunk ** 2))))
    return energies


def analyze_cut_point(audio_i16, cut_time, word, sr=SR):
    """Analyze energy profile around a cut point."""
    cut_sample = int(cut_time * sr)
    analysis = {"word": word, "cut_time": cut_time}

    # Energy in 5ms windows: -50ms to +50ms around cut point
    profile = {}
    for offset_ms in [-50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20]:
        offset_samples = int(sr * offset_ms / 1000)
        start = max(0, cut_sample + offset_samples)
        end = min(len(audio_i16), start + int(sr * 0.005))
        if start < end:
            chunk = audio_i16[start:end].astype(np.float64)
            profile[offset_ms] = int(np.sqrt(np.mean(chunk ** 2)))
        else:
            profile[offset_ms] = 0
    analysis["energy_profile"] = profile

    # Pre-silence energy (5ms window right before cut)
    pre_start = max(0, cut_sample - int(sr * 0.005))
    pre = audio_i16[pre_start:cut_sample].astype(np.float64)
    analysis["pre_energy"] = int(np.sqrt(np.mean(pre ** 2))) if len(pre) > 0 else 0

    # Post-silence energy (5ms window right after cut)
    post_end = min(len(audio_i16), cut_sample + int(sr * 0.005))
    post = audio_i16[cut_sample:post_end].astype(np.float64)
    analysis["post_energy"] = int(np.sqrt(np.mean(post ** 2))) if len(post) > 0 else 0

    # Check if silence region is clean (50ms after cut)
    sil_end = min(len(audio_i16), cut_sample + int(sr * 0.050))
    sil = audio_i16[cut_sample:sil_end]
    non_zero = int(np.count_nonzero(sil))
    analysis["silence_non_zero"] = non_zero
    analysis["silence_clean"] = non_zero == 0

    return analysis


# ======================================================================
print("=" * 70)
print("STEP 1: Generate audio with phoneme-level commas (v21)")
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
    print(json.dumps(resp, indent=2)[:1000])
    sys.exit(1)

out = resp["output"]
audio_bytes = base64.b64decode(out["audio_base64"])
audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
word_ts = out.get("word_timestamps", [])
cut_points = out.get("phrase_cut_points", [])

print("  Duration: {:.2f}s | Words: {} | Cut points: {}".format(
    out.get("duration_seconds", 0), len(word_ts), len(cut_points)))

# Save
os.makedirs("output", exist_ok=True)
save_wav("output/v21_phoneme_commas.wav", audio_i16)
print("  Saved: output/v21_phoneme_commas.wav")

# Build phrase audio with 1s silence between phrases
print()
print("=" * 70)
print("STEP 2: Build phrase audio with 1s silence at cut points")
print("=" * 70)

silence_1s = np.zeros(SR, dtype=np.int16)
segments = []
prev_end = 0
for i, cp in enumerate(cut_points):
    cut_sample = int(cp["time"] * SR)
    segments.append(audio_i16[prev_end:cut_sample])
    segments.append(silence_1s)
    prev_end = cut_sample
segments.append(audio_i16[prev_end:])
phrase_audio = np.concatenate(segments)
save_wav("output/v21_phrases.wav", phrase_audio)
print("  Saved: output/v21_phrases.wav ({:.1f}s)".format(len(phrase_audio) / SR))

# ======================================================================
print()
print("=" * 70)
print("STEP 3: Analyze the 5 problem words")
print("=" * 70)

# Find cut points for the 5 problem words
problem_cuts = []
for pw in PROBLEM_WORDS:
    found = False
    for cp in cut_points:
        if cp.get("after_word", "").lower().rstrip(".,;:!?") == pw.lower():
            problem_cuts.append(cp)
            found = True
            break
    if not found:
        # Try partial match
        for cp in cut_points:
            if pw.lower() in cp.get("after_word", "").lower():
                problem_cuts.append(cp)
                found = True
                break
    if not found:
        print("  WARNING: Could not find cut point for '{}'".format(pw))
        problem_cuts.append(None)

print()
print("  {:>15s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>7s}".format(
    "WORD", "CUT(s)", "PRE_RMS", "POST_RMS", "SIL_CLEAN", "VERDICT"))
print("  " + "-" * 65)

for i, pw in enumerate(PROBLEM_WORDS):
    cp = problem_cuts[i]
    if cp is None:
        print("  {:>15s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>7s}".format(
            pw, "N/A", "N/A", "N/A", "N/A", "MISS"))
        continue

    a = analyze_cut_point(audio_i16, cp["time"], pw)
    clean = "YES" if a["silence_clean"] else "NO"
    verdict = "OK" if a["pre_energy"] < 50 else "CHECK"
    print("  {:>15s}  {:>8.3f}  {:>8d}  {:>8d}  {:>10s}  {:>7s}".format(
        pw, cp["time"], a["pre_energy"], a["post_energy"], clean, verdict))

    # Detailed energy profile
    print("    Energy profile (5ms windows around cut):")
    for ms, rms in sorted(a["energy_profile"].items()):
        bar = "#" * min(50, rms // 50)
        label = "<-- CUT" if ms == 0 else ""
        print("      {:>+4d}ms: rms={:>5d}  {}  {}".format(ms, rms, bar, label))
    print()

# ======================================================================
print("=" * 70)
print("STEP 4: Compare against previous version (if exists)")
print("=" * 70)

prev_path = "output/test_step1_paused.wav"
if os.path.exists(prev_path):
    with wave.open(prev_path, "rb") as wf:
        prev_audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    print("  Previous audio: {} ({:.1f}s)".format(prev_path, len(prev_audio) / SR))
    print("  New audio:      output/v21_phoneme_commas.wav ({:.1f}s)".format(len(audio_i16) / SR))
    print("  Duration diff:  {:.3f}s".format((len(audio_i16) - len(prev_audio)) / SR))
else:
    print("  No previous audio found at {}".format(prev_path))
    print("  Run test_phrase_workflow.py first to generate baseline, then compare.")

print()
print("=" * 70)
print("DONE - Listen to output/v21_phoneme_commas.wav and output/v21_phrases.wav")
print("Compare against previous test_step1_paused.wav / test_step3_phrases.wav")
print("=" * 70)
