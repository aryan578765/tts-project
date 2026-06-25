"""
DEEP ANALYSIS: Analyze ALL 27 phrase boundaries for quality.
For each boundary:
  1. Extract word clip from raw audio (baseline)
  2. Extract word clip from paused audio (with crossfade)
  3. Measure energy profile, fade quality, coarticulation
  4. Rate each boundary: CLEAN, ACCEPTABLE, ABRUPT
  5. Save individual clips for listening
"""
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

os.makedirs("output/deep_analysis", exist_ok=True)

def save_wav(path, audio, sr=24000):
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

def energy_at(audio, sr, time_s, window_ms=10):
    center = int(time_s * sr)
    half = int(sr * window_ms / 2000)
    s = max(0, center - half)
    e = min(len(audio), center + half)
    if e <= s: return 0
    return np.sqrt(np.mean(audio[s:e] ** 2))

def pitch_estimate(audio, sr, time_s, window_ms=30):
    """Simple zero-crossing rate as proxy for pitch activity."""
    center = int(time_s * sr)
    half = int(sr * window_ms / 2000)
    s = max(0, center - half)
    e = min(len(audio), center + half)
    if e - s < 10: return 0
    chunk = audio[s:e]
    zc = np.sum(np.abs(np.diff(np.sign(chunk))) > 0)
    return zc / (len(chunk) / sr)  # zero-crossings per second

# ============================================================
# Step 1: Generate RAW audio (no pauses)
# ============================================================
print("STEP 1: Generating RAW audio (no pauses)...")
r1 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out1 = r1.json().get("output", {})
ts_raw = out1.get("word_timestamps", [])
audio_raw_bytes = base64.b64decode(out1["audio_base64"])
with open("output/deep_analysis/raw_audio.wav", "wb") as f:
    f.write(audio_raw_bytes)
sr, raw = read_wav("output/deep_analysis/raw_audio.wav")
print("  Raw: {:.2f}s, {} words".format(len(raw)/sr, len(ts_raw)))

# ============================================================
# Step 2: Generate PAUSED audio (50ms pauses)
# ============================================================
print("\nSTEP 2: Generating PAUSED audio (50ms pauses)...")
r2 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "pause_after": PAUSE_AFTER, "micro_pause_ms": 50
}}, timeout=300)
out2 = r2.json().get("output", {})
ts_paused = out2.get("word_timestamps", [])
cut_points = out2.get("phrase_cut_points", [])
audio_paused_bytes = base64.b64decode(out2["audio_base64"])
with open("output/deep_analysis/paused_audio.wav", "wb") as f:
    f.write(audio_paused_bytes)
sr2, paused = read_wav("output/deep_analysis/paused_audio.wav")
print("  Paused: {:.2f}s, {} words, {} cut points".format(len(paused)/sr2, len(ts_paused), len(cut_points)))

# ============================================================
# Step 3: Deep analysis of ALL phrase boundaries
# ============================================================
print("\n" + "=" * 90)
print("DEEP ANALYSIS: ALL {} PHRASE BOUNDARIES".format(len(PAUSE_AFTER)))
print("=" * 90)

results = []

for idx in PAUSE_AFTER:
    if idx >= len(ts_raw) or idx >= len(ts_paused):
        continue
    
    raw_w = ts_raw[idx]
    paused_w = ts_paused[idx]
    word = raw_w["word"]
    
    # Next word info
    next_word = ts_raw[idx+1]["word"] if idx+1 < len(ts_raw) else "END"
    gap_ms = (ts_raw[idx+1]["start"] - raw_w["end"]) * 1000 if idx+1 < len(ts_raw) else 999
    
    # --- RAW audio analysis ---
    raw_end = raw_w["end"]
    e_raw_at_end = energy_at(raw, sr, raw_end)
    e_raw_m20 = energy_at(raw, sr, raw_end - 0.02)
    e_raw_p20 = energy_at(raw, sr, raw_end + 0.02)
    e_raw_p50 = energy_at(raw, sr, raw_end + 0.05)
    e_raw_p80 = energy_at(raw, sr, raw_end + 0.08)
    zcr_at_end = pitch_estimate(raw, sr, raw_end)
    zcr_past = pitch_estimate(raw, sr, raw_end + 0.05)
    
    # Energy decay rate: how fast does energy drop after word end?
    decay_50ms = (e_raw_at_end - e_raw_p50) / max(e_raw_at_end, 1) * 100
    
    # Coarticulation: is there still speech past word end?
    coarticulated = e_raw_p50 > e_raw_at_end * 0.5
    
    # Natural boundary: does the word end with low energy?
    natural_end = e_raw_at_end < 300
    
    # Pitch continuity: high ZCR past word end = speech continues
    pitch_continues = zcr_past > 2000
    
    # --- Classification ---
    issues = []
    if coarticulated:
        issues.append("COARTICULATED")
    if not natural_end and gap_ms < 100:
        issues.append("HIGH_ENERGY_TIGHT_GAP")
    if pitch_continues:
        issues.append("PITCH_CONTINUES")
    if e_raw_p80 > 500 and gap_ms < 100:
        issues.append("NEXT_WORD_BLEED")
    
    if len(issues) == 0:
        rating = "CLEAN"
    elif len(issues) == 1 and "HIGH_ENERGY_TIGHT_GAP" in issues:
        rating = "ACCEPTABLE"
    else:
        rating = "ABRUPT"
    
    # Find cut point
    cp = None
    for c in cut_points:
        if c.get("after_word_idx") == idx:
            cp = c
            break
    
    result = {
        "idx": idx, "word": word, "next_word": next_word,
        "gap_ms": gap_ms, "rating": rating, "issues": issues,
        "e_at_end": e_raw_at_end, "e_m20": e_raw_m20,
        "e_p20": e_raw_p20, "e_p50": e_raw_p50, "e_p80": e_raw_p80,
        "decay_50ms": decay_50ms, "zcr_end": zcr_at_end, "zcr_past": zcr_past,
        "coarticulated": coarticulated, "natural_end": natural_end,
        "cp_time": cp["time"] if cp else None,
    }
    results.append(result)
    
    # Save clips for listening
    # Raw: word + 200ms context
    rs = max(0, int((raw_w["start"] - 0.1) * sr))
    re = min(len(raw), int((raw_w["end"] + 0.15) * sr))
    save_wav("output/deep_analysis/raw_{:03d}_{}.wav".format(idx, word.replace(".", "").replace(",", "")), raw[rs:re], sr)
    
    # Paused: word + transition
    ps = max(0, int((paused_w["start"] - 0.1) * sr2))
    pe = min(len(paused), int((paused_w["end"] + 0.2) * sr2))
    save_wav("output/deep_analysis/paused_{:03d}_{}.wav".format(idx, word.replace(".", "").replace(",", "")), paused[ps:pe], sr2)

# ============================================================
# Print results
# ============================================================
print("\n{:>5} {:>15} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {}".format(
    "Idx", "Word", "Next", "Gap(ms)", "E@end", "E+20ms", "E+50ms", "E+80ms", "Rating", "Issues"))
print("-" * 120)

clean_count = 0
acceptable_count = 0
abrupt_count = 0

for r in results:
    marker = ""
    if r["rating"] == "ABRUPT":
        marker = " <<<< PROBLEM"
        abrupt_count += 1
    elif r["rating"] == "ACCEPTABLE":
        acceptable_count += 1
    else:
        clean_count += 1
    
    print("{:>5} {:>15} {:>10} {:>7.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>10} {}{}".format(
        r["idx"], r["word"], r["next_word"], r["gap_ms"],
        r["e_at_end"], r["e_p20"], r["e_p50"], r["e_p80"],
        r["rating"], ", ".join(r["issues"]), marker))

print("\n" + "=" * 90)
print("SUMMARY: {} CLEAN | {} ACCEPTABLE | {} ABRUPT".format(clean_count, acceptable_count, abrupt_count))
print("=" * 90)

# Categorize abrupt words by issue type
print("\n--- ABRUPT WORDS BY ISSUE TYPE ---")
issue_groups = {}
for r in results:
    if r["rating"] == "ABRUPT":
        for issue in r["issues"]:
            if issue not in issue_groups:
                issue_groups[issue] = []
            issue_groups[issue].append("'{}' (idx {})".format(r["word"], r["idx"]))

for issue, words in issue_groups.items():
    print("\n  {}: {}".format(issue, ", ".join(words)))

# Final pitch analysis for problem words
print("\n--- PITCH (ZCR) ANALYSIS FOR ABRUPT WORDS ---")
for r in results:
    if r["rating"] == "ABRUPT":
        print("  '{}' (idx {}): ZCR at end={:.0f}, ZCR past end={:.0f} | {}".format(
            r["word"], r["idx"], r["zcr_end"], r["zcr_past"],
            "pitch continues" if r["zcr_past"] > 2000 else "pitch drops"))
