"""
Root Cause Analysis: Why do some words sound abrupt at phrase boundaries?
Tests 4 possible causes:
  1. Kokoro model (word already sounds cut in raw audio)
  2. Timestamp accuracy (alignment cuts word too early)
  3. Post-processing (crossfade/tail introduces artifacts)
  4. Phrase cutting (cutting logic causes issues)
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

# Problem words reported by Mariam
PROBLEM_WORDS = [13, 153]  # "Conference", "itself"
# Also check some words she said sound fine for comparison
GOOD_WORDS = [2, 87, 131]  # "overhaul", "bills", "behalf."

ALL_CHECK = PROBLEM_WORDS + GOOD_WORDS

os.makedirs("output/diagnosis", exist_ok=True)

def save_wav(path, audio, sr=24000):
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

# ============================================================
# TEST 1: Raw audio (no pauses) - is the model the problem?
# ============================================================
print("=" * 70)
print("TEST 1: Generate RAW audio (no pauses) - check model output")
print("=" * 70)

r1 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out1 = r1.json().get("output", {})
ts_raw = out1.get("word_timestamps", [])
audio_raw = base64.b64decode(out1["audio_base64"])
with open("output/diagnosis/raw_audio.wav", "wb") as f:
    f.write(audio_raw)
sr, raw = read_wav("output/diagnosis/raw_audio.wav")
print("Raw audio: {:.2f}s, {} words".format(len(raw)/sr, len(ts_raw)))

# Extract each problem word + 200ms before and after from RAW audio
for idx in ALL_CHECK:
    if idx >= len(ts_raw):
        continue
    w = ts_raw[idx]
    label = "PROBLEM" if idx in PROBLEM_WORDS else "GOOD"
    start = max(0, int((w["start"] - 0.2) * sr))
    end = min(len(raw), int((w["end"] + 0.2) * sr))
    clip = raw[start:end]
    save_wav("output/diagnosis/raw_word_{:03d}_{}.wav".format(idx, w["word"].replace(".", "")), clip, sr)
    
    # Check energy at word end (does the word fade naturally?)
    word_end = int(w["end"] * sr)
    # Energy in 10ms windows around word end
    print("\n  [{}] Word {}: '{}' ({:.3f}s - {:.3f}s)".format(label, idx, w["word"], w["start"], w["end"]))
    for offset_ms in [-50, -30, -20, -10, 0, 10, 20, 30, 50]:
        pos = word_end + int(sr * offset_ms / 1000)
        win_start = max(0, pos - int(sr * 0.005))
        win_end = min(len(raw), pos + int(sr * 0.005))
        if win_end > win_start:
            e = np.sqrt(np.mean(raw[win_start:win_end] ** 2))
        else:
            e = 0
        marker = " <-- word end" if offset_ms == 0 else ""
        print("    end{:+d}ms: energy={:.0f}{}".format(offset_ms, e, marker))

# ============================================================
# TEST 2: Audio WITH pauses - check what crossfade does
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Generate audio WITH pauses (50ms) - check crossfade effect")
print("=" * 70)

r2 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "word_boundaries": True,
    "pause_after": PAUSE_AFTER, "micro_pause_ms": 50, "crossfade_ms": 20.0
}}, timeout=300)
out2 = r2.json().get("output", {})
ts_paused = out2.get("word_timestamps", [])
cut_points = out2.get("phrase_cut_points", [])
audio_paused = base64.b64decode(out2["audio_base64"])
with open("output/diagnosis/paused_audio.wav", "wb") as f:
    f.write(audio_paused)
sr2, paused = read_wav("output/diagnosis/paused_audio.wav")
print("Paused audio: {:.2f}s, {} words, {} cut points".format(len(paused)/sr2, len(ts_paused), len(cut_points)))

# Extract each problem word from PAUSED audio
for idx in ALL_CHECK:
    if idx >= len(ts_paused):
        continue
    w = ts_paused[idx]
    label = "PROBLEM" if idx in PROBLEM_WORDS else "GOOD"
    start = max(0, int((w["start"] - 0.2) * sr2))
    end = min(len(paused), int((w["end"] + 0.3) * sr2))
    clip = paused[start:end]
    save_wav("output/diagnosis/paused_word_{:03d}_{}.wav".format(idx, w["word"].replace(".", "")), clip, sr2)
    
    # Find the cut point for this word
    cp_time = None
    for cp in cut_points:
        if cp.get("after_word_idx") == idx:
            cp_time = cp["time"]
            break
    
    print("\n  [{}] Word {}: '{}' ({:.3f}s - {:.3f}s) | cut_point: {}".format(
        label, idx, w["word"], w["start"], w["end"],
        "{:.4f}s".format(cp_time) if cp_time else "N/A"))
    
    # Energy around word end in paused audio
    word_end = int(w["end"] * sr2)
    for offset_ms in [-50, -30, -20, -10, 0, 10, 20, 30, 50, 70, 100]:
        pos = word_end + int(sr2 * offset_ms / 1000)
        win_start = max(0, pos - int(sr2 * 0.005))
        win_end = min(len(paused), pos + int(sr2 * 0.005))
        if win_end > win_start:
            e = np.sqrt(np.mean(paused[win_start:win_end] ** 2))
        else:
            e = 0
        marker = " <-- word end" if offset_ms == 0 else (" <-- tail zone" if offset_ms in [30, 50] else "")
        print("    end{:+d}ms: energy={:.0f}{}".format(offset_ms, e, marker))

# ============================================================
# TEST 3: Compare raw vs paused word clips
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: ROOT CAUSE SUMMARY")
print("=" * 70)

for idx in ALL_CHECK:
    if idx >= len(ts_raw) or idx >= len(ts_paused):
        continue
    raw_w = ts_raw[idx]
    paused_w = ts_paused[idx]
    label = "PROBLEM" if idx in PROBLEM_WORDS else "GOOD"
    
    # Check if raw word already sounds cut (high energy at boundary)
    raw_end = int(raw_w["end"] * sr)
    raw_e_at_end = np.sqrt(np.mean(raw[max(0,raw_end-120):raw_end+120] ** 2))
    
    # Check energy 50ms past raw word end (is there still speech?)
    raw_e_past = np.sqrt(np.mean(raw[raw_end:min(len(raw), raw_end+int(sr*0.05))] ** 2))
    
    # Check if next word starts immediately (coarticulation)
    if idx + 1 < len(ts_raw):
        gap_ms = (ts_raw[idx+1]["start"] - raw_w["end"]) * 1000
        next_word = ts_raw[idx+1]["word"]
    else:
        gap_ms = 999
        next_word = "N/A"
    
    cause = ""
    if raw_e_past > 500:
        cause = "TIMESTAMP: alignment cuts too early, speech continues past word end"
    elif raw_e_at_end > 2000 and gap_ms < 30:
        cause = "COARTICULATION: word blends into next word, no natural boundary"
    elif raw_e_at_end > 2000:
        cause = "MODEL: word ends with high energy (plosive/fricative ending)"
    else:
        cause = "CLEAN: word ends naturally"
    
    print("\n  [{}] '{}' (idx {}):".format(label, raw_w["word"], idx))
    print("    Energy at word end: {:.0f}".format(raw_e_at_end))
    print("    Energy 50ms past end: {:.0f}".format(raw_e_past))
    print("    Gap to next word: {:.0f}ms ('{}')".format(gap_ms, next_word))
    print("    >> ROOT CAUSE: {}".format(cause))

print("\n" + "=" * 70)
print("AUDIO FILES SAVED in output/diagnosis/")
print("  raw_word_XXX_*.wav = word clip from raw (no pause) audio")
print("  paused_word_XXX_*.wav = word clip from paused audio")
print("  Listen and compare to identify the exact issue")
print("=" * 70)
