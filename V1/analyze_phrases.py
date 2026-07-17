"""
DEEP AUDIO ANALYSIS of test_step3_phrases.wav
Examines the actual waveform around every problem word to assess quality.
"""
import wave, numpy as np

PROBLEM_WORDS = {"Conference", "apps", "bills", "information", "dramatic", "itself", "features"}

# Load the phrase audio (with 1s silences)
with wave.open("output/test_step3_phrases.wav", "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

print(f"Audio: {len(audio)/sr:.2f}s, {sr}Hz\n")

# Also load the paused audio (with 50ms silences) for comparison
with wave.open("output/test_step1_paused.wav", "rb") as wf:
    raw2 = wf.readframes(wf.getnframes())
    paused_audio = np.frombuffer(raw2, dtype=np.int16).astype(np.float32)

# The cut data from the test run
cuts = [
    {"word": "overhaul", "cut": 1.6928, "word_end": 1.480},
    {"word": "announcement", "cut": 3.2658, "word_end": 3.131},
    {"word": "Conference", "cut": 5.4665, "word_end": 5.381},
    {"word": "month,", "cut": 6.7390, "word_end": 6.292},
    {"word": "strategy", "cut": 9.0427, "word_end": 8.622},
    {"word": "features", "cut": 12.2157, "word_end": 11.873},
    {"word": "software.", "cut": 14.3863, "word_end": 13.724},
    {"word": "Siri", "cut": 19.0452, "word_end": 18.575},
    {"word": "brings,", "cut": 21.7681, "word_end": 21.266},
    {"word": "apps", "cut": 24.1597, "word_end": 24.077},
    {"word": "use,", "cut": 26.4469, "word_end": 26.007},
    {"word": "problems.", "cut": 29.2830, "word_end": 29.158},
    {"word": "bills", "cut": 32.7389, "word_end": 32.689},
    {"word": "breaches,", "cut": 36.5899, "word_end": 36.200},
    {"word": "information", "cut": 39.4760, "word_end": 39.351},
    {"word": "things.", "cut": 42.7464, "word_end": 42.062},
    {"word": "dramatic", "cut": 45.7632, "word_end": 45.733},
    {"word": "context", "cut": 49.3857, "word_end": 49.023},
    {"word": "behalf.", "cut": 51.5564, "word_end": 50.954},
    {"word": "AI", "cut": 54.2420, "word_end": 54.105},
    {"word": "bot", "cut": 55.8726, "word_end": 55.795},
    {"word": "itself", "cut": 58.7760, "word_end": 58.686},
    {"word": "capable.", "cut": 60.9515, "word_end": 60.356},
    {"word": "features", "cut": 64.7802, "word_end": 64.748},
    {"word": "using.", "cut": 68.9836, "word_end": 68.359},
    {"word": "beta", "cut": 71.8344, "word_end": 71.350},
    {"word": "beta,", "cut": 74.2202, "word_end": 73.780},
]

# In test_step3_phrases.wav, each cut point has 1s silence inserted.
# So the actual position in the phrases audio shifts by +1s for each cut.
# Let's compute the actual positions in the phrases audio.

print("=" * 80)
print("ANALYSIS OF EVERY CUT BOUNDARY IN test_step3_phrases.wav")
print("=" * 80)

# First, compute where each cut lands in the phrases audio
phrase_cuts = []
cumulative_silence = 0.0
for c in cuts:
    actual_time = c["cut"] + cumulative_silence
    phrase_cuts.append({
        "word": c["word"],
        "cut_in_paused": c["cut"],
        "cut_in_phrases": actual_time,
        "word_end": c["word_end"],
        "gap_ms": (c["cut"] - c["word_end"]) * 1000,
    })
    cumulative_silence += 1.0  # 1s silence added at each cut

print(f"\n{'Word':>15s}  {'Gap':>7s}  {'Last 50ms energy':>18s}  {'Severity':>15s}")
print("-" * 70)

for pc in phrase_cuts:
    word = pc["word"]
    cut_sample = int(pc["cut_in_phrases"] * sr)
    gap_ms = pc["gap_ms"]
    
    # Energy in the last 50ms of audio before the 1s silence
    pre_start = max(0, cut_sample - int(sr * 0.05))  # 50ms before
    pre_end = cut_sample
    pre_chunk = audio[pre_start:pre_end]
    
    if len(pre_chunk) > 0:
        pre_energy = float(np.sqrt(np.mean(pre_chunk ** 2)))
        peak_amp = float(np.max(np.abs(pre_chunk)))
    else:
        pre_energy = 0
        peak_amp = 0
    
    # Classify severity
    clean_word = word.strip(".,;:!?'\"")
    is_problem = clean_word in PROBLEM_WORDS
    
    if pre_energy < 50:
        severity = "PERFECT"
    elif pre_energy < 200:
        severity = "GOOD"
    elif pre_energy < 500:
        severity = "AUDIBLE CUT"
    elif pre_energy < 1000:
        severity = "BAD CUT"
    else:
        severity = "VERY BAD CUT"
    
    marker = " <<<" if is_problem else ""
    print(f"  {word:>15s}  {gap_ms:6.1f}ms  rms={pre_energy:7.0f} peak={peak_amp:6.0f}  {severity}{marker}")

# Detailed analysis of just the problem words
print(f"\n{'=' * 80}")
print("DETAILED WAVEFORM ANALYSIS OF PROBLEM WORDS")
print(f"{'=' * 80}")

for pc in phrase_cuts:
    clean_word = pc["word"].strip(".,;:!?'\"")
    if clean_word not in PROBLEM_WORDS:
        continue
    
    cut_sample = int(pc["cut_in_phrases"] * sr)
    
    print(f"\n--- {pc['word']} (gap={pc['gap_ms']:.1f}ms) ---")
    print(f"  Energy profile going into silence (5ms windows, last 100ms):")
    
    for offset_ms in range(-100, 20, 5):
        pos = cut_sample + int(sr * offset_ms / 1000)
        ws = int(sr * 0.005)
        if pos < 0 or pos + ws > len(audio):
            continue
        seg = audio[pos:pos + ws]
        e = float(np.sqrt(np.mean(seg ** 2)))
        bar = "#" * min(50, int(e / 100))
        marker = ""
        if offset_ms == 0:
            marker = " <-- CUT (silence starts here)"
        elif offset_ms == 5:
            marker = " <-- (should be silence)"
        print(f"    {offset_ms:+4d}ms: rms={e:7.0f} {bar}{marker}")
    
    # What does the listener actually hear?
    # The last non-zero sample before silence
    search_back = min(cut_sample, int(sr * 0.2))  # 200ms
    last_speech = cut_sample
    for s in range(cut_sample - 1, cut_sample - search_back, -1):
        if s >= 0 and abs(audio[s]) > 50:
            last_speech = s
            break
    
    speech_to_silence_ms = (cut_sample - last_speech) / sr * 1000
    print(f"  Last audible sample: {speech_to_silence_ms:.1f}ms before silence")
    
    if speech_to_silence_ms < 5:
        print(f"  >> VERDICT: Speech ends ABRUPTLY at cut point — listener hears word chopped off")
    elif speech_to_silence_ms < 20:
        print(f"  >> VERDICT: Very short natural decay — might sound slightly abrupt")
    else:
        print(f"  >> VERDICT: Natural decay into silence — should sound clean")

# Summary
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

perfect = 0
good = 0
audible = 0
bad = 0
for pc in phrase_cuts:
    cut_sample = int(pc["cut_in_phrases"] * sr)
    pre_start = max(0, cut_sample - int(sr * 0.05))
    pre_chunk = audio[pre_start:cut_sample]
    if len(pre_chunk) > 0:
        e = float(np.sqrt(np.mean(pre_chunk ** 2)))
    else:
        e = 0
    if e < 50: perfect += 1
    elif e < 200: good += 1
    elif e < 500: audible += 1
    else: bad += 1

total = len(phrase_cuts)
print(f"  PERFECT  (rms<50):   {perfect}/{total} — natural silence before cut")
print(f"  GOOD     (rms<200):  {good}/{total} — very low energy, barely audible")
print(f"  AUDIBLE  (rms<500):  {audible}/{total} — listener may hear a slight cut")
print(f"  BAD      (rms>=500): {bad}/{total} — listener will hear word chopped off")
print(f"\nProblem words that are BAD/AUDIBLE cuts cannot be fixed without")
print(f"modifying the audio (fades) or changing the phrase boundaries.")
print(f"This is a physics limitation — the speaker blends these words together.")
