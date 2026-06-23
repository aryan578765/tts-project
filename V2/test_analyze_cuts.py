"""Analyze both phrase-cut WAV files: compare cut quality and detect clipping."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import wave
import numpy as np

def analyze_wav(path):
    """Analyze a WAV file for phrase cuts and silence gaps."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        if wf.getsampwidth() == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        else:
            audio = np.frombuffer(raw, dtype=np.float32)

    duration = len(audio) / sr
    print("File: {}".format(os.path.basename(path)))
    print("Duration: {:.2f}s | Sample rate: {} | Samples: {}".format(duration, sr, len(audio)))

    # Find silence regions (1s gaps = phrase boundaries)
    # A silence region is where abs(amplitude) < threshold for >= 0.5s
    threshold = 100  # int16 scale
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)

    # Calculate RMS energy in windows
    energy = []
    for i in range(0, len(audio) - window_samples, window_samples):
        chunk = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        energy.append((i / sr, rms))

    # Find silence regions (RMS < threshold for consecutive windows)
    silence_regions = []
    in_silence = False
    silence_start = 0
    for t, rms in energy:
        if rms < threshold:
            if not in_silence:
                silence_start = t
                in_silence = True
        else:
            if in_silence:
                silence_end = t
                duration_s = silence_end - silence_start
                if duration_s >= 0.5:  # only count pauses >= 500ms
                    silence_regions.append((silence_start, silence_end, duration_s))
                in_silence = False

    print("Found {} phrase breaks (silence >= 500ms):".format(len(silence_regions)))
    phrases = []
    prev_end = 0
    for i, (s_start, s_end, s_dur) in enumerate(silence_regions):
        phrase_dur = s_start - prev_end
        phrases.append((prev_end, s_start, phrase_dur))
        print("  Phrase {:>2}: {:.2f}s - {:.2f}s ({:.2f}s) | Silence: {:.2f}s - {:.2f}s ({:.3f}s)".format(
            i + 1, prev_end, s_start, phrase_dur, s_start, s_end, s_dur))
        prev_end = s_end

    # Last phrase after final silence
    if prev_end < len(audio) / sr:
        final_dur = (len(audio) / sr) - prev_end
        phrases.append((prev_end, len(audio) / sr, final_dur))
        print("  Phrase {:>2}: {:.2f}s - {:.2f}s ({:.2f}s)".format(
            len(phrases), prev_end, len(audio) / sr, final_dur))

    # Check for abrupt cuts (high energy right before silence)
    print("\nCut quality analysis (energy at phrase boundaries):")
    abrupt_cuts = 0
    for i, (s_start, s_end, s_dur) in enumerate(silence_regions):
        # Check energy in 20ms before silence start
        pre_start = max(0, int((s_start - 0.02) * sr))
        pre_end = int(s_start * sr)
        if pre_end > pre_start:
            pre_energy = np.sqrt(np.mean(audio[pre_start:pre_end] ** 2))
        else:
            pre_energy = 0

        # Check energy in 20ms after silence end
        post_start = int(s_end * sr)
        post_end = min(len(audio), int((s_end + 0.02) * sr))
        if post_end > post_start:
            post_energy = np.sqrt(np.mean(audio[post_start:post_end] ** 2))
        else:
            post_energy = 0

        # If energy is high right at the cut, it's abrupt
        cut_quality = "clean" if pre_energy < 500 else "ABRUPT"
        if pre_energy >= 500:
            abrupt_cuts += 1
        print("  Break {:>2} at {:.2f}s: pre_energy={:.0f}, post_energy={:.0f} -> {}".format(
            i + 1, s_start, pre_energy, post_energy, cut_quality))

    print("\nSummary: {} total breaks, {} abrupt cuts, {} clean cuts".format(
        len(silence_regions), abrupt_cuts, len(silence_regions) - abrupt_cuts))
    print()
    return phrases, silence_regions, abrupt_cuts

print("=" * 70)
print("CORRECT TIMESTAMPS (API returned)")
print("=" * 70)
p1, s1, a1 = analyze_wav("output/test_phrases_correct_ts.wav")

print("=" * 70)
print("WRONG TIMESTAMPS (probe, no pauses)")
print("=" * 70)
p2, s2, a2 = analyze_wav("output/test_phrases_wrong_ts.wav")

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print("CORRECT: {} abrupt cuts out of {} breaks".format(a1, len(s1)))
print("WRONG:   {} abrupt cuts out of {} breaks".format(a2, len(s2)))
if a2 > a1:
    print("\n-> WRONG timestamps produce {} MORE abrupt cuts, confirming the bug.".format(a2 - a1))
elif a1 == a2:
    print("\n-> Same abrupt cut count. Check audio by ear for subtle differences.")
