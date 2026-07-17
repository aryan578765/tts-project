"""Analyze energy at ALL phrase boundaries in the punctuated audio."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import wave, numpy as np

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

def energy_at(audio, sr, time_s, window_ms=10):
    center = int(time_s * sr)
    half = int(sr * window_ms / 2000)
    s, e = max(0, center - half), min(len(audio), center + half)
    if e <= s: return 0
    return np.sqrt(np.mean(audio[s:e] ** 2))

# Analyze the punctuated short version vs original
print("=" * 70)
print("COMPARISON: Original (paused) vs Punctuated")
print("=" * 70)

for name, path in [("original_paused", "output/solutions/original.wav"),
                    ("punctuated", "output/solutions/punctuated.wav")]:
    sr, audio = read_wav(path)
    print("\n--- {} ({:.2f}s) ---".format(name, len(audio)/sr))
    
    # Phrase boundary words and their end times (from test output)
    if name == "original_paused":
        boundaries = [
            (2, "overhaul", 1.681), (8, "announcement", 3.572),
            (13, "Conference", 6.303), (16, "month", 7.333),
        ]
    else:
        boundaries = [
            (2, "overhaul.", 1.621), (8, "announcement.", 3.742),
            (13, "Conference.", 6.744), (16, "month.", 8.105),
        ]
    
    for idx, word, end_t in boundaries:
        # Energy at word end and in the following gap
        e_end = energy_at(audio, sr, end_t)
        e_p10 = energy_at(audio, sr, end_t + 0.01)
        e_p30 = energy_at(audio, sr, end_t + 0.03)
        e_p50 = energy_at(audio, sr, end_t + 0.05)
        e_p100 = energy_at(audio, sr, end_t + 0.1)
        e_p200 = energy_at(audio, sr, end_t + 0.2)
        e_p300 = energy_at(audio, sr, end_t + 0.3)
        
        print("  {:>15} end={:.3f}s: end={:.0f} +10ms={:.0f} +30ms={:.0f} +50ms={:.0f} +100ms={:.0f} +200ms={:.0f} +300ms={:.0f}".format(
            word, end_t, e_end, e_p10, e_p30, e_p50, e_p100, e_p200, e_p300))

# Now analyze the FULL punctuated version
print("\n" + "=" * 70)
print("FULL PUNCTUATED AUDIO: Silence detection at natural breaks")
print("=" * 70)

sr, audio = read_wav("output/solutions/full_punctuated.wav")
peak = float(np.max(np.abs(audio)))
silence_threshold = peak * 0.02

# Find all silence regions > 100ms
window = int(sr * 0.02)  # 20ms windows
silence_regions = []
in_silence = False
silence_start = 0

for i in range(0, len(audio) - window, window):
    t = i / sr
    rms = np.sqrt(np.mean(audio[i:i+window] ** 2))
    is_silent = rms < silence_threshold
    
    if is_silent and not in_silence:
        silence_start = t
        in_silence = True
    elif not is_silent and in_silence:
        dur = t - silence_start
        if dur >= 0.1:  # > 100ms
            silence_regions.append((silence_start, t, dur))
        in_silence = False

if in_silence:
    dur = len(audio)/sr - silence_start
    if dur >= 0.1:
        silence_regions.append((silence_start, len(audio)/sr, dur))

print("Found {} natural silence regions (>100ms):".format(len(silence_regions)))
for i, (s, e, d) in enumerate(silence_regions):
    # Check energy at center
    center = (s + e) / 2
    ec = energy_at(audio, sr, center)
    print("  {:>2}. {:.2f}s - {:.2f}s ({:.0f}ms) energy={:.0f}".format(i+1, s, e, d*1000, ec))

print("\nTotal natural pauses: {}".format(len(silence_regions)))
print("Compare: we need 27 phrase cuts, found {} natural breaks".format(len(silence_regions)))
