"""Analyze SSML test WAV files - check if model generates pauses or speaks tags."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import wave, numpy as np

def analyze_wav(path, label):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    
    duration = len(audio) / sr
    print("\n" + "=" * 60)
    print("FILE: {} ({})".format(os.path.basename(path), label))
    print("Duration: {:.2f}s | Samples: {} | Rate: {}".format(duration, len(audio), sr))
    
    # Analyze energy over time in 50ms windows
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    peak = float(np.max(np.abs(audio)))
    silence_threshold = peak * 0.02  # 2% of peak
    
    print("\nEnergy timeline (50ms windows):")
    print("{:>6}  {:>10}  {:>8}".format("time", "rms", "status"))
    
    silence_regions = []
    in_silence = False
    silence_start = 0
    
    for i in range(0, len(audio) - window_samples, window_samples):
        t = i / sr
        chunk = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        
        is_silent = rms < silence_threshold
        
        if is_silent:
            if not in_silence:
                silence_start = t
                in_silence = True
        else:
            if in_silence:
                silence_dur = t - silence_start
                if silence_dur >= 0.05:  # >= 50ms
                    silence_regions.append((silence_start, t, silence_dur))
                in_silence = False
        
        # Print every 200ms
        if int(t * 1000) % 200 == 0:
            status = "SILENCE" if is_silent else "speech"
            bar = "#" * min(int(rms / max(peak, 1) * 40), 40)
            print("{:>5.1f}s  {:>10.0f}  {:>8}  {}".format(t, rms, status, bar))
    
    # Handle trailing silence
    if in_silence:
        silence_dur = duration - silence_start
        if silence_dur >= 0.05:
            silence_regions.append((silence_start, duration, silence_dur))
    
    print("\nSilence regions (>= 50ms):")
    for s, e, d in silence_regions:
        print("  {:.2f}s - {:.2f}s ({:.0f}ms)".format(s, e, d * 1000))
    
    print("\nTotal silence: {:.2f}s ({:.0f}%)".format(
        sum(d for _, _, d in silence_regions),
        sum(d for _, _, d in silence_regions) / max(duration, 0.001) * 100
    ))
    print("Total speech: {:.2f}s ({:.0f}%)".format(
        duration - sum(d for _, _, d in silence_regions),
        (duration - sum(d for _, _, d in silence_regions)) / max(duration, 0.001) * 100
    ))
    
    return silence_regions

# Analyze SSML word-level break test
s1 = analyze_wav("output/test_ssml_break.wav", 
    'SSML: "I chose <break 200ms/> what felt <break 200ms/> right."')

# Analyze SSML sentence-level break test  
s2 = analyze_wav("output/test_ssml_sentence.wav",
    'SSML: "I chose what felt right. <break 500ms/> It is not dramatic."')

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print("Plain text duration: 1.536s")
print("SSML word breaks:    {:.2f}s (expected ~1.9s with 2x200ms)".format(7.019))
print("SSML sentence break: {:.2f}s (expected ~3.5s with 1x500ms)".format(5.845))

extra1 = 7.019 - 1.536
extra2 = 5.845 - (1.536 + 1.5)  # ~1.5s for "It is not dramatic"
print("\nExtra duration from SSML word breaks: {:.2f}s (expected 0.4s)".format(extra1))
print("Extra duration from SSML sentence break: {:.2f}s (expected 0.5s)".format(extra2))

if extra1 > 2.0:
    print("\n=> Model is likely SPEAKING the SSML tags, not just pausing!")
else:
    print("\n=> Model generates proper pauses from SSML tags.")
