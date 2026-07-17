"""
DEEP AUDIO ANALYSIS
===================
Compares the RAW unmodified audio vs the modified audio at every 
problematic word boundary to find exactly what's being destroyed.
"""
import os, requests, json, base64, wave, struct, numpy as np

API_KEY = os.environ.get("RUNPOD_API_KEY", "rpa_FM53IJS8SMMWARXINSDL5D9E706CT4GUYJAZL9DLhn48fy")
URL = "https://api.runpod.ai/v2/54td14oe86jexh/runsync"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software. Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things. Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable. Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

PAUSE_AFTER = [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107, 115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]
PROBLEM_WORDS = {"Conference", "apps", "bills", "information", "dramatic", "itself", "features"}

def decode_wav(b64):
    raw = base64.b64decode(b64)
    buf = wave.open(__import__('io').BytesIO(raw), 'rb')
    sr = buf.getframerate()
    frames = buf.readframes(buf.getnframes())
    audio = np.array(struct.unpack(f'{len(frames)//2}h', frames), dtype=np.float32)
    return audio, sr

def energy_profile(audio, sr, start_s, end_s, window_ms=5):
    """Return time, energy arrays for a segment."""
    ws = int(sr * window_ms / 1000)
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    seg = audio[s:e]
    times = []
    energies = []
    for i in range(0, len(seg) - ws, ws):
        times.append(start_s + (i + ws//2) / sr)
        energies.append(np.sqrt(np.mean(seg[i:i+ws]**2)))
    return times, energies

print("=" * 70)
print("STEP 1: Fetch RAW audio (NO micro-pauses, NO modifications)")
print("=" * 70)
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=120)
raw_out = r.json()["output"]
raw_audio, sr = decode_wav(raw_out["audio_base64"])
raw_ts = raw_out["word_timestamps"]
print(f"Raw audio: {len(raw_audio)/sr:.2f}s, {len(raw_ts)} words")

print("\n" + "=" * 70)
print("STEP 2: Fetch MODIFIED audio (50ms micro-pauses)")
print("=" * 70)
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True, "micro_pause_ms": 50, "pause_after": PAUSE_AFTER
}}, timeout=120)
mod_out = r.json()["output"]
mod_audio, _ = decode_wav(mod_out["audio_base64"])
mod_ts = mod_out["word_timestamps"]
mod_cuts = mod_out.get("phrase_cut_points", [])
print(f"Modified audio: {len(mod_audio)/sr:.2f}s, {len(mod_ts)} words, {len(mod_cuts)} cuts")

print("\n" + "=" * 70)
print("STEP 3: DEEP ANALYSIS of every problematic word boundary")
print("=" * 70)

for pa_idx in PAUSE_AFTER:
    if pa_idx >= len(raw_ts):
        continue
    word = raw_ts[pa_idx]["word"]
    clean_word = word.strip(".,;:!?'\"")
    if clean_word not in PROBLEM_WORDS:
        continue
    
    # --- RAW AUDIO ANALYSIS ---
    word_end_s = raw_ts[pa_idx]["end"]
    word_start_s = raw_ts[pa_idx]["start"]
    if pa_idx + 1 < len(raw_ts):
        next_start_s = raw_ts[pa_idx + 1]["start"]
        next_word = raw_ts[pa_idx + 1]["word"]
    else:
        next_start_s = word_end_s + 0.5
        next_word = "[END]"
    
    gap_ms = (next_start_s - word_end_s) * 1000
    
    # Energy at word end in RAW audio
    word_end_sample = int(word_end_s * sr)
    ws = int(sr * 0.005)  # 5ms window
    
    # Check energy in 10ms windows from word_end forward for 200ms
    print(f"\n{'='*60}")
    print(f"WORD: '{word}' (idx={pa_idx}) -> NEXT: '{next_word}'")
    print(f"  Word span: {word_start_s:.3f}s - {word_end_s:.3f}s")
    print(f"  Gap to next word: {gap_ms:.1f}ms")
    print(f"  --- RAW audio energy profile (5ms windows after FA word end) ---")
    
    # Show energy for 200ms after word end
    for offset_ms in range(0, 200, 10):
        pos = word_end_sample + int(sr * offset_ms / 1000)
        if pos + ws > len(raw_audio):
            break
        e = np.sqrt(np.mean(raw_audio[pos:pos+ws]**2))
        bar = "#" * min(50, int(e / 50))
        marker = ""
        if offset_ms == 0:
            marker = " <-- FA word end"
        if abs(word_end_s + offset_ms/1000 - next_start_s) < 0.01:
            marker = " <-- NEXT word start"
        print(f"    +{offset_ms:3d}ms: energy={e:7.1f} {bar}{marker}")
    
    # Find the ACTUAL silence point in raw audio (where energy drops below 50)
    actual_silence_ms = None
    for offset_ms in range(0, 500, 5):
        pos = word_end_sample + int(sr * offset_ms / 1000)
        if pos + ws > len(raw_audio):
            break
        e = np.sqrt(np.mean(raw_audio[pos:pos+ws]**2))
        if e < 50:
            actual_silence_ms = offset_ms
            break
    
    if actual_silence_ms is not None:
        print(f"  >> Audio actually reaches silence at +{actual_silence_ms}ms after FA word end")
    else:
        print(f"  >> Audio NEVER reaches silence within 500ms after FA word end!")
    
    # --- MODIFIED AUDIO ANALYSIS ---
    # Find the cut point for this word
    cut_info = None
    for c in mod_cuts:
        if c.get("after_word_idx") == pa_idx:
            cut_info = c
            break
    
    if cut_info:
        cut_time = cut_info["time"]
        cut_sample = int(cut_time * sr)
        
        # Where does the fade-out start? (5ms before cut)
        fade_start_sample = cut_sample - int(sr * 0.005)
        fade_start_ms_after_end = (fade_start_sample / sr - word_end_s) * 1000
        
        print(f"\n  --- MODIFIED audio analysis ---")
        print(f"  Cut point: {cut_time:.4f}s ({(cut_time - word_end_s)*1000:.1f}ms after FA word end)")
        print(f"  Fade-out starts: {fade_start_ms_after_end:.1f}ms after FA word end")
        
        # Check: is there still speech energy at the fade-out point?
        if fade_start_sample > 0 and fade_start_sample + ws < len(mod_audio):
            e_at_fade = np.sqrt(np.mean(mod_audio[fade_start_sample:fade_start_sample+ws]**2))
            print(f"  Energy at fade-out start: {e_at_fade:.1f}")
            if e_at_fade > 200:
                print(f"  *** PROBLEM: Fade is cutting through ACTIVE SPEECH (energy={e_at_fade:.0f})! ***")
            elif e_at_fade > 50:
                print(f"  ** WARNING: Fade is cutting through low-level audio (energy={e_at_fade:.0f}) **")
            else:
                print(f"  OK: Fade starts in near-silence")
        
        # Check energy profile 50ms before and after cut in modified audio
        print(f"  --- Modified audio energy around cut point ---")
        for offset_ms in range(-50, 60, 10):
            pos = cut_sample + int(sr * offset_ms / 1000)
            if pos < 0 or pos + ws > len(mod_audio):
                continue
            e = np.sqrt(np.mean(mod_audio[pos:pos+ws]**2))
            bar = "#" * min(50, int(e / 50))
            marker = ""
            if offset_ms == 0:
                marker = " <-- CUT POINT"
            print(f"    {offset_ms:+4d}ms: energy={e:7.1f} {bar}{marker}")

        # CRITICAL: Compare raw vs modified audio in the 100ms before cut
        print(f"\n  --- RAW vs MODIFIED comparison (last 100ms before cut) ---")
        for offset_ms in range(-100, 10, 10):
            raw_pos = int((word_end_s + (cut_time - word_end_s) + offset_ms/1000) * sr)
            mod_pos = cut_sample + int(sr * offset_ms / 1000)
            if raw_pos < 0 or raw_pos + ws > len(raw_audio):
                continue
            if mod_pos < 0 or mod_pos + ws > len(mod_audio):
                continue
            raw_e = np.sqrt(np.mean(raw_audio[raw_pos:raw_pos+ws]**2))
            mod_e = np.sqrt(np.mean(mod_audio[mod_pos:mod_pos+ws]**2))
            diff_pct = ((mod_e - raw_e) / max(raw_e, 1)) * 100
            flag = ""
            if abs(diff_pct) > 20 and raw_e > 100:
                flag = f" *** MODIFIED BY {diff_pct:+.0f}% ***"
            print(f"    {offset_ms:+4d}ms: RAW={raw_e:7.1f}  MOD={mod_e:7.1f}  diff={diff_pct:+.1f}%{flag}")

print("\n" + "=" * 70)
print("STEP 4: Check if NATURAL GAPS exist (could we cut WITHOUT any modification?)")
print("=" * 70)

for pa_idx in PAUSE_AFTER:
    if pa_idx >= len(raw_ts):
        continue
    word = raw_ts[pa_idx]["word"]
    word_end = raw_ts[pa_idx]["end"]
    if pa_idx + 1 < len(raw_ts):
        next_start = raw_ts[pa_idx + 1]["start"]
    else:
        continue
    
    gap_ms = (next_start - word_end) * 1000
    
    # Find minimum energy in the gap (in raw audio)
    gap_start = int(word_end * sr)
    gap_end = int(next_start * sr)
    if gap_end <= gap_start or gap_end > len(raw_audio):
        continue
    
    min_energy = float("inf")
    for pos in range(gap_start, min(gap_end, len(raw_audio) - ws), ws):
        e = np.sqrt(np.mean(raw_audio[pos:pos+ws]**2))
        min_energy = min(min_energy, e)
    
    clean_word = word.strip(".,;:!?'\"")
    marker = " <<<" if clean_word in PROBLEM_WORDS else ""
    can_cut = "YES" if min_energy < 100 else "NO"
    print(f"  {word:15s} gap={gap_ms:6.1f}ms  min_energy={min_energy:7.1f}  natural_cut={can_cut}{marker}")
