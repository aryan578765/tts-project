import wave, numpy as np
import json
import sys

# Load word timestamps from step 1 (I don't have them easily accessible from a JSON, I'll run the API again)
import os, requests, base64

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
URL = "https://api.runpod.ai/v2/54td14oe86jexh/runsync"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software. Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things. Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable. Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

print("Fetching audio and timestamps...")
r = requests.post(URL, headers=HEADERS, json={"input": {
    "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out = r.json().get("output", {})
word_ts = out.get("word_timestamps", [])
audio_bytes = base64.b64decode(out["audio_base64"])
audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
# skip WAV header which is 44 bytes
audio = audio[22:] 
sr = 24000

# The words the user complained about:
target_words = ["Conference", "apps", "problems.", "bills", "information", "dramatic", "AI", "itself", "features"]

print("\nAnalyzing boundaries of target words...")
for i, wt in enumerate(word_ts):
    word = wt["word"]
    clean_word = word.strip(".,;")
    if clean_word in target_words or word in target_words:
        word_end_s = wt["end"]
        word_end = int(word_end_s * sr)
        
        if i + 1 < len(word_ts):
            next_start_s = word_ts[i + 1]["start"]
            next_start = int(next_start_s * sr)
            next_word = word_ts[i + 1]["word"]
        else:
            next_start = min(len(audio), word_end + int(sr*0.5))
            next_word = "[END]"
            
        gap_ms = (next_start_s - word_end_s) * 1000
        
        # Analyze energy at word_end and up to next_start
        # Does the energy actually drop at word_end?
        window = int(sr * 0.01) # 10ms
        energy_at_end = np.sqrt(np.mean(audio[word_end-window:word_end]**2))
        energy_after_end = np.sqrt(np.mean(audio[word_end:word_end+window]**2))
        
        best_pos = word_end
        best_energy = float("inf")
        search_end = min(len(audio)-window, max(word_end + int(sr*0.1), next_start))
        for pos in range(word_end, search_end, window//4):
            e = np.sqrt(np.mean(audio[pos:pos+window]**2))
            if e < best_energy:
                best_energy = e
                best_pos = pos + window//2
                
        trough_dist_ms = (best_pos - word_end) / sr * 1000
        
        print(f"\nWord: {word:12} -> Next: {next_word:12} | Gap: {gap_ms:.1f}ms")
        print(f"  Energy at FA word end: {energy_at_end:.0f}")
        print(f"  Energy just AFTER end: {energy_after_end:.0f}")
        print(f"  Lowest energy trough is {trough_dist_ms:.1f}ms PAST the FA word end (Energy: {best_energy:.0f})")
