"""
Test Solution A: Punctuation Injection
Instead of post-processing pauses, inject periods/commas BEFORE synthesis
so the model naturally drops pitch at phrase boundaries.
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, base64, wave, numpy as np

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

os.makedirs("output/solutions", exist_ok=True)

def save_b64_wav(data, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

# Test a short segment with problem words "Conference" and "itself"
# Original text (no modification)
ORIGINAL = "Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month but Apple's broader AI strategy is taking shape."

# Punctuation-injected (add periods at phrase boundaries)
PUNCTUATED = "Siri's AI overhaul. May have been the headline announcement. At Apple's Worldwide Developers Conference. Earlier this month. But Apple's broader AI strategy is taking shape."

# Test both
tests = {
    "original": {"text": ORIGINAL, "pause_after": [2, 8, 13, 16], "micro_pause_ms": 50},
    "punctuated": {"text": PUNCTUATED, "pause_after": None, "micro_pause_ms": 0},
    "punctuated_with_pause": {"text": PUNCTUATED, "pause_after": None, "micro_pause_ms": 0},
}

for name, params in tests.items():
    print("\n" + "=" * 60)
    print("TEST: {}".format(name.upper()))
    print("Text: {}".format(params["text"][:80]))
    print("=" * 60)
    
    payload = {
        "input": {
            "text": params["text"],
            "voice": "af_heart", "lang_code": "a", "speed": 1.0,
            "timestamps": True,
        }
    }
    if params.get("pause_after"):
        payload["input"]["pause_after"] = params["pause_after"]
        payload["input"]["micro_pause_ms"] = params["micro_pause_ms"]
    
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
    out = r.json().get("output", {})
    
    if "error" in out:
        print("ERROR: {}".format(out["error"]))
        continue
    
    ts = out.get("word_timestamps", [])
    print("Duration: {:.2f}s | Words: {}".format(out.get("duration_seconds", 0), len(ts)))
    
    save_b64_wav(out["audio_base64"], "output/solutions/{}.wav".format(name))
    print("Saved: output/solutions/{}.wav".format(name))
    
    # Print all word timestamps
    for i, w in enumerate(ts):
        marker = ""
        if params.get("pause_after") and i in params["pause_after"]:
            marker = " <-- PAUSE"
        print("  {:>3} {:>15} {:.3f}s - {:.3f}s{}".format(i, w["word"], w["start"], w["end"], marker))

# Now test with full text
print("\n" + "=" * 60)
print("FULL TEXT TEST: Punctuation-injected version")
print("=" * 60)

# Full original text
FULL_ORIGINAL = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things.
Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""

# Punctuation-injected: add periods at phrase boundaries
FULL_PUNCTUATED = """Siri's AI overhaul. May have been the headline announcement. At Apple's Worldwide Developers Conference. Earlier this month. But Apple's broader AI strategy is taking shape through a series of smaller features. Embedded across its software.
Rather than asking consumers to adopt the new AI-powered version of Siri. To get all the benefits that AI brings. The company is weaving AI into the apps. And services people already use. With a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills. Among friends, secure your passwords after data breaches. Automate tasks, and organize information. With less manual effort, among other things.
Individually, these features may not be as dramatic. As a Siri that finally understands your personal context. And can take action on your behalf. But combined, they showcase a vision for AI. That's less about chatting with a bot. And more about making Apple's software itself. Feel smarter and more capable.
Beyond Siri AI, here are the smaller AI features. In iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta. And will soon arrive in the public beta. Before iOS twenty-seven's general public release later this fall."""

r_full = requests.post(URL, headers=HEADERS, json={"input": {
    "text": FULL_PUNCTUATED,
    "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=300)
out_full = r_full.json().get("output", {})
print("Duration: {:.2f}s | Words: {}".format(out_full.get("duration_seconds", 0), len(out_full.get("word_timestamps", []))))
save_b64_wav(out_full["audio_base64"], "output/solutions/full_punctuated.wav")
print("Saved: output/solutions/full_punctuated.wav")
print("\nListen and compare: original paused vs punctuated versions")
print("The punctuated version should have natural prosodic breaks at each period.")
