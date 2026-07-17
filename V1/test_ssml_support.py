"""Quick test: Does kokoro-onnx support SSML break tags?"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

# Test 1: Plain text (baseline)
print("=" * 60)
print("TEST 1: Plain text (baseline)")
r1 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": "I chose what felt right.",
    "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=120)
out1 = r1.json().get("output", {})
print("Duration: {}s".format(out1.get("duration_seconds")))
ts1 = out1.get("word_timestamps", [])
for w in ts1:
    print("  {} {:.3f}-{:.3f}".format(w["word"], w["start"], w["end"]))

# Test 2: SSML with break tags
print("\n" + "=" * 60)
print("TEST 2: SSML break tags")
ssml_text = 'I chose <break time="200ms"/> what felt <break time="200ms"/> right.'
r2 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": ssml_text,
    "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=120)
out2 = r2.json().get("output", {})
print("Duration: {}s".format(out2.get("duration_seconds")))
if "error" in out2:
    print("ERROR: {}".format(out2["error"]))
else:
    ts2 = out2.get("word_timestamps", [])
    for w in ts2:
        print("  {} {:.3f}-{:.3f}".format(w["word"], w["start"], w["end"]))
    # Save for listening
    if "audio_base64" in out2:
        with open("output/test_ssml_break.wav", "wb") as f:
            f.write(base64.b64decode(out2["audio_base64"]))
        print("Saved: output/test_ssml_break.wav")

# Test 3: SSML with phoneme-style text
print("\n" + "=" * 60)
print("TEST 3: Sentence-level break")
ssml_text3 = 'I chose what felt right. <break time="500ms"/> It is not dramatic.'
r3 = requests.post(URL, headers=HEADERS, json={"input": {
    "text": ssml_text3,
    "voice": "af_heart", "lang_code": "a", "speed": 1.0,
    "timestamps": True
}}, timeout=120)
out3 = r3.json().get("output", {})
print("Duration: {}s".format(out3.get("duration_seconds")))
if "error" in out3:
    print("ERROR: {}".format(out3["error"]))
else:
    ts3 = out3.get("word_timestamps", [])
    for w in ts3:
        print("  {} {:.3f}-{:.3f}".format(w["word"], w["start"], w["end"]))
    if "audio_base64" in out3:
        with open("output/test_ssml_sentence.wav", "wb") as f:
            f.write(base64.b64decode(out3["audio_base64"]))
        print("Saved: output/test_ssml_sentence.wav")

# Compare durations
print("\n" + "=" * 60)
print("COMPARISON")
print("Plain: {}s".format(out1.get("duration_seconds")))
print("SSML word-level: {}s".format(out2.get("duration_seconds")))
print("SSML sentence: {}s".format(out3.get("duration_seconds")))
d1 = out1.get("duration_seconds", 0)
d2 = out2.get("duration_seconds", 0)
d3 = out3.get("duration_seconds", 0)
if d2 > d1 + 0.1:
    print("=> SSML break tags ADD duration - they WORK!")
elif d2 == d1 or abs(d2 - d1) < 0.1:
    print("=> SSML break tags have NO effect - they DON'T work with kokoro-onnx")
