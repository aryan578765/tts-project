"""Generate micro-pause audio: every word separated by <break time="10ms"/>."""
import requests
import json
import base64
import time
import os

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not API_KEY:
    print("ERROR: Set RUNPOD_API_KEY environment variable")
    exit(1)
ENDPOINT_ID = "3z59kpduf0vkil"

# Original sentence
original = "I didn't rush this decision. I took my time, listened carefully, and chose what felt right."

# Insert <break time="10ms"/> between every word
words = original.split()
ssml_text = ' <break time="10ms"/> '.join(words)

print(f"SSML text ({len(ssml_text)} chars):")
print(ssml_text[:200])

payload = {
    "input": {
        "text": ssml_text,
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "a",
        "ssml": True,
    }
}

print(f"\nSubmitting job to Kokoro endpoint {ENDPOINT_ID}...")
url = f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/runsync"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

resp = requests.post(url, json=payload, headers=headers, timeout=300)
print(f"Status code: {resp.status_code}")

data = resp.json()
status = data.get("status", "")
print(f"Job status: {status}")

# Poll if needed
if status in ("IN_QUEUE", "IN_PROGRESS"):
    job_id = data.get("id")
    print(f"Polling job {job_id}...")
    for i in range(60):
        time.sleep(3)
        poll_resp = requests.get(
            f"{RUNPOD_BASE_URL}/{ENDPOINT_ID}/status/{job_id}",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        poll_data = poll_resp.json()
        poll_status = poll_data.get("status", "")
        print(f"  Attempt {i+1}: {poll_status}")
        if poll_status in ("COMPLETED", "FAILED"):
            data = poll_data
            break

if data.get("status") == "COMPLETED":
    output = data.get("output", {})
    audio_b64 = output.get("audio_base64") or output.get("audio") or output.get("audio_data", "")
    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        out_path = os.path.join("audio_output", "kokoro", "ssml_tests", "micro_pause_every_word.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print(f"\nSaved {len(audio_bytes)} bytes to {out_path}")
        duration = output.get("duration_seconds", output.get("duration", "unknown"))
        print(f"Duration: {duration}s")
    else:
        print(f"No audio in output keys: {list(output.keys())}")
        print(f"Full output: {json.dumps(output)[:500]}")
else:
    error = data.get("error", "unknown")
    print(f"Job failed: {error}")
    print(f"Full response: {json.dumps(data)[:500]}")
