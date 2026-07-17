"""Test all features Mariam requested: threshold 65ms, pause_after, smart+timestamps."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

TEXT = "I chose what felt right. It's not dramatic. It's just calm confidence."

tests = [
    {
        "name": "1. Smart pause 50ms threshold (current default)",
        "payload": {
            "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
            "timestamps": True, "word_boundaries": True,
            "micro_pause_ms": 10, "smart_pause": True
        }
    },
    {
        "name": "2. Smart pause 65ms threshold (Mariam's request)",
        "payload": {
            "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
            "timestamps": True, "word_boundaries": True,
            "micro_pause_ms": 10, "smart_pause": True,
            "smart_pause_threshold_ms": 65
        }
    },
    {
        "name": "3. Targeted pause_after [1, 3, 7] (pause wherever you want)",
        "payload": {
            "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
            "timestamps": True, "word_boundaries": True,
            "micro_pause_ms": 15, "pause_after": [1, 3, 7]
        }
    },
    {
        "name": "4. Smart pause + timestamps together (Mariam's Q3)",
        "payload": {
            "text": TEXT, "voice": "af_heart", "lang_code": "a", "speed": 1.0,
            "timestamps": True, "word_boundaries": True,
            "micro_pause_ms": 10, "smart_pause": True,
            "smart_pause_threshold_ms": 65
        }
    },
]

for t in tests:
    print("\n" + "=" * 60)
    print("TEST: {}".format(t["name"]))
    print("=" * 60)
    payload = {"input": t["payload"]}
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        data = r.json()
        if "output" in data:
            out = data["output"]
            if "error" in out:
                print("ERROR: {}".format(out["error"]))
                continue
            dur = out.get("duration_seconds", "?")
            rtf = out.get("rtf", "?")
            print("Duration: {}s | RTF: {}".format(dur, rtf))

            if "word_timestamps" in out:
                wts = out["word_timestamps"]
                print("Timestamps ({} words):".format(len(wts)))
                for w in wts:
                    print("  {:15s}  {:.3f}s - {:.3f}s".format(w["word"], w["start"], w["end"]))
            else:
                print("NO TIMESTAMPS")

            if "word_boundaries" in out:
                bds = out["word_boundaries"]
                tight_count = sum(1 for b in bds if b.get("status") == "tight")
                coart_count = sum(1 for b in bds if b.get("status") == "coarticulated")
                clean_count = sum(1 for b in bds if b.get("status") == "clean")
                print("Boundaries ({} total): {} clean, {} tight, {} coarticulated".format(
                    len(bds), clean_count, tight_count, coart_count))
                for b in bds:
                    print("  {:25s}  {:6.1f}ms  {}  sep={}".format(
                        b["pair"], b["gap_ms"], b.get("status", "?"), b.get("can_separate", "?")))
            else:
                print("NO BOUNDARIES")

            # Save audio for comparison
            if "audio_base64" in out:
                fname = "test_mariam_{}.wav".format(t["name"][0])
                with open("output/{}".format(fname), "wb") as f:
                    f.write(base64.b64decode(out["audio_base64"]))
                print("Audio saved: output/{}".format(fname))
        else:
            print("Error: {}".format(json.dumps(data, indent=2)[:500]))
    except Exception as e:
        print("Request failed: {}".format(e))

print("\n" + "=" * 60)
print("ALL TESTS DONE")
print("=" * 60)
