"""Test CJK timestamps support (Chinese + Japanese + English control)."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = "54td14oe86jexh"
URL = "https://api.runpod.ai/v2/{}/runsync".format(ENDPOINT)
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

tests = [
    {"name": "Chinese", "text": "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u60f3\u51fa\u53bb\u6563\u6b65\u3002", "lang": "z", "voice": "zf_xiaobei"},
    {"name": "Japanese", "text": "\u3053\u3093\u306b\u3061\u306f\u3001\u4eca\u65e5\u306f\u3044\u3044\u5929\u6c17\u3067\u3059\u306d\u3002", "lang": "j", "voice": "jf_alpha"},
    {"name": "English", "text": "Hello, this is a test of timestamps.", "lang": "a", "voice": "af_heart"},
]

for t in tests:
    print("\n=== {} ===".format(t["name"]))
    print("Text: {}".format(t["text"]))
    payload = {
        "input": {
            "text": t["text"],
            "voice": t["voice"],
            "lang_code": t["lang"],
            "speed": 1.0,
            "timestamps": True,
            "word_boundaries": True,
        }
    }
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        data = r.json()
        if "output" in data:
            out = data["output"]
            dur = out.get("duration_seconds", "?")
            rtf = out.get("rtf", "?")
            print("Duration: {}s | RTF: {}".format(dur, rtf))
            if "word_timestamps" in out:
                wts = out["word_timestamps"]
                print("Timestamps ({} words):".format(len(wts)))
                for w in wts:
                    print("  {:15s}  {:.3f}s - {:.3f}s".format(w["word"], w["start"], w["end"]))
            else:
                print("NO TIMESTAMPS RETURNED")
            if "word_boundaries" in out:
                bds = out["word_boundaries"]
                print("Boundaries ({}):".format(len(bds)))
                for b in bds[:5]:
                    print("  {:25s}  {:6.1f}ms  {}".format(b["pair"], b["gap_ms"], b.get("status", "?")))
            else:
                print("NO BOUNDARIES RETURNED")
        else:
            print("Error: " + json.dumps(data, indent=2)[:500])
    except Exception as e:
        print("Request failed: {}".format(e))

print("\n=== ALL TESTS DONE ===")
