"""
Generate All Audio — RunPod Serverless
=======================================
Reads test_texts.json and generates audio for all 13 test texts using a
RunPod serverless endpoint. Saves WAV files and logs generation times.

Usage:
    python generate_all_audio.py --model kokoro --endpoint-id 3z59kpduf0vkil --api-key YOUR_KEY
    python generate_all_audio.py --model cosyvoice --endpoint-id x3yzb0ve3mgx82 --api-key YOUR_KEY
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_TEXTS_PATH = PROJECT_ROOT / "test_texts" / "test_texts.json"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "audio_output"

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


def load_test_texts() -> list[dict]:
    """Load the 13 test texts from JSON."""
    if not TEST_TEXTS_PATH.exists():
        print(f"[ERROR] Test texts not found: {TEST_TEXTS_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(TEST_TEXTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def submit_job(endpoint_id: str, api_key: str, payload: dict, timeout: int = 300) -> dict:
    """Submit a job to RunPod serverless and wait for completion.
    
    Uses /runsync for synchronous execution. If the job doesn't complete
    within the sync timeout, falls back to polling /status.
    """
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}

    if resp.status_code == 401:
        return {"error": "Unauthorized – check your RunPod API key"}
    if resp.status_code == 404:
        return {"error": f"Endpoint {endpoint_id} not found"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

    status = data.get("status", "")

    # If completed synchronously
    if status == "COMPLETED":
        return data

    # If still in progress, poll
    job_id = data.get("id")
    if not job_id:
        return {"error": f"No job ID in response: {data}"}

    if status in ("IN_QUEUE", "IN_PROGRESS"):
        print(f"  [INFO] Job {job_id} is {status}. Polling status...")
        return poll_job(endpoint_id, api_key, job_id, timeout)

    # Job failed immediately
    if status == "FAILED":
        error = data.get("error", "Unknown error")
        return {"error": f"Job failed: {error}", "raw": data}

    return {"error": f"Unknown status '{status}': {data}"}


def poll_job(endpoint_id: str, api_key: str, job_id: str, timeout: int = 300) -> dict:
    """Poll a RunPod job until it completes or times out."""
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    poll_interval = 2.0

    while time.time() - start < timeout:
        time.sleep(poll_interval)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            data = resp.json()
        except Exception as exc:
            print(f"  [WARN] Poll error: {exc}")
            continue

        status = data.get("status", "")

        if status == "COMPLETED":
            return data
        elif status == "FAILED":
            error = data.get("error", "Unknown error")
            return {"error": f"Job failed: {error}", "raw": data}
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            elapsed = time.time() - start
            print(f"  [INFO] Still {status}... ({elapsed:.0f}s elapsed)")
            # Increase poll interval gradually
            poll_interval = min(poll_interval * 1.2, 10.0)
        else:
            return {"error": f"Unknown status '{status}'"}

    return {"error": f"Job timed out after {timeout}s"}


def extract_audio(result: dict) -> bytes | None:
    """Extract base64-encoded audio from RunPod job result."""
    # RunPod wraps handler output in "output" field
    output = result.get("output", result)

    if isinstance(output, dict):
        # Check for error from handler
        if "error" in output:
            print(f"  [ERROR] Handler error: {output['error']}", file=sys.stderr)
            return None

        # Look for audio_base64 field (our handler returns this)
        for key in ("audio_base64", "audio", "audio_data"):
            if key in output and isinstance(output[key], str):
                try:
                    return base64.b64decode(output[key])
                except Exception:
                    continue

    return None


def get_model_payload(model: str, text: str) -> dict:
    """Build the RunPod job payload based on model type."""
    if model == "kokoro":
        return {
            "input": {
                "text": text,
                "voice": "af_heart",
                "speed": 1.0,
                "lang_code": "a",
                "ssml": False,
            }
        }
    elif model == "cosyvoice":
        return {
            "input": {
                "text": text,
                "voice": "英文女",
                "speed": 1.0,
                "language": "en",
                "mode": "sft",
            }
        }
    else:
        # Generic payload
        return {
            "input": {
                "text": text,
            }
        }


def save_audio(audio_bytes: bytes, output_path: Path) -> bool:
    """Save audio bytes to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return True
    except IOError as exc:
        print(f"  [ERROR] Failed to save: {exc}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio for all 13 TTS test texts via RunPod Serverless"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (kokoro or cosyvoice)"
    )
    parser.add_argument(
        "--endpoint-id", required=True, help="RunPod endpoint ID"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--format", default="wav", choices=["wav", "mp3", "ogg", "flac"],
        help="Audio format (default: wav)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Job timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: audio_output/{model}/)"
    )
    parser.add_argument(
        "--tests", type=str, default=None,
        help="Comma-separated test IDs to run (e.g., test_1,test_3). Default: all"
    )

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[ERROR] RunPod API key required. Use --api-key or set RUNPOD_API_KEY env var",
              file=sys.stderr)
        sys.exit(1)

    # Load test texts
    test_texts = load_test_texts()

    # Filter tests if specified
    if args.tests:
        selected = set(args.tests.split(","))
        test_texts = [t for t in test_texts if t["id"] in selected]
        if not test_texts:
            print(f"[ERROR] No matching tests found for: {args.tests}", file=sys.stderr)
            sys.exit(1)

    # Prepare output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = AUDIO_OUTPUT_DIR / args.model

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Audio Generation: {args.model}")
    print(f"Endpoint ID:      {args.endpoint_id}")
    print(f"Output:           {output_dir}")
    print(f"Format:           {args.format}")
    print(f"Tests:            {len(test_texts)}")
    print(f"{'=' * 60}\n")

    # Generation log
    log_entries = []
    total_gen_time = 0.0
    success_count = 0
    fail_count = 0

    for entry in test_texts:
        test_id = entry["id"]
        title = entry["title"]
        text = entry["text"]

        print(f"[{test_id}] {title}")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        # Build model-specific payload
        payload = get_model_payload(args.model, text)

        # Submit job
        start = time.perf_counter()
        result = submit_job(args.endpoint_id, api_key, payload, args.timeout)
        gen_time = time.perf_counter() - start

        # Check for errors
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            if "raw" in result:
                print(f"    Raw: {json.dumps(result['raw'], indent=2)[:500]}")
            fail_count += 1
            total_gen_time += gen_time
            log_entries.append({
                "test_id": test_id,
                "title": title,
                "text_length_chars": len(text),
                "generation_time_s": round(gen_time, 4),
                "success": False,
                "error": result["error"],
            })
            print()
            continue

        # Extract audio
        audio_bytes = extract_audio(result)

        if audio_bytes:
            filename = f"{test_id}.{args.format}"
            output_path = output_dir / filename

            if save_audio(audio_bytes, output_path):
                file_size_kb = len(audio_bytes) / 1024
                duration = result.get("output", {}).get("duration_seconds", "?")
                print(f"  ✓ Saved: {output_path} ({file_size_kb:.1f} KB)")
                print(f"  ⏱ Generation time: {gen_time:.3f}s | Audio duration: {duration}s")
                success_count += 1
            else:
                print(f"  ✗ Failed to save")
                fail_count += 1
        else:
            print(f"  ✗ No audio in response (time: {gen_time:.3f}s)")
            print(f"    Response keys: {list(result.keys())}")
            if "output" in result:
                print(f"    Output: {json.dumps(result['output'], indent=2)[:300]}")
            fail_count += 1

        total_gen_time += gen_time

        log_entries.append({
            "test_id": test_id,
            "title": title,
            "text_length_chars": len(text),
            "generation_time_s": round(gen_time, 4),
            "success": audio_bytes is not None,
            "file_size_bytes": len(audio_bytes) if audio_bytes else 0,
            "duration_seconds": result.get("output", {}).get("duration_seconds"),
        })

        print()

    # Save generation log
    log_path = output_dir / "generation_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "endpoint_id": args.endpoint_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_generation_time_s": round(total_gen_time, 4),
                "success_count": success_count,
                "fail_count": fail_count,
                "entries": log_entries,
            },
            f,
            indent=2,
        )

    # Summary
    print(f"{'=' * 60}")
    print(f"GENERATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:              {args.model}")
    print(f"  Succeeded:          {success_count}/{len(test_texts)}")
    print(f"  Failed:             {fail_count}/{len(test_texts)}")
    print(f"  Total gen time:     {total_gen_time:.3f}s")
    print(f"  Output directory:   {output_dir}")
    print(f"  Generation log:     {log_path}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
